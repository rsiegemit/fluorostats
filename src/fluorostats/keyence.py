"""Keyence BZ-X reader — assemble a z-stack from a folder of per-slice OME-TIFFs
and parse the ``.gci`` project metadata.

Keyence BZ-X exports a z-stack as one OME-TIFF per slice, named
``Image _Z###_CHF.bz.ome.tif`` (all acquired channels stored inside each file),
alongside 8-bit ``_Overlay`` / ``Preview`` derivatives and an ``Image .gci``
metadata archive (a zip of XML). Only the ``CHF`` files + ``.gci`` carry the raw
16-bit multi-channel data and acquisition parameters; the rest are derivable.

Public API:
    parse_gci(gci_path) -> dict
    load_keyence_stack(folder, downsample=1, channel=None) -> (array, meta)

``load_keyence_stack`` returns ``(C, Z, Y, X)`` (or ``(Z, Y, X)`` when ``channel``
is given) plus a metadata dict with pixel size, z-step, channel names/exposures.

Note: Keyence stores ``System.Double`` values as the raw little-endian Int64 bit
pattern of the IEEE-754 double, and lengths in nanometres; both are decoded here.
"""
from __future__ import annotations

import glob
import re
import struct
import warnings
import zipfile
from pathlib import Path

import numpy as np

__all__ = ["parse_gci", "load_keyence_stack", "is_keyence_folder"]


def is_keyence_folder(folder) -> bool:
    """True if ``folder`` looks like a Keyence BZ-X export (has ``*_CHF*`` slices
    or a ``.gci`` archive). Used by ``fluorostats.io.load_volume`` to auto-detect
    a Keyence folder passed in place of a single file."""
    folder = Path(folder)
    if not folder.is_dir():
        return False
    return bool(next(folder.glob("*_CHF*.tif"), None) or next(folder.glob("*.gci"), None))


def _int64_to_double(value) -> float | None:
    """Reinterpret a Keyence ``System.Double`` (stored as an Int64 bit pattern)."""
    try:
        return struct.unpack("<d", struct.pack("<q", int(value)))[0]
    except (ValueError, struct.error, TypeError):
        return None


def parse_gci(gci_path) -> dict:
    """Parse a Keyence ``.gci`` archive into an acquisition-metadata dict.

    Returns keys: ``px_um`` (µm/pixel), ``z_step_um``, ``n_slices``,
    ``objective``, ``channels`` ({index: {name, exposure_ms}}), plus
    ``z_step_note`` recording the unit assumption.
    """
    gci_path = Path(gci_path)
    meta: dict = {
        "px_um": None, "z_step_um": None, "n_slices": None, "objective": None,
        "channels": {},
        "z_step_note": "Stack/Pitch read as 0.1-µm units; confirm against the acquisition GUI.",
    }
    with zipfile.ZipFile(gci_path) as z:
        for name in z.namelist():
            handler = _gci_handler(name)
            if handler is not None:
                meta = handler(name, z.read(name).decode("utf-8", "ignore"), meta)
    if meta["z_step_um"] is not None:
        warnings.warn(f"{gci_path.name}: Stack/Pitch read as 0.1-µm units (z step "
                      f"{meta['z_step_um']} µm); confirm against the acquisition GUI.",
                      RuntimeWarning, stacklevel=2)
    return meta


# Each .gci entry handler returns ``meta`` updated with what the entry holds, as a new dict.

def _gci_pixel_size(name: str, text: str, meta: dict) -> dict:
    m = re.search(r'Calibration Type="System.Double">(-?\d+)', text)
    nm = _int64_to_double(m.group(1)) if m else None
    return {**meta, "px_um": round(nm / 1000.0, 5)} if nm else meta      # nm -> µm


def _gci_objective(name: str, text: str, meta: dict) -> dict:
    m = re.search(r"<LensName[^>]*>([^<]+)", text)
    return {**meta, "objective": m.group(1)} if m else meta


def _gci_stack(name: str, text: str, meta: dict) -> dict:
    tn = re.search(r"TotalNumber[^>]*>(\d+)", text)
    pitch = re.search(r"Pitch[^>]*>(\d+)", text)
    return {**meta,
            **({"n_slices": int(tn.group(1))} if tn else {}),
            **({"z_step_um": int(pitch.group(1)) / 10.0} if pitch else {})}


def _with_channel(meta: dict, idx: str, **fields) -> dict:
    """``meta`` with ``fields`` merged into channel ``idx``, as a new dict."""
    channels = meta["channels"]
    return {**meta, "channels": {**channels, idx: {**channels.get(idx, {}), **fields}}}


def _gci_channel_name(name: str, text: str, meta: dict) -> dict:
    idx = re.search(r"Channel(\d+)", name).group(1)
    m = re.search(r"<Comment[^>]*>([^<]*)", text)
    return _with_channel(meta, idx, name=m.group(1) if m else f"CH{idx}")


def _gci_exposure(name: str, text: str, meta: dict) -> dict:
    idx = re.search(r"Channel(\d+)", name).group(1)
    num = re.search(r"Numerator[^>]*>(\d+)", text)
    den = re.search(r"Denominator[^>]*>(\d+)", text)
    if not (num and den and int(den.group(1))):
        return meta
    return _with_channel(meta, idx, exposure_ms=1000.0 * int(num.group(1)) / int(den.group(1)))


def _gci_handler(name: str):
    """The parser for one .gci archive entry, or None (first match wins)."""
    if name.endswith("Image/properties.xml"):
        return _gci_pixel_size
    if name.endswith("Lens/properties.xml"):
        return _gci_objective
    if name.endswith("Stack/properties.xml") and "RangeSelection" not in name:
        return _gci_stack
    if re.search(r"Channel(\d+)/properties.xml$", name):
        return _gci_channel_name
    if re.search(r"Channel(\d+)/Shooting/Parameter/ExposureTime/properties.xml$", name):
        return _gci_exposure
    return None


def _z_index(path: str) -> int:
    m = re.search(r"_Z(\d+)", Path(path).name)
    return int(m.group(1)) if m else 0


def _warn_z_gaps(indices: list[int], folder: Path) -> None:
    """Warn when the slices' ``_Z`` indices skip or repeat a value: the stack is
    assembled in index order from the files present, so z positions shift."""
    missing = sorted(set(range(min(indices), max(indices) + 1)) - set(indices))
    duplicated = sorted({i for i in indices if indices.count(i) > 1})
    problems = [f"{what} z slice(s): {found}"
                for what, found in (("missing", missing), ("duplicated", duplicated)) if found]
    if problems:
        warnings.warn(f"{folder.name}: {'; '.join(problems)}. The stack is built from the files "
                      "present in z-index order, so z positions after this are shifted.",
                      RuntimeWarning, stacklevel=3)


def load_keyence_stack(folder, downsample: int = 1, channel: int | None = None):
    """Assemble the raw z-stack from a Keyence BZ-X export folder.

    Parameters
    ----------
    folder : path
        Folder containing ``*_CHF*.tif`` per-slice OME-TIFFs and (optionally) an
        ``*.gci`` metadata archive.
    downsample : int
        Integer XY stride (1 = full resolution).
    channel : int, optional
        If given, return only that channel as ``(Z, Y, X)``; otherwise all
        channels as ``(C, Z, Y, X)``.

    Returns
    -------
    (array, meta) : tuple[np.ndarray, dict]
        ``meta`` includes parsed ``.gci`` fields (when present) plus
        ``vox_z_um`` / ``vox_xy_um`` (accounting for ``downsample``) and
        ``n_slices_loaded``.
    """
    import tifffile

    folder = Path(folder)
    files = sorted(glob.glob(str(folder / "*_CHF*.tif")), key=_z_index)
    if not files:  # fall back to any per-slice tif with a _Z index
        files = sorted((f for f in glob.glob(str(folder / "*.tif")) if re.search(r"_Z\d+", f)),
                       key=_z_index)
    if not files:
        raise FileNotFoundError(f"no Keyence per-slice TIFFs (*_CHF*.tif) in {folder}")
    _warn_z_gaps([_z_index(f) for f in files], folder)

    planes = []
    for f in files:
        a = tifffile.imread(f)                      # (C, Y, X) or (Y, X)
        if a.ndim == 2:
            a = a[None]
        a = a[:, ::downsample, ::downsample]
        planes.append(a[channel] if channel is not None else a)
    arr = np.stack(planes, axis=1) if channel is None else np.stack(planes, axis=0)

    gci = next((p for p in folder.glob("*.gci")), None)
    meta = parse_gci(gci) if gci else {"px_um": None, "z_step_um": None, "channels": {}}
    meta["n_slices_loaded"] = len(files)
    meta["downsample"] = downsample
    if meta.get("px_um"):
        meta["vox_xy_um"] = round(meta["px_um"] * downsample, 5)
    meta["vox_z_um"] = meta.get("z_step_um")
    return arr, meta
