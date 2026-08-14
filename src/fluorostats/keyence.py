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
The z-step is stored as ``Stack/Pitch`` in 0.1-µm units (documented convention);
confirm against the acquisition GUI for a given dataset.
"""
from __future__ import annotations

import glob
import re
import struct
import zipfile
from pathlib import Path

import numpy as np

__all__ = ["parse_gci", "load_keyence_stack"]


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
        read = lambda name: z.read(name).decode("utf-8", "ignore")
        for name in z.namelist():
            if name.endswith("Image/properties.xml"):
                m = re.search(r'Calibration Type="System.Double">(-?\d+)', read(name))
                if m:
                    nm = _int64_to_double(m.group(1))
                    if nm:
                        meta["px_um"] = round(nm / 1000.0, 5)      # nm -> µm
            elif name.endswith("Lens/properties.xml"):
                m = re.search(r"<LensName[^>]*>([^<]+)", read(name))
                if m:
                    meta["objective"] = m.group(1)
            elif name.endswith("Stack/properties.xml") and "RangeSelection" not in name:
                x = read(name)
                tn = re.search(r"TotalNumber[^>]*>(\d+)", x)
                pitch = re.search(r"Pitch[^>]*>(\d+)", x)
                if tn:
                    meta["n_slices"] = int(tn.group(1))
                if pitch:
                    meta["z_step_um"] = int(pitch.group(1)) / 10.0
            elif re.search(r"Channel(\d+)/properties.xml$", name):
                idx = re.search(r"Channel(\d+)", name).group(1)
                m = re.search(r"<Comment[^>]*>([^<]*)", read(name))
                meta["channels"].setdefault(idx, {})["name"] = m.group(1) if m else f"CH{idx}"
            elif re.search(r"Channel(\d+)/Shooting/Parameter/ExposureTime/properties.xml$", name):
                idx = re.search(r"Channel(\d+)", name).group(1)
                num = re.search(r"Numerator[^>]*>(\d+)", read(name))
                den = re.search(r"Denominator[^>]*>(\d+)", read(name))
                if num and den and int(den.group(1)):
                    meta["channels"].setdefault(idx, {})["exposure_ms"] = \
                        1000.0 * int(num.group(1)) / int(den.group(1))
    return meta


def _z_index(path: str) -> int:
    m = re.search(r"_Z(\d+)", Path(path).name)
    return int(m.group(1)) if m else 0


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
