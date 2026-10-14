"""Universal microscopy file loaders with auto-detection.

Supported formats:
  Core (always available):
    .tif, .tiff     — TIFF / OME-TIFF (via tifffile)
    .png, .jpg, .jpeg, .bmp — standard images (via imageio)
    .npy            — NumPy arrays

  Optional (install extras):
    .oib, .oif      — Olympus FluoView     (pip install oiffile)
    .czi            — Zeiss ZEN             (pip install czifile)
    .nd2            — Nikon NIS-Elements    (pip install nd2)
    .lif            — Leica LAS X          (pip install readlif)

All loaders return (array, metadata_dict) where array is canonicalized to:
  - 3D volumes: (C, Z, Y, X)
  - 2D images:  (C, Y, X)
with channel axis always present (C=1 if single-channel).

Metadata dict carries:
  - voxel_size_um: tuple of floats (z, y, x) for 3D or (y, x) for 2D
  - voxel_size_source: str — where each axis's size came from (e.g. "z:file,y:default,x:default")
  - channel_names: list[str]
  - format: str — detected file format

This module is the entry point and the format registry; the per-format readers
live in ``io_vendors`` and the array layout rules in ``io_canonical``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np

from . import io_vendors, keyence
from .io_canonical import canonicalize_image, canonicalize_volume
from .io_vendors import READER_EXTRAS


def _record_calibration(meta: dict, axes: str, path: Path) -> None:
    """Record which voxel axes came from the file and warn about defaulted ones.

    Loaders note the axes they read in ``meta["_read_axes"]``; every other axis
    of ``voxel_size_um`` is the 1.0 µm placeholder. Sets ``voxel_size_source``
    (e.g. ``"z:file,y:default,x:default"``) and warns when any axis defaulted,
    because calibrated lengths, areas, volumes and densities along it are then
    in pixel units. Values in ``voxel_size_um`` are not changed. A loader that
    took an axis from the file with a caveat (e.g. an unknown length unit)
    names it in ``meta["_axis_source"]`` ({axis: source}) and warns itself.
    """
    read = meta.pop("_read_axes", set())
    noted = meta.pop("_axis_source", {})
    meta["voxel_size_source"] = ",".join(
        f"{a}:{noted.get(a) or ('file' if a in read else 'default')}" for a in axes)
    missing = [a for a in axes if a not in read and a not in noted]
    if missing:
        warnings.warn(
            f"{path.name}: no {'/'.join(missing)} calibration found in the file; assuming "
            "1.0 µm per pixel on that axis, so lengths, areas, volumes and densities along "
            "it are in pixel units. Check voxel_size_um.",
            RuntimeWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_auto(path: str | Path) -> tuple[np.ndarray, dict]:
    """Auto-detect format and load as volume or image.

    PNG/JPG/BMP load straight as 2D. Other formats try volume loading
    first and fall back to 2D on any error (or when the volume has Z=1).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in IMAGE_ONLY_SUFFIXES:
        return load_image(path)

    try:
        arr, meta = _read_volume(path)
    except Exception as exc:
        # A genuine volume-read failure (not a format we recognise as 2D) should not
        # be silently retried as a 2D read — surface it so a real error is visible.
        warnings.warn(f"load_volume failed for {path.name} ({type(exc).__name__}: {exc}); "
                      "falling back to a 2D image read.", RuntimeWarning, stacklevel=2)
        return load_image(path)
    if arr.shape[1] == 1:  # a single plane: (C, 1, Y, X) -> (C, Y, X), no z calibration
        arr = arr[:, 0, :, :]
        meta = {**meta, "voxel_size_um": meta["voxel_size_um"][1:]}
        _record_calibration(meta, "yx", path)
    else:
        _record_calibration(meta, "zyx", path)
    return arr, meta


def load_volume(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a 3D volume. Returns (array[C,Z,Y,X], metadata).

    A directory is auto-detected as a Keyence BZ-X export (folder of per-slice
    ``*_CHF*`` OME-TIFFs + ``.gci``) and assembled via ``fluorostats.keyence``.
    """
    path = Path(path)
    arr, meta = _read_volume(path)
    _record_calibration(meta, "zyx", path)
    return arr, meta


def _read_volume(path: Path) -> tuple[np.ndarray, dict]:
    """(C, Z, Y, X) array and metadata, before the calibration record is made."""
    if path.is_dir():
        fmt = next((f for f in DIRECTORY_FORMATS if f.probe(path)), None)
        if fmt is None:
            kinds = " or ".join(f.description for f in DIRECTORY_FORMATS)
            raise ValueError(f"Directory is not a recognised {kinds}: {path}")
        return fmt.volume(path)

    suffix = path.suffix.lower()

    fmt = FORMATS.get(suffix)
    if fmt is None or fmt.volume is None:
        raise ValueError(
            f"Unsupported volume format: {suffix}\n"
            f"Supported: {', '.join(sorted(VOLUME_SUFFIXES))}"
        )
    return fmt.volume(path)


def load_image(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a 2D image. Returns (array[C,Y,X], metadata)."""
    path = Path(path)
    suffix = path.suffix.lower()

    fmt = FORMATS.get(suffix)
    loader = fmt.image if fmt is not None and fmt.image is not None else _load_standard_image
    arr, meta = loader(path)
    _record_calibration(meta, "yx", path)
    return arr, meta


def supported_formats() -> dict[str, str]:
    """Return dict of extension -> description for all supported formats."""
    formats: dict[str, str] = {}
    for ext, fmt in FORMATS.items():
        if fmt.package is None:
            formats[ext] = fmt.description
            continue
        try:
            io_vendors.require(fmt.package, fmt.extra)
            formats[ext] = fmt.description
        except ImportError:
            formats[ext] = f"{fmt.description} (requires: pip install {fmt.package})"
    return formats


# ---------------------------------------------------------------------------
# Keyence BZ-X export folder
# ---------------------------------------------------------------------------

def _load_keyence_volume(path: Path) -> tuple[np.ndarray, dict]:
    arr, meta = keyence.load_keyence_stack(path)
    px, zs = meta.get("px_um") or 1.0, meta.get("z_step_um") or 1.0
    meta["voxel_size_um"] = (zs, px, px)
    meta["_read_axes"] = ({"y", "x"} if meta.get("px_um") else set()) | (
        {"z"} if meta.get("z_step_um") else set())
    # keyence.parse_gci warns that the Stack/Pitch unit is assumed.
    meta["_axis_source"] = {"z": "file-assumed-0.1um"} if meta.get("z_step_um") else {}
    chd = meta.get("channels") or {}
    meta["channel_names"] = [chd.get(str(i), {}).get("name", f"CH{i}") for i in range(arr.shape[0])]
    return arr, meta


# ---------------------------------------------------------------------------
# NumPy .npy
# ---------------------------------------------------------------------------

def _load_npy_volume(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(str(path))
    meta: dict = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": [], "format": "numpy"}
    data = canonicalize_volume(data, meta)
    return data, meta


def _load_npy_image(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(str(path))
    meta: dict = {"voxel_size_um": (1.0, 1.0), "channel_names": [], "format": "numpy"}
    data = canonicalize_image(data, meta)
    return data, meta


# ---------------------------------------------------------------------------
# Standard images (PNG, JPEG, BMP)
# ---------------------------------------------------------------------------

def _load_standard_image(path: Path) -> tuple[np.ndarray, dict]:
    data = iio.imread(str(path))
    meta: dict = {"voxel_size_um": (1.0, 1.0), "channel_names": [], "format": path.suffix.lstrip(".").lower()}
    data = canonicalize_image(data, meta)
    return data, meta


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Format registry: the one place each supported format is described (SIN-4).
# The suffix sets, `load_volume` / `load_image` dispatch, `supported_formats`
# and the CLI's reader list all derive from it. Insertion order is the order
# `supported_formats` reports. Defined last so it can name the loaders above.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Format:
    """How one file suffix is read."""

    description: str
    volume: Callable[[Path], tuple[np.ndarray, dict]] | None   # -> (C, Z, Y, X)
    image: Callable[[Path], tuple[np.ndarray, dict]] | None    # -> (C, Y, X)
    package: str | None = None    # optional vendor reader distribution
    extra: str | None = None      # pyproject extra that installs it


FORMATS: dict[str, Format] = {
    ".tif": Format("TIFF / OME-TIFF", io_vendors.load_tiff_volume, io_vendors.load_tiff_image),
    ".tiff": Format("TIFF / OME-TIFF", io_vendors.load_tiff_volume, io_vendors.load_tiff_image),
    ".png": Format("PNG", None, _load_standard_image),
    ".jpg": Format("JPEG", None, _load_standard_image),
    ".jpeg": Format("JPEG", None, _load_standard_image),
    ".bmp": Format("BMP", None, _load_standard_image),
    ".npy": Format("NumPy array", _load_npy_volume, _load_npy_image),
    ".oib": Format("Olympus FluoView (OIB)", io_vendors.load_olympus_volume, None,
                   "oiffile", READER_EXTRAS["oiffile"]),
    ".oif": Format("Olympus FluoView (OIF)", io_vendors.load_olympus_volume, None,
                   "oiffile", READER_EXTRAS["oiffile"]),
    ".czi": Format("Zeiss ZEN (CZI)", io_vendors.load_czi_volume, None,
                   "czifile", READER_EXTRAS["czifile"]),
    ".nd2": Format("Nikon NIS-Elements (ND2)", io_vendors.load_nd2_volume, None,
                   "nd2", READER_EXTRAS["nd2"]),
    ".lif": Format("Leica LAS X (LIF)", io_vendors.load_lif_volume, None,
                   "readlif", READER_EXTRAS["readlif"]),
}


@dataclass(frozen=True)
class DirectoryFormat:
    """How one kind of folder export (read by ``load_volume``) is recognised and read."""

    description: str
    probe: Callable[[Path], bool]                          # True if the folder is this kind
    volume: Callable[[Path], tuple[np.ndarray, dict]]      # -> (C, Z, Y, X)


DIRECTORY_FORMATS: tuple[DirectoryFormat, ...] = (
    DirectoryFormat("Keyence BZ-X export", keyence.is_keyence_folder, _load_keyence_volume),
)

VOLUME_SUFFIXES = {ext for ext, f in FORMATS.items() if f.volume is not None}
IMAGE_SUFFIXES = {ext for ext, f in FORMATS.items() if f.image is not None}
IMAGE_ONLY_SUFFIXES = IMAGE_SUFFIXES - VOLUME_SUFFIXES
ALL_SUFFIXES = VOLUME_SUFFIXES | IMAGE_SUFFIXES
OPTIONAL_READERS = tuple(dict.fromkeys(f.package for f in FORMATS.values() if f.package))
