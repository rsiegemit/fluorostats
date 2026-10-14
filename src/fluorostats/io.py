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
  - channel_names: list[str]
  - format: str — detected file format
"""

from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Lazy optional imports with helpful error messages
# ---------------------------------------------------------------------------

def _reader(package: str):
    """Import an optional vendor reader, naming the extra that installs it."""
    extra = next(f.extra for f in FORMATS.values() if f.package == package)
    return _require(package, extra)


def _require(package: str, extra: str):
    """Import a package or raise a helpful install message."""
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"Reading this format requires the '{package}' package.\n"
            f"Install it with: pip install {package}\n"
            f"(or, from a fluorostats checkout: pip install -e \".[{extra}]\")"
        ) from None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_auto(path: Path) -> tuple[np.ndarray, dict]:
    """Auto-detect format and load as volume or image.

    PNG/JPG/BMP load straight as 2D. Other formats try volume loading
    first and fall back to 2D on any error (or when the volume has Z=1).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in IMAGE_ONLY_SUFFIXES:
        return load_image(path)

    try:
        arr, meta = load_volume(path)
        # If Z=1, treat as 2D
        if arr.shape[1] == 1:
            arr = arr[:, 0, :, :]  # (C, 1, Y, X) -> (C, Y, X)
            meta["voxel_size_um"] = meta["voxel_size_um"][1:]
        return arr, meta
    except Exception as exc:
        # A genuine volume-read failure (not a format we recognise as 2D) should not
        # be silently retried as a 2D read — surface it so a real error is visible.
        warnings.warn(f"load_volume failed for {path.name} ({type(exc).__name__}: {exc}); "
                      "falling back to a 2D image read.", RuntimeWarning, stacklevel=2)
        return load_image(path)


def load_volume(path: Path) -> tuple[np.ndarray, dict]:
    """Load a 3D volume. Returns (array[C,Z,Y,X], metadata).

    A directory is auto-detected as a Keyence BZ-X export (folder of per-slice
    ``*_CHF*`` OME-TIFFs + ``.gci``) and assembled via ``fluorostats.keyence``.
    """
    path = Path(path)

    if path.is_dir():
        from . import keyence
        if not keyence.is_keyence_folder(path):
            raise ValueError(f"Directory is not a recognised Keyence BZ-X export: {path}")
        arr, meta = keyence.load_keyence_stack(path)
        px, zs = meta.get("px_um") or 1.0, meta.get("z_step_um") or 1.0
        meta["voxel_size_um"] = (zs, px, px)
        chd = meta.get("channels") or {}
        meta["channel_names"] = [chd.get(str(i), {}).get("name", f"CH{i}") for i in range(arr.shape[0])]
        return arr, meta

    suffix = path.suffix.lower()

    fmt = FORMATS.get(suffix)
    if fmt is None or fmt.volume is None:
        raise ValueError(
            f"Unsupported volume format: {suffix}\n"
            f"Supported: {', '.join(sorted(VOLUME_SUFFIXES))}"
        )
    return fmt.volume(path)


def load_image(path: Path) -> tuple[np.ndarray, dict]:
    """Load a 2D image. Returns (array[C,Y,X], metadata)."""
    path = Path(path)
    suffix = path.suffix.lower()

    fmt = FORMATS.get(suffix)
    loader = fmt.image if fmt is not None and fmt.image is not None else _load_standard_image
    return loader(path)


def supported_formats() -> dict[str, str]:
    """Return dict of extension -> description for all supported formats."""
    formats: dict[str, str] = {}
    for ext, fmt in FORMATS.items():
        if fmt.package is None:
            formats[ext] = fmt.description
            continue
        try:
            _require(fmt.package, fmt.extra)
            formats[ext] = fmt.description
        except ImportError:
            formats[ext] = f"{fmt.description} (requires: pip install {fmt.package})"
    return formats


# ---------------------------------------------------------------------------
# Olympus .oib / .oif
# ---------------------------------------------------------------------------

def _load_olympus_volume(path: Path) -> tuple[np.ndarray, dict]:
    oiffile = _reader("oiffile")

    oif = oiffile.OifFile(str(path))
    arr = oif.asarray()

    meta = _parse_olympus_metadata(oif)
    meta["format"] = "olympus"
    arr = _canonicalize_volume(arr, meta)
    oif.close()
    return arr, meta


def _parse_olympus_metadata(oif) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

    try:
        settings = oif.mainfile
    except Exception:
        return meta

    voxel_x, voxel_y, voxel_z = 1.0, 1.0, 1.0
    channel_names: list[str] = []

    for section_name, section in settings.items():
        section_str = str(section_name)

        if re.match(r"Channel \d+ Parameters$", section_str):
            dye = _section_get(section, "DyeName")
            if dye and dye != "(null)" and dye != "None":
                channel_names.append(dye)

        if section_str == "Reference Image Parameter":
            voxel_x = _safe_float(_section_get(section, "WidthConvertValue"), voxel_x)
            voxel_y = _safe_float(_section_get(section, "HeightConvertValue"), voxel_y)

        if "Axis" in section_str and "Parameters Common" in section_str:
            axis_code = _section_get(section, "AxisCode")
            if axis_code == "Z":
                interval = _safe_float(_section_get(section, "Interval"), 0.0)
                pix_unit = _section_get(section, "PixUnit") or ""
                if interval > 0:
                    if pix_unit.lower() == "nm":
                        voxel_z = interval / 1000.0
                    elif pix_unit.lower() == "um":
                        voxel_z = interval
                    else:
                        warnings.warn(f"Olympus PixUnit {pix_unit!r} not recognised; "
                                      "assuming nm (interval/1000). Verify the z voxel size.",
                                      RuntimeWarning, stacklevel=2)
                        voxel_z = interval / 1000.0

    meta["channel_names"] = channel_names
    meta["voxel_size_um"] = (voxel_z, voxel_y, voxel_x)
    return meta


def _section_get(section, key: str):
    try:
        if isinstance(section, dict):
            return section.get(key)
        return getattr(section, key, None) or section[key]
    except (KeyError, TypeError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Zeiss .czi
# ---------------------------------------------------------------------------

def _load_czi_volume(path: Path) -> tuple[np.ndarray, dict]:
    czifile = _reader("czifile")

    with czifile.CziFile(str(path)) as czi:
        arr = czi.asarray()
        meta = _parse_czi_metadata(czi)

    meta["format"] = "zeiss_czi"
    arr = _squeeze_singleton_dims(arr)
    arr = _canonicalize_volume(arr, meta)
    return arr, meta


def _parse_czi_metadata(czi) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

    try:
        xml_str = czi.metadata()
        if xml_str:
            root = ET.fromstring(xml_str)

            vz, vy, vx = 1.0, 1.0, 1.0
            for dist in root.iter("Distance"):
                axis_id = dist.get("Id", "")
                value_el = dist.find("Value")
                if value_el is not None and value_el.text:
                    val_um = float(value_el.text) * 1e6
                    if axis_id == "X":
                        vx = val_um
                    elif axis_id == "Y":
                        vy = val_um
                    elif axis_id == "Z":
                        vz = val_um
            meta["voxel_size_um"] = (vz, vy, vx)

            for ch in root.iter("Channel"):
                name = ch.get("Name", "")
                if name:
                    meta["channel_names"].append(name)
            if not meta["channel_names"]:
                for dye in root.iter("DyeName"):
                    if dye.text:
                        meta["channel_names"].append(dye.text)
    except Exception:
        pass

    return meta


# ---------------------------------------------------------------------------
# Nikon .nd2
# ---------------------------------------------------------------------------

def _load_nd2_volume(path: Path) -> tuple[np.ndarray, dict]:
    nd2 = _reader("nd2")

    with nd2.ND2File(str(path)) as f:
        arr = f.asarray()
        meta = _parse_nd2_metadata(f)

    meta["format"] = "nikon_nd2"
    arr = _squeeze_singleton_dims(arr)
    arr = _canonicalize_volume(arr, meta)
    return arr, meta


def _parse_nd2_metadata(f) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

    try:
        if hasattr(f, "voxel_size"):
            vs = f.voxel_size()
            meta["voxel_size_um"] = (
                _safe_float(getattr(vs, "z", 1.0), 1.0),
                _safe_float(getattr(vs, "y", 1.0), 1.0),
                _safe_float(getattr(vs, "x", 1.0), 1.0),
            )
    except Exception:
        pass

    try:
        if hasattr(f, "metadata") and hasattr(f.metadata, "channels"):
            for ch in f.metadata.channels:
                name = getattr(ch.channel, "name", "") or getattr(ch, "name", "")
                if name:
                    meta["channel_names"].append(name)
    except Exception:
        pass

    return meta


# ---------------------------------------------------------------------------
# Leica .lif
# ---------------------------------------------------------------------------

def _load_lif_volume(path: Path) -> tuple[np.ndarray, dict]:
    readlif = _reader("readlif")

    reader = readlif.LifFile(str(path))

    if len(reader.image_list) == 0:
        raise ValueError(f"No images found in LIF file: {path}")

    img = reader.get_image(0)
    meta = _parse_lif_metadata(img)
    meta["format"] = "leica_lif"

    n_channels = img.channels
    n_z = img.dims.z
    frames = []
    for z in range(n_z):
        for c in range(n_channels):
            frame = np.array(img.get_frame(z=z, t=0, c=c))
            frames.append(frame)

    h, w = frames[0].shape[:2]
    if frames[0].ndim == 3:
        frames = [f[:, :, 0] if f.ndim == 3 else f for f in frames]

    # frames were appended Z-major, C-minor (for z: for c:), i.e. index = z*n_channels + c,
    # so reshape to (Z, C, H, W) THEN transpose to (C, Z, H, W). Reshaping straight to
    # (C, Z, ...) interleaves channels for multi-channel z-stacks (only 1-channel is safe).
    arr = np.array(frames).reshape(n_z, n_channels, h, w).transpose(1, 0, 2, 3)
    arr = _canonicalize_volume(arr, meta)
    return arr, meta


def _parse_lif_metadata(img) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

    try:
        scale = img.scale
        if scale:
            voxel_x = 1.0 / scale[0] * 1e6 if scale[0] > 0 else 1.0
            voxel_y = 1.0 / scale[1] * 1e6 if len(scale) > 1 and scale[1] > 0 else voxel_x
            voxel_z = 1.0 / scale[2] * 1e6 if len(scale) > 2 and scale[2] > 0 else 1.0
            meta["voxel_size_um"] = (voxel_z, voxel_y, voxel_x)
    except Exception:
        pass

    try:
        for i in range(img.channels):
            meta["channel_names"].append(f"Ch{i + 1}")
    except Exception:
        pass

    return meta


# ---------------------------------------------------------------------------
# TIFF / OME-TIFF
# ---------------------------------------------------------------------------

def _load_tiff_volume(path: Path) -> tuple[np.ndarray, dict]:
    data = tifffile.imread(str(path))
    meta: dict = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": [], "format": "tiff"}

    with tifffile.TiffFile(str(path)) as tif:
        if tif.ome_metadata:
            meta = _parse_ome_metadata(tif.ome_metadata, meta)
            meta["format"] = "ome_tiff"
        elif tif.imagej_metadata:
            meta = _parse_imagej_metadata(tif.imagej_metadata, meta)
            meta["format"] = "imagej_tiff"

    data = _canonicalize_volume(data, meta)
    return data, meta


def _load_tiff_image(path: Path) -> tuple[np.ndarray, dict]:
    data = tifffile.imread(str(path))
    meta: dict = {"voxel_size_um": (1.0, 1.0), "channel_names": [], "format": "tiff"}
    data = _canonicalize_image(data, meta)
    return data, meta


def _parse_ome_metadata(ome_xml: str, meta: dict) -> dict:
    try:
        root = ET.fromstring(ome_xml)
        pixels = root.find(".//{*}Pixels")
        if pixels is not None:
            x = _safe_float(pixels.get("PhysicalSizeX"), 1.0)
            y = _safe_float(pixels.get("PhysicalSizeY"), 1.0)
            z = _safe_float(pixels.get("PhysicalSizeZ"), 1.0)
            meta["voxel_size_um"] = (z, y, x)

            for ch in pixels.findall("{*}Channel"):
                name = ch.get("Name", "") or ch.get("Fluor", "")
                if name:
                    meta["channel_names"].append(name)
    except ET.ParseError:
        pass
    return meta


def _parse_imagej_metadata(ij_meta: dict, meta: dict) -> dict:
    try:
        if "spacing" in ij_meta:
            meta["voxel_size_um"] = (
                float(ij_meta["spacing"]),
                meta["voxel_size_um"][1],
                meta["voxel_size_um"][2],
            )
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------------------
# NumPy .npy
# ---------------------------------------------------------------------------

def _load_npy_volume(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(str(path))
    meta: dict = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": [], "format": "numpy"}
    data = _canonicalize_volume(data, meta)
    return data, meta


def _load_npy_image(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(str(path))
    meta: dict = {"voxel_size_um": (1.0, 1.0), "channel_names": [], "format": "numpy"}
    data = _canonicalize_image(data, meta)
    return data, meta


# ---------------------------------------------------------------------------
# Standard images (PNG, JPEG, BMP)
# ---------------------------------------------------------------------------

def _load_standard_image(path: Path) -> tuple[np.ndarray, dict]:
    data = iio.imread(str(path))
    meta: dict = {"voxel_size_um": (1.0, 1.0), "channel_names": [], "format": path.suffix.lstrip(".").lower()}
    data = _canonicalize_image(data, meta)
    return data, meta


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------

def _squeeze_singleton_dims(arr: np.ndarray) -> np.ndarray:
    """Remove singleton dimensions (common in CZI: B, S, T axes)."""
    while arr.ndim > 4 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    while arr.ndim > 4:
        arr = arr[0]
    return arr


def _canonicalize_volume(arr: np.ndarray, meta: dict) -> np.ndarray:
    """Ensure array is (C, Z, Y, X)."""
    ndim = arr.ndim
    if ndim == 3:
        arr = arr[np.newaxis]
    elif ndim == 4:
        # Axis-order heuristic: a leading axis larger than this is far more likely to be
        # Z (many slices) than C (few channels), so treat the array as channel-LAST and
        # move the trailing axis to front. This is genuinely ambiguous for a channel-last
        # stack with few z-slices — pass explicit channel metadata upstream to be sure.
        _MAX_LEADING_CHANNELS = 10
        if arr.shape[0] > _MAX_LEADING_CHANNELS:
            warnings.warn(f"4D array leading axis = {arr.shape[0]} (> {_MAX_LEADING_CHANNELS}); "
                          "assuming channel-last and transposing to (C, Z, Y, X).",
                          RuntimeWarning, stacklevel=3)
            arr = np.moveaxis(arr, -1, 0)
    elif ndim == 5:
        arr = arr[0]
    elif ndim == 2:
        arr = arr[np.newaxis, np.newaxis]
    elif ndim > 5:
        arr = _squeeze_singleton_dims(arr)
        if arr.ndim != 4:  # pragma: no cover - squeeze always yields 4D
            return _canonicalize_volume(arr, meta)
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} with {ndim} dims")

    n_ch = arr.shape[0]
    if len(meta["channel_names"]) < n_ch:
        for i in range(len(meta["channel_names"]), n_ch):
            meta["channel_names"].append(f"Ch{i + 1}")

    return arr


def _canonicalize_image(arr: np.ndarray, meta: dict) -> np.ndarray:
    """Ensure 2D image is (C, Y, X)."""
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3:
        if arr.shape[2] <= 4:
            arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Unexpected 2D image shape: {arr.shape}")

    n_ch = arr.shape[0]
    if len(meta["channel_names"]) < n_ch:
        for i in range(len(meta["channel_names"]), n_ch):
            meta["channel_names"].append(f"Ch{i + 1}")

    return arr


def _safe_float(val, default: float) -> float:
    try:
        v = float(val)
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


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
    ".tif": Format("TIFF / OME-TIFF", _load_tiff_volume, _load_tiff_image),
    ".tiff": Format("TIFF / OME-TIFF", _load_tiff_volume, _load_tiff_image),
    ".png": Format("PNG", None, _load_standard_image),
    ".jpg": Format("JPEG", None, _load_standard_image),
    ".jpeg": Format("JPEG", None, _load_standard_image),
    ".bmp": Format("BMP", None, _load_standard_image),
    ".npy": Format("NumPy array", _load_npy_volume, _load_npy_image),
    ".oib": Format("Olympus FluoView (OIB)", _load_olympus_volume, None, "oiffile", "olympus"),
    ".oif": Format("Olympus FluoView (OIF)", _load_olympus_volume, None, "oiffile", "olympus"),
    ".czi": Format("Zeiss ZEN (CZI)", _load_czi_volume, None, "czifile", "zeiss"),
    ".nd2": Format("Nikon NIS-Elements (ND2)", _load_nd2_volume, None, "nd2", "nikon"),
    ".lif": Format("Leica LAS X (LIF)", _load_lif_volume, None, "readlif", "leica"),
}

VOLUME_SUFFIXES = {ext for ext, f in FORMATS.items() if f.volume is not None}
IMAGE_SUFFIXES = {ext for ext, f in FORMATS.items() if f.image is not None}
IMAGE_ONLY_SUFFIXES = IMAGE_SUFFIXES - VOLUME_SUFFIXES
ALL_SUFFIXES = VOLUME_SUFFIXES | IMAGE_SUFFIXES
OPTIONAL_READERS = tuple(dict.fromkeys(f.package for f in FORMATS.values() if f.package))
