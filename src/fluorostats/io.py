"""Universal microscopy file loaders with auto-detection.

Supported formats:
  Core (always available):
    .tif, .tiff     — TIFF / OME-TIFF (via tifffile)
    .png, .jpg, .jpeg, .bmp — standard images (via imageio)
    .npy            — NumPy arrays

  Optional (install extras):
    .oib, .oif      — Olympus FluoView     (pip install fluorostats[olympus])
    .czi            — Zeiss ZEN             (pip install fluorostats[zeiss])
    .nd2            — Nikon NIS-Elements    (pip install fluorostats[nikon])
    .lif            — Leica LAS X          (pip install fluorostats[leica])

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
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile

# Format groups
VOLUME_SUFFIXES = {".oib", ".oif", ".czi", ".nd2", ".lif", ".tif", ".tiff", ".npy"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
ALL_SUFFIXES = VOLUME_SUFFIXES | IMAGE_SUFFIXES


# ---------------------------------------------------------------------------
# Lazy optional imports with helpful error messages
# ---------------------------------------------------------------------------

def _require(package: str, extra: str):
    """Import a package or raise a helpful install message."""
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"Reading this format requires the '{package}' package.\n"
            f"Install it with: pip install fluorostats[{extra}]"
        ) from None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_auto(path: Path) -> tuple[np.ndarray, dict]:
    """Auto-detect format and load as volume or image.

    Tries volume loading first; falls back to 2D if the file has
    fewer than 3 spatial dimensions.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        return load_image(path)

    try:
        arr, meta = load_volume(path)
        # If Z=1, treat as 2D
        if arr.shape[1] == 1:
            arr = arr[:, 0, :, :]  # (C, 1, Y, X) -> (C, Y, X)
            meta["voxel_size_um"] = meta["voxel_size_um"][1:]
        return arr, meta
    except Exception:
        return load_image(path)


def load_volume(path: Path) -> tuple[np.ndarray, dict]:
    """Load a 3D volume. Returns (array[C,Z,Y,X], metadata)."""
    path = Path(path)
    suffix = path.suffix.lower()

    loaders = {
        ".oib": _load_olympus_volume,
        ".oif": _load_olympus_volume,
        ".czi": _load_czi_volume,
        ".nd2": _load_nd2_volume,
        ".lif": _load_lif_volume,
        ".tif": _load_tiff_volume,
        ".tiff": _load_tiff_volume,
        ".npy": _load_npy_volume,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported volume format: {suffix}\n"
            f"Supported: {', '.join(sorted(loaders.keys()))}"
        )
    return loader(path)


def load_image(path: Path) -> tuple[np.ndarray, dict]:
    """Load a 2D image. Returns (array[C,Y,X], metadata)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        return _load_tiff_image(path)
    if suffix == ".npy":
        return _load_npy_image(path)
    return _load_standard_image(path)


def supported_formats() -> dict[str, str]:
    """Return dict of extension -> description for all supported formats."""
    formats = {
        ".tif": "TIFF / OME-TIFF",
        ".tiff": "TIFF / OME-TIFF",
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".bmp": "BMP",
        ".npy": "NumPy array",
    }
    optional = [
        ("oiffile", "olympus", {".oib": "Olympus FluoView (OIB)", ".oif": "Olympus FluoView (OIF)"}),
        ("czifile", "zeiss", {".czi": "Zeiss ZEN (CZI)"}),
        ("nd2", "nikon", {".nd2": "Nikon NIS-Elements (ND2)"}),
        ("readlif", "leica", {".lif": "Leica LAS X (LIF)"}),
    ]
    for pkg, extra, ext_map in optional:
        try:
            _require(pkg, extra)
            formats.update(ext_map)
        except ImportError:
            for ext, desc in ext_map.items():
                formats[ext] = f"{desc} (requires: pip install fluorostats[{extra}])"
    return formats


# ---------------------------------------------------------------------------
# Olympus .oib / .oif
# ---------------------------------------------------------------------------

def _load_olympus_volume(path: Path) -> tuple[np.ndarray, dict]:
    oiffile = _require("oiffile", "olympus")

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
    czifile = _require("czifile", "zeiss")

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
    nd2 = _require("nd2", "nikon")

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
    readlif = _require("readlif", "leica")

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

    arr = np.array(frames).reshape(n_channels, n_z, h, w)
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
        if arr.shape[0] > 10:
            arr = np.moveaxis(arr, -1, 0)
    elif ndim == 5:
        arr = arr[0]
    elif ndim == 2:
        arr = arr[np.newaxis, np.newaxis]
    elif ndim > 5:
        arr = _squeeze_singleton_dims(arr)
        if arr.ndim != 4:
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
