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

from . import keyence

# ---------------------------------------------------------------------------
# Lazy optional imports with helpful error messages
# ---------------------------------------------------------------------------

def _reader(package: str):
    """Import an optional vendor reader, naming the extra that installs it."""
    extra = next(f.extra for f in FORMATS.values() if f.package == package)
    return _require(package, extra)


def _valid_size(val) -> bool:
    """True when ``val`` is a usable (positive, numeric) physical size."""
    return _safe_float(val, -1.0) > 0


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
            _require(fmt.package, fmt.extra)
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
    read: set[str] = set()

    for section_name, section in settings.items():
        section_str = str(section_name)

        if re.match(r"Channel \d+ Parameters$", section_str):
            dye = _section_get(section, "DyeName")
            if dye and dye != "(null)" and dye != "None":
                channel_names.append(dye)

        if section_str == "Reference Image Parameter":
            wx, hy = _section_get(section, "WidthConvertValue"), _section_get(section, "HeightConvertValue")
            voxel_x = _safe_float(wx, voxel_x)
            voxel_y = _safe_float(hy, voxel_y)
            read |= {a for a, v in (("x", wx), ("y", hy)) if _valid_size(v)}

        if "Axis" in section_str and "Parameters Common" in section_str:
            z = _olympus_z_um(section)
            if z is not None:
                voxel_z = z
                read.add("z")

    meta["channel_names"] = channel_names
    meta["voxel_size_um"] = (voxel_z, voxel_y, voxel_x)
    meta["_read_axes"] = read
    return meta


def _olympus_z_um(section) -> float | None:
    """Z step (µm) from an Olympus axis section; None unless it is a Z axis with
    a positive interval."""
    if _section_get(section, "AxisCode") != "Z":
        return None
    interval = _safe_float(_section_get(section, "Interval"), 0.0)
    pix_unit = _section_get(section, "PixUnit") or ""
    if interval <= 0:
        return None
    if pix_unit.lower() == "um":
        return interval
    if pix_unit.lower() != "nm":
        warnings.warn(f"Olympus PixUnit {pix_unit!r} not recognised; "
                      "assuming nm (interval/1000). Verify the z voxel size.",
                      RuntimeWarning, stacklevel=3)
    return interval / 1000.0


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
        axes = _czi_axes(czi)
        meta = _parse_czi_metadata(czi)

    meta["format"] = "zeiss_czi"
    arr = _canonicalize_volume(arr if axes else _squeeze_singleton_dims(arr), meta,
                               axes, _CZI_INDEX_AXES)
    return arr, meta


# CZI dimensions other than C/Z/Y/X/S that are reduced to their first index.
_CZI_INDEX_AXES = {"T": "time point", "B": "block", "V": "view", "H": "phase",
                   "I": "illumination", "R": "rotation", "M": "mosaic tile",
                   "A": "acquisition", "s": "scene"}
# czifile < 2026 letters: '0' = colour samples, 'S' = scene (renamed 's').
_LEGACY_CZI_LETTERS = str.maketrans({"S": "s", "0": "S"})


def _czi_axes(czi) -> str | None:
    """Axis letters of ``czi.asarray()``: czifile >= 2026 reads and labels the
    first scene; older czifile labels the whole file. None if neither is exposed."""
    scenes = getattr(czi, "scenes", None)
    if scenes is not None:
        return next(iter(scenes.values())).axes
    axes = getattr(czi, "axes", None)
    return axes.translate(_LEGACY_CZI_LETTERS) if axes else None


def _czi_voxel_um(root):
    """(z, y, x) voxel size in µm from CZI ``Distance`` elements (metres), and the
    set of axes it found."""
    size = {"X": 1.0, "Y": 1.0, "Z": 1.0}
    read: set[str] = set()
    for dist in root.iter("Distance"):
        value_el = dist.find("Value")
        axis_id = dist.get("Id", "")
        if value_el is not None and value_el.text:
            val_um = float(value_el.text) * 1e6
            if axis_id in size:
                size[axis_id] = val_um
                read.add(axis_id.lower())
    return (size["Z"], size["Y"], size["X"]), read


def _czi_channel_names(root) -> list[str]:
    """Channel names from ``Channel@Name``, falling back to ``DyeName`` text."""
    names = [ch.get("Name", "") for ch in root.iter("Channel")]
    names = [n for n in names if n]
    if not names:
        names = [dye.text for dye in root.iter("DyeName") if dye.text]
    return names


def _parse_czi_metadata(czi) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

    try:
        xml_str = czi.metadata()
        if xml_str:
            root = ET.fromstring(xml_str)

            meta["voxel_size_um"], meta["_read_axes"] = _czi_voxel_um(root)
            meta["channel_names"] = _czi_channel_names(root)
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
        axes = "".join(f.sizes) if hasattr(f, "sizes") else None
        meta = _parse_nd2_metadata(f)

    meta["format"] = "nikon_nd2"
    arr = _canonicalize_volume(arr if axes else _squeeze_singleton_dims(arr), meta,
                               axes, {"T": "time point", "P": "position"})
    return arr, meta


def _parse_nd2_metadata(f) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

    try:
        if hasattr(f, "voxel_size"):
            vs = f.voxel_size()
            meta["_read_axes"] = {a for a in "zyx" if _valid_size(getattr(vs, a, None))}
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
    if len(reader.image_list) > 1 or getattr(img.dims, "t", 1) > 1:
        warnings.warn(f"{path.name}: reading image 0 at time point 0 only "
                      f"({len(reader.image_list)} images, {getattr(img.dims, 't', 1)} time points "
                      "in the file).", RuntimeWarning, stacklevel=3)
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
        # readlif's scale is (x, y, z, t) in pixels per µm for x, y and z (None when
        # the file gives no length for an axis), so the voxel size is its reciprocal.
        scale = img.scale
        if scale:
            def um(i: int):
                s = scale[i] if len(scale) > i else None
                return 1.0 / s if s and s > 0 else None

            voxel_x = um(0) or 1.0
            voxel_y = um(1) or voxel_x
            voxel_z = um(2) or 1.0
            meta["voxel_size_um"] = (voxel_z, voxel_y, voxel_x)
            meta["_read_axes"] = {a for a, i in (("x", 0), ("y", 1), ("z", 2))
                                  if len(scale) > i and scale[i] and scale[i] > 0}
            if um(1) is None and um(0) is not None:
                meta["_axis_source"] = {"y": "copied-from-x"}
                warnings.warn(f"the LIF file gives no y length; y voxel size copied from x "
                              f"({voxel_x} µm).", RuntimeWarning, stacklevel=3)
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
    meta: dict = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": [], "format": "tiff"}

    with tifffile.TiffFile(str(path)) as tif:
        data, axes = tif.series[0].asarray(), tif.series[0].axes
        if tif.ome_metadata:
            meta = _parse_ome_metadata(tif.ome_metadata, meta)
            meta["format"] = "ome_tiff"
        elif tif.imagej_metadata:
            meta = _parse_imagej_metadata(tif.imagej_metadata, meta)
            meta["format"] = "imagej_tiff"

    data = _canonicalize_volume(data, meta, axes, _TIFF_INDEX_AXES)
    return data, meta


def _load_tiff_image(path: Path) -> tuple[np.ndarray, dict]:
    with tifffile.TiffFile(str(path)) as tif:
        data, axes = tif.series[0].asarray(), tif.series[0].axes
    meta: dict = {"voxel_size_um": (1.0, 1.0), "channel_names": [], "format": "tiff"}
    data = _canonicalize_image(data, meta, axes)
    return data, meta


# tifffile axis letters reduced to their first index ('Q', 'I' and the like say
# nothing about what an axis is, so files carrying them fall back to the shape).
_TIFF_INDEX_AXES = {"T": "time point"}


# Length units as OME (UnitsLength) and ImageJ write them -> µm per unit.
# ImageJ descriptions are ASCII, so Fiji spells µm "micron" or "\\u00B5m".
_UM_PER_UNIT = {"pm": 1e-6, "Å": 1e-4, "nm": 1e-3, "µm": 1.0, "μm": 1.0, "um": 1.0,
                "micron": 1.0, "microns": 1.0, "\\u00B5m": 1.0, "\\u00b5m": 1.0,
                "mm": 1e3, "cm": 1e4, "m": 1e6}


def _length_um(value: float, unit: str | None, axis: str) -> tuple[float, str]:
    """``value`` in ``unit`` converted to µm, with its ``voxel_size_source``.

    An unknown or missing unit keeps the value as if it were µm, records the
    source ``"file-unit-unknown"`` and warns.
    """
    factor = _UM_PER_UNIT.get(unit)
    if factor is not None:
        return value * factor, "file"
    said = "no length unit" if unit is None else f"length unit {unit!r}, which is not recognised"
    warnings.warn(f"the file gives the {axis} voxel size {value} with {said}; keeping it as µm. "
                  "Check voxel_size_um.", RuntimeWarning, stacklevel=3)
    return value, "file-unit-unknown"


def _parse_ome_metadata(ome_xml: str, meta: dict) -> dict:
    """``meta`` plus the OME voxel size (converted to µm) and channel names, as a
    new dict; ``meta`` itself is returned unchanged for unparseable XML."""
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return meta
    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        return meta
    size, noted = {"z": 1.0, "y": 1.0, "x": 1.0}, {}
    read = [a for a in "zyx" if _valid_size(pixels.get(f"PhysicalSize{a.upper()}"))]
    for a in read:
        key = f"PhysicalSize{a.upper()}"
        size[a], source = _length_um(float(pixels.get(key)), pixels.get(f"{key}Unit", "µm"), a)
        if source != "file":                    # the OME default unit is µm
            noted[a] = source
    names = [ch.get("Name", "") or ch.get("Fluor", "") for ch in pixels.findall("{*}Channel")]
    return {**meta, "voxel_size_um": (size["z"], size["y"], size["x"]), "_read_axes": set(read),
            "_axis_source": noted, "channel_names": meta["channel_names"] + [n for n in names if n]}


def _parse_imagej_metadata(ij_meta: dict, meta: dict) -> dict:
    """``meta`` plus the ImageJ z spacing converted to µm, as a new dict. ImageJ
    x/y resolution tags are not read."""
    if "spacing" not in ij_meta:
        return meta
    try:
        spacing = float(ij_meta["spacing"])
    except (TypeError, ValueError):
        warnings.warn(f"ImageJ spacing {ij_meta['spacing']!r} is not a number; the z spacing "
                      "is not read.", RuntimeWarning, stacklevel=3)
        return meta
    z, source = _length_um(spacing, ij_meta.get("zunit", ij_meta.get("unit")), "z")
    return {**meta, "voxel_size_um": (z, *meta["voxel_size_um"][1:]), "_read_axes": {"z"},
            "_axis_source": {} if source == "file" else {"z": source}}


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
    if arr.ndim > 4:
        warnings.warn(f"array of shape {arr.shape} has more than 4 non-singleton axes; keeping "
                      "only the first index of each extra leading axis (e.g. time point, scene).",
                      RuntimeWarning, stacklevel=4)
    while arr.ndim > 4:
        arr = arr[0]
    return arr


# Axis letters kept in a canonical array (tifffile convention); S = colour samples,
# which become extra channels.
_CANONICAL_AXES = "CSZYX"


def _by_axis_names(arr: np.ndarray, axes: str, index_axes: dict[str, str]) -> np.ndarray | None:
    """(C, Z, Y, X) from the reader's axis letters (C, S, Z, Y, X), or None when
    the letters do not fit the array or a non-singleton axis is neither one of
    them nor an ``index_axes`` letter (then only the shape can be used).

    Colour samples become extra channels (channel-major). Each ``index_axes``
    axis (time point, position, ...) is reduced to its first index, announced.
    """
    sizes = dict(zip(axes, arr.shape))
    if len(axes) != arr.ndim or len(sizes) != arr.ndim or not {"Y", "X"} <= set(axes):
        return None
    if any(n > 1 and a not in _CANONICAL_AXES and a not in index_axes for a, n in sizes.items()):
        return None
    for a, n in sizes.items():
        if a in index_axes and n > 1:
            warnings.warn(f"keeping only {index_axes[a]} 0 of {n} (axis {a!r} of the file).",
                          RuntimeWarning, stacklevel=4)
    arr = arr[tuple(slice(None) if a in _CANONICAL_AXES else 0 for a in axes)]
    order = [a for a in "CSZ" if a not in sizes] + [a for a in axes if a in _CANONICAL_AXES]
    arr = arr[(np.newaxis,) * (len(order) - arr.ndim)]
    arr = arr.transpose([order.index(a) for a in _CANONICAL_AXES])
    return arr.reshape(-1, *arr.shape[2:])


def _canonicalize_volume(arr: np.ndarray, meta: dict, axes: str | None = None,
                         index_axes: dict[str, str] | None = None) -> np.ndarray:
    """Ensure array is (C, Z, Y, X): by axis name when the reader labels its axes
    (``axes``), otherwise by shape."""
    named = _by_axis_names(arr, axes, index_axes or {}) if axes else None
    arr = named if named is not None else _volume_by_shape(arr)
    n_ch = arr.shape[0]
    if len(meta["channel_names"]) < n_ch:
        for i in range(len(meta["channel_names"]), n_ch):
            meta["channel_names"].append(f"Ch{i + 1}")
    return arr


def _volume_by_shape(arr: np.ndarray) -> np.ndarray:
    """(C, Z, Y, X) guessed from the array shape alone (no axis labels)."""
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
        if arr.shape[0] > 1:
            warnings.warn(f"5D array of shape {arr.shape}: keeping only the first of "
                          f"{arr.shape[0]} entries along the leading axis (e.g. time point).",
                          RuntimeWarning, stacklevel=3)
        arr = arr[0]
    elif ndim == 2:
        arr = arr[np.newaxis, np.newaxis]
    elif ndim > 5:
        arr = _squeeze_singleton_dims(arr)
        if arr.ndim != 4:  # pragma: no cover - squeeze always yields 4D
            return _volume_by_shape(arr)
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} with {ndim} dims")
    return arr


def _canonicalize_image(arr: np.ndarray, meta: dict, axes: str | None = None) -> np.ndarray:
    """Ensure 2D image is (C, Y, X): by axis name when the reader labels a single
    plane (``axes``), otherwise by shape (a labelled z-stack keeps the shape rule)."""
    named = _by_axis_names(arr, axes, _TIFF_INDEX_AXES) if axes else None
    if named is not None and named.shape[1] == 1:
        arr = named[:, 0]
    elif arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3:
        if arr.shape[2] <= 4:
            arr = np.moveaxis(arr, -1, 0)
        elif arr.shape[0] > 4:
            warnings.warn(f"3D array of shape {arr.shape} read as a 2D image: its {arr.shape[0]} "
                          "planes are treated as channels. If it is a z-stack, load it with "
                          "load_volume / quant3d instead.", RuntimeWarning, stacklevel=3)
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
