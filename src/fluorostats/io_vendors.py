"""Per-format readers and metadata parsers: TIFF (OME, ImageJ), Olympus
FluoView, Zeiss CZI, Nikon ND2 and Leica LIF.

Each ``load_*`` takes a path and returns ``(array, meta)`` with the array in
``(C, Z, Y, X)`` (``load_tiff_image``: ``(C, Y, X)``) and ``meta`` holding
``voxel_size_um`` in µm, ``channel_names`` and ``format``, plus the private
``_read_axes`` / ``_axis_source`` notes that ``fluorostats.io`` turns into
``voxel_size_source``. The vendor libraries are optional and imported on use.
``fluorostats.io`` registers these readers and is the public entry point.
"""

from __future__ import annotations

import importlib
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import tifffile

from .io_canonical import canonicalize_image, canonicalize_volume, squeeze_singleton_dims

# Optional vendor reader distribution -> pyproject extra that installs it.
READER_EXTRAS = {"oiffile": "olympus", "czifile": "zeiss", "nd2": "nikon", "readlif": "leica"}


def require(package: str, extra: str | None) -> ModuleType:
    """Import a package or raise an ImportError naming the pip extra that installs it."""
    try:
        return importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"Reading this format requires the '{package}' package.\n"
            f"Install it with: pip install {package}\n"
            f'(or, from a fluorostats checkout: pip install -e ".[{extra}]")'
        ) from None


def _reader(package: str):
    """Import an optional vendor reader, naming the extra that installs it."""
    return require(package, READER_EXTRAS[package])


def _valid_size(val) -> bool:
    """True when ``val`` is a usable (positive, numeric) physical size."""
    return _safe_float(val, -1.0) > 0


def _safe_float(val, default: float) -> float:
    try:
        v = float(val)
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


def _warn_unread(what: str, exc: Exception) -> None:
    """Announce a metadata element that could not be read (its default is kept)."""
    warnings.warn(
        f"{what} could not be read ({type(exc).__name__}: {exc}); the default is kept.",
        RuntimeWarning,
        stacklevel=3,
    )


def _positional_names(names: list[str | None]) -> list[str]:
    """Channel names in file order, an empty one becoming ``Ch<i+1>`` in place so
    every later name keeps its channel."""
    return [n or f"Ch{i + 1}" for i, n in enumerate(names)]


# ---------------------------------------------------------------------------
# Olympus .oib / .oif
# ---------------------------------------------------------------------------


def load_olympus_volume(path: Path) -> tuple[np.ndarray, dict]:
    """Olympus .oib/.oif -> ((C, Z, Y, X), meta); z step from the axis section (nm or µm)."""
    oiffile = _reader("oiffile")

    oif = oiffile.OifFile(str(path))
    arr = oif.asarray()

    meta = _parse_olympus_metadata(oif)
    meta["format"] = "olympus"
    arr, meta = canonicalize_volume(arr, meta)
    oif.close()
    return arr, meta


def _parse_olympus_metadata(oif) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

    settings = oif.mainfile
    voxel_x, voxel_y, voxel_z = 1.0, 1.0, 1.0
    channel_names: list[str | None] = []
    read: set[str] = set()

    for section_name, section in settings.items():
        section_str = str(section_name)

        if re.match(r"Channel \d+ Parameters$", section_str):
            dye = _section_get(section, "DyeName")
            channel_names.append(dye if dye not in ("(null)", "None") else None)

        if section_str == "Reference Image Parameter":
            wx, hy = (
                _section_get(section, "WidthConvertValue"),
                _section_get(section, "HeightConvertValue"),
            )
            voxel_x = _safe_float(wx, voxel_x)
            voxel_y = _safe_float(hy, voxel_y)
            read |= {a for a, v in (("x", wx), ("y", hy)) if _valid_size(v)}

        if "Axis" in section_str and "Parameters Common" in section_str:
            z = _olympus_z_um(section)
            if z is not None:
                voxel_z = z
                read.add("z")

    meta["channel_names"] = _positional_names(channel_names)
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
        warnings.warn(
            f"Olympus PixUnit {pix_unit!r} not recognised; "
            "assuming nm (interval/1000). Verify the z voxel size.",
            RuntimeWarning,
            stacklevel=3,
        )
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


def load_czi_volume(path: Path) -> tuple[np.ndarray, dict]:
    """Zeiss .czi -> ((C, Z, Y, X), meta), first scene; sizes from ``Distance`` (m -> µm)."""
    czifile = _reader("czifile")

    with czifile.CziFile(str(path)) as czi:
        arr = czi.asarray()
        axes = _czi_axes(czi)
        meta = _parse_czi_metadata(czi)

    meta["format"] = "zeiss_czi"
    arr, meta = canonicalize_volume(
        arr if axes else squeeze_singleton_dims(arr), meta, axes, _CZI_INDEX_AXES
    )
    return arr, meta


# CZI dimensions other than C/Z/Y/X/S that are reduced to their first index.
_CZI_INDEX_AXES = {
    "T": "time point",
    "B": "block",
    "V": "view",
    "H": "phase",
    "I": "illumination",
    "R": "rotation",
    "M": "mosaic tile",
    "A": "acquisition",
    "s": "scene",
}
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


def _czi_voxel_um(root: ET.Element) -> tuple[tuple[float, float, float], set[str]]:
    """(z, y, x) voxel size in µm from CZI ``Distance`` elements (metres), and the
    set of axes it found."""
    size = {"X": 1.0, "Y": 1.0, "Z": 1.0}
    read: set[str] = set()
    for dist in root.iter("Distance"):
        value_el = dist.find("Value")
        axis_id = dist.get("Id", "")
        if value_el is None or not value_el.text or axis_id not in size:
            continue
        try:
            size[axis_id] = float(value_el.text) * 1e6
        except ValueError as exc:
            _warn_unread(f"CZI Distance {axis_id!r} value {value_el.text!r}", exc)
            continue
        read.add(axis_id.lower())
    return (size["Z"], size["Y"], size["X"]), read


def _czi_channel_names(root: ET.Element) -> list[str]:
    """Names of the image's channels (``Information/Image/Dimensions/Channels``;
    the experiment tracks and display settings also hold ``Channel`` elements,
    acquired or not) from ``Channel@Name``, falling back to their ``DyeName``."""
    channels = root.findall(".//Information/Image/Dimensions/Channels/Channel")
    names = [ch.get("Name") for ch in channels]
    if not any(names):
        names = [ch.findtext("DyeName") for ch in channels]
    return _positional_names(names)


def _parse_czi_metadata(czi) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

    xml_str = czi.metadata()
    if not xml_str:
        return meta
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as exc:
        _warn_unread("CZI metadata XML", exc)
        return meta
    meta["voxel_size_um"], meta["_read_axes"] = _czi_voxel_um(root)
    meta["channel_names"] = _czi_channel_names(root)
    return meta


# ---------------------------------------------------------------------------
# Nikon .nd2
# ---------------------------------------------------------------------------


def load_nd2_volume(path: Path) -> tuple[np.ndarray, dict]:
    """Nikon .nd2 -> ((C, Z, Y, X), meta); sizes from ``voxel_size()`` (µm)."""
    nd2 = _reader("nd2")

    with nd2.ND2File(str(path)) as f:
        arr = f.asarray()
        axes = "".join(f.sizes) if hasattr(f, "sizes") else None
        meta = _parse_nd2_metadata(f)

    meta["format"] = "nikon_nd2"
    arr, meta = canonicalize_volume(
        arr if axes else squeeze_singleton_dims(arr),
        meta,
        axes,
        {"T": "time point", "P": "position"},
    )
    return arr, meta


def _nd2_calibrated_axes(f, vs) -> set[str]:
    """Axes whose voxel size the ND2 file calibrates. ``voxel_size()`` gives 1 for
    an axis nd2 cannot determine, so the first channel's ``volume.axesCalibrated``
    (x, y, z) decides. The legacy reader never flags z and gives 1 when it finds
    no z step, so there z counts as read when it is not 1."""
    channels = getattr(f.metadata, "channels", None) or []
    flags = dict(zip("xyz", channels[0].volume.axesCalibrated if channels else (False,) * 3))
    if getattr(f, "is_legacy", False):
        flags["z"] = getattr(vs, "z", 1) != 1
    return {a for a in "zyx" if flags[a] and _valid_size(getattr(vs, a, None))}


def _parse_nd2_metadata(f) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

    try:
        vs = f.voxel_size()
        meta["_read_axes"] = _nd2_calibrated_axes(f, vs)
        meta["voxel_size_um"] = tuple(_safe_float(getattr(vs, a, 1.0), 1.0) for a in "zyx")
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        _warn_unread("ND2 voxel size", exc)
    try:
        meta["channel_names"] = _positional_names(
            [
                getattr(ch.channel, "name", "") or getattr(ch, "name", "")
                for ch in f.metadata.channels or []
            ]
        )
    except (AttributeError, TypeError) as exc:
        _warn_unread("ND2 channel names", exc)
    return meta


# ---------------------------------------------------------------------------
# Leica .lif
# ---------------------------------------------------------------------------


def load_lif_volume(path: Path) -> tuple[np.ndarray, dict]:
    """Leica .lif -> ((C, Z, Y, X), meta), image 0 at time point 0."""
    readlif = _reader("readlif")

    reader = readlif.LifFile(str(path))

    if len(reader.image_list) == 0:
        raise ValueError(f"No images found in LIF file: {path}")

    img = reader.get_image(0)
    if len(reader.image_list) > 1 or getattr(img.dims, "t", 1) > 1:
        warnings.warn(
            f"{path.name}: reading image 0 at time point 0 only "
            f"({len(reader.image_list)} images, {getattr(img.dims, 't', 1)} time points "
            "in the file).",
            RuntimeWarning,
            stacklevel=3,
        )
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
    arr, meta = canonicalize_volume(arr, meta, "CZYX")
    return arr, meta


def _parse_lif_metadata(img) -> dict:
    meta: dict = {
        "voxel_size_um": (1.0, 1.0, 1.0),
        "channel_names": [],
    }

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
        meta["_read_axes"] = {
            a
            for a, i in (("x", 0), ("y", 1), ("z", 2))
            if len(scale) > i and scale[i] and scale[i] > 0
        }
        if um(1) is None and um(0) is not None:
            meta["_axis_source"] = {"y": "copied-from-x"}
            warnings.warn(
                f"the LIF file gives no y length; y voxel size copied from x ({voxel_x} µm).",
                RuntimeWarning,
                stacklevel=3,
            )
    meta["channel_names"] = [f"Ch{i + 1}" for i in range(img.channels)]
    return meta


# ---------------------------------------------------------------------------
# TIFF / OME-TIFF
# ---------------------------------------------------------------------------


def load_tiff_volume(path: Path) -> tuple[np.ndarray, dict]:
    """TIFF / OME-TIFF / ImageJ TIFF -> ((C, Z, Y, X), meta), series 0."""
    with tifffile.TiffFile(str(path)) as tif:
        data, axes = tif.series[0].asarray(), tif.series[0].axes
        meta = _tiff_volume_meta(tif)
    data, meta = canonicalize_volume(data, meta, axes, _TIFF_INDEX_AXES)
    return data, meta


def read_tiff_volume_meta(path: Path) -> dict:
    """The ``meta`` of `load_tiff_volume` before channel names are padded, read
    from the TIFF header and metadata only (no pixel data)."""
    with tifffile.TiffFile(str(path)) as tif:
        return _tiff_volume_meta(tif)


def _tiff_volume_meta(tif: tifffile.TiffFile) -> dict:
    """Volume ``meta`` from the OME-XML or ImageJ metadata of an open TIFF."""
    meta: dict = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": [], "format": "tiff"}
    if tif.ome_metadata:
        return {**_parse_ome_metadata(tif.ome_metadata, meta), "format": "ome_tiff"}
    if tif.imagej_metadata:
        xy = _imagej_xy_um(tif, tif.imagej_metadata)
        return {**_parse_imagej_metadata(tif.imagej_metadata, meta, xy), "format": "imagej_tiff"}
    return meta


def load_tiff_image(path: Path) -> tuple[np.ndarray, dict]:
    """TIFF -> ((C, Y, X), meta), series 0; the y/x pixel size and channel names
    are read as `load_tiff_volume` reads them (OME-XML first, then ImageJ)."""
    with tifffile.TiffFile(str(path)) as tif:
        data, axes = tif.series[0].asarray(), tif.series[0].axes
        volume_meta = _tiff_volume_meta(tif)
    meta = {
        **volume_meta,
        "voxel_size_um": volume_meta["voxel_size_um"][1:],
        "_read_axes": volume_meta.get("_read_axes", set()) - {"z"},
        "_axis_source": {a: s for a, s in volume_meta.get("_axis_source", {}).items() if a != "z"},
    }
    data, meta = canonicalize_image(data, meta, axes, _TIFF_INDEX_AXES)
    return data, meta


# tifffile axis letters reduced to their first index ('Q', 'I' and the like say
# nothing about what an axis is, so files carrying them fall back to the shape).
_TIFF_INDEX_AXES = {"T": "time point"}


# Length units as OME (UnitsLength) and ImageJ write them -> µm per unit.
# ImageJ descriptions are ASCII, so Fiji spells µm "micron" or "\\u00B5m".
_UM_PER_UNIT: dict[str | None, float] = {
    "pm": 1e-6,
    "Å": 1e-4,
    "nm": 1e-3,
    "µm": 1.0,
    "μm": 1.0,
    "um": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "\\u00B5m": 1.0,
    "\\u00b5m": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
    "inch": 25400.0,
}


def _length_um(value: float, unit: str | None, axis: str) -> tuple[float, str]:
    """``value`` in ``unit`` converted to µm, with its ``voxel_size_source``.

    An unknown or missing unit keeps the value as if it were µm, records the
    source ``"file-unit-unknown"`` and warns.
    """
    factor = _UM_PER_UNIT.get(unit)
    if factor is not None:
        return value * factor, "file"
    said = "no length unit" if unit is None else f"length unit {unit!r}, which is not recognised"
    warnings.warn(
        f"the file gives the {axis} voxel size {value} with {said}; keeping it as µm. "
        "Check voxel_size_um.",
        RuntimeWarning,
        stacklevel=3,
    )
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
        size[a], source = _length_um(float(pixels.attrib[key]), pixels.get(f"{key}Unit", "µm"), a)
        if source != "file":  # the OME default unit is µm
            noted[a] = source
    names = _positional_names(
        [ch.get("Name") or ch.get("Fluor") for ch in pixels.findall("{*}Channel")]
    )
    return {
        **meta,
        "voxel_size_um": (size["z"], size["y"], size["x"]),
        "_read_axes": set(read),
        "_axis_source": noted,
        "channel_names": meta["channel_names"] + names,
    }


def _imagej_xy_um(tif: tifffile.TiffFile, ij_meta: dict) -> dict | None:
    """ImageJ y/x pixel size in µm from the resolution tags, ``{axis: (µm, source)}``.

    ImageJ writes pixels per ``unit``; with no unit (its uncalibrated "pixel")
    the tags carry no length, so None is returned and the axes stay defaulted.
    """
    unit = ij_meta.get("unit")
    if unit in (None, "", "pixel", "pixels"):
        return None
    tags = cast(tifffile.TiffPage, tif.pages[0]).tags  # the first page is never a TiffFrame
    out = {}
    for axis, key in (("y", "YResolution"), ("x", "XResolution")):
        tag = tags.get(key)
        num, den = tag.value if tag is not None else (0, 0)
        if not num or not den:
            return None
        out[axis] = _length_um(den / num, unit, axis)
    return out


def _parse_imagej_metadata(ij_meta: dict, meta: dict, xy: dict | None = None) -> dict:
    """``meta`` plus the ImageJ voxel size converted to µm, as a new dict: z from
    ``spacing``, y/x from ``xy`` (see `_imagej_xy_um`)."""
    size = dict(zip("zyx", meta["voxel_size_um"]))
    read, noted = set(), {}
    if "spacing" in ij_meta and not _valid_size(ij_meta["spacing"]):
        warnings.warn(
            f"ImageJ spacing {ij_meta['spacing']!r} is not a positive number; the z "
            "spacing is not read.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif "spacing" in ij_meta:
        spacing = float(ij_meta["spacing"])
        size["z"], source = _length_um(spacing, ij_meta.get("zunit", ij_meta.get("unit")), "z")
        read.add("z")
        noted.update({} if source == "file" else {"z": source})
    for axis, (value, source) in (xy or {}).items():
        size[axis] = value
        read.add(axis)
        noted.update({} if source == "file" else {axis: source})
    if not read:
        return meta
    return {
        **meta,
        "voxel_size_um": (size["z"], size["y"], size["x"]),
        "_read_axes": read,
        "_axis_source": noted,
    }
