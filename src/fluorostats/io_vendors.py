"""Per-format readers and metadata parsers for the vendor formats: Olympus
FluoView, Zeiss CZI, Nikon ND2 and Leica LIF (TIFF is read by ``io_tiff``).

Each ``load_*`` takes a path and returns ``(array, meta)`` with the array in
``(C, Z, Y, X)`` and ``meta`` holding
``voxel_size_um`` in µm, ``channel_names`` and ``format``, plus the private
``_read_axes`` / ``_axis_source`` notes that ``fluorostats.io`` turns into
``voxel_size_source``. The vendor libraries are optional and imported on use.
``fluorostats.io`` registers these readers and is the public entry point. The
metadata helpers (``valid_size``, ``warn_unread``, ``positional_names``) are
shared with ``io_tiff``.
"""

from __future__ import annotations

import importlib
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import numpy as np
import tifffile

from .io_canonical import canonicalize_volume, squeeze_singleton_dims

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


def valid_size(val) -> bool:
    """True when ``val`` is a usable (positive, numeric) physical size."""
    return _safe_float(val, -1.0) > 0


def _safe_float(val, default: float) -> float:
    try:
        v = float(val)
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


def warn_unread(what: str, exc: Exception) -> None:
    """Announce a metadata element that could not be read (its default is kept)."""
    warnings.warn(
        f"{what} could not be read ({type(exc).__name__}: {exc}); the default is kept.",
        RuntimeWarning,
        stacklevel=3,
    )


def positional_names(names: list[str | None]) -> list[str]:
    """Channel names in file order, an empty one becoming ``Ch<i+1>`` in place so
    every later name keeps its channel."""
    return [n or f"Ch{i + 1}" for i, n in enumerate(names)]


# ---------------------------------------------------------------------------
# Olympus .oib / .oif
# ---------------------------------------------------------------------------


# An Olympus image is one TIFF per plane, named by its axis indices
# (``s_C001Z002T003.tif``). The pattern is anchored at the end of the name: a
# file prefix can hold letter-digit runs (``y2_40X_2_C001Z001.tif``) that
# oiffile's own pattern takes for axes, and its axis order does not describe
# ``OifFile.asarray()`` either.
_OLYMPUS_INDEX = r"_?(?:([CZTLAP])(\d{1,4}))?"
_OLYMPUS_PLANE_NAME = r"(?i)_?([CZTLAP])(\d{1,4})" + _OLYMPUS_INDEX * 5 + r"\.tif$"
# Olympus axes reduced to their first index, announced.
_OLYMPUS_INDEX_AXES = {"T": "time point", "L": "lambda step"}


def load_olympus_volume(path: Path) -> tuple[np.ndarray, dict]:
    """Olympus .oib/.oif -> ((C, Z, Y, X), meta), ordered by the axis letters in
    the plane file names; z step from the axis section (nm or µm)."""
    oiffile = _reader("oiffile")
    with oiffile.OifFile(str(path)) as oif:
        files = list(oif.series[0])
        planes = tifffile.TiffSequence(files, imread=oif.asarray, pattern=_OLYMPUS_PLANE_NAME)
        arr, axes = planes.asarray(), planes.axes.upper()
        sizes = _olympus_axis_sizes(oif.mainfile)
        meta = _parse_olympus_metadata(oif)
    _check_olympus_planes(dict(zip(axes, planes.shape)), len(files), sizes, path)
    meta["format"] = "olympus"
    return canonicalize_volume(arr, meta, axes + "YX", _OLYMPUS_INDEX_AXES)


def _olympus_axis_sizes(mainfile) -> dict[str, int]:
    """``{axis code: MaxSize}`` of the mainfile's axes other than X and Y that the
    image has (an absent axis is listed with MaxSize 0)."""
    sizes = {}
    for name, section in mainfile.items():
        code = _section_get(section, "AxisCode")
        n = int(_safe_float(_section_get(section, "MaxSize"), 0))
        if "Axis" in str(name) and "Parameters Common" in str(name) and code not in "XY" and n:
            sizes[code] = n
    return sizes


def _check_olympus_planes(
    found: dict[str, int], n_files: int, expected: dict[str, int], path: Path
) -> None:
    """Raise when the plane files do not make the image the mainfile describes,
    rather than return a stack with planes missing or out of place."""
    wrong = [
        f"{a}: {found.get(a, 0)} planes, mainfile {n}"
        for a, n in expected.items()
        if found.get(a, 1 if n == 1 else 0) != n
    ]
    if n_files != int(np.prod(list(found.values()))):
        wrong.append(f"{n_files} plane files for {found}")
    if wrong:
        raise ValueError(
            f"{path.name}: the plane files do not match the image the file describes "
            f"({'; '.join(wrong)}), so it is not read."
        )


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
            read |= {a for a, v in (("x", wx), ("y", hy)) if valid_size(v)}

        if "Axis" in section_str and "Parameters Common" in section_str:
            z = _olympus_z_um(section)
            if z is not None:
                voxel_z = z
                read.add("z")

    meta["channel_names"] = positional_names(channel_names)
    meta["voxel_size_um"] = (voxel_z, voxel_y, voxel_x)
    meta["_read_axes"] = read
    return meta


def _olympus_z_um(section) -> float | None:
    """Z step (µm) from an Olympus axis section; None unless it is a Z axis of
    more than one slice with a positive interval."""
    if (
        _section_get(section, "AxisCode") != "Z"
        or _safe_float(_section_get(section, "MaxSize"), 0) <= 1
    ):
        return None  # not z, or a z axis the image does not have (MaxSize 0 or 1)
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
            warn_unread(f"CZI Distance {axis_id!r} value {value_el.text!r}", exc)
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
    return positional_names(names)


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
        warn_unread("CZI metadata XML", exc)
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
    return {a for a in "zyx" if flags[a] and valid_size(getattr(vs, a, None))}


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
        warn_unread("ND2 voxel size", exc)
    try:
        meta["channel_names"] = positional_names(
            [
                getattr(ch.channel, "name", "") or getattr(ch, "name", "")
                for ch in f.metadata.channels or []
            ]
        )
    except (AttributeError, TypeError) as exc:
        warn_unread("ND2 channel names", exc)
    return meta


# ---------------------------------------------------------------------------
# Leica .lif
# ---------------------------------------------------------------------------


def load_lif_volume(path: Path) -> tuple[np.ndarray, dict]:
    """Leica .lif -> ((C, Z, Y, X), meta), image 0 at time point 0."""
    _reader("readlif")  # names the extra when readlif is missing
    # readlif's package holds only __version__; LifFile lives in readlif.reader.
    reader = importlib.import_module("readlif.reader").LifFile(str(path))

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
    # A negative length (a stack acquired top-down) gives its magnitude.
    scale = img.scale
    if scale:

        def um(i: int):
            s = scale[i] if len(scale) > i else None
            return 1.0 / abs(s) if s else None

        voxel_x = um(0) or 1.0
        voxel_y = um(1) or voxel_x
        voxel_z = um(2) or 1.0
        meta["voxel_size_um"] = (voxel_z, voxel_y, voxel_x)
        meta["_read_axes"] = {a for a, i in (("x", 0), ("y", 1), ("z", 2)) if um(i) is not None}
        if um(1) is None and um(0) is not None:
            meta["_axis_source"] = {"y": "copied-from-x"}
            warnings.warn(
                f"the LIF file gives no y length; y voxel size copied from x ({voxel_x} µm).",
                RuntimeWarning,
                stacklevel=3,
            )
    meta["channel_names"] = [f"Ch{i + 1}" for i in range(img.channels)]
    return meta
