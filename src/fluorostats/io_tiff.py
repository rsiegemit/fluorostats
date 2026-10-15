"""TIFF reader and metadata parsers: OME-TIFF, ImageJ TIFF and plain TIFF.

``load_tiff_volume`` -> ``(C, Z, Y, X)`` and ``load_tiff_image`` -> ``(C, Y, X)``,
with ``meta`` as the vendor readers give it (see ``io_vendors``): the voxel size
in µm converted from the file's length unit, ``channel_names`` and ``format``.
``fluorostats.io`` registers these readers and is the public entry point.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import numpy as np
import tifffile

from .io_canonical import canonicalize_image, canonicalize_volume
from .io_vendors import positional_names, valid_size, warn_first_only, warn_unread


def load_tiff_volume(path: Path) -> tuple[np.ndarray, dict]:
    """TIFF / OME-TIFF / ImageJ TIFF -> ((C, Z, Y, X), meta), series 0."""
    with tifffile.TiffFile(str(path)) as tif:
        data, axes = tif.series[0].asarray(), tif.series[0].axes
        meta = _tiff_volume_meta(tif)
        _warn_first_series(tif, path)
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


def _warn_first_series(tif: tifffile.TiffFile, path: Path) -> None:
    """Announce an OME-TIFF holding several images (series), of which the loaders
    read the first. Other TIFFs are not counted: their extra series are
    thumbnails or reduced copies, and pyramid levels are not series."""
    if tif.is_ome:
        warn_first_only(path, len(tif.series), "series")


def load_tiff_image(path: Path) -> tuple[np.ndarray, dict]:
    """TIFF -> ((C, Y, X), meta), series 0; the y/x pixel size and channel names
    are read as `load_tiff_volume` reads them (OME-XML first, then ImageJ)."""
    with tifffile.TiffFile(str(path)) as tif:
        data, axes = tif.series[0].asarray(), tif.series[0].axes
        volume_meta = _tiff_volume_meta(tif)
        _warn_first_series(tif, path)
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
    except ET.ParseError as exc:
        warn_unread("OME-XML (voxel size and channel names)", exc)
        return meta
    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        return meta
    size, noted = {"z": 1.0, "y": 1.0, "x": 1.0}, {}
    read = [a for a in "zyx" if valid_size(pixels.get(f"PhysicalSize{a.upper()}"))]
    for a in read:
        key = f"PhysicalSize{a.upper()}"
        size[a], source = _length_um(float(pixels.attrib[key]), pixels.get(f"{key}Unit", "µm"), a)
        if source != "file":  # the OME default unit is µm
            noted[a] = source
    names = positional_names(
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


def _magnitude(value):
    """``|value|`` as a float, or ``value`` itself when it is not a number."""
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return value


def _parse_imagej_metadata(ij_meta: dict, meta: dict, xy: dict | None = None) -> dict:
    """``meta`` plus the ImageJ voxel size converted to µm, as a new dict: z from
    ``spacing``, y/x from ``xy`` (see `_imagej_xy_um`)."""
    size = dict(zip("zyx", meta["voxel_size_um"]))
    read, noted = set(), {}
    spacing = _magnitude(ij_meta.get("spacing"))  # negative: a stack acquired top-down
    if "spacing" in ij_meta and not valid_size(spacing):
        warnings.warn(
            f"ImageJ spacing {ij_meta['spacing']!r} is not a nonzero number; the z "
            "spacing is not read.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif "spacing" in ij_meta:
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
