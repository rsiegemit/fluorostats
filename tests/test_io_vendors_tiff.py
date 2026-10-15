"""TIFF metadata parser tests: OME-XML and ImageJ calibration."""

import types
import warnings

import numpy as np
import pytest
import tifffile

from fluorostats import io, io_vendors


def test_ome_metadata_parse_error():
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    out = io_vendors._parse_ome_metadata("<not valid xml", meta)
    assert out["voxel_size_um"] == (1.0, 1.0, 1.0)


def test_imagej_metadata_unparseable_spacing_warns():
    # spacing present but non-float -> announced, defaults kept
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    with pytest.warns(RuntimeWarning, match="'notanumber' is not a positive number"):
        out = io_vendors._parse_imagej_metadata({"spacing": "notanumber"}, meta)
    assert out["voxel_size_um"] == (1.0, 1.0, 1.0)


def test_ome_metadata_channel_name_append():
    ome_xml = (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image><Pixels PhysicalSizeX="0.5" PhysicalSizeY="0.5" PhysicalSizeZ="2.0">'
        '<Channel Name="EGFP"/>'
        '<Channel Fluor="mCherry"/>'
        "</Pixels></Image></OME>"
    )
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    out = io_vendors._parse_ome_metadata(ome_xml, meta)
    assert out["channel_names"] == ["EGFP", "mCherry"]
    assert out["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta == {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}  # not mutated


def test_ome_metadata_without_pixels_returns_meta_unchanged():
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    assert io_vendors._parse_ome_metadata("<OME><Image/></OME>", meta) == meta


def test_imagej_metadata_returns_new_dict():
    meta = {"voxel_size_um": (1.0, 0.5, 0.5), "channel_names": []}
    out = io_vendors._parse_imagej_metadata({"spacing": 2.0, "unit": "micron"}, meta)
    assert out["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta == {"voxel_size_um": (1.0, 0.5, 0.5), "channel_names": []}
    assert io_vendors._parse_imagej_metadata({}, meta) is meta


@pytest.mark.parametrize("unit, scale", [("micron", 1.0), ("nm", 1e-3)])
def test_imagej_xy_resolution_is_read_with_its_unit(tmp_path, unit, scale):
    """ImageJ stores x/y pixel size as resolution tags (pixels per `unit`)."""
    p = tmp_path / "ij.tif"
    tifffile.imwrite(
        p,
        np.zeros((3, 8, 8), np.uint16),
        imagej=True,
        resolution=(2.0, 4.0),
        metadata={"axes": "ZYX", "spacing": 3.0, "unit": unit},
    )
    vol, meta = io.load_volume(p)
    assert meta["voxel_size_um"] == pytest.approx((3.0 * scale, 0.25 * scale, 0.5 * scale))
    assert meta["voxel_size_source"] == "z:file,y:file,x:file"
    img, imeta = io.load_image(p)
    assert imeta["voxel_size_um"] == pytest.approx((0.25 * scale, 0.5 * scale))
    assert imeta["voxel_size_source"] == "y:file,x:file"


def test_imagej_resolution_without_a_unit_is_not_calibration(tmp_path):
    """No unit is ImageJ's uncalibrated 'pixel': the tags say nothing about µm."""
    p = tmp_path / "ij.tif"
    tifffile.imwrite(
        p,
        np.zeros((3, 8, 8), np.uint16),
        imagej=True,
        resolution=(2.0, 4.0),
        metadata={"axes": "ZYX"},
    )
    with pytest.warns(RuntimeWarning, match="no y/x calibration"):
        img, imeta = io.load_image(p)
    assert imeta["voxel_size_um"] == (1.0, 1.0)


@pytest.mark.parametrize(
    "tags",
    [
        {},
        {
            "XResolution": types.SimpleNamespace(value=(0, 1)),
            "YResolution": types.SimpleNamespace(value=(0, 1)),
        },
    ],
)
def test_imagej_xy_missing_or_zero_resolution_is_not_calibration(tags):
    tif = types.SimpleNamespace(pages=[types.SimpleNamespace(tags=tags)])
    assert io_vendors._imagej_xy_um(tif, {"unit": "micron"}) is None


def test_ome_unnamed_channel_keeps_its_position(tmp_path):
    # Regression: an empty Name used to be dropped, so 'GFP' moved to channel 0 and
    # the signal channel was labelled 'Ch2'.
    p = tmp_path / "unnamed.ome.tif"
    data = np.zeros((2, 3, 8, 8), np.uint16)
    data[1] = 5
    tifffile.imwrite(p, data, ome=True, metadata={"axes": "CZYX", "Channel": {"Name": ["", "GFP"]}})
    with pytest.warns(RuntimeWarning, match="calibration"):
        arr, meta = io.load_volume(p)
    assert meta["channel_names"] == ["Ch1", "GFP"]
    assert arr[meta["channel_names"].index("GFP")].max() == 5


def test_2d_ome_tiff_reads_calibration_and_channel_names(tmp_path):
    # Regression: the 2D reader read only ImageJ y/x, so a 2D OME-TIFF loaded with
    # 1.0 µm pixels and positional names while load_auto gave the file's values.
    p = tmp_path / "cyx.ome.tif"
    data = np.zeros((2, 8, 8), np.uint16)
    data[1] = 5
    tifffile.imwrite(
        p,
        data,
        ome=True,
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 0.25,
            "PhysicalSizeY": 0.25,
            "Channel": {"Name": ["DAPI", "GFP"]},
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # fully calibrated in y/x: no warning
        arr, meta = io.load_image(p)
    assert meta["voxel_size_um"] == (0.25, 0.25)
    assert meta["voxel_size_source"] == "y:file,x:file"
    assert meta["channel_names"] == ["DAPI", "GFP"]
    assert meta["format"] == "ome_tiff"
    assert arr[meta["channel_names"].index("GFP")].max() == 5
    _, auto_meta = io.load_auto(p)
    assert {k: auto_meta[k] for k in meta} == meta


@pytest.mark.parametrize("spacing", [0.0, -2.0, float("nan")])
def test_imagej_non_positive_spacing_is_not_calibration(tmp_path, spacing):
    # Regression: spacing 0 or negative was taken as the z voxel size, labelled 'file'.
    p = tmp_path / "ij.tif"
    tifffile.imwrite(
        p,
        np.zeros((3, 8, 8), np.uint16),
        imagej=True,
        resolution=(2.0, 2.0),
        metadata={"axes": "ZYX", "spacing": spacing, "unit": "um"},
    )
    with pytest.warns(RuntimeWarning, match="not a positive number"):
        vol, meta = io.load_volume(p)
    assert meta["voxel_size_um"] == (1.0, 0.5, 0.5)
    assert meta["voxel_size_source"] == "z:default,y:file,x:file"
    with pytest.warns(RuntimeWarning, match="not a positive number"):
        assert io.read_voxel_size(p) == ((1.0, 0.5, 0.5), "z:default,y:file,x:file")


def test_imagej_inch_unit_is_converted(tmp_path):
    # Regression: 'inch' (ImageJ's unit for DPI-tagged images) was kept as µm.
    p = tmp_path / "inch.tif"
    tifffile.imwrite(
        p, np.zeros((8, 9), np.uint8), imagej=True, metadata={"unit": "inch"}, resolution=(72, 72)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, meta = io.load_image(p)
    assert meta["voxel_size_um"] == pytest.approx((25400 / 72, 25400 / 72))
    assert meta["voxel_size_source"] == "y:file,x:file"
