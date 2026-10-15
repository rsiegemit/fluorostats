"""TIFF metadata parser tests: OME-XML and ImageJ calibration."""

import types

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
    with pytest.warns(RuntimeWarning, match="'notanumber' is not a number"):
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
