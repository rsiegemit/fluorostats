"""TIFF metadata parser tests: OME-XML and ImageJ calibration."""

import types
import warnings

import numpy as np
import pytest
import tifffile

from fluorostats import io, io_tiff


def test_ome_metadata_parse_error():
    # announced like the CZI path: the channel names and calibration are lost
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    with pytest.warns(
        RuntimeWarning, match="OME-XML .voxel size and channel names. could not be read"
    ):
        out = io_tiff._parse_ome_metadata("<not valid xml", meta)
    assert out is meta


def test_imagej_metadata_unparseable_spacing_warns():
    # spacing present but non-float -> announced, defaults kept
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    with pytest.warns(RuntimeWarning, match="'notanumber' is not a nonzero number"):
        out = io_tiff._parse_imagej_metadata({"spacing": "notanumber"}, meta)
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
    out = io_tiff._parse_ome_metadata(ome_xml, meta)
    assert out["channel_names"] == ["EGFP", "mCherry"]
    assert out["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta == {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}  # not mutated


def test_ome_metadata_without_pixels_returns_meta_unchanged():
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    assert io_tiff._parse_ome_metadata("<OME><Image/></OME>", meta) == meta


def test_imagej_metadata_returns_new_dict():
    meta = {"voxel_size_um": (1.0, 0.5, 0.5), "channel_names": []}
    out = io_tiff._parse_imagej_metadata({"spacing": 2.0, "unit": "micron"}, meta)
    assert out["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta == {"voxel_size_um": (1.0, 0.5, 0.5), "channel_names": []}
    assert io_tiff._parse_imagej_metadata({}, meta) is meta


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
    assert io_tiff._imagej_xy_um(tif, {"unit": "micron"}) is None


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


@pytest.mark.parametrize("spacing", [0.0, float("nan")])
def test_imagej_zero_spacing_is_not_calibration(tmp_path, spacing):
    # Regression: spacing 0 was taken as the z voxel size, labelled 'file'.
    p = tmp_path / "ij.tif"
    tifffile.imwrite(
        p,
        np.zeros((3, 8, 8), np.uint16),
        imagej=True,
        resolution=(2.0, 2.0),
        metadata={"axes": "ZYX", "spacing": spacing, "unit": "um"},
    )
    with pytest.warns(RuntimeWarning, match="not a nonzero number"):
        vol, meta = io.load_volume(p)
    assert meta["voxel_size_um"] == (1.0, 0.5, 0.5)
    assert meta["voxel_size_source"] == "z:default,y:file,x:file"
    with pytest.warns(RuntimeWarning, match="not a nonzero number"):
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


def _ome_rgb(path, shape, axes):
    """OME RGB file whose channel c, sample s holds the value 10 * c + s."""
    a = np.zeros(shape, np.uint8)
    for c in range(2):
        for s in range(3):
            a[c, ..., s] = 10 * c + s
    tifffile.imwrite(
        path,
        a,
        ome=True,
        photometric="rgb",
        metadata={"axes": axes, "Channel": {"Name": ["DAPI", "GFP"]}},
    )


@pytest.mark.parametrize(
    "load, shape, axes",
    [
        (io.load_volume, (2, 3, 8, 9, 3), "CZYXS"),
        (io.load_image, (2, 8, 9, 3), "CYXS"),
    ],
)
def test_ome_rgb_channel_names_follow_their_samples(tmp_path, load, shape, axes):
    # Regression: the samples were expanded channel-major but the names were not, so
    # 'GFP' named DAPI's second sample and GFP's data was under Ch4-Ch6.
    p = tmp_path / "rgb.ome.tif"
    _ome_rgb(p, shape, axes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # uncalibrated
        arr, meta = load(p)
    assert meta["channel_names"] == [f"{n}[{s}]" for n in ("DAPI", "GFP") for s in range(3)]
    assert [int(arr[i].max()) for i in range(6)] == [10 * c + s for c in range(2) for s in range(3)]


def test_imagej_negative_spacing_gives_its_magnitude(tmp_path):
    # A stack acquired top-down has a negative spacing; it was read as -2.0 µm.
    p = tmp_path / "ij.tif"
    tifffile.imwrite(
        p,
        np.zeros((3, 8, 8), np.uint16),
        imagej=True,
        resolution=(2.0, 2.0),
        metadata={"axes": "ZYX", "spacing": -2.0, "unit": "um"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, meta = io.load_volume(p)
    assert meta["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta["voxel_size_source"] == "z:file,y:file,x:file"


def test_ome_reading_the_first_of_several_series_is_announced(tmp_path):
    # Regression: only the first OME Image was read, without a word.
    p = tmp_path / "two.ome.tif"
    with tifffile.TiffWriter(p, ome=True) as tif:
        for value in (1, 2):
            tif.write(np.full((3, 8, 8), value, np.uint16), metadata={"axes": "ZYX"})
    with pytest.warns(RuntimeWarning, match="two.ome.tif: reading the first of 2 series only"):
        vol, _ = io.load_volume(p)
    assert vol.shape == (1, 3, 8, 8) and vol.max() == 1
    with pytest.warns(RuntimeWarning, match="the first of 2 series only"):
        io.load_image(p)
