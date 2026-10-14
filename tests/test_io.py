"""Tests for io module: canonicalization and round-trip loading.

The proprietary loaders (.oib/.oif/.czi/.nd2/.lif) are NOT exercised here:
they require optional third-party deps plus real vendor files. Only the
core always-available formats (TIFF, PNG, NumPy) are round-tripped.
"""

import numpy as np
import pytest
import tifffile

from fluorostats import io
from fluorostats.io import (
    load_volume,
    load_image,
    load_auto,
    supported_formats,
)


@pytest.fixture
def tmp_tiff_volume(tmp_path):
    """Create a synthetic 3-channel, 5-slice TIFF volume."""
    arr = np.random.default_rng(42).integers(0, 4095, (3, 5, 32, 32), dtype=np.uint16)
    path = tmp_path / "test_vol.tif"
    tifffile.imwrite(str(path), arr)
    return path, arr


@pytest.fixture
def tmp_tiff_2d(tmp_path):
    """Create a synthetic single-channel TIFF image."""
    arr = np.random.default_rng(42).integers(0, 255, (64, 64), dtype=np.uint8)
    path = tmp_path / "test_img.tif"
    tifffile.imwrite(str(path), arr)
    return path, arr


class TestLoadVolume:
    def test_tiff_volume_shape(self, tmp_tiff_volume):
        path, original = tmp_tiff_volume
        arr, meta = load_volume(path)
        # Should be canonicalized to (C, Z, Y, X)
        assert arr.ndim == 4
        assert arr.shape[0] == 3  # channels
        assert arr.shape[1] == 5  # z slices

    def test_tiff_volume_values_preserved(self, tmp_tiff_volume):
        path, original = tmp_tiff_volume
        arr, meta = load_volume(path)
        np.testing.assert_array_equal(arr, original)

    def test_metadata_has_required_keys(self, tmp_tiff_volume):
        path, _ = tmp_tiff_volume
        _, meta = load_volume(path)
        assert "voxel_size_um" in meta
        assert "channel_names" in meta
        assert len(meta["voxel_size_um"]) == 3
        assert len(meta["channel_names"]) == 3

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported"):
            load_volume(path)


class TestLoadImage:
    def test_2d_grayscale_canonicalized(self, tmp_tiff_2d):
        path, original = tmp_tiff_2d
        arr, meta = load_image(path)
        assert arr.ndim == 3  # (C, Y, X)
        assert arr.shape[0] == 1
        assert arr.shape[1:] == original.shape

    def test_png_rgb_canonicalized(self, tmp_path):
        import imageio.v3 as iio
        rgb = np.random.default_rng(42).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        iio.imwrite(str(path), rgb)

        arr, meta = load_image(path)
        assert arr.ndim == 3
        assert arr.shape[0] == 3  # C first
        assert arr.shape[1:] == (48, 64)

    def test_npy_2d_image_roundtrip(self, tmp_path):
        arr2d = np.random.default_rng(0).random((40, 50)).astype(np.float32)
        path = tmp_path / "img2d.npy"
        np.save(str(path), arr2d)
        arr, meta = load_image(path)
        assert arr.shape == (1, 40, 50)
        assert meta["channel_names"] == ["Ch1"]
        assert len(meta["voxel_size_um"]) == 2


class TestNpyVolume:
    def test_npy_3d_volume_gets_channel_axis(self, tmp_path):
        vol = np.random.default_rng(0).random((6, 32, 32)).astype(np.float32)
        path = tmp_path / "vol3d.npy"
        np.save(str(path), vol)
        arr, meta = load_volume(path)
        # (Z, Y, X) -> (C=1, Z, Y, X)
        assert arr.shape == (1, 6, 32, 32)
        assert meta["format"] == "numpy"
        assert len(meta["voxel_size_um"]) == 3

    def test_npy_2d_raises_in_volume_loader(self, tmp_path):
        # 2D input to the volume canonicaliser becomes (1,1,Y,X)
        arr2d = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        path = tmp_path / "flat.npy"
        np.save(str(path), arr2d)
        arr, _ = load_volume(path)
        assert arr.shape == (1, 1, 16, 16)


class TestLoadAuto:
    def test_load_auto_png_goes_2d(self, tmp_path):
        import imageio.v3 as iio
        rgb = np.random.default_rng(1).integers(0, 255, (24, 32, 3), dtype=np.uint8)
        path = tmp_path / "auto.png"
        iio.imwrite(str(path), rgb)
        arr, meta = load_auto(path)
        assert arr.ndim == 3 and arr.shape[0] == 3

    def test_load_auto_z1_volume_collapses_to_2d(self, tmp_path):
        # single-Z 3-channel TIFF -> load_auto should drop the Z axis
        arr = np.random.default_rng(2).integers(0, 4095, (2, 1, 20, 20), dtype=np.uint16)
        path = tmp_path / "z1.tif"
        tifffile.imwrite(str(path), arr, photometric="minisblack")
        out, meta = load_auto(path)
        assert out.ndim == 3  # (C, Y, X)
        assert len(meta["voxel_size_um"]) == 2

    def test_load_auto_multiz_volume_stays_3d(self, tmp_path):
        arr = np.random.default_rng(3).integers(0, 4095, (2, 4, 20, 20), dtype=np.uint16)
        path = tmp_path / "z4.tif"
        tifffile.imwrite(str(path), arr, photometric="minisblack")
        out, meta = load_auto(path)
        assert out.ndim == 4 and out.shape[1] == 4


class TestTiffMetadata:
    def test_ome_tiff_physical_sizes_parsed(self, tmp_path):
        arr = np.random.default_rng(0).integers(0, 4095, (2, 3, 16, 16), dtype=np.uint16)
        path = tmp_path / "ome.ome.tif"
        tifffile.imwrite(str(path), arr, ome=True,
                         metadata={"axes": "CZYX", "PhysicalSizeX": 0.5,
                                   "PhysicalSizeY": 0.5, "PhysicalSizeZ": 2.0})
        _, meta = load_volume(path)
        assert meta["format"] == "ome_tiff"
        assert meta["voxel_size_um"] == (2.0, 0.5, 0.5)
        assert len(meta["channel_names"]) == 2

    def test_imagej_tiff_spacing_parsed(self, tmp_path):
        arr = np.random.default_rng(0).integers(0, 4095, (4, 16, 16), dtype=np.uint16)
        path = tmp_path / "ij.tif"
        tifffile.imwrite(str(path), arr, imagej=True,
                         metadata={"spacing": 3.0, "unit": "um"})
        _, meta = load_volume(path)
        assert meta["format"] == "imagej_tiff"
        # z spacing picked up from the ImageJ metadata
        assert meta["voxel_size_um"][0] == 3.0


class TestCanonicalizeEdgeCases:
    def test_5d_volume_squeezed_to_czyx(self, tmp_path):
        # (T=1, C, Z, Y, X) -> canonicalize_volume drops the leading axis
        arr = np.random.default_rng(0).integers(0, 4095, (1, 2, 3, 16, 16), dtype=np.uint16)
        path = tmp_path / "vol5d.npy"
        np.save(str(path), arr)
        out, _ = load_volume(path)
        assert out.shape == (2, 3, 16, 16)

    def test_channel_last_volume_moved_to_front(self, tmp_path):
        # (Z, Y, X, C) with a large leading axis triggers moveaxis to (C,Z,Y,X)
        arr = np.random.default_rng(0).integers(0, 255, (20, 16, 16, 3), dtype=np.uint8)
        path = tmp_path / "chlast.npy"
        np.save(str(path), arr)
        out, _ = load_volume(path)
        assert out.shape[0] == 3  # channel moved to front


class TestSupportedFormatsAndDirs:
    def test_supported_formats_has_core_and_optional(self):
        fmts = supported_formats()
        assert fmts[".tif"] == "TIFF / OME-TIFF"
        assert fmts[".png"] == "PNG"
        assert fmts[".npy"] == "NumPy array"
        # Optional vendor formats are always listed (with or without a "requires" note)
        assert ".czi" in fmts and ".nd2" in fmts

    def test_non_keyence_directory_raises(self, tmp_path):
        d = tmp_path / "not_keyence"
        d.mkdir()
        (d / "readme.txt").write_text("nothing microscopy here")
        with pytest.raises(ValueError, match="Keyence"):
            load_volume(d)


# ---------------------------------------------------------------------------
# io.py:435 — OME channel-name append (valid Channel with a Name)
# ---------------------------------------------------------------------------
def test_ome_metadata_channel_name_append():
    ome_xml = (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image><Pixels PhysicalSizeX="0.5" PhysicalSizeY="0.5" PhysicalSizeZ="2.0">'
        '<Channel Name="EGFP"/>'
        '<Channel Fluor="mCherry"/>'
        '</Pixels></Image></OME>'
    )
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    out = io._parse_ome_metadata(ome_xml, meta)
    assert out["channel_names"] == ["EGFP", "mCherry"]
    assert out["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta == {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}   # not mutated


def test_ome_metadata_without_pixels_returns_meta_unchanged():
    meta = {"voxel_size_um": (1.0, 1.0, 1.0), "channel_names": []}
    assert io._parse_ome_metadata("<OME><Image/></OME>", meta) == meta


def test_imagej_metadata_returns_new_dict():
    meta = {"voxel_size_um": (1.0, 0.5, 0.5), "channel_names": []}
    out = io._parse_imagej_metadata({"spacing": 2.0, "unit": "micron"}, meta)
    assert out["voxel_size_um"] == (2.0, 0.5, 0.5)
    assert meta == {"voxel_size_um": (1.0, 0.5, 0.5), "channel_names": []}
    assert io._parse_imagej_metadata({}, meta) is meta


def test_voxel_size_source_records_defaulted_axes_and_warns(tmp_path):
    # ImageJ TIFFs carry z spacing but their x/y resolution is not read: the
    # defaulted axes are recorded and announced, never silently taken as microns.
    import warnings

    import tifffile

    p = tmp_path / "ij.tif"
    tifffile.imwrite(p, np.zeros((3, 1, 8, 8), np.uint16), imagej=True,
                     metadata={"spacing": 2.5, "unit": "micron", "axes": "ZCYX"})
    with pytest.warns(RuntimeWarning, match="no y/x calibration"):
        _, meta = load_volume(p)
    assert meta["voxel_size_source"] == "z:file,y:default,x:default"
    assert meta["voxel_size_um"] == (2.5, 1.0, 1.0)

    ome = tmp_path / "ome.tif"
    tifffile.imwrite(ome, np.zeros((3, 8, 8), np.uint16), ome=True,
                     metadata={"axes": "ZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
                               "PhysicalSizeZ": 2.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # fully calibrated: no warning
        _, meta2 = load_volume(ome)
    assert meta2["voxel_size_source"] == "z:file,y:file,x:file"


def test_image_without_calibration_warns_for_both_axes(tmp_path):
    import imageio.v3 as iio

    p = tmp_path / "a.png"
    iio.imwrite(p, np.zeros((8, 8), np.uint8))
    with pytest.warns(RuntimeWarning, match="no y/x calibration"):
        _, meta = load_image(p)
    assert meta["voxel_size_source"] == "y:default,x:default"


def test_loaders_announce_discarded_axes(tmp_path):
    # Extra leading axes (time points, scenes) are reduced to their first index:
    # that is announced, as is a z-stack being read as a 2D image.
    import tifffile

    p5 = tmp_path / "t5.npy"
    np.save(p5, np.zeros((2, 1, 3, 8, 8), np.float32))          # (T, C, Z, Y, X)
    with pytest.warns(RuntimeWarning, match="first of 2 entries"):
        arr, _ = load_volume(p5)
    assert arr.shape == (1, 3, 8, 8)

    p6 = tmp_path / "t6.npy"
    np.save(p6, np.zeros((2, 3, 1, 3, 8, 8), np.float32))
    with pytest.warns(RuntimeWarning, match="more than 4 non-singleton axes"):
        arr6, _ = load_volume(p6)
    assert arr6.shape == (1, 3, 8, 8)

    pz = tmp_path / "zstack.tif"
    tifffile.imwrite(pz, np.zeros((6, 8, 8), np.uint16))
    with pytest.warns(RuntimeWarning, match="planes are treated as channels"):
        img, _ = load_image(pz)
    assert img.shape == (6, 8, 8)


# ---------------------------------------------------------------------------
# Axis order comes from the file's axis labels, not from the array shape
# ---------------------------------------------------------------------------

def _two_channel(order: str, n_z: int, value: int = 7) -> np.ndarray:
    """A (Z, C, Y, X) or (C, Z, Y, X) stack whose channel 0 is 0 and channel 1 is `value`."""
    czyx = np.stack([np.zeros((n_z, 16, 16)), np.full((n_z, 16, 16), value)]).astype(np.uint16)
    return czyx if order == "CZYX" else czyx.transpose(1, 0, 2, 3)


@pytest.mark.parametrize("n_z", [5, 30])
def test_imagej_zcyx_hyperstack_channels_by_name(tmp_path, n_z):
    p = tmp_path / "ij.tif"
    tifffile.imwrite(p, _two_channel("ZCYX", n_z), imagej=True, metadata={"axes": "ZCYX"})
    arr, meta = load_volume(p)
    assert arr.shape == (2, n_z, 16, 16)
    assert (arr[0].mean(), arr[1].mean()) == (0, 7)
    assert len(meta["channel_names"]) == 2


@pytest.mark.parametrize("order", ["CZYX", "ZCYX"])
def test_ome_stack_channels_by_name(tmp_path, order):
    p = tmp_path / "s.ome.tif"
    tifffile.imwrite(p, _two_channel(order, 5), ome=True, metadata={"axes": order})
    arr, _ = load_volume(p)
    assert arr.shape == (2, 5, 16, 16)
    assert (arr[0].mean(), arr[1].mean()) == (0, 7)


def test_ome_time_series_keeps_first_time_point_by_name(tmp_path):
    p = tmp_path / "t.ome.tif"
    tzcyx = np.stack([_two_channel("ZCYX", 5), _two_channel("ZCYX", 5, value=9)])
    tifffile.imwrite(p, tzcyx, ome=True, metadata={"axes": "TZCYX"})
    with pytest.warns(RuntimeWarning, match="time point 0 of 2"):
        arr, _ = load_volume(p)
    assert arr.shape == (2, 5, 16, 16)
    assert (arr[0].mean(), arr[1].mean()) == (0, 7)


def test_ome_single_plane_multichannel_is_channels_not_z(tmp_path):
    p = tmp_path / "c.ome.tif"
    tifffile.imwrite(p, np.stack([np.full((16, 16), v, np.uint16) for v in (1, 2, 3)]),
                     ome=True, metadata={"axes": "CYX"})
    vol, _ = load_volume(p)
    assert vol.shape == (3, 1, 16, 16)
    arr, meta = load_auto(p)
    assert arr.shape == (3, 16, 16)
    assert [arr[c].mean() for c in range(3)] == [1, 2, 3]
    assert len(meta["voxel_size_um"]) == 2


def test_rgb_tiff_samples_become_channels(tmp_path):
    p = tmp_path / "rgb.tif"
    rgb = np.stack([np.full((64, 48), v, np.uint8) for v in (10, 20, 30)], -1)
    tifffile.imwrite(p, rgb, photometric="rgb")
    vol, _ = load_volume(p)
    assert vol.shape == (3, 1, 64, 48)
    arr, _ = load_auto(p)
    assert arr.shape == (3, 64, 48)
    assert [arr[c].mean() for c in range(3)] == [10, 20, 30]
    img, _ = load_image(p)
    np.testing.assert_array_equal(img, arr)


def test_unlabelled_tiff_keeps_shape_heuristic(tmp_path):
    # A plain TIFF has no axis names ('QQYX'), so the shape rule still applies.
    p = tmp_path / "plain.tif"
    tifffile.imwrite(p, np.zeros((20, 8, 8, 3), np.uint8))
    with pytest.warns(RuntimeWarning, match="assuming channel-last"):
        arr, _ = load_volume(p)
    assert arr.shape == (3, 20, 8, 8)


# ---------------------------------------------------------------------------
# Physical-size units are converted to µm; unknown units are kept and flagged
# ---------------------------------------------------------------------------

def _ome_with_units(tmp_path, **sizes_and_units):
    p = tmp_path / "u.ome.tif"
    tifffile.imwrite(p, np.zeros((3, 8, 8), np.uint16), ome=True,
                     metadata={"axes": "ZYX", **sizes_and_units})
    return p


def test_ome_physical_size_units_converted_to_um(tmp_path):
    p = _ome_with_units(tmp_path, PhysicalSizeX=500, PhysicalSizeXUnit="nm",
                        PhysicalSizeY=500, PhysicalSizeYUnit="nm",
                        PhysicalSizeZ=0.002, PhysicalSizeZUnit="mm")
    _, meta = load_volume(p)
    assert meta["voxel_size_um"] == pytest.approx((2.0, 0.5, 0.5))
    assert meta["voxel_size_source"] == "z:file,y:file,x:file"


def test_ome_unknown_unit_kept_flagged_and_warned(tmp_path):
    p = _ome_with_units(tmp_path, PhysicalSizeX=0.5, PhysicalSizeY=0.5,
                        PhysicalSizeZ=3.0, PhysicalSizeZUnit="pixel")
    with pytest.warns(RuntimeWarning, match="length unit 'pixel', which is not recognised"):
        _, meta = load_volume(p)
    assert meta["voxel_size_um"] == (3.0, 0.5, 0.5)
    assert meta["voxel_size_source"] == "z:file-unit-unknown,y:file,x:file"


# ImageJ descriptions are ASCII: Fiji writes µm as the escape "\\u00B5m" or as "micron".
@pytest.mark.parametrize("unit, z_um", [("nm", 2.0), ("micron", 2000.0), ("\\u00B5m", 2000.0),
                                        ("mm", 2e6)])
def test_imagej_spacing_unit_converted_to_um(tmp_path, unit, z_um):
    p = tmp_path / "ij.tif"
    tifffile.imwrite(p, np.zeros((3, 1, 8, 8), np.uint16), imagej=True,
                     metadata={"spacing": 2000.0, "unit": unit, "axes": "ZCYX"})
    _, meta = load_volume(p)
    assert meta["voxel_size_um"][0] == pytest.approx(z_um)
    assert meta["voxel_size_source"].startswith("z:file,")


@pytest.mark.parametrize("extra, said", [({}, "no length unit"), ({"unit": "inch"}, "'inch'")])
def test_imagej_spacing_without_known_unit_is_flagged(tmp_path, extra, said):
    p = tmp_path / "ij.tif"
    tifffile.imwrite(p, np.zeros((3, 1, 8, 8), np.uint16), imagej=True,
                     metadata={"spacing": 2.0, "axes": "ZCYX", **extra})
    with pytest.warns(RuntimeWarning, match=said):
        _, meta = load_volume(p)
    assert meta["voxel_size_um"][0] == 2.0
    assert meta["voxel_size_source"].startswith("z:file-unit-unknown,")
