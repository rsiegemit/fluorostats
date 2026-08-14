"""Tests for io module: canonicalization and round-trip loading.

The proprietary loaders (.oib/.oif/.czi/.nd2/.lif) are NOT exercised here:
they require optional third-party deps plus real vendor files. Only the
core always-available formats (TIFF, PNG, NumPy) are round-tripped.
"""

import numpy as np
import pytest
import tifffile

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
