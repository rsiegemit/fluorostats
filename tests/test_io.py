"""Tests for io module: canonicalization and round-trip loading."""

import numpy as np
import pytest
import tifffile
from pathlib import Path

from fluorostats.io import load_volume, load_image


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
