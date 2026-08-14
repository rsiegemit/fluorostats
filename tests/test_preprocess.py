"""Tests for fluorostats.preprocess: channel selection, denoise, bg, auto-crop."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats import preprocess


# ---------------------------------------------------------------------------
# select_green_channel — name match, override (int/str), fallback heuristics
# ---------------------------------------------------------------------------

def test_select_green_channel_matches_name_pattern():
    arr = np.arange(3 * 4).reshape(3, 4)
    out = preprocess.select_green_channel(arr, ["DAPI", "Alexa 488", "PI"])
    np.testing.assert_array_equal(out, arr[1])


def test_select_green_channel_int_override():
    arr = np.arange(3 * 4).reshape(3, 4)
    out = preprocess.select_green_channel(arr, ["a", "b", "c"], override=2)
    np.testing.assert_array_equal(out, arr[2])


def test_select_green_channel_str_override():
    arr = np.arange(3 * 4).reshape(3, 4)
    out = preprocess.select_green_channel(arr, ["red", "myGreenLabel", "blue"],
                                          override="green")
    np.testing.assert_array_equal(out, arr[1])


def test_select_green_channel_int_override_out_of_range_raises():
    arr = np.zeros((2, 3))
    with pytest.raises(ValueError, match="out of range"):
        preprocess.select_green_channel(arr, ["a", "b"], override=5)


def test_select_green_channel_str_override_no_match_raises():
    arr = np.zeros((2, 3))
    with pytest.raises(ValueError, match="No channel matching"):
        preprocess.select_green_channel(arr, ["red", "blue"], override="green")


def test_select_green_channel_fallbacks_by_count():
    # No name match -> heuristic by channel count
    single = np.arange(4).reshape(1, 4)
    np.testing.assert_array_equal(
        preprocess.select_green_channel(single, ["x"]), single[0])

    rgb = np.arange(3 * 4).reshape(3, 4)
    np.testing.assert_array_equal(
        preprocess.select_green_channel(rgb, ["x", "y", "z"]), rgb[1])  # index 1

    two = np.arange(2 * 4).reshape(2, 4)
    np.testing.assert_array_equal(
        preprocess.select_green_channel(two, ["x", "y"]), two[1])  # Ch2

    four = np.arange(4 * 4).reshape(4, 4)
    np.testing.assert_array_equal(
        preprocess.select_green_channel(four, ["a", "b", "c", "d"]), four[0])


# ---------------------------------------------------------------------------
# denoise — 2D and 3D per-slice
# ---------------------------------------------------------------------------

def test_denoise_2d_preserves_shape_and_range():
    img = np.random.default_rng(0).random((16, 16)) * 100
    out = preprocess.denoise(img, sigma=1.0)
    assert out.shape == img.shape
    # preserve_range keeps values in the original scale (not 0..1)
    assert out.max() > 1.0


def test_denoise_3d_blurs_each_slice():
    stack = np.random.default_rng(0).random((4, 16, 16)) * 100
    out = preprocess.denoise(stack, sigma=1.0)
    assert out.shape == stack.shape
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# background_subtract — 2D and 3D stack
# ---------------------------------------------------------------------------

def test_background_subtract_2d():
    img = np.zeros((32, 32), dtype=np.float64)
    img[14:18, 14:18] = 200.0
    out = preprocess.background_subtract(img, radius=5)
    assert out.shape == img.shape
    assert out.max() > 0  # the bright feature survives top-hat


def test_background_subtract_3d_stack():
    stack = np.zeros((3, 32, 32), dtype=np.float64)
    stack[:, 14:18, 14:18] = 200.0
    out = preprocess.background_subtract(stack, radius=5)
    assert out.shape == stack.shape


# ---------------------------------------------------------------------------
# auto_crop — multichannel input, and no-border passthrough
# ---------------------------------------------------------------------------

def test_auto_crop_multichannel_trims_border():
    # Build a (C, Y, X) image with a uniform gray frame and a textured centre.
    img = np.zeros((2, 40, 40), dtype=np.float64)
    img[:, 8:32, 8:32] = np.random.default_rng(0).random((2, 24, 24)) * 255
    cropped, (y0, y1, x0, x1) = preprocess.auto_crop(img, margin=0)
    assert cropped.ndim == 3 and cropped.shape[0] == 2
    # A border was detected, so the crop is strictly smaller than the input.
    assert (y1 - y0) < 40 and (x1 - x0) < 40


def test_auto_crop_uniform_image_returns_unchanged():
    # Zero-variance everywhere -> no active rows/cols -> full-frame passthrough.
    img = np.zeros((20, 20), dtype=np.float64)
    out, coords = preprocess.auto_crop(img)
    assert out.shape == img.shape
    assert coords == (0, 20, 0, 20)
