"""Tests for fluorostats.depth (axial intensity profiling)."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats.depth import (
    intensity_depth_profile,
    subtract_background,
    normalize_to_surface,
    auc_depth,
    resample_to_depth,
)


def _ramp_volume(per_slice, ny=4, nx=5):
    """(Z, Y, X) volume whose slice i is uniformly per_slice[i]."""
    per_slice = np.asarray(per_slice, dtype=float)
    return np.ones((len(per_slice), ny, nx)) * per_slice[:, None, None]


# ---------------------------------------------------------------------------
# intensity_depth_profile
# ---------------------------------------------------------------------------

def test_profile_mean_and_depth_axis():
    vol = _ramp_volume([10, 8, 6, 4])
    prof = intensity_depth_profile(vol, voxel_size_z_um=5.0)
    np.testing.assert_allclose(prof.mean, [10, 8, 6, 4])
    np.testing.assert_allclose(prof.depth_um, [0, 5, 10, 15])
    # uniform slices -> zero spatial spread
    np.testing.assert_allclose(prof.sem, 0.0)


def test_profile_selects_channel_from_4d():
    vol4d = np.stack([_ramp_volume([1, 1, 1]), _ramp_volume([9, 9, 9])])  # (C,Z,Y,X)
    prof = intensity_depth_profile(vol4d, 1.0, channel=1)
    np.testing.assert_allclose(prof.mean, [9, 9, 9])


def test_profile_rejects_bad_spacing():
    with pytest.raises(ValueError):
        intensity_depth_profile(_ramp_volume([1, 2]), voxel_size_z_um=0.0)


def test_median_reducer_ignores_outlier():
    vol = _ramp_volume([10, 10])
    vol[0, 0, 0] = 1000.0  # one hot pixel
    prof = intensity_depth_profile(vol, 1.0, reducer="median")
    np.testing.assert_allclose(prof.mean[0], 10.0)


# ---------------------------------------------------------------------------
# subtract_background
# ---------------------------------------------------------------------------

def test_subtract_scalar_background():
    prof = intensity_depth_profile(_ramp_volume([10, 8, 6]), 1.0)
    out = subtract_background(prof, 5.0)
    np.testing.assert_allclose(out.mean, [5, 3, 1])


def test_subtract_clips_negative():
    prof = intensity_depth_profile(_ramp_volume([3, 1]), 1.0)
    out = subtract_background(prof, 2.0)
    np.testing.assert_allclose(out.mean, [1, 0])  # 1-2 -> clipped to 0


def test_subtract_profile_background_is_depth_matched():
    # signal at 1 um spacing, blank at 2 um spacing but same flat value
    signal = intensity_depth_profile(_ramp_volume([10, 10, 10, 10]), 1.0)
    blank = intensity_depth_profile(_ramp_volume([4, 4]), 2.0)
    out = subtract_background(signal, blank)
    np.testing.assert_allclose(out.mean, [6, 6, 6, 6])


def test_resample_holds_endpoints():
    blank = intensity_depth_profile(_ramp_volume([5, 5]), 2.0)  # depths 0,2
    got = resample_to_depth(blank, np.array([0.0, 1.0, 10.0]))
    np.testing.assert_allclose(got, [5, 5, 5])


# ---------------------------------------------------------------------------
# normalize_to_surface
# ---------------------------------------------------------------------------

def test_normalize_to_first_slices():
    prof = intensity_depth_profile(_ramp_volume([100, 100, 50, 0]), 1.0)
    out = normalize_to_surface(prof, n_surface=2)
    assert out.surface_reference == pytest.approx(100.0)
    np.testing.assert_allclose(out.normalized, [1.0, 1.0, 0.5, 0.0])


def test_normalize_rejects_nonpositive_reference():
    prof = intensity_depth_profile(_ramp_volume([0, 0, 5]), 1.0)
    with pytest.raises(ValueError):
        normalize_to_surface(prof, n_surface=2)


# ---------------------------------------------------------------------------
# auc_depth
# ---------------------------------------------------------------------------

def test_auc_flat_curve_equals_window():
    # normalized flat curve == 1 everywhere -> AUC over 0..100 == 100
    depth = np.arange(0, 201, 4.8)
    vals = np.ones_like(depth)
    assert auc_depth(depth, vals, z_min=0.0, z_max=100.0) == pytest.approx(100.0)


def test_auc_interpolates_endpoint_between_slices():
    # slices at 0,4.8,9.6...; window ends at 100 (between slices)
    depth = np.arange(0, 201, 4.8)
    vals = np.ones_like(depth)
    # linear decay to zero at 200 um
    vals = 1 - depth / 200.0
    # analytic integral of (1 - z/200) from 0..100 = 100 - (100^2)/(2*200) = 75
    assert auc_depth(depth, vals, z_max=100.0) == pytest.approx(75.0, rel=1e-6)


def test_auc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        auc_depth(np.arange(3), np.arange(4))
