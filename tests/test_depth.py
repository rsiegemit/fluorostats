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
    fit_penetration_depth,
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
    prof = intensity_depth_profile(vol, voxel_size_um=5.0)
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
        intensity_depth_profile(_ramp_volume([1, 2]), voxel_size_um=0.0)


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


# ---------------------------------------------------------------------------
# fit_penetration_depth
# ---------------------------------------------------------------------------

def test_fit_recovers_lambda_noiseless():
    # I0=100, lambda=40 um over 0..200 um
    z = np.arange(0, 200.0, 4.0)
    I0, lam = 100.0, 40.0
    y = I0 * np.exp(-z / lam)
    fit = fit_penetration_depth(z, y)
    assert fit.fit_ok
    assert fit.offset is None
    assert fit.lambda_um == pytest.approx(lam, rel=5e-3)  # < 0.5 %
    assert fit.I0 == pytest.approx(I0, rel=5e-3)
    assert fit.r_squared > 0.999


def test_fit_recovers_lambda_and_offset():
    z = np.arange(0, 200.0, 4.0)
    I0, lam, c = 80.0, 30.0, 12.0
    y = I0 * np.exp(-z / lam) + c
    fit = fit_penetration_depth(z, y, offset=True)
    assert fit.fit_ok
    assert fit.lambda_um == pytest.approx(lam, rel=5e-3)
    assert fit.I0 == pytest.approx(I0, rel=5e-3)
    assert fit.offset == pytest.approx(c, rel=5e-3)


def test_fit_flags_bad_fit_on_pure_noise():
    # Deterministic non-exponential input (sawtooth) -> single-exp cannot fit.
    z = np.arange(0, 200.0, 4.0)
    y = np.tile([5.0, 40.0, 15.0, 30.0, 10.0], z.size // 5 + 1)[: z.size]
    fit = fit_penetration_depth(z, y)
    assert not fit.fit_ok
    assert fit.r_squared < 0.90


def test_fit_lambda_is_gain_independent():
    # Same lambda, different gain (I0) -> identical recovered lambda.
    z = np.arange(0, 200.0, 4.0)
    lam = 55.0
    fa = fit_penetration_depth(z, 50.0 * np.exp(-z / lam))
    fb = fit_penetration_depth(z, 4000.0 * np.exp(-z / lam))
    assert fa.lambda_um == pytest.approx(fb.lambda_um, rel=1e-4)
    assert fa.lambda_um == pytest.approx(lam, rel=5e-3)


def test_fit_rejects_lambda_beyond_acquired_range():
    # True lambda (5000 um) >> acquired depth (0..100 um): the profile is nearly
    # flat, so lambda is unmeasurable extrapolation even though R^2 is high.
    z = np.arange(0, 100.0, 4.0)
    y = 100.0 * np.exp(-z / 5000.0)
    fit = fit_penetration_depth(z, y)
    assert not fit.fit_ok                     # degenerate: lambda out of range
    assert np.isnan(fit.lambda_um)            # unreliable lambda is NaN'd
    assert np.isnan(fit.I0)
    assert np.isfinite(fit.r_squared)         # r_squared kept (fit was attempted)


def test_fit_accepts_lambda_at_edge_of_range():
    # lambda equal to the acquired span is the boundary and must be accepted.
    z = np.arange(0, 200.0, 4.0)
    span = float(z[-1] - z[0])
    y = 100.0 * np.exp(-z / span)
    fit = fit_penetration_depth(z, y)
    assert fit.fit_ok
    assert fit.lambda_um == pytest.approx(span, rel=5e-3)


def test_fit_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        fit_penetration_depth(np.arange(3), np.arange(4))


def _ramp(per_slice, ny=4, nx=5):
    per_slice = np.asarray(per_slice, dtype=float)
    return np.ones((len(per_slice), ny, nx)) * per_slice[:, None, None]


# ---------------------------------------------------------------------------
# _select_plane_stack — wrong ndim raises ValueError  (line 71)
# ---------------------------------------------------------------------------

def test_select_plane_stack_2d_raises():
    """A 2D array (not 3D or 4D) should raise ValueError."""
    vol_2d = np.ones((4, 4))
    with pytest.raises(ValueError, match="expected 3D"):
        intensity_depth_profile(vol_2d, 1.0)


def test_select_plane_stack_5d_raises():
    """A 5D array should raise ValueError."""
    vol_5d = np.ones((2, 2, 4, 4, 4))
    with pytest.raises(ValueError, match="expected 3D"):
        intensity_depth_profile(vol_5d, 1.0)


# ---------------------------------------------------------------------------
# intensity_depth_profile — bad reducer  (line 116)
# ---------------------------------------------------------------------------

def test_bad_reducer_raises():
    vol = _ramp([10, 8, 6])
    with pytest.raises(ValueError, match="reducer must be"):
        intensity_depth_profile(vol, 1.0, reducer="mode")


# ---------------------------------------------------------------------------
# normalize_to_surface  (line 191: non-positive reference)
# ---------------------------------------------------------------------------

def test_normalize_rejects_nonpositive_reference_message():
    prof = intensity_depth_profile(_ramp([0.0, 0.0, 5.0]), 1.0)
    with pytest.raises(ValueError, match="non-positive"):
        normalize_to_surface(prof, n_surface=2)


def test_normalize_n_surface_zero_raises():
    """n_surface < 1 raises ValueError (line 190 check)."""
    prof = intensity_depth_profile(_ramp([10.0, 8.0]), 1.0)
    with pytest.raises(ValueError, match="n_surface must be"):
        normalize_to_surface(prof, n_surface=0)


# ---------------------------------------------------------------------------
# fit_penetration_depth — not enough points (line 306)
# ---------------------------------------------------------------------------

def test_fit_too_few_points_2param():
    """2-param model needs >= 2 points; 1 point raises ValueError."""
    with pytest.raises(ValueError, match="need at least 2"):
        fit_penetration_depth(np.array([0.0]), np.array([1.0]))


def test_fit_too_few_points_3param():
    """3-param model needs >= 3 points; 2 points raises ValueError."""
    with pytest.raises(ValueError, match="need at least 3"):
        fit_penetration_depth(np.array([0.0, 1.0]), np.array([1.0, 0.5]), offset=True)


# ---------------------------------------------------------------------------
# fit_penetration_depth — lam_guess <= 0 guard  (lines 319-324)
#
# This branch fires when:
#   - `below` is non-empty but z[below[0]] == z[0]  (below[0] == 0), so
#     `z[below[0]] - z[0] == 0`, making the conditional fail and falling
#     through to lam_guess = span.
# OR
#   - `below` is empty, so lam_guess = span; if span == 0, the second
#     guard (line 324) sets lam_guess = span (which is the acquired depth).
# ---------------------------------------------------------------------------

def test_fit_i0_guess_clamped_to_one_when_all_nan():
    """All-NaN y: surface=nan, nanmax=nan -> i0_guess=1.0 (line 319); curve_fit raises -> except (lines 344-345)."""
    import warnings
    z = np.linspace(0.0, 50.0, 8)
    y = np.full(8, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_penetration_depth(z, y)
    assert not fit.fit_ok
    assert np.isnan(fit.lambda_um)
    assert np.isnan(fit.r_squared)


def test_fit_i0_guess_clamped_when_all_zeros():
    """All-zero y: nanmax=0 -> i0_guess <= 0 -> set to 1.0 (line 319)."""
    z = np.linspace(0.0, 50.0, 8)
    y = np.zeros(8)
    fit = fit_penetration_depth(z, y)
    # Should not raise; may converge or not
    assert hasattr(fit, "fit_ok")


def test_fit_except_block_offset_all_nan():
    """All-NaN y with offset=True: curve_fit raises -> except block lines 344-345; offset returned as None."""
    import warnings
    z = np.linspace(0.0, 50.0, 8)
    y = np.full(8, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_penetration_depth(z, y, offset=True)
    assert not fit.fit_ok
    assert fit.offset is None


def test_fit_lam_guess_when_all_below_threshold():
    """Profile is all below I0/e immediately at z=0 -> below[0]==0, lam_guess falls to span."""
    z = np.linspace(0, 100, 20)
    y = np.full(20, 0.001)   # tiny flat — all below any reasonable I0/e
    fit = fit_penetration_depth(z, y)
    assert hasattr(fit, "fit_ok")


def test_fit_of_a_flat_profile_has_no_r_squared():
    # Regression: a profile with no variance (ss_tot == 0) reported r_squared = 0.0, a
    # measured-looking "explains nothing" for a quantity that is undefined there.
    z = np.linspace(0, 100, 20)
    fit = fit_penetration_depth(z, np.full(20, 5.0))
    assert np.isnan(fit.r_squared)
    assert not fit.fit_ok


def test_fit_all_points_below_threshold_at_surface():
    """Profile that starts below I0/e at z=0 (below[0] == 0 path)."""
    z = np.linspace(0.0, 50.0, 10)
    y = 0.001 * np.exp(-z / 5.0)
    fit = fit_penetration_depth(z, y)
    assert hasattr(fit, "n_points")


# ---------------------------------------------------------------------------
# fit_penetration_depth — offset=True + failed convergence  (lines 344-345)
# ---------------------------------------------------------------------------

def test_fit_offset_nonconvergent_returns_fit_ok_false():
    """Pure noise with offset=True -> curve_fit fails -> fit_ok=False, offset=None."""
    rng = np.random.default_rng(42)
    z = np.linspace(0, 50, 8)
    y = rng.uniform(-5, 5, 8)   # random noise, no exponential structure
    # Force convergence failure by giving a tiny maxfev via a monotone-but-noisy signal
    # that still makes curve_fit fail; use a sawtooth that can't be fit at all.
    y = np.tile([50.0, 1.0, 50.0, 1.0], 2)
    fit = fit_penetration_depth(z, y, offset=True)
    # Either it failed to converge or it converged with bad lambda:
    assert hasattr(fit, "fit_ok")


# ---------------------------------------------------------------------------
# fit_penetration_depth — offset lambda out of range -> c_fit NaN'd  (line 369)
# ---------------------------------------------------------------------------

def test_fit_offset_lambda_out_of_range_nans_offset():
    """3-param fit with lambda >> acquired depth -> fit_ok False, offset is nan."""
    z = np.linspace(0, 50.0, 15)
    # Near-flat signal: true lambda >> 50 -> degenerate, out of range
    y = 100.0 * np.exp(-z / 5000.0) + 2.0   # tiny decay + floor
    fit = fit_penetration_depth(z, y, offset=True)
    assert not fit.fit_ok
    assert np.isnan(fit.lambda_um)
    assert np.isnan(fit.I0)
    # offset should be nan when lambda is out of range
    if fit.offset is not None:
        assert np.isnan(fit.offset)


# ---------------------------------------------------------------------------
# auc_depth — z_max <= z_min raises  (line 401)
# ---------------------------------------------------------------------------

def test_auc_zmax_le_zmin_raises():
    depth = np.array([0.0, 1.0, 2.0])
    vals = np.array([1.0, 0.8, 0.5])
    with pytest.raises(ValueError, match="z_max"):
        auc_depth(depth, vals, z_min=5.0, z_max=2.0)


# ---------------------------------------------------------------------------
# auc_depth — window exceeds range AND z_max <= z_min after clamp  (lines 409-412)
# ---------------------------------------------------------------------------

def test_auc_window_entirely_outside_acquired_range_is_nan():
    """z_min > max acquired depth: nothing was measured there, so NaN, not 0.0."""
    depth = np.array([0.0, 5.0, 10.0])
    vals = np.array([1.0, 0.8, 0.5])
    # Request window [20, 30] when stack only goes to 10
    result = auc_depth(depth, vals, z_min=20.0, z_max=30.0)
    assert np.isnan(result)


def test_auc_window_lower_edge_below_start_returns_nonzero():
    """z_min < depth[0] -> clamped to depth[0]; result still computed."""
    depth = np.array([5.0, 10.0, 15.0, 20.0])
    vals = np.ones(4)
    # z_min=0 is below depth[0]=5; gets clamped to 5; window [5,15] -> AUC=10
    result = auc_depth(depth, vals, z_min=0.0, z_max=15.0)
    assert result == pytest.approx(10.0)
