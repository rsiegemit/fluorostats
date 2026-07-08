"""Tests for fluorostats.viability."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats.viability import (
    live_dead_fractions, viability_depth_profile,
    viability_2d_vs_3d, attenuation_correct,
)


def _stack_with_depth_gradient(nz=20, size=32):
    """Live signal present only in the top half — a depth-death gradient."""
    live = np.zeros((nz, size, size), np.float32)
    live[: nz // 2] = 100.0   # shallow slices bright, deep slices empty
    return live


def test_live_dead_fractions_basic():
    live = np.zeros((4, 10, 10), np.float32); live[:, :5, :] = 100
    dead = np.zeros((4, 10, 10), np.float32); dead[:, 5:, :] = 100
    r = live_dead_fractions(live, dead)
    assert r["live_fraction"] == pytest.approx(0.5, abs=0.01)
    assert r["viability"] == pytest.approx(0.5, abs=0.05)


def test_depth_profile_reveals_gradient():
    live = _stack_with_depth_gradient()
    prof = viability_depth_profile(live)["live_by_z"]
    assert prof[:10].mean() > 0.5      # shallow slices have signal
    assert prof[10:].mean() == pytest.approx(0.0, abs=1e-6)  # deep empty


def test_2d_overestimates_vs_3d():
    live = _stack_with_depth_gradient()
    r = viability_2d_vs_3d(live)
    # MIP collapses depth -> looks fully covered; 3D fraction is ~half
    assert r["live_fraction_mip"] > r["live_fraction_3d"]
    assert r["overestimate_mip"] > 1.5


def test_attenuation_correct_equalises_slice_means():
    vol = np.ones((5, 8, 8), np.float32)
    for z in range(5):
        vol[z] *= (1.0 - 0.15 * z)   # simulated depth decay
    corr = attenuation_correct(vol)
    means = corr.reshape(5, -1).mean(axis=1)
    assert np.allclose(means, means[0], rtol=1e-4)   # all slices equalised
    assert not np.shares_memory(corr, vol)


def test_accepts_premasked_input():
    mask = np.zeros((4, 10, 10), bool); mask[:, :5, :] = True
    r = live_dead_fractions(None, live_mask=mask)
    assert r["live_fraction"] == pytest.approx(0.5, abs=0.01)
