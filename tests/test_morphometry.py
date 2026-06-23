"""Tests for fluorostats.morphometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluorostats.morphometry import (
    lateral_homogeneity,
    depth_profile,
    depth_span,
    depth_centroid,
)


# ---------------------------------------------------------------------------
# Lateral homogeneity
# ---------------------------------------------------------------------------

def test_lateral_homogeneity_uniform_volume_is_near_zero():
    vol = np.ones((4, 64, 64), dtype=np.float32)
    out = lateral_homogeneity(vol, tiles=8)
    assert out["n_tiles"] == 64
    assert out["lateral_gini"] == pytest.approx(0.0, abs=1e-6)
    assert out["lateral_cv"] == pytest.approx(0.0, abs=1e-6)


def test_lateral_homogeneity_concentrated_volume_is_high():
    vol = np.zeros((4, 64, 64), dtype=np.float32)
    vol[:, :8, :8] = 100.0  # all signal in one 8x8 tile
    out = lateral_homogeneity(vol, tiles=8)
    assert out["lateral_gini"] > 0.95
    assert out["lateral_cv"] > 5.0


def test_lateral_homogeneity_tiny_volume_returns_nan():
    vol = np.ones((2, 4, 4), dtype=np.float32)
    out = lateral_homogeneity(vol, tiles=8)
    assert out["n_tiles"] == 0
    assert math.isnan(out["lateral_gini"])


# ---------------------------------------------------------------------------
# Depth profile / span / centroid
# ---------------------------------------------------------------------------

def test_depth_profile_shape_and_values():
    vol = np.zeros((5, 4, 4), dtype=np.float32)
    vol[2] = 10.0
    prof = depth_profile(vol)
    assert prof.shape == (5,)
    assert prof[2] == pytest.approx(10.0)
    assert prof[[0, 1, 3, 4]].sum() == 0


def test_depth_span_returns_um_when_voxel_given():
    vol = np.zeros((10, 4, 4), dtype=np.float32)
    vol[3:7] = 5.0
    res = depth_span(vol, voxel_size_um=(2.0, 1.0, 1.0), relative_threshold=0.1)
    assert res["z_lo"] == 3 and res["z_hi"] == 6
    assert res["span_slices"] == 4
    assert res["span_um"] == pytest.approx(8.0)


def test_depth_span_empty_volume():
    vol = np.zeros((5, 4, 4), dtype=np.float32)
    res = depth_span(vol)
    assert res["span_slices"] == 0


def test_depth_centroid_returns_um():
    vol = np.zeros((10, 4, 4), dtype=np.float32)
    vol[5] = 1.0
    res = depth_centroid(vol, voxel_size_um=(2.0, 1.0, 1.0))
    assert res["z_centroid"] == pytest.approx(10.0)
    assert res["z_p50"] == pytest.approx(10.0)
