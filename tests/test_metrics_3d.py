"""Tests for fluorostats.metrics_3d."""

import numpy as np
import pytest

from fluorostats import metrics_3d
from fluorostats.metrics_3d import volume_fraction, connectivity_metrics, skeleton_metrics


def test_metrics3d_empty_mask():
    # nothing to measure: no components, and the largest-component share is NaN
    # (not a measured 0.0)
    m = metrics_3d.connectivity_metrics(np.zeros((4, 4, 4), bool))
    assert m["n_components"] == 0
    assert np.isnan(m["largest_component_fraction"])


class TestVolumeMetrics:
    def test_volume_fraction_empty(self):
        mask = np.zeros((5, 32, 32), dtype=bool)
        assert volume_fraction(mask) == 0.0

    def test_volume_fraction_full(self):
        mask = np.ones((5, 32, 32), dtype=bool)
        assert volume_fraction(mask) == 1.0

    def test_volume_fraction_half(self):
        mask = np.zeros((10, 32, 32), dtype=bool)
        mask[:5] = True
        assert abs(volume_fraction(mask) - 0.5) < 1e-10

    def test_connectivity_single_blob(self):
        mask = np.zeros((10, 32, 32), dtype=bool)
        mask[3:7, 10:22, 10:22] = True
        metrics = connectivity_metrics(mask)
        assert metrics["n_components"] == 1

    def test_connectivity_two_blobs(self):
        mask = np.zeros((10, 64, 64), dtype=bool)
        mask[2:4, 5:10, 5:10] = True
        mask[7:9, 50:55, 50:55] = True
        metrics = connectivity_metrics(mask)
        assert metrics["n_components"] == 2

    def test_skeleton_straight_line(self):
        """A thin straight line should produce a known skeleton length."""
        mask = np.zeros((1, 1, 100), dtype=bool)
        mask[0, 0, 10:90] = True
        voxel = (1.0, 1.0, 2.0)  # 2 µm per voxel in X
        metrics = skeleton_metrics(mask, voxel_size_um=voxel)
        # 80 voxels * 2 µm = 160 µm expected length
        assert metrics["total_length_um"] > 100, f"Length too short: {metrics['total_length_um']}"

    def test_skeleton_empty_mask(self):
        mask = np.zeros((5, 32, 32), dtype=bool)
        metrics = skeleton_metrics(mask)
        assert metrics["total_length_um"] == 0.0
        assert metrics["n_branches"] == 0


# Regression (release hardening): the 6-/26-connectivity pair must not be unified.
def test_connectivity_dual_convention_pinned():
    """n_components (6-conn foreground) and euler (26-conn) are the intentional DUAL
    pair — pin it so a refactor can't silently unify the two connectivities."""
    from scipy.ndimage import label
    from skimage.measure import euler_number
    m = np.zeros((10, 10, 20), bool)
    m[2:8, 2:8, 2:8] = True          # two disjoint solid blobs (each chi = 1)
    m[2:8, 2:8, 12:18] = True
    res = metrics_3d.connectivity_metrics(m)
    assert res["n_components"] == int(label(m)[1]) == 2
    assert res["euler_number"] == int(euler_number(m, connectivity=3)) == 2


def test_density_zero_volume_is_nan_with_warning():
    from fluorostats.metrics_3d import density_per_mm3, normalise_skeleton_metrics
    with pytest.warns(RuntimeWarning, match="non-positive"):
        assert np.isnan(density_per_mm3(5, (0, 10, 10), (1.0, 1.0, 1.0)))
    with pytest.warns(RuntimeWarning, match="non-positive"):
        out = normalise_skeleton_metrics({"total_length_um": 3.0}, (0, 10, 10), (1.0, 1.0, 1.0))
    assert np.isnan(out["length_density_um_per_mm3"])
    assert np.isnan(out["branch_density_per_mm3"])
