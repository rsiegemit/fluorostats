"""Tests for fluorostats.metrics_2d."""

import numpy as np
import pytest

from fluorostats import metrics_2d
from fluorostats.metrics_2d import area_fraction


class TestAreaMetrics:
    def test_area_fraction_empty(self):
        mask = np.zeros((64, 64), dtype=bool)
        assert area_fraction(mask) == 0.0

    def test_area_fraction_full(self):
        mask = np.ones((64, 64), dtype=bool)
        assert area_fraction(mask) == 1.0

    def test_area_fraction_quarter(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[:32, :32] = True
        assert abs(area_fraction(mask) - 0.25) < 1e-10


def test_coverage_metrics():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    m = metrics_2d.coverage_metrics(mask)
    assert m["area_fraction"] == pytest.approx(100 / 400)
    assert m["n_components"] == 1
    assert metrics_2d.coverage_metrics(np.zeros((5, 5), bool))["area_fraction"] == 0


def test_coverage_metrics_empty_mask_is_nan_not_zero():
    # No clusters: counts and coverage are a measured 0; cluster sizes and the
    # largest-cluster share are undefined, so NaN.
    m = metrics_2d.coverage_metrics(np.zeros((5, 5), bool), voxel_size_um=0.5)
    assert m["area_fraction"] == 0.0 and m["n_components"] == 0
    for k in ("largest_component_fraction", "mean_cluster_area_px", "median_cluster_area_px",
              "mean_cluster_area_um2", "median_cluster_area_um2"):
        assert np.isnan(m[k]), k
