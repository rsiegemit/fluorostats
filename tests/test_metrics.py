"""Tests for 3D and 2D metrics on synthetic data."""

import numpy as np
import pytest

from fluorostats.metrics_3d import volume_fraction, connectivity_metrics, skeleton_metrics
from fluorostats.metrics_2d import area_fraction


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
