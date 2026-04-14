"""Tests for segmentation: known-geometry synthetic volumes."""

import numpy as np
import pytest

from fluorostats.segment import binarize
from fluorostats.metrics_3d import volume_fraction


class TestBinarize3D:
    def test_bright_block_detected(self):
        """A bright block in a dark volume should be segmented."""
        vol = np.zeros((10, 64, 64), dtype=np.float64)
        vol[3:7, 20:44, 20:44] = 1000.0
        # Add background noise
        rng = np.random.default_rng(42)
        vol += rng.normal(10, 5, vol.shape)

        mask = binarize(vol, method="otsu", min_size=16)

        # The block should be mostly segmented
        block_mask = mask[3:7, 20:44, 20:44]
        assert block_mask.mean() > 0.9, f"Block coverage too low: {block_mask.mean()}"

        # Background should be mostly empty
        bg_mask = mask.copy()
        bg_mask[3:7, 20:44, 20:44] = False
        assert bg_mask.mean() < 0.05, f"Background noise too high: {bg_mask.mean()}"

    def test_volume_fraction_accuracy(self):
        """Volume fraction of a known-volume object should be accurate."""
        vol = np.zeros((20, 64, 64), dtype=np.float64)
        # Sphere-ish block: 4*24*24 = 2304 voxels out of 20*64*64 = 81920
        vol[8:12, 20:44, 20:44] = 2000.0
        rng = np.random.default_rng(42)
        vol += rng.normal(10, 5, vol.shape)

        mask = binarize(vol, method="otsu", min_size=16)
        vf = volume_fraction(mask)

        expected = (4 * 24 * 24) / (20 * 64 * 64)
        assert abs(vf - expected) < 0.01, f"VF {vf:.4f} too far from expected {expected:.4f}"


class TestBinarize2D:
    def test_bright_region_2d(self):
        """A bright square in a 2D image should be segmented."""
        img = np.zeros((128, 128), dtype=np.float64)
        img[30:90, 30:90] = 500.0
        rng = np.random.default_rng(42)
        img += rng.normal(10, 5, img.shape)

        mask = binarize(img, method="otsu", min_size=16)

        region_mask = mask[30:90, 30:90]
        assert region_mask.mean() > 0.95

    def test_li_threshold(self):
        """Li threshold should also work."""
        img = np.zeros((64, 64), dtype=np.float64)
        img[10:50, 10:50] = 300.0
        rng = np.random.default_rng(42)
        img += rng.normal(10, 5, img.shape)

        mask = binarize(img, method="li", min_size=16)
        assert mask.sum() > 0
