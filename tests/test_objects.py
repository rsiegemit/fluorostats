"""Tests for fluorostats.objects."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats.objects import (
    label_3d,
    object_volumes_voxels,
    equivalent_diameters_um,
    object_centroids,
    object_density_per_mm3,
    centroid_homogeneity,
)


def _two_blobs() -> np.ndarray:
    mask = np.zeros((10, 20, 20), dtype=bool)
    mask[1:4, 1:4, 1:4] = True       # 3x3x3 = 27 voxels
    mask[5:9, 12:17, 12:17] = True   # 4x5x5 = 100 voxels
    return mask


def test_label_3d_finds_two_blobs():
    mask = _two_blobs()
    labels, n = label_3d(mask)
    assert n == 2
    assert labels.max() == 2


def test_label_3d_min_size_filters():
    mask = _two_blobs()
    _, n = label_3d(mask, min_size=50)
    assert n == 1  # only the 100-voxel blob survives


def test_object_volumes_voxels_matches_construction():
    mask = _two_blobs()
    labels, _ = label_3d(mask)
    sizes = sorted(object_volumes_voxels(labels).tolist())
    assert sizes == [27, 100]


def test_equivalent_diameters_um_uses_voxel_spacing():
    mask = _two_blobs()
    labels, _ = label_3d(mask)
    diams = equivalent_diameters_um(labels, voxel_size_um=(1.0, 1.0, 1.0))
    # equivalent sphere diameter for V=27 ≈ 3.722; V=100 ≈ 5.759
    expected = sorted([2 * (3 * 27 / (4 * np.pi)) ** (1 / 3),
                       2 * (3 * 100 / (4 * np.pi)) ** (1 / 3)])
    assert sorted(diams.tolist()) == pytest.approx(expected, rel=1e-6)


def test_object_centroids_have_expected_shape_and_position():
    mask = _two_blobs()
    labels, _ = label_3d(mask)
    centroids = object_centroids(labels)
    assert centroids.shape == (2, 3)
    # Each centroid should fall inside the originating bounding box
    z0, y0, x0 = centroids[0]
    assert 1 <= z0 <= 8 and 1 <= y0 <= 16 and 1 <= x0 <= 16


def test_object_density_per_mm3_handles_microns():
    # FOV = 100x100x100 µm = 1e-3 mm³; 50 objects → 5e4 / mm³
    d = object_density_per_mm3(50, shape_zyx=(100, 100, 100),
                                voxel_size_um=(1.0, 1.0, 1.0))
    assert d == pytest.approx(5e4, rel=1e-6)


def test_centroid_homogeneity_uniform_is_low_and_clustered_is_high():
    # 64 centroids — one per tile in an 8x8 grid → uniform
    centroids = []
    for yi in range(8):
        for xi in range(8):
            centroids.append([0, (yi + 0.5) * 8, (xi + 0.5) * 8])
    uniform = centroid_homogeneity(np.array(centroids),
                                    shape_zyx=(1, 64, 64), tiles=8)
    assert uniform["centroid_gini"] == pytest.approx(0.0, abs=1e-6)

    # 64 centroids — all in one tile → highly concentrated
    cluster = np.array([[0, 1, 1]] * 64)
    out = centroid_homogeneity(cluster, shape_zyx=(1, 64, 64), tiles=8)
    assert out["centroid_gini"] > 0.95
