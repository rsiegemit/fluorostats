"""Tests for fluorostats.spatial (tiling, slabbing, spatial heterogeneity)."""
import numpy as np
import pytest

from fluorostats import spatial


def test_tile_bounds_partitions_fully():
    b = list(spatial.tile_bounds((10, 8), (2, 2)))
    assert len(b) == 4
    assert b[0] == (0, 5, 0, 4)
    assert b[-1] == (5, 10, 4, 8)          # last cell reaches the edge


def test_tile_reduce_values_and_nan():
    field = np.arange(16, dtype=float).reshape(4, 4)
    g = spatial.tile_reduce(field, lambda t: t.mean(), grid=(2, 2))
    assert g.shape == (2, 2)
    assert g[0, 0] == field[:2, :2].mean()
    # func returning None -> NaN
    g2 = spatial.tile_reduce(field, lambda t: None, grid=(2, 2))
    assert np.isnan(g2).all()


def test_tile_point_density_counts_and_area():
    pts = np.array([[1.0, 1.0], [1.0, 2.0], [6.0, 6.0]])   # 2 in top-left, 1 bottom-right
    counts = spatial.tile_point_density(pts, (8, 8), (2, 2))
    assert counts[0, 0] == 2 and counts[1, 1] == 1
    dens = spatial.tile_point_density(pts, (8, 8), (2, 2), per_area=True, pixel_area=1.0)
    assert dens[0, 0] == pytest.approx(2 / 16)
    empty = spatial.tile_point_density(np.zeros((0, 2)), (8, 8), (2, 2))
    assert empty.sum() == 0
    # 3D (z,y,x) centroids use the last two columns
    p3 = np.array([[5.0, 1.0, 1.0]])
    assert spatial.tile_point_density(p3, (8, 8), (2, 2))[0, 0] == 1


def test_slab_reduce_profile_and_empty_slabs():
    vol = np.ones((10, 4, 4), dtype=float)
    vol[5:] = 3.0
    centers, vals = spatial.slab_reduce(vol, lambda s: s.mean(), n_slabs=2)
    assert centers.tolist() == [0.25, 0.75]
    assert vals[0] == 1.0 and vals[1] == 3.0
    # more slabs than slices -> some empty (hi<=lo) -> NaN, centers still set
    c2, v2 = spatial.slab_reduce(np.ones((3, 2, 2)), lambda s: s.mean(), n_slabs=5)
    assert np.isnan(v2).any() and np.all(c2 >= 0)
    # func -> None becomes NaN
    _, v3 = spatial.slab_reduce(vol, lambda s: None, n_slabs=2)
    assert np.isnan(v3).all()


def test_morans_i_clustered_dispersed_uniform():
    clustered = np.array([[0.0, 0, 0], [0, 0, 0], [1, 1, 1]])  # smooth block
    dispersed = np.indices((4, 4)).sum(0) % 2 * 1.0            # checkerboard
    assert spatial.morans_i(clustered) > 0
    assert spatial.morans_i(dispersed) < 0
    assert np.isfinite(spatial.morans_i(dispersed, connectivity="queen"))
    assert np.isnan(spatial.morans_i(np.ones((3, 3))))         # zero variance
    two = np.full((3, 3), np.nan); two[0, 0] = 1; two[0, 1] = 2
    assert np.isnan(spatial.morans_i(two))                     # <3 valid cells


def test_morans_i_no_adjacent_valid_pairs():
    g = np.full((3, 3), np.nan)
    for (i, j), v in zip([(0, 0), (0, 2), (2, 0), (2, 2)], [1.0, 2.0, 3.0, 4.0]):
        g[i, j] = v                                            # corners only -> no rook neighbours
    assert np.isnan(spatial.morans_i(g))                       # s0 == 0 branch


def test_spatial_heterogeneity_summary():
    g = np.array([[1.0, 2.0], [3.0, 4.0]])
    h = spatial.spatial_heterogeneity(g)
    assert h["n"] == 4 and h["cv"] > 0 and np.isfinite(h["morans_i"])
    assert spatial.spatial_heterogeneity(np.ones((2, 2)))["cv"] == 0.0       # uniform -> CV 0
