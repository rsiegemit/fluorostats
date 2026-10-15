"""Tests for fluorostats.objects."""

from __future__ import annotations

import numpy as np
from fluorostats import stats
import pytest

from fluorostats import metrics_3d, objects
from fluorostats.objects import (
    label_3d,
    object_volumes_voxels,
    equivalent_diameters_um,
    object_centroids,
    object_density_per_mm3,
    centroid_homogeneity,
    count_local_maxima,
)


def test_count_local_maxima_counts_separated_dots():
    img = np.zeros((100, 100), np.float32)
    for y in (20, 50, 80):
        for x in (20, 50, 80):
            img[y, x] = 100.0  # 9 well-separated peaks
    r = count_local_maxima(img, min_distance=5, threshold_rel=0.2)
    assert r["count"] == 9
    assert r["coords"].shape == (9, 2)


def test_count_local_maxima_empty_image():
    r = count_local_maxima(np.zeros((10, 10), np.float32))
    assert r["count"] == 0


def _two_blobs() -> np.ndarray:
    mask = np.zeros((10, 20, 20), dtype=bool)
    mask[1:4, 1:4, 1:4] = True  # 3x3x3 = 27 voxels
    mask[5:9, 12:17, 12:17] = True  # 4x5x5 = 100 voxels
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
    expected = sorted(
        [2 * (3 * 27 / (4 * np.pi)) ** (1 / 3), 2 * (3 * 100 / (4 * np.pi)) ** (1 / 3)]
    )
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
    d = object_density_per_mm3(50, shape_zyx=(100, 100, 100), voxel_size_um=(1.0, 1.0, 1.0))
    assert d == pytest.approx(5e4, rel=1e-6)


def test_centroid_homogeneity_uniform_is_low_and_clustered_is_high():
    # 64 centroids — one per tile in an 8x8 grid → uniform
    centroids = []
    for yi in range(8):
        for xi in range(8):
            centroids.append([0, (yi + 0.5) * 8, (xi + 0.5) * 8])
    uniform = centroid_homogeneity(np.array(centroids), shape_zyx=(1, 64, 64), tiles=8)
    assert uniform["centroid_gini"] == pytest.approx(0.0, abs=1e-6)

    # 64 centroids — all in one tile → highly concentrated
    cluster = np.array([[0, 1, 1]] * 64)
    out = centroid_homogeneity(cluster, shape_zyx=(1, 64, 64), tiles=8)
    assert out["centroid_gini"] > 0.95


def test_label_3d_min_size_removes_everything():
    """min_size larger than any component returns an all-zero label image."""
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[0, 0, 0] = True  # single voxel
    labels, n = objects.label_3d(mask, min_size=100)
    assert n == 0
    assert labels.max() == 0


def test_clear_border_labels_drops_edge_objects():
    labels = np.zeros((12, 12), dtype=int)
    labels[0:3, 0:3] = 1  # touches border -> removed
    labels[5:8, 5:8] = 2  # interior -> kept
    cleared, n = objects.clear_border_labels(labels)
    assert n == 1
    assert cleared.max() == 1


def test_watershed_split_empty_mask_returns_zero():
    empty = np.zeros((5, 10, 10), dtype=bool)
    labels, n = objects.watershed_split(empty)
    assert n == 0 and labels.max() == 0


def test_watershed_split_min_size_filter():
    """Two convex blobs of different size; min_size drops the small one."""
    mask = np.zeros((6, 30, 30), dtype=bool)
    mask[2:4, 2:5, 2:5] = True  # tiny
    mask[1:5, 12:22, 12:22] = True  # large
    _, n = objects.watershed_split(mask, min_size=50, min_distance=1)
    assert n == 1


def _touching_discs() -> np.ndarray:
    yy, xx = np.mgrid[:40, :60]
    return ((yy - 20) ** 2 + (xx - 20) ** 2 < 100) | ((yy - 20) ** 2 + (xx - 37) ** 2 < 100)


def test_watershed_split_min_size_keeps_the_split():
    """Size filtering must not re-merge the touching objects watershed split."""
    mask = _touching_discs()
    _, n_all = objects.watershed_split(mask, min_size=0)
    labels, n = objects.watershed_split(mask, min_size=10)
    assert n_all == 2
    assert n == 2
    assert set(np.unique(labels)) == {0, 1, 2}


@pytest.mark.parametrize("shape", ["irregular", "ellipse"])
def test_watershed_split_does_not_oversplit_one_object(shape):
    """A single irregular blob or an elongated ellipse is one object: shallow
    distance-transform bumps from its outline or its ridge must not seed splits."""
    yy, xx = np.mgrid[:80, :80]
    if shape == "irregular":
        th = np.arctan2(yy - 40, xx - 40)
        mask = np.hypot(yy - 40, xx - 40) <= 12 + 2 * np.sin(3 * th) + 1.5 * np.sin(5 * th + 1)
    else:
        mask = ((yy - 40) / 8) ** 2 + ((xx - 40) / 22) ** 2 <= 1
    _, n = objects.watershed_split(mask, min_distance=4)
    assert n == 1


def test_watershed_split_keeps_objects_smaller_than_min_distance():
    """An object too small to seed is still one object, not dropped."""
    yy, xx = np.mgrid[:40, :40]
    mask = np.hypot(yy - 20, xx - 20) <= 3
    labels, n = objects.watershed_split(mask, min_distance=5)
    assert n == 1
    assert (labels > 0).sum() == mask.sum()


def test_watershed_split_neck_criterion():
    """Two r = 10 discs split when the neck is at least min_distance narrower than
    their inscribed diameter (2r - 2w >= min_distance), and stay one otherwise."""
    yy, xx = np.mgrid[:40, :80]

    def pair(d):
        return (np.hypot(yy - 20, xx - 30) <= 10) | (np.hypot(yy - 20, xx - 30 - d) <= 10)

    assert objects.watershed_split(pair(18), min_distance=8)[1] == 2  # neck w ~ 4.4 px
    assert objects.watershed_split(pair(8), min_distance=8)[1] == 1  # neck w ~ 9.2 px


def test_count_local_maxima_with_mask_and_smoothing():
    img = np.zeros((40, 40), np.float32)
    img[10, 10] = 100.0
    img[30, 30] = 100.0
    fg = np.zeros((40, 40), dtype=bool)
    fg[:20, :20] = True  # only the first peak is inside the mask
    r = objects.count_local_maxima(
        img, min_distance=3, threshold_rel=0.2, mask=fg, smooth_sigma=1.0
    )
    assert r["count"] == 1


def test_object_volumes_voxels_empty_label():
    assert objects.object_volumes_voxels(np.zeros((3, 3, 3), int)).size == 0


def test_object_centroids_empty_label():
    assert objects.object_centroids(np.zeros((3, 3, 3), int)).shape == (0, 3)


def test_object_density_per_mm3_zero_volume_is_nan():
    with pytest.warns(RuntimeWarning, match="non-positive"):
        d = objects.object_density_per_mm3(5, (0, 10, 10), (1.0, 1.0, 1.0))
    assert np.isnan(d)


def test_object_density_per_mm3_is_metrics_3d_density():
    args = (17, (4, 30, 20), (2.0, 0.5, 0.25))
    assert objects.object_density_per_mm3(*args) == metrics_3d.density_per_mm3(*args)


def test_object_centroids_empty_2d_label_has_two_columns():
    assert objects.object_centroids(np.zeros((5, 5), int)).shape == (0, 2)


def test_centroid_homogeneity_empty_is_nan():
    out = objects.centroid_homogeneity(np.zeros((0, 3)), (1, 64, 64))
    assert out["n_objects"] == 0
    assert np.isnan(out["centroid_gini"])


def test_objects_empty_and_single():
    z = np.zeros((5, 5, 5), np.int64)
    _, n = objects.watershed_split(z.astype(bool))
    assert n == 0
    assert objects.equivalent_diameters_um(z, (1.0, 1.0, 1.0)).size == 0
    assert "centroid_gini" in objects.centroid_homogeneity(np.array([[0.0, 1.0, 1.0]]), (2, 4, 4))
    assert "centroid_gini" in objects.centroid_homogeneity(np.zeros((0, 3)), (2, 4, 4))


# ---------------------------------------------------------------------------
# objects.py:90 — watershed_split size-filter drops every label
# ---------------------------------------------------------------------------
def test_watershed_split_all_labels_below_min_size():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True  # single-voxel object -> non-empty mask
    labels, n = objects.watershed_split(mask, min_size=1000)
    assert n == 0
    assert labels.max() == 0


# ---------------------------------------------------------------------------
# stats.gini / stats.coefficient_of_variation guards (empty / all-zero)
# ---------------------------------------------------------------------------
def test_objects_gini_cv_guards():
    assert np.isnan(stats.gini(np.array([])))  # 248: size == 0
    assert stats.gini(np.zeros(4)) == 0.0  # 251: sum == 0
    assert np.isnan(stats.coefficient_of_variation(np.zeros(4)))  # 261: mean == 0


# --------------------------------------------------------------------------- #
# objects.angular_homogeneity / radial_distribution
# --------------------------------------------------------------------------- #
def test_angular_homogeneity_uniform_vs_clustered():
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ring_pts = np.column_stack([100 + 50 * np.sin(ang), 100 + 50 * np.cos(ang)])
    uni = objects.angular_homogeneity(ring_pts, (100, 100))
    arc = np.linspace(0, 0.4, 60)  # tight arc
    clus = np.column_stack([100 + 50 * np.sin(arc), 100 + 50 * np.cos(arc)])
    cl = objects.angular_homogeneity(clus, (100, 100))
    assert uni["resultant_length"] < 0.1
    assert cl["resultant_length"] > 0.9
    assert cl["angular_gini"] > uni["angular_gini"]
    assert objects.angular_homogeneity(np.zeros((0, 2)), (0, 0))["n"] == 0


def test_radial_distribution_shell_and_degenerate():
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    at_r = np.column_stack([100 + 50 * np.sin(ang), 100 + 50 * np.cos(ang)])
    r = objects.radial_distribution(at_r, (100, 100), n_bins=10)
    assert r["mean_norm_radius"] == pytest.approx(1.0, abs=0.05)
    assert r["fractions"].sum() == pytest.approx(1.0)
    assert objects.radial_distribution(np.zeros((0, 2)), (0, 0))["n"] == 0
    # all points at the centre -> r_max <= 0 branch
    z = objects.radial_distribution(np.array([[5.0, 5.0], [5.0, 5.0]]), (5, 5))
    assert z["mean_norm_radius"] == 0.0
    # explicit r_max
    assert objects.radial_distribution(at_r, (100, 100), r_max=100.0)["r_max"] == 100.0


# --------------------------------------------------------------------------- #
# objects.object_shape_metrics
# --------------------------------------------------------------------------- #
def test_object_shape_metrics_elongation_and_scaling():
    lab = np.zeros((40, 40), int)
    lab[5:8, 5:35] = 1  # long thin horizontal bar -> elongated
    lab[20:30, 20:30] = 2  # square -> ~round
    s = objects.object_shape_metrics(lab)
    assert s["elongation"][0] > 3.0
    assert s["elongation"][1] < 1.5
    assert 0.0 <= s["orientation_deg"][0] < 180.0
    assert s["solidity"][1] == pytest.approx(1.0, abs=0.05)
    # µm scaling multiplies axis lengths by the mean pixel pitch
    s2 = objects.object_shape_metrics(lab, voxel_size_um=(2.0, 2.0))
    assert s2["major_axis"][0] == pytest.approx(2.0 * s["major_axis"][0], rel=1e-6)


def test_object_shape_metrics_empty_and_ndim_guard():
    e = objects.object_shape_metrics(np.zeros((10, 10), int))
    assert e["elongation"].size == 0
    with pytest.raises(ValueError):
        objects.object_shape_metrics(np.zeros((3, 3, 3), int))


# --------------------------------------------------------------------------- #
# objects.object_mask_association
# --------------------------------------------------------------------------- #
def test_object_mask_association_on_and_off_structure():
    mask = np.zeros((50, 50), bool)
    mask[25, :] = True  # a horizontal strand
    on = np.array([[25.0, 10.0], [25.0, 40.0]])  # sit on the strand
    off = np.array([[5.0, 10.0], [45.0, 40.0]])  # 20 px away
    r_on = objects.object_mask_association(on, mask)
    r_off = objects.object_mask_association(off, mask)
    assert r_on["median_distance"] == pytest.approx(0.0)
    assert r_on["frac_within"] == 1.0
    assert r_off["median_distance"] > 15 and r_off["frac_within"] == 0.0
    # µm scaling + explicit threshold
    r = objects.object_mask_association(off, mask, voxel_size_um=(2.0, 2.0), max_dist_um=50.0)
    assert r["frac_within"] == 1.0  # 20 px * 2 µm = 40 µm < 50
    # 3D centroids (z,y,x) against a 2D mask use the last two columns
    p3 = np.array([[7.0, 25.0, 10.0]])
    assert objects.object_mask_association(p3, mask)["median_distance"] == pytest.approx(0.0)


def test_object_mask_association_degenerate():
    r = objects.object_mask_association(np.zeros((0, 2)), np.ones((5, 5), bool))
    assert r["n"] == 0 and np.isnan(r["median_distance"])
    r2 = objects.object_mask_association(np.array([[1.0, 1.0]]), np.zeros((5, 5), bool))
    assert np.isnan(r2["frac_within"])  # empty mask


# --------------------------------------------------------------------------- #
# objects.nearest_neighbor_stats
# --------------------------------------------------------------------------- #
def test_nearest_neighbor_clustered_vs_regular():
    rng = np.random.default_rng(0)
    # two tight clusters -> clustered (R < 1)
    clus = np.vstack([rng.normal([10, 10], 1.0, (40, 2)), rng.normal([90, 90], 1.0, (40, 2))])
    # regular lattice -> dispersed (R > 1)
    gy, gx = np.mgrid[0:10, 0:10]
    grid = np.column_stack([gy.ravel() * 10.0, gx.ravel() * 10.0])
    rc = objects.nearest_neighbor_stats(clus)
    rg = objects.nearest_neighbor_stats(grid)
    assert rc["clark_evans_R"] < 0.9 and rc["pattern"] == "clustered"
    assert rg["clark_evans_R"] > 1.1 and rg["pattern"] == "dispersed"
    assert rc["mean_nn_dist"] > 0


def test_nearest_neighbor_3d_and_degenerate():
    rng = np.random.default_rng(1)
    p3 = rng.uniform(0, 50, (60, 3))
    r = objects.nearest_neighbor_stats(p3, voxel_size_um=(2.0, 0.5, 0.5))
    assert np.isfinite(r["clark_evans_R"]) and r["n"] == 60  # 3D branch
    assert objects.nearest_neighbor_stats(np.zeros((1, 2)))["pattern"] == "n/a"  # <2 points
    # unsupported dimensionality -> R is NaN but spacing still returned
    r4 = objects.nearest_neighbor_stats(rng.uniform(0, 1, (5, 4)))
    assert np.isnan(r4["clark_evans_R"]) and r4["mean_nn_dist"] > 0


def test_nearest_neighbor_flat_axis_uses_the_matching_dimension():
    # 3D centroids all in one z-plane are a 2D pattern: same R as the 2D points
    # (the 3D Clark-Evans constant on an area made random points look dispersed).
    rng = np.random.default_rng(2)
    p2 = rng.uniform(0, 100, (200, 2))
    p3 = np.column_stack([np.full(200, 4.0), p2])
    r2 = objects.nearest_neighbor_stats(p2)
    r3 = objects.nearest_neighbor_stats(p3)
    assert r3["clark_evans_R"] == pytest.approx(r2["clark_evans_R"])
    assert r3["pattern"] == r2["pattern"] == "random"


@pytest.mark.parametrize(
    "pts",
    [
        np.column_stack([np.zeros(5), np.arange(5.0)]),  # collinear: one axis varies
        np.ones((4, 3)),  # coincident: none varies
    ],
)
def test_nearest_neighbor_fewer_than_two_varying_axes_is_nan(pts):
    with pytest.warns(RuntimeWarning, match="fewer than 2 axes"):
        r = objects.nearest_neighbor_stats(pts)
    assert np.isnan(r["clark_evans_R"]) and r["pattern"] == "n/a"
    assert r["n"] == len(pts)


# Regression (release hardening): label gaps produced spurious zero-volume objects.
def test_object_volumes_ignore_label_gaps():
    """Non-contiguous label images must not yield spurious zero-volume objects."""
    labels = np.array([[0, 1, 1], [0, 3, 3]])  # label 2 missing
    vols = objects.object_volumes_voxels(labels)
    assert sorted(vols.tolist()) == [2, 2]
    assert (vols > 0).all()


# Regression (release hardening): label gaps produced zero equivalent diameters.
def test_equivalent_diameters_no_zero_from_gaps():
    labels = np.array([[[0, 1, 1], [0, 3, 3]]])
    d = objects.equivalent_diameters_um(labels, (1.0, 1.0, 1.0))
    assert len(d) == 2 and (d > 0).all()


def test_centroid_homogeneity_accepts_2d_centroids():
    # Regression: (N, 2) centroids raised IndexError (columns 1/2 hard-coded); a 2D
    # call must equal the 3D call on the same y/x positions.
    from fluorostats.objects import centroid_homogeneity

    yx = np.array([[3.0, 4.0], [20.0, 30.0], [40.0, 10.0], [60.0, 60.0]])
    zyx = np.column_stack([np.zeros(4), yx])
    assert centroid_homogeneity(yx, (64, 64)) == centroid_homogeneity(zyx, (5, 64, 64))


def test_centroid_homogeneity_is_gini_of_tile_point_density():
    """One point-binning rule: centroid_homogeneity summarises spatial.tile_point_density."""
    from fluorostats import spatial, stats

    pts = np.random.default_rng(5).uniform(0, 500, (60, 2))
    counts = spatial.tile_point_density(pts, (500, 500), grid=(8, 8)).ravel()
    res = centroid_homogeneity(pts, (500, 500), tiles=8)
    assert res["centroid_gini"] == pytest.approx(stats.gini(counts))
    assert res["centroid_cv"] == pytest.approx(stats.coefficient_of_variation(counts))


def test_watershed_splits_touching_blobs():
    # two overlapping disks forming a dumbbell -> CC labeling sees 1, watershed 2
    mask = np.zeros((40, 70), bool)
    yy, xx = np.ogrid[:40, :70]
    mask |= (yy - 20) ** 2 + (xx - 25) ** 2 <= 13**2
    mask |= (yy - 20) ** 2 + (xx - 45) ** 2 <= 13**2
    _, n_cc = label_3d(mask)
    _, n_ws = objects.watershed_split(mask, min_distance=5)
    assert n_cc == 1
    assert n_ws == 2


def test_clear_border_removes_edge_objects():
    lab = np.zeros((30, 30), int)
    lab[0:5, 10:15] = 1  # touches top border
    lab[12:18, 12:18] = 2  # interior
    _, n = objects.clear_border_labels(lab)
    assert n == 1  # only the interior object survives


def test_clear_border_labels_keeps_touching_objects_apart():
    """Clearing the border must not re-merge touching labels (e.g. a watershed split)."""
    lab = np.zeros((20, 20), int)
    lab[0:3, 0:3] = 7  # touches the border -> removed
    lab[5:10, 5:10] = 3
    lab[10:15, 5:10] = 9  # touches label 3
    cleared, n = objects.clear_border_labels(lab)
    assert n == 2
    assert set(np.unique(cleared)) == {0, 1, 2}
    assert (cleared[5:10, 5:10] == 1).all() and (cleared[10:15, 5:10] == 2).all()


def test_clear_border_labels_after_watershed_keeps_the_split():
    labels, n = objects.watershed_split(_touching_discs())
    assert n == 2
    assert objects.clear_border_labels(labels)[1] == 2
