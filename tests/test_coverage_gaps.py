"""Targeted tests for remaining coverage gaps across small modules.

Each test names the specific behaviour / branch it exercises rather than
asserting something trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats import objects, segment, viability, skeleton, morphometry


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------

def test_label_3d_min_size_removes_everything():
    """min_size larger than any component returns an all-zero label image."""
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[0, 0, 0] = True  # single voxel
    labels, n = objects.label_3d(mask, min_size=100)
    assert n == 0
    assert labels.max() == 0


def test_clear_border_labels_drops_edge_objects():
    labels = np.zeros((12, 12), dtype=int)
    labels[0:3, 0:3] = 1     # touches border -> removed
    labels[5:8, 5:8] = 2     # interior -> kept
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
    mask[2:4, 2:5, 2:5] = True        # tiny
    mask[1:5, 12:22, 12:22] = True    # large
    _, n = objects.watershed_split(mask, min_size=50, min_distance=1)
    assert n == 1


def test_count_local_maxima_with_mask_and_smoothing():
    img = np.zeros((40, 40), np.float32)
    img[10, 10] = 100.0
    img[30, 30] = 100.0
    fg = np.zeros((40, 40), dtype=bool)
    fg[:20, :20] = True  # only the first peak is inside the mask
    r = objects.count_local_maxima(img, min_distance=3, threshold_rel=0.2,
                                   mask=fg, smooth_sigma=1.0)
    assert r["count"] == 1


def test_object_volumes_voxels_empty_label():
    assert objects.object_volumes_voxels(np.zeros((3, 3, 3), int)).size == 0


def test_object_centroids_empty_label():
    assert objects.object_centroids(np.zeros((3, 3, 3), int)).shape == (0, 3)


def test_object_density_per_mm3_zero_volume():
    assert objects.object_density_per_mm3(5, (0, 10, 10), (1.0, 1.0, 1.0)) == 0.0


def test_centroid_homogeneity_empty_is_nan():
    out = objects.centroid_homogeneity(np.zeros((0, 3)), (1, 64, 64))
    assert out["n_objects"] == 0
    assert np.isnan(out["centroid_gini"])


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------

def test_choose_threshold_selects_li_for_sparse_signal():
    """A few saturating pixels drive Otsu's threshold sky-high so it keeps
    <1% of the image; the dim diffuse block is only recovered by Li."""
    img = np.zeros((128, 128), dtype=np.float64)
    img[:2, :2] = 4000.0        # 4 extremely bright px -> Otsu split point
    img[50:70, 50:70] = 20.0    # dim diffuse foreground Li catches
    out = segment.choose_threshold_method(img)
    assert out["method"] == "li"
    assert "fractions" in out and "otsu" in out["fractions"]


def test_binarize_consensus_path():
    img = np.zeros((32, 32), dtype=np.float64)
    img[8:24, 8:24] = 200.0
    mask = segment.binarize(img, method="consensus", min_size=0)
    assert mask.dtype == bool
    assert mask[16, 16]


def test_binarize_auto_path_runs():
    img = np.zeros((32, 32), dtype=np.float64)
    img[8:24, 8:24] = 200.0
    mask = segment.binarize(img, method="auto", min_size=0)
    assert mask.any()


def test_binarize_flat_image_is_all_false():
    """A perfectly flat image has no signal above threshold."""
    flat = np.full((16, 16), 7.0, dtype=np.float64)
    mask = segment.binarize(flat, method="otsu", min_size=0)
    assert not mask.any()


def test_binarize_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown threshold method"):
        segment.binarize(np.zeros((8, 8)), method="nope")


# ---------------------------------------------------------------------------
# viability
# ---------------------------------------------------------------------------

def _two_channel_stack():
    live = np.zeros((4, 40, 40), np.float32)
    dead = np.zeros((4, 40, 40), np.float32)
    # a handful of bright blobs per channel
    for (y, x) in [(10, 10), (10, 30), (30, 10)]:
        live[:, y - 3:y + 3, x - 3:x + 3] = 200.0
    dead[:, 30 - 3:30 + 3, 30 - 3:30 + 3] = 200.0
    return live, dead


def test_choose_count_method_returns_regime():
    live, _ = _two_channel_stack()
    out = viability.choose_count_method(live)
    assert out["method"] in {"cc", "maxima"}
    assert {"snr", "crowding", "n_cc", "smooth_sigma", "reason"} <= set(out)


def test_choose_count_method_no_cells():
    out = viability.choose_count_method(np.zeros((3, 20, 20), np.float32))
    assert out["method"] == "cc"
    assert out["n_cc"] == 0


def test_live_dead_by_count_method_all_reports_consensus():
    live, dead = _two_channel_stack()
    out = viability.live_dead_by_count(live, dead, method="all")
    assert set(out["by_method"]) == {"cc", "watershed", "maxima"}
    assert "consensus" in out and "spread" in out
    assert out["viability"] == out["consensus"]


def test_live_dead_by_count_auto_reports_reasoning():
    live, dead = _two_channel_stack()
    out = viability.live_dead_by_count(live, dead, method="auto")
    assert "method_live" in out and "reason_live" in out


def test_attenuation_correct_first_reference():
    vol = np.ones((4, 8, 8), np.float32)
    vol[0] *= 4.0  # bright top slice
    out = viability.attenuation_correct(vol, reference="first")
    # every slice scaled so its mean matches slice 0's mean
    means = out.mean(axis=(1, 2))
    assert np.allclose(means, means[0])


def test_attenuation_correct_mean_reference():
    vol = np.ones((3, 8, 8), np.float32)
    vol[1] *= 2.0
    out = viability.attenuation_correct(vol, reference="mean")
    assert out.shape == vol.shape


# ---------------------------------------------------------------------------
# skeleton
# ---------------------------------------------------------------------------

def test_skeleton_metrics_prune_true_runs():
    """prune=True path with an explicit micrometre spur threshold."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 5:35] = True          # main line
    mask[15:21, 20] = True         # short spur
    out = skeleton.skeleton_metrics(mask, voxel_size_um=(1.0, 1.0),
                                    prune=True, min_branch_length_um=8.0)
    assert set(out) == {"total_length_um", "n_branches", "n_junctions",
                        "n_junction_nodes", "mean_branch_length_um"}
    assert out["total_length_um"] > 0


def test_skeleton_metrics_empty_mask_returns_zeros():
    out = skeleton.skeleton_metrics(np.zeros((10, 10), dtype=bool))
    assert out["total_length_um"] == 0.0
    assert out["n_branches"] == 0


# ---------------------------------------------------------------------------
# morphometry — zero-signal guard branches
# ---------------------------------------------------------------------------

def test_depth_span_zero_volume():
    out = morphometry.depth_span(np.zeros((5, 8, 8)), voxel_size_um=(2.0, 1.0, 1.0))
    assert out["span_slices"] == 0
    assert out["span_um"] == 0.0


def test_depth_centroid_zero_volume():
    out = morphometry.depth_centroid(np.zeros((5, 8, 8)))
    assert out["z_centroid"] == 0.0 and out["z_p50"] == 0.0


def test_lateral_homogeneity_too_small_for_tiles():
    """Image smaller than the tile grid returns the n_tiles==0 guard."""
    out = morphometry.lateral_homogeneity(np.ones((2, 4, 4)), tiles=8)
    assert out["n_tiles"] == 0
    assert np.isnan(out["lateral_gini"])
