"""Tests for fluorostats.skeleton — pruning, node counting, backward compat."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fluorostats.skeleton import (
    skeleton_metrics,
    prune_skeleton,
    n_junction_nodes,
)
from fluorostats import metrics_3d, skeleton


def _plus(n=64, a=20):
    """2D plus/cross: 4 arms from a centre -> 1 junction node, 4 branches."""
    v = np.zeros((n, n), bool)
    c = n // 2
    v[c, c - a : c + a + 1] = True
    v[c - a : c + a + 1, c] = True
    return v


def _spurred_line(n=64, L=40, spur=3):
    """A straight line with a tiny spur — the spur adds a spurious junction
    that pruning should remove."""
    v = np.zeros((n, n), bool)
    c = n // 2
    v[c, c - L // 2 : c + L // 2] = True
    v[c : c + spur, c] = True  # short perpendicular spur
    return v


# ---------------------------------------------------------------------------
# Backward compatibility — prune=False must not change existing keys/values
# ---------------------------------------------------------------------------


def test_default_is_unpruned_and_keys_present():
    mask = _plus()
    m = skeleton_metrics(mask, voxel_size_um=(1, 1, 1))
    for key in ("total_length_um", "n_branches", "n_junctions", "mean_branch_length_um"):
        assert key in m
    # new additive key
    assert "n_junction_nodes" in m


def test_plus_has_one_junction_node_and_four_branches():
    m = skeleton_metrics(_plus(), voxel_size_um=(1, 1, 1))
    assert m["n_branches"] == 4
    assert m["n_junction_nodes"] == 1


def test_reexport_matches_module_and_warns():
    """metrics_3d.skeleton_metrics is skeleton.skeleton_metrics, with a DeprecationWarning."""
    for name, func in (
        ("skeleton_metrics", skeleton_metrics),
        ("prune_skeleton", prune_skeleton),
        ("n_junction_nodes", n_junction_nodes),
    ):
        with pytest.warns(DeprecationWarning, match=f"metrics_3d.{name} is deprecated"):
            assert getattr(metrics_3d, name) is func
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        metrics_3d.nope


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def test_prune_removes_spur_junction():
    mask = _spurred_line(spur=3)
    raw = skeleton_metrics(mask, prune=False)
    pruned = skeleton_metrics(mask, prune=True, min_branch_length_um=10)
    # the raw skeleton sees the spur as a junction; pruning removes it
    assert raw["n_junction_nodes"] >= 1
    assert pruned["n_junction_nodes"] == 0


def test_prune_preserves_main_length_roughly():
    mask = _spurred_line(L=40, spur=3)
    pruned = skeleton_metrics(mask, prune=True, min_branch_length_um=10)
    # main line ~39 long; pruning a 3px spur keeps most length
    assert pruned["total_length_um"] > 30


def test_prune_skeleton_accepts_an_array_voxel_size():
    # Regression: `voxel_size_um or 1` raised "truth value ... is ambiguous" for an array.
    from skimage.morphology import skeletonize

    skel = skeletonize(_spurred_line())
    np.testing.assert_array_equal(
        prune_skeleton(skel, 10, voxel_size_um=np.array([1.0, 1.0])),
        prune_skeleton(skel, 10, voxel_size_um=(1.0, 1.0)),
    )


def test_prune_skeleton_returns_new_array():
    from skimage.morphology import skeletonize

    skel = skeletonize(_spurred_line())
    before = skel.copy()
    out = prune_skeleton(skel, min_branch_length=10)
    assert np.array_equal(skel, before)  # input not mutated
    assert out.dtype == bool


# ---------------------------------------------------------------------------
# Node counting
# ---------------------------------------------------------------------------


def test_n_junction_nodes_straight_line_is_zero():
    from skimage.morphology import skeletonize

    v = np.zeros((32, 32), bool)
    v[16, 4:28] = True
    assert n_junction_nodes(skeletonize(v)) == 0


def test_n_junction_nodes_plus_is_one():
    from skimage.morphology import skeletonize

    assert n_junction_nodes(skeletonize(_plus())) == 1


def test_empty_mask_returns_zeros():
    m = skeleton_metrics(np.zeros((16, 16), bool))
    assert m["total_length_um"] == 0.0
    assert m["n_junction_nodes"] == 0


def test_skeleton_metrics_prune_true_runs():
    """prune=True path with an explicit micrometre spur threshold."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 5:35] = True  # main line
    mask[15:21, 20] = True  # short spur
    out = skeleton.skeleton_metrics(
        mask, voxel_size_um=(1.0, 1.0), prune=True, min_branch_length_um=8.0
    )
    assert set(out) == {
        "total_length_um",
        "n_branches",
        "n_junctions",
        "n_junction_nodes",
        "mean_branch_length_um",
    }
    assert out["total_length_um"] > 0


def test_skeleton_metrics_empty_mask_returns_zeros():
    out = skeleton.skeleton_metrics(np.zeros((10, 10), dtype=bool))
    assert out["total_length_um"] == 0.0
    assert out["n_branches"] == 0


def test_skeleton_prune_spur_free_and_default_length():
    line = np.zeros((3, 30, 30), bool)
    line[1, 15, 5:25] = True  # a single straight branch, no spurs
    m = skeleton.skeleton_metrics(line, (1.0, 1.0, 1.0), prune=True)  # default min length
    assert "total_length_um" in m


# ---------------------------------------------------------------------------
# skeleton.py:65 — prune loop break when skeleton has < 3 voxels
# ---------------------------------------------------------------------------
def test_prune_skeleton_too_few_voxels():
    s = np.zeros((10, 10), dtype=bool)
    s[5, 5] = True
    s[5, 6] = True  # 2 voxels < 3 -> break immediately
    out = skeleton.prune_skeleton(s, min_branch_length=10.0)
    assert out.sum() == 2


# ---------------------------------------------------------------------------
# skeleton.py:78 — prune loop break when a skeleton has no short spurs
# ---------------------------------------------------------------------------
def test_prune_skeleton_no_spurs():
    line = np.zeros((3, 20), dtype=bool)
    line[1, :] = True  # a straight line: no type-1 spur branches
    out = skeleton.prune_skeleton(line, min_branch_length=100.0)
    assert out.sum() == 20


# ---------------------------------------------------------------------------
# Spur threshold is in micrometres along the branch, per axis (anisotropic)
# ---------------------------------------------------------------------------
def test_prune_threshold_is_micrometres_in_2d():
    """2D image with 0.5 µm pixels: an 8-px (4 µm) spur is below a 5 µm cutoff.
    The z size of a (z, y, x) voxel tuple must not enter the 2D threshold."""
    v = np.zeros((64, 64), bool)
    v[32, 10:54] = True
    v[33:41, 32] = True
    m = skeleton_metrics(v, (5.0, 0.5, 0.5), prune=True, min_branch_length_um=5)
    assert m["n_junction_nodes"] == 0
    assert m["n_branches"] == 1


def test_prune_keeps_long_z_spur_on_anisotropic_stack():
    """A 3-slice z spur at 5 µm steps is 15 µm long: a 5 µm cutoff must keep it."""
    w = np.zeros((8, 40, 40), bool)
    w[1, 20, 5:35] = True
    w[2:5, 20, 20] = True
    m = skeleton_metrics(w, (5.0, 0.5, 0.5), prune=True, min_branch_length_um=5)
    assert m["n_branches"] == 3
    assert m["n_junction_nodes"] == 1


# ---------------------------------------------------------------------------
# Failure is NaN (not measured), never zeros; isolated points are a real zero
# ---------------------------------------------------------------------------
def test_skeleton_failure_returns_nan_and_warns(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated skan failure")

    monkeypatch.setattr(skeleton.skan, "summarize", boom)
    with pytest.warns(RuntimeWarning, match="simulated skan failure"):
        m = skeleton_metrics(_plus())
    assert set(m) == set(skeleton.EMPTY_METRICS)
    assert all(np.isnan(v) for v in m.values())


def _isolated_points():
    v = np.zeros((20, 20), bool)
    v[1, 1] = v[5, 5] = v[9, 9] = True
    return v


def test_isolated_points_measure_zero_without_warning():
    """A skeleton of isolated points has no branch: zero length, not a failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = skeleton_metrics(_isolated_points())
        pruned = prune_skeleton(_isolated_points())
    assert m.keys() == skeleton.EMPTY_METRICS.keys()
    assert m["total_length_um"] == 0.0
    assert m["n_branches"] == m["n_junctions"] == m["n_junction_nodes"] == 0
    assert np.isnan(m["mean_branch_length_um"])  # mean of zero branches
    assert pruned.sum() == 3


def test_diagonally_touching_junction_voxels_are_one_node():
    # Two degree-3 voxels that touch only at a corner are adjacent under the full
    # connectivity the neighbour count uses, so they merge into one node.
    s = np.zeros((9, 9), bool)
    for r, c in [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (2, 4),
        (1, 5),
        (0, 6),  # junction (3, 3)
        (4, 4),
        (5, 3),
        (6, 2),
        (7, 1),
        (5, 5),
        (6, 6),
        (7, 7),
    ]:  # junction (4, 4)
        s[r, c] = True
    assert n_junction_nodes(s) == 1


def test_pruning_never_deletes_every_branch_of_a_junction():
    """A Y whose three arms are all shorter than the threshold keeps a path: only
    the shortest arm goes, so the structure is not reported as a measured zero."""
    y = np.zeros((30, 30), bool)
    y[15, 15:23] = True  # 7-px arm
    for i in range(8):
        y[15 - i, 15 - i] = y[15 + i, 15 - i] = True  # two 9.9-px diagonal arms
    pruned = skeleton_metrics(y, (1, 1), prune=True)
    assert pruned["total_length_um"] == pytest.approx(2 * 7 * np.sqrt(2), abs=0.5)
    assert pruned["n_branches"] == 1


def test_pruning_keeps_a_through_path_at_a_junction():
    """A 5 um line (x spacing 0.1 um) with an 8 um side branch, pruned at 5 um: both
    line halves are short, but only one may go, leaving the 2.5 um + 8 um path."""
    t = np.zeros((41, 61), bool)
    t[20, 5:56] = True
    t[12:20, 30] = True
    r = skeleton_metrics(t, (1.0, 0.1), prune=True, min_branch_length_um=5)
    assert r["total_length_um"] == pytest.approx(10.5, abs=0.2)
