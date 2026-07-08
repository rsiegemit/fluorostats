"""Tests for fluorostats.skeleton — pruning, node counting, backward compat."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats.skeleton import (
    skeleton_metrics, prune_skeleton, n_junction_nodes,
)
from fluorostats import metrics_3d


def _plus(n=64, a=20):
    """2D plus/cross: 4 arms from a centre -> 1 junction node, 4 branches."""
    v = np.zeros((n, n), bool)
    c = n // 2
    v[c, c - a:c + a + 1] = True
    v[c - a:c + a + 1, c] = True
    return v


def _spurred_line(n=64, L=40, spur=3):
    """A straight line with a tiny spur — the spur adds a spurious junction
    that pruning should remove."""
    v = np.zeros((n, n), bool)
    c = n // 2
    v[c, c - L // 2:c + L // 2] = True
    v[c:c + spur, c] = True   # short perpendicular spur
    return v


# ---------------------------------------------------------------------------
# Backward compatibility — prune=False must not change existing keys/values
# ---------------------------------------------------------------------------

def test_default_is_unpruned_and_keys_present():
    mask = _plus()
    m = skeleton_metrics(mask, voxel_size_um=(1, 1, 1))
    for key in ("total_length_um", "n_branches", "n_junctions",
                "mean_branch_length_um"):
        assert key in m
    # new additive key
    assert "n_junction_nodes" in m


def test_plus_has_one_junction_node_and_four_branches():
    m = skeleton_metrics(_plus(), voxel_size_um=(1, 1, 1))
    assert m["n_branches"] == 4
    assert m["n_junction_nodes"] == 1


def test_reexport_matches_module():
    """metrics_3d.skeleton_metrics is the same object as skeleton.skeleton_metrics."""
    assert metrics_3d.skeleton_metrics is skeleton_metrics
    assert metrics_3d.prune_skeleton is prune_skeleton
    assert metrics_3d.n_junction_nodes is n_junction_nodes


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


def test_prune_skeleton_returns_new_array():
    from skimage.morphology import skeletonize
    skel = skeletonize(_spurred_line())
    before = skel.copy()
    out = prune_skeleton(skel, min_branch_length_px=10)
    assert np.array_equal(skel, before)   # input not mutated
    assert out.dtype == bool


# ---------------------------------------------------------------------------
# Node counting
# ---------------------------------------------------------------------------

def test_n_junction_nodes_straight_line_is_zero():
    from skimage.morphology import skeletonize
    v = np.zeros((32, 32), bool); v[16, 4:28] = True
    assert n_junction_nodes(skeletonize(v)) == 0


def test_n_junction_nodes_plus_is_one():
    from skimage.morphology import skeletonize
    assert n_junction_nodes(skeletonize(_plus())) == 1


def test_empty_mask_returns_zeros():
    m = skeleton_metrics(np.zeros((16, 16), bool))
    assert m["total_length_um"] == 0.0
    assert m["n_junction_nodes"] == 0
