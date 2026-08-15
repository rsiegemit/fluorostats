"""Tests covering uncovered branches in fluorostats.depth_batch.

Targets (from coverage report):
  88:   load_manifest — back-compat auc_window_um (singular) path
  91:   load_manifest — no groups raises ValueError
  186:  _common_grid — single-slice stack guard (np.diff -> [])
  272:  write_group_summary_csv — agg is None branch
  312:  plot_depth_curves — agg is None branch (group with no primary stacks)
  351:  plot_auc — group with no vals (role != primary) skips continue
  358:  plot_auc — sem > 0 -> errorbar branch (need >1 stack per group)
  458-461: run — multiple replicate blanks averaging branch
  467:  run — blank valid.all() False -> trim trailing zero slices
  531-532: _print_summary — Welch t-test branch (2 groups with fit_ok)
  552-559: _print_lambda_summary — Welch t-test on λ branch
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fluorostats import depth_batch
from fluorostats.depth_batch import (
    load_manifest,
    _common_grid,
    write_group_summary_csv,
    aggregate_group,
)

tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# Helper: write a synthetic decay tiff
# ---------------------------------------------------------------------------

def _decay_stack(path, lam=20.0, n_z=16, yx=8, bg=5.0, i0=150.0):
    """Write a synthetic z-stack whose per-slice mean is i0*exp(-z/lam)+bg."""
    z = np.arange(n_z)[:, None, None]
    vol = np.broadcast_to(
        (i0 * np.exp(-z / lam) + bg).astype(np.float32), (n_z, yx, yx)
    )
    tifffile.imwrite(path, np.ascontiguousarray(vol))


def _flat_stack(path, value=5.0, n_z=16, yx=8):
    """Write a spatially-flat stack (used as blank)."""
    vol = np.full((n_z, yx, yx), value, dtype=np.float32)
    tifffile.imwrite(path, vol)


def _zero_tail_blank(path, n_z=16, yx=8, good_slices=8):
    """Blank with last slices == 0 to exercise the trim-trailing-zeros branch."""
    vol = np.zeros((n_z, yx, yx), dtype=np.float32)
    vol[:good_slices] = 5.0   # only first half has signal
    tifffile.imwrite(path, vol)


# ---------------------------------------------------------------------------
# load_manifest — back-compat singular auc_window_um  (line 88)
# ---------------------------------------------------------------------------

def test_load_manifest_singular_auc_window(tmp_path):
    """auc_window_um (singular) should be accepted and wrapped into windows."""
    blank = tmp_path / "blank.tif"
    _flat_stack(blank)
    s0 = tmp_path / "s0.tif"
    _decay_stack(s0)
    m = {
        "auc_window_um": [0, 10],   # singular (back-compat), NOT auc_windows_um
        "output_dir": str(tmp_path / "out"),
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    loaded = load_manifest(mpath)
    assert loaded["windows"] == [(0.0, 10.0)]


# ---------------------------------------------------------------------------
# load_manifest — no groups raises ValueError  (line 91)
# ---------------------------------------------------------------------------

def test_load_manifest_no_groups_raises(tmp_path):
    m = {"auc_windows_um": [[0, 10]], "groups": []}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="at least one group"):
        load_manifest(mpath)


def test_load_manifest_missing_groups_key_raises(tmp_path):
    m = {"auc_windows_um": [[0, 10]]}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="at least one group"):
        load_manifest(mpath)


# ---------------------------------------------------------------------------
# _common_grid — single-slice stack guard  (line 186)
# ---------------------------------------------------------------------------

def test_common_grid_single_slice_stack():
    """A stack with only 1 slice gives depth_um of size 1; np.diff -> []; guard fires."""
    # Simulate a 1-slice profile (depth_um has 1 element)
    single = {"depth_um": np.array([0.0]), "bg_subtracted": np.array([5.0]),
               "normalized": np.array([1.0]), "max_depth_um": 0.0}
    multi = {"depth_um": np.array([0.0, 2.0, 4.0]), "bg_subtracted": np.array([5.0, 4.0, 3.0]),
              "normalized": np.array([1.0, 0.8, 0.6]), "max_depth_um": 4.0}
    # Should not raise; step defaults to 1.0 for the single-slice member
    grid = _common_grid([single, multi])
    assert grid.size >= 1


# ---------------------------------------------------------------------------
# write_group_summary_csv — agg is None branch  (line 272)
# ---------------------------------------------------------------------------

def test_write_group_summary_csv_skips_none_agg(tmp_path):
    """Groups with agg=None are silently skipped."""
    out = tmp_path / "summary.csv"
    aggregates = {"A": None, "B": None}
    write_group_summary_csv(out, aggregates)
    assert out.exists()
    content = out.read_text()
    # Only the header row, no data rows
    lines = [l for l in content.splitlines() if l.strip()]
    assert len(lines) == 1   # header only


# ---------------------------------------------------------------------------
# aggregate_group — no primary stacks returns None  (indirectly tests line 272)
# ---------------------------------------------------------------------------

def test_aggregate_group_empty_returns_none():
    assert aggregate_group([]) is None


# ---------------------------------------------------------------------------
# Full run with multiple replicate blanks  (lines 458-461)
# ---------------------------------------------------------------------------

def test_run_with_replicate_blanks(tmp_path):
    """background as a list of paths triggers the replicate-averaging branch."""
    blank1 = tmp_path / "blank1.tif"
    blank2 = tmp_path / "blank2.tif"
    _flat_stack(blank1, value=5.0)
    _flat_stack(blank2, value=6.0)

    s0 = tmp_path / "s0.tif"
    _decay_stack(s0)

    manifest = {
        "channel": 0, "reducer": "mean", "n_surface": 2,
        "auc_windows_um": [[0, 8]],
        "output_dir": str(tmp_path / "out"),
        "groups": [{
            "name": "A",
            "background": [str(blank1), str(blank2)],   # list of blanks
            "stacks": [{"label": "s0", "path": str(s0)}],
        }],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    out = depth_batch.run(mpath)
    assert (out / "auc_per_stack.csv").exists()


# ---------------------------------------------------------------------------
# Full run — blank with trailing zero slices (trim branch)  (line 467)
# ---------------------------------------------------------------------------

def test_run_blank_trailing_zeros_trimmed(tmp_path):
    """Blank with trailing zero slices triggers the valid.all() False branch."""
    blank = tmp_path / "blank_zeros.tif"
    _zero_tail_blank(blank, n_z=16, good_slices=8)

    s0 = tmp_path / "s0.tif"
    _decay_stack(s0)

    manifest = {
        "channel": 0, "reducer": "mean", "n_surface": 2,
        "auc_windows_um": [[0, 8]],
        "output_dir": str(tmp_path / "out"),
        "groups": [{
            "name": "A",
            "background": str(blank),
            "stacks": [{"label": "s0", "path": str(s0)}],
        }],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    out = depth_batch.run(mpath)
    assert (out / "auc_per_stack.csv").exists()


# ---------------------------------------------------------------------------
# plot_depth_curves — agg is None branch  (line 312)
# ---------------------------------------------------------------------------

def test_plot_depth_curves_with_none_agg(tmp_path):
    """plot_depth_curves handles a group whose aggregate is None (no primary stacks)."""
    from fluorostats.depth_batch import plot_depth_curves

    # Build a minimal row (role=aux so no primary stacks -> agg=None)
    depth_um = np.linspace(0, 10, 8)
    res = {
        "depth_um": depth_um,
        "bg_subtracted": np.ones(8),
        "normalized": np.ones(8),
        "raw_mean": np.ones(8),
    }
    rows = [{"group": "A", "stack": "s0", "role": "aux", "result": res}]
    aggregates = {"A": None}
    colors = {"A": "#3F6AB3"}

    out_base = tmp_path / "fig_test"
    # Should complete without error
    plot_depth_curves(rows, aggregates, colors, "bg_subtracted",
                      "intensity", "title", out_base)
    assert out_base.with_suffix(".png").exists()


# ---------------------------------------------------------------------------
# plot_auc — group with no vals (all aux) skips the continue  (line 351)
# ---------------------------------------------------------------------------

def test_plot_auc_group_no_primary_vals(tmp_path):
    """plot_auc group whose stacks are all role=aux -> vals is empty -> continue."""
    from fluorostats.depth_batch import plot_auc, parse_window

    window = parse_window([0, 10])
    lab = "0_10um"
    depth_um = np.linspace(0, 10, 8)
    res = {
        "depth_um": depth_um,
        "auc": {lab: {"normalized": 5.0, "absolute": 3.0}},
    }
    rows = [{"group": "A", "stack": "s0", "role": "aux", "result": res}]
    colors = {"A": "#3F6AB3"}
    out_base = tmp_path / "auc_test"

    plot_auc(rows, colors, window, out_base, use_normalized=True)
    assert out_base.with_suffix(".png").exists()


# ---------------------------------------------------------------------------
# plot_auc — sem > 0 branch (errorbar)  (line 358): need >1 primary stack
# ---------------------------------------------------------------------------

def test_plot_auc_sem_errorbar(tmp_path):
    """plot_auc with 3 primary stacks -> sem > 0 -> errorbar branch exercised."""
    from fluorostats.depth_batch import plot_auc, parse_window

    window = parse_window([0, 10])
    lab = "0_10um"

    def _row(i, val):
        return {
            "group": "A", "stack": f"s{i}", "role": "primary",
            "result": {
                "depth_um": np.linspace(0, 10, 8),
                "auc": {lab: {"normalized": val, "absolute": val * 0.5}},
            },
        }

    rows = [_row(0, 10.0), _row(1, 12.0), _row(2, 11.0)]
    colors = {"A": "#3F6AB3"}
    out_base = tmp_path / "auc_sem"

    plot_auc(rows, colors, window, out_base, use_normalized=True)
    assert out_base.with_suffix(".png").exists()


# ---------------------------------------------------------------------------
# _print_summary and _print_lambda_summary Welch t-test branches  (lines 531-532, 552-559)
# These are hit by a full run with 2 groups where >=2 primary stacks each have fit_ok.
# ---------------------------------------------------------------------------

def test_run_two_groups_with_fit_ok_hits_ttest_branches(tmp_path):
    """Run with 2 groups, 2 stacks each where both groups have fit_ok=True -> λ t-test branch fires.

    λ must be WITHIN the acquired depth range for fit_ok to be True.
    With n_z=40 slices and default 1µm voxel spacing, max depth = 39µm.
    Use lam=5 and lam=12, both < 39µm.
    """
    blank = tmp_path / "blank.tif"
    _flat_stack(blank, value=2.0, n_z=40, yx=8)

    stacks_a, stacks_b = [], []
    for i in range(2):
        pa = tmp_path / f"a_{i}.tif"
        pb = tmp_path / f"b_{i}.tif"
        # lam=5 and lam=12 are both well within 39µm depth range
        _decay_stack(pa, lam=5.0, n_z=40)
        _decay_stack(pb, lam=12.0, n_z=40)
        stacks_a.append({"label": f"a{i}", "path": str(pa)})
        stacks_b.append({"label": f"b{i}", "path": str(pb)})

    manifest = {
        "channel": 0, "reducer": "mean", "n_surface": 2,
        "auc_windows_um": [[0, 10], "full"],
        "output_dir": str(tmp_path / "out"),
        "groups": [
            {"name": "fast", "background": str(blank), "stacks": stacks_a},
            {"name": "slow", "background": str(blank), "stacks": stacks_b},
        ],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    out = depth_batch.run(mpath)
    assert (out / "auc_per_stack.csv").exists()
    assert (out / "group_depth_summary.csv").exists()
