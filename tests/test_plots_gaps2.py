"""Gap-filling tests for fluorostats.plots — targets lines missing from coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unittest.mock import patch

from fluorostats.plots import (
    _add_pvalue_annotations,
    _pvalue_stars,
    compute_pvalues,
    effect_size_heatmap,
    forest_plot,
    modality_panel,
    condition_strip,
)


# ---------------------------------------------------------------------------
# _pvalue_stars — lines 254, 258
# ---------------------------------------------------------------------------

def test_pvalue_stars_triple_and_double_star():
    # p < 0.001 → "***"; p in [0.001, 0.01) → "**"
    assert _pvalue_stars(0.0009) == "***"
    assert _pvalue_stars(0.001) == "**"   # exactly at threshold: not < 0.001
    assert _pvalue_stars(0.005) == "**"
    assert _pvalue_stars(0.009) == "**"


def test_pvalue_stars_single_star():
    # p in [0.01, 0.05) → "*"  (line 258)
    assert _pvalue_stars(0.01) == "*"
    assert _pvalue_stars(0.04) == "*"
    assert _pvalue_stars(0.049) == "*"


def test_pvalue_stars_ns():
    # p >= 0.05 → "ns"
    assert _pvalue_stars(0.05) == "ns"
    assert _pvalue_stars(0.99) == "ns"


# ---------------------------------------------------------------------------
# _add_pvalue_annotations — lines 213, 219, 222-223
# ---------------------------------------------------------------------------

def _make_ax_with_ylim():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 10)
    return fig, ax


def test_add_pvalue_annotations_more_than_4_conditions_adjacent_pairs():
    """n > 4 → adjacent pairs only (line 213)."""
    rng = np.random.default_rng(0)
    # 5 groups: first 4 similar, last one very different → some adjacent sig
    data = [rng.normal(0, 0.01, 10) for _ in range(4)]
    data.append(rng.normal(100, 0.01, 10))  # 5th group
    x = np.arange(5)
    conditions = ["a", "b", "c", "d", "e"]
    fig, ax = _make_ax_with_ylim()
    _add_pvalue_annotations(ax, data, x, conditions)
    plt.close(fig)


def test_add_pvalue_annotations_skips_small_groups():
    """Groups with < 2 samples are skipped (line 219)."""
    # Pair (0,1) has only 1 sample each → skipped
    data = [np.array([1.0]), np.array([2.0]), np.array([1.0, 2.0, 3.0, 4.0])]
    x = np.arange(3)
    conditions = ["a", "b", "c"]
    fig, ax = _make_ax_with_ylim()
    _add_pvalue_annotations(ax, data, x, conditions)
    plt.close(fig)


def test_add_pvalue_annotations_all_nonsig_returns_early():
    """When no significant pairs, nothing is drawn (early return via sig_pairs empty)."""
    rng = np.random.default_rng(42)
    # Three nearly-identical groups → no significant differences
    data = [rng.normal(5.0, 0.001, 10) for _ in range(3)]
    x = np.arange(3)
    conditions = ["a", "b", "c"]
    fig, ax = _make_ax_with_ylim()
    _add_pvalue_annotations(ax, data, x, conditions)
    plt.close(fig)


def test_add_pvalue_annotations_mannwhitney_valueerror_via_mock():
    """Mock mannwhitneyu to raise ValueError → except branch (lines 222-223)."""
    data = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    x = np.arange(2)
    conditions = ["a", "b"]
    fig, ax = _make_ax_with_ylim()
    # The function does `from scipy.stats import mannwhitneyu` locally, so
    # patching scipy.stats.mannwhitneyu is the right target.
    with patch("scipy.stats.mannwhitneyu", side_effect=ValueError("identical")):
        _add_pvalue_annotations(ax, data, x, conditions)
    plt.close(fig)


# ---------------------------------------------------------------------------
# compute_pvalues — lines 294-295 (ValueError continue)
# ---------------------------------------------------------------------------

def test_compute_pvalues_skips_too_small_groups():
    """Groups with < 2 samples are skipped in compute_pvalues."""
    df = pd.DataFrame({
        "condition": ["a", "b", "b", "b"],
        "metric": [1.0, 2.0, 3.0, 4.0],
    })
    result = compute_pvalues(df, ["metric"])
    # 'a' has only 1 sample → skipped
    assert result.empty


def test_compute_pvalues_skips_missing_metric():
    """Metrics not in df are skipped."""
    df = pd.DataFrame({"condition": ["a", "b"], "metric": [1.0, 2.0]})
    result = compute_pvalues(df, ["does_not_exist"])
    assert result.empty


# ---------------------------------------------------------------------------
# effect_size_heatmap — lines 351, 376
# ---------------------------------------------------------------------------

def test_effect_size_heatmap_no_col_col_creates_all_stratum():
    """When col_col not in df, all rows collapse into an 'all' column (line 351)."""
    df = pd.DataFrame([
        {"metric": "vf", "cliffs_delta": 0.8, "sig_q05": True},
        {"metric": "len", "cliffs_delta": 0.3, "sig_q05": False},
    ])
    # 'stratum' column absent → line 351 fires
    fig = effect_size_heatmap(df, col_col="stratum")
    assert fig is not None
    plt.close(fig)


def test_effect_size_heatmap_with_title(tmp_path):
    """title kwarg triggers ax.set_title (line 376)."""
    df = pd.DataFrame([
        {"metric": "vf", "stratum": "top", "cliffs_delta": 0.8, "sig_q05": True},
    ])
    out = tmp_path / "hm_title.png"
    effect_size_heatmap(df, out_path=out, title="My Title")
    assert out.exists()


# ---------------------------------------------------------------------------
# forest_plot — line 416
# ---------------------------------------------------------------------------

def test_forest_plot_with_title(tmp_path):
    """title kwarg triggers ax.set_title (line 416)."""
    fc = pd.DataFrame([
        {"label": "VF", "fold_change_median": 2.0, "ci_low": 1.0, "ci_high": 4.0},
    ])
    out = tmp_path / "forest_title.png"
    forest_plot(fc, out_path=out, title="Forest Title")
    assert out.exists()


def test_forest_plot_no_reference_no_log_scale(tmp_path):
    """reference=None skips axvline; log_scale=False skips log axis."""
    fc = pd.DataFrame([
        {"label": "A", "fold_change_median": 2.0, "ci_low": 1.0, "ci_high": 3.0},
    ])
    out = tmp_path / "forest_nolog.png"
    forest_plot(fc, out_path=out, reference=None, log_scale=False)
    assert out.exists()


# ---------------------------------------------------------------------------
# modality_panel — lines 464, 511-512, 522
# ---------------------------------------------------------------------------

def _make_modality_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "modality": np.repeat(["Live/Dead", "Immuno"], 12),
        "condition": np.tile(np.repeat(["A", "B"], 6), 2),
        "vf": rng.normal(size=24),
        "len": rng.normal(size=24),
    })


def test_modality_panel_missing_metric_sets_axis_invisible(tmp_path):
    """When metric not in sub.columns, ax.set_visible(False) fires (line 464)."""
    df = _make_modality_df()
    out = tmp_path / "mod_missing.png"
    # 'missing_col' does not exist in df → the axis should be set invisible
    modality_panel(df, metrics=["vf", "missing_col"], out_path=out)
    assert out.exists()


def test_modality_panel_multiple_rows_hides_x_tick_labels(tmp_path):
    """With nrows > 1, rows 0..nrows-2 have x-tick labels hidden (lines 511-512)."""
    df = _make_modality_df()
    out = tmp_path / "mod_multirow.png"
    # Two modalities → nrows=2, so row 0 (i=0) will hit i < nrows-1
    modality_panel(df, metrics=["vf"], out_path=out)
    assert out.exists()


def test_modality_panel_with_title(tmp_path):
    """title kwarg triggers fig.suptitle (line 522)."""
    df = _make_modality_df()
    out = tmp_path / "mod_title.png"
    modality_panel(df, metrics=["vf"], out_path=out, title="My Modality Panel")
    assert out.exists()


# ---------------------------------------------------------------------------
# condition_strip — line 483 (legend when hue + multiple levels, show_median=False)
# ---------------------------------------------------------------------------

def test_condition_strip_no_median():
    """show_median=False skips the median line drawing."""
    df = pd.DataFrame({
        "cond": ["a", "a", "b", "b"],
        "hue": ["x", "y", "x", "y"],
        "val": [1.0, 2.0, 3.0, 4.0],
    })
    fig, ax = plt.subplots()
    condition_strip(df, "cond", "val", hue_col="hue", ax=ax, show_median=False)
    plt.close(fig)


# ---------------------------------------------------------------------------
# compute_pvalues — lines 294-295 (ValueError continue via mock)
# ---------------------------------------------------------------------------

def test_compute_pvalues_mannwhitney_valueerror_via_mock():
    """Mock mannwhitneyu to raise ValueError → except continue (lines 294-295)."""
    df = pd.DataFrame({
        "condition": ["a", "a", "a", "b", "b", "b"],
        "metric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    with patch("scipy.stats.mannwhitneyu", side_effect=ValueError("identical")):
        result = compute_pvalues(df, ["metric"])
    # ValueError was caught → no rows appended → empty result
    assert result.empty


# ---------------------------------------------------------------------------
# condition_strip — line 464: vals.size == 0 → continue (empty (x, hue, marker) cross)
# ---------------------------------------------------------------------------

def test_condition_strip_empty_cross_skips_scatter():
    """When a (x_level, hue_level) combination has no rows, vals.size==0 → continue (line 464)."""
    # Condition "a" only has hue "x"; condition "b" only has hue "y".
    # The cross (a, y) and (b, x) will be empty → vals.size == 0 → continue.
    df = pd.DataFrame({
        "cond": ["a", "a", "b", "b"],
        "hue": ["x", "x", "y", "y"],
        "val": [1.0, 2.0, 3.0, 4.0],
    })
    fig, ax = plt.subplots()
    condition_strip(df, "cond", "val", hue_col="hue", ax=ax)
    plt.close(fig)
