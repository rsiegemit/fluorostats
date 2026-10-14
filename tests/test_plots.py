"""Tests for fluorostats.plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unittest.mock import patch

from fluorostats import plots
from fluorostats.plots import (
    _add_pvalue_annotations,
    _pvalue_stars,
    effect_size_heatmap,
    forest_plot,
    condition_strip,
    modality_panel,
    bar_mean_sem,
    boxplot_by_condition,
    summary_panel,
    compute_pvalues,
)


def _replicate_df():
    """Two conditions with a clear separation -> significant p-values."""
    rng = np.random.default_rng(0)
    lo = rng.normal(1.0, 0.05, 6)
    hi = rng.normal(5.0, 0.05, 6)
    return pd.DataFrame({
        "condition": ["ctrl"] * 6 + ["treat"] * 6,
        "volume_fraction": np.concatenate([lo, hi]),
        "n_components": np.concatenate([lo * 10, hi * 10]),
    })


def test_bar_mean_sem_with_pvalue_brackets(tmp_path):
    """show_pvalues=True on separated data draws significance brackets."""
    out = tmp_path / "bar.png"
    bar_mean_sem(_replicate_df(), "volume_fraction", out, show_pvalues=True)
    assert out.exists() and out.stat().st_size > 1000


def test_boxplot_by_condition_writes_file(tmp_path):
    out = tmp_path / "box.png"
    boxplot_by_condition(_replicate_df(), "volume_fraction", out,
                         ylabel="Volume Fraction")
    assert out.exists() and out.stat().st_size > 1000


def test_summary_panel_with_pvalues(tmp_path):
    out = tmp_path / "summary.png"
    summary_panel(_replicate_df(), ["volume_fraction", "n_components"], out,
                  show_pvalues=True)
    assert out.exists() and out.stat().st_size > 1000


def test_summary_panel_no_available_metrics_is_noop(tmp_path):
    out = tmp_path / "empty.png"
    df = pd.DataFrame({"condition": ["a", "b"]})
    summary_panel(df, ["does_not_exist"], out)
    assert not out.exists()


def test_compute_pvalues_flags_significance():
    pvals = compute_pvalues(_replicate_df(), ["volume_fraction", "missing_metric"])
    assert set(pvals["metric"]) == {"volume_fraction"}
    assert (pvals["significance"] != "ns").all()


def test_compute_pvalues_names_test_correction_and_effect_size():
    # pvalues.csv named neither the test nor the correction and had no effect size.
    from fluorostats import stats

    df = _replicate_df()
    df.loc[df.index[:3], "n_components"] = np.nan
    pvals = compute_pvalues(df, ["volume_fraction", "n_components"])
    assert list(pvals.columns) == ["metric", "condition_a", "condition_b", "n_a", "n_b",
                                   "p_value", "q_value", "significance", "cliffs_delta",
                                   "test", "correction"]
    assert (pvals["test"] == "Mann-Whitney U, two-sided").all()
    assert (pvals["correction"] == "Benjamini-Hochberg FDR across all rows").all()
    for row in pvals.itertuples():
        a = df.loc[df["condition"] == row.condition_a, row.metric].dropna()
        b = df.loc[df["condition"] == row.condition_b, row.metric].dropna()
        ref = stats.mann_whitney(a, b)
        assert (row.n_a, row.n_b) == (len(a), len(b))
        assert row.p_value == ref["p"] and row.cliffs_delta == ref["cliffs_delta"] == -1.0
    assert list(pvals["q_value"]) == list(stats.bh_fdr(pvals["p_value"].to_numpy()))


def test_condition_strip_default_axes_and_palette():
    """ax=None + no hue/marker exercises the auto-figure and default palette."""
    df = pd.DataFrame({
        "cond": ["a", "a", "b", "b", "b"],
        "val": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    ax = condition_strip(df, "cond", "val")
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def _stats_df():
    return pd.DataFrame([
        {"metric": "vf", "stratum": "top",    "cliffs_delta":  0.8, "sig_q05": True},
        {"metric": "vf", "stratum": "middle", "cliffs_delta":  0.9, "sig_q05": True},
        {"metric": "vf", "stratum": "bottom", "cliffs_delta":  0.2, "sig_q05": False},
        {"metric": "len","stratum": "top",    "cliffs_delta":  0.5, "sig_q05": True},
        {"metric": "len","stratum": "middle", "cliffs_delta":  0.7, "sig_q05": True},
        {"metric": "len","stratum": "bottom", "cliffs_delta": -0.1, "sig_q05": False},
    ])


def test_effect_size_heatmap_writes_file(tmp_path):
    out = tmp_path / "heatmap.png"
    effect_size_heatmap(_stats_df(), out_path=out)
    assert out.exists() and out.stat().st_size > 1000


def test_forest_plot_writes_file(tmp_path):
    fc = pd.DataFrame([
        {"label": "VF",      "fold_change_median": 9.5,  "ci_low": 2.5, "ci_high": 38},
        {"label": "Length",  "fold_change_median": 16.0, "ci_low": 2.5, "ci_high": 235},
        {"label": "Junction","fold_change_median": 23.0, "ci_low": 2.6, "ci_high": 4097},
    ])
    out = tmp_path / "forest.png"
    forest_plot(fc, out_path=out)
    assert out.exists() and out.stat().st_size > 1000


def test_condition_strip_runs(tmp_path):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "day": np.repeat([1, 7, 14], 16),
        "material": np.tile(np.repeat(["A", "B"], 8), 3),
        "batch": np.tile(["old", "new"], 24),
        "value": rng.normal(size=48),
    })
    fig, ax = plt.subplots()
    out_ax = condition_strip(df, "day", "value", hue_col="material",
                             marker_col="batch", ax=ax)
    assert out_ax is ax
    plt.close(fig)


def test_modality_panel_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "modality": np.repeat(["Live/Dead", "Immuno-488"], 12),
        "condition": np.tile(np.repeat(["A", "B"], 6), 2),
        "vf":  rng.normal(size=24),
        "len": rng.normal(size=24),
    })
    out = tmp_path / "modality.png"
    modality_panel(df, metrics=["vf", "len"], out_path=out)
    assert out.exists() and out.stat().st_size > 1000


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


# Regression (release hardening): >8 conditions reused Set2 colors.
def test_plots_many_conditions_no_color_collision(tmp_path):
    """>8 conditions previously reused Set2 colors; just assert all renderers run."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame([{"condition": f"C{c:02d}", "volume_fraction": float(rng.random())}
                       for c in range(10) for _ in range(3)])
    plots.bar_mean_sem(df, "volume_fraction", tmp_path / "bar.png")
    plots.boxplot_by_condition(df, "volume_fraction", tmp_path / "box.png")
    plots.summary_panel(df, ["volume_fraction"], tmp_path / "panel.png")
    assert (tmp_path / "bar.png").exists() and (tmp_path / "box.png").exists()


def test_summary_stars_follow_the_corrected_table(tmp_path):
    # A vs C: completely separated, n=4 each -> raw MWU p = 2/70 = 0.029 ("*"
    # uncorrected), but BH across the three pairs gives q = 0.086 ("ns"). The
    # figure must agree with compute_pvalues, which stars the corrected q.
    import matplotlib.pyplot as plt

    from fluorostats import plots

    df = pd.DataFrame({"condition": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                       "m": [1, 2, 3, 4, 1, 6, 9, 14, 11, 12, 13, 14]})
    table = plots.compute_pvalues(df, ["m"])
    ac = table[(table.condition_a == "A") & (table.condition_b == "C")].iloc[0]
    assert ac.p_value < 0.05 and ac.significance == "ns"

    drawn = {}
    real_savefig = plt.Figure.savefig

    def capture(fig, *a, **k):
        drawn["stars"] = [t.get_text() for ax in fig.axes for t in ax.texts]
        return real_savefig(fig, *a, **k)

    plt.Figure.savefig = capture
    try:
        plots.summary_panel(df, ["m"], tmp_path / "s.png", show_pvalues=True, pvalues=table)
        assert drawn["stars"] == [s for s in table.significance if s != "ns"] == []
        plots.summary_panel(df, ["m"], tmp_path / "s2.png", show_pvalues=True)  # per-panel BH
        assert drawn["stars"] == []
        # and where the corrected table does call pairs significant, they are drawn
        sig = df.assign(m=[1, 2, 3, 4, 2, 3, 4, 5, 11, 12, 13, 14])
        sig_table = plots.compute_pvalues(sig, ["m"])
        plots.summary_panel(sig, ["m"], tmp_path / "s3.png", show_pvalues=True, pvalues=sig_table)
        assert drawn["stars"] == [s for s in sig_table.significance if s != "ns"] == ["*", "*"]
    finally:
        plt.Figure.savefig = real_savefig
