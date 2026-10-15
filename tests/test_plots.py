"""Tests for fluorostats.plots."""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unittest.mock import patch

from fluorostats import plots
from fluorostats.plots import (
    _add_pvalue_annotations,
    effect_size_heatmap,
    forest_plot,
    condition_strip,
    modality_panel,
    bar_mean_sem,
    boxplot_by_condition,
    summary_panel,
)


def _replicate_df():
    """Two conditions with a clear separation -> significant p-values."""
    rng = np.random.default_rng(0)
    lo = rng.normal(1.0, 0.05, 6)
    hi = rng.normal(5.0, 0.05, 6)
    return pd.DataFrame(
        {
            "condition": ["ctrl"] * 6 + ["treat"] * 6,
            "volume_fraction": np.concatenate([lo, hi]),
            "n_components": np.concatenate([lo * 10, hi * 10]),
        }
    )


def test_bar_mean_sem_with_pvalue_brackets(tmp_path):
    """show_pvalues=True on separated data draws significance brackets."""
    out = tmp_path / "bar.png"
    bar_mean_sem(_replicate_df(), "volume_fraction", out, show_pvalues=True)
    assert out.exists() and out.stat().st_size > 1000


def test_boxplot_by_condition_writes_file(tmp_path):
    out = tmp_path / "box.png"
    boxplot_by_condition(_replicate_df(), "volume_fraction", out, ylabel="Volume Fraction")
    assert out.exists() and out.stat().st_size > 1000


def test_summary_panel_with_pvalues(tmp_path):
    out = tmp_path / "summary.png"
    summary_panel(_replicate_df(), ["volume_fraction", "n_components"], out, show_pvalues=True)
    assert out.exists() and out.stat().st_size > 1000


def test_summary_panel_no_available_metrics_is_noop(tmp_path):
    out = tmp_path / "empty.png"
    df = pd.DataFrame({"condition": ["a", "b"]})
    summary_panel(df, ["does_not_exist"], out)
    assert not out.exists()


def test_condition_strip_default_axes_and_palette():
    """ax=None + no hue/marker exercises the auto-figure and default palette."""
    df = pd.DataFrame(
        {
            "cond": ["a", "a", "b", "b", "b"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ax = condition_strip(df, "cond", "val")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    assert len(ax.collections) == 2 and len(ax.lines) == 2  # points + median per level
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def _stats_df():
    return pd.DataFrame(
        [
            {"metric": "vf", "stratum": "top", "cliffs_delta": 0.8, "sig_q05": True},
            {"metric": "vf", "stratum": "middle", "cliffs_delta": 0.9, "sig_q05": True},
            {"metric": "vf", "stratum": "bottom", "cliffs_delta": 0.2, "sig_q05": False},
            {"metric": "len", "stratum": "top", "cliffs_delta": 0.5, "sig_q05": True},
            {"metric": "len", "stratum": "middle", "cliffs_delta": 0.7, "sig_q05": True},
            {"metric": "len", "stratum": "bottom", "cliffs_delta": -0.1, "sig_q05": False},
        ]
    )


def test_effect_size_heatmap_writes_file(tmp_path):
    out = tmp_path / "heatmap.png"
    effect_size_heatmap(_stats_df(), out_path=out)
    assert out.exists() and out.stat().st_size > 1000


def test_forest_plot_writes_file(tmp_path):
    fc = pd.DataFrame(
        [
            {"label": "VF", "fold_change_median": 9.5, "ci_low": 2.5, "ci_high": 38},
            {"label": "Length", "fold_change_median": 16.0, "ci_low": 2.5, "ci_high": 235},
            {"label": "Junction", "fold_change_median": 23.0, "ci_low": 2.6, "ci_high": 4097},
        ]
    )
    out = tmp_path / "forest.png"
    forest_plot(fc, out_path=out)
    assert out.exists() and out.stat().st_size > 1000


def test_condition_strip_runs(tmp_path):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "day": np.repeat([1, 7, 14], 16),
            "material": np.tile(np.repeat(["A", "B"], 8), 3),
            "batch": np.tile(["old", "new"], 24),
            "value": rng.normal(size=48),
        }
    )
    fig, ax = plt.subplots()
    out_ax = condition_strip(df, "day", "value", hue_col="material", marker_col="batch", ax=ax)
    assert out_ax is ax
    plt.close(fig)


def test_modality_panel_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "modality": np.repeat(["Live/Dead", "Immuno-488"], 12),
            "condition": np.tile(np.repeat(["A", "B"], 6), 2),
            "vf": rng.normal(size=24),
            "len": rng.normal(size=24),
        }
    )
    out = tmp_path / "modality.png"
    modality_panel(df, metrics=["vf", "len"], out_path=out)
    assert out.exists() and out.stat().st_size > 1000


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
    # only the d-e neighbour pair differs; all pairs would also bracket a-e etc.
    assert [t.get_text() for t in ax.texts] == ["***"] and len(ax.lines) == 1
    plt.close(fig)


def test_add_pvalue_annotations_skips_small_groups():
    """Groups with < 2 samples are skipped (line 219)."""
    # Group a has one sample: its pairs are not tested, so b-c is the only test and
    # its BH q equals its p (0.0079, "**"); testing a's pairs would make it "*".
    data = [np.array([1.0]), np.array([1.0, 2, 3, 4, 5]), np.array([10.0, 11, 12, 13, 14])]
    x = np.arange(3)
    conditions = ["a", "b", "c"]
    fig, ax = _make_ax_with_ylim()
    _add_pvalue_annotations(ax, data, x, conditions)
    assert [t.get_text() for t in ax.texts] == ["**"]
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
    assert len(ax.texts) == len(ax.lines) == 0 and ax.get_ylim() == (0, 10)
    plt.close(fig)


def test_add_pvalue_annotations_skips_a_pair_whose_p_is_nan():
    # separated n=10 groups: "***" unless the uncomputed test is skipped
    data = [np.arange(10.0), np.arange(10.0) + 100]
    x = np.arange(2)
    conditions = ["a", "b"]
    fig, ax = _make_ax_with_ylim()
    # stats.mann_whitney calls scipy.stats.mannwhitneyu, so that is the patch target.
    with patch("scipy.stats.mannwhitneyu", return_value=(0.0, float("nan"))):
        _add_pvalue_annotations(ax, data, x, conditions)
    assert len(ax.texts) == 0
    plt.close(fig)


# ---------------------------------------------------------------------------
# effect_size_heatmap — lines 351, 376
# ---------------------------------------------------------------------------


def test_effect_size_heatmap_no_col_col_creates_all_stratum():
    """When col_col not in df, all rows collapse into an 'all' column (line 351)."""
    df = pd.DataFrame(
        [
            {"metric": "vf", "cliffs_delta": 0.8, "sig_q05": True},
            {"metric": "len", "cliffs_delta": 0.3, "sig_q05": False},
        ]
    )
    # 'stratum' column absent → line 351 fires
    fig = effect_size_heatmap(df, col_col="stratum")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["all"]
    assert [t.get_text() for t in ax.texts] == ["0.30", "0.80*"]  # * marks sig_q05
    plt.close(fig)


def test_effect_size_heatmap_with_title(tmp_path):
    """title kwarg triggers ax.set_title (line 376)."""
    df = pd.DataFrame(
        [
            {"metric": "vf", "stratum": "top", "cliffs_delta": 0.8, "sig_q05": True},
        ]
    )
    fig = effect_size_heatmap(df, title="My Title")
    ax = fig.axes[0]
    assert ax.get_title() == "My Title"
    assert [t.get_text() for t in ax.texts] == ["0.80*"]  # value, starred as significant
    plt.close(fig)


# ---------------------------------------------------------------------------
# forest_plot — line 416
# ---------------------------------------------------------------------------


def test_forest_plot_with_title(tmp_path):
    """title kwarg triggers ax.set_title (line 416)."""
    fc = pd.DataFrame(
        [
            {"label": "VF", "fold_change_median": 2.0, "ci_low": 1.0, "ci_high": 4.0},
        ]
    )
    fig = forest_plot(fc, title="Forest Title")
    ax = fig.axes[0]
    assert ax.get_title() == "Forest Title"
    assert ax.patches[0].get_width() == 2.0  # bar at the median
    assert [t.get_text() for t in ax.get_yticklabels()] == ["VF"]
    plt.close(fig)


def test_forest_plot_no_reference_no_log_scale(tmp_path):
    """reference=None skips axvline; log_scale=False skips log axis."""
    fc = pd.DataFrame(
        [
            {"label": "A", "fold_change_median": 2.0, "ci_low": 1.0, "ci_high": 3.0},
        ]
    )
    fig = forest_plot(fc, reference=None, log_scale=False)
    ax = fig.axes[0]
    with_ref = forest_plot(fc).axes[0]
    assert ax.get_xscale() == "linear" and with_ref.get_xscale() == "log"
    assert len(with_ref.lines) == len(ax.lines) + 1  # only the reference line
    assert list(with_ref.lines[-1].get_xdata()) == [1.0, 1.0]
    plt.close("all")


# ---------------------------------------------------------------------------
# modality_panel — lines 464, 511-512, 522
# ---------------------------------------------------------------------------


def _make_modality_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "modality": np.repeat(["Live/Dead", "Immuno"], 12),
            "condition": np.tile(np.repeat(["A", "B"], 6), 2),
            "vf": rng.normal(size=24),
            "len": rng.normal(size=24),
        }
    )


def test_modality_panel_missing_metric_sets_axis_invisible(tmp_path):
    """When metric not in sub.columns, ax.set_visible(False) fires (line 464)."""
    df = _make_modality_df()
    # 'missing_col' does not exist in df → its column of axes is hidden
    fig = modality_panel(df, metrics=["vf", "missing_col"])
    axes = np.array(fig.axes).reshape(2, 2)
    assert [ax.get_visible() for ax in axes[:, 0]] == [True, True]
    assert [ax.get_visible() for ax in axes[:, 1]] == [False, False]
    plt.close(fig)


def test_modality_panel_multiple_rows_hides_x_tick_labels(tmp_path):
    """With nrows > 1, rows 0..nrows-2 have x-tick labels hidden (lines 511-512)."""
    df = _make_modality_df()
    # Two modalities (rows sorted: Immuno, Live/Dead) → only the bottom row is labelled
    fig = modality_panel(df, metrics=["vf"])
    top, bottom = fig.axes
    assert [t.get_text() for t in top.get_xticklabels()] == ["", ""]
    assert [t.get_text() for t in bottom.get_xticklabels()] == ["A", "B"]
    assert top.get_ylabel() == "Immuno\nVf" and bottom.get_ylabel() == "Live/Dead\nVf"
    plt.close(fig)


def test_modality_panel_with_title(tmp_path):
    """title kwarg triggers fig.suptitle (line 522)."""
    df = _make_modality_df()
    fig = modality_panel(df, metrics=["vf"], title="My Modality Panel")
    assert fig.get_suptitle() == "My Modality Panel"
    plt.close(fig)


# ---------------------------------------------------------------------------
# condition_strip — line 483 (legend when hue + multiple levels, show_median=False)
# ---------------------------------------------------------------------------


def test_condition_strip_no_median():
    """show_median=False skips the median line drawing."""
    df = pd.DataFrame(
        {
            "cond": ["a", "a", "b", "b"],
            "hue": ["x", "y", "x", "y"],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fig, ax = plt.subplots()
    condition_strip(df, "cond", "val", hue_col="hue", ax=ax, show_median=False)
    assert len(ax.lines) == 0 and len(ax.collections) == 4
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["x", "y"]
    plt.close(fig)


# ---------------------------------------------------------------------------
# condition_strip — line 464: vals.size == 0 → continue (empty (x, hue, marker) cross)
# ---------------------------------------------------------------------------


def test_condition_strip_empty_cross_skips_scatter():
    """When a (x_level, hue_level) combination has no rows, vals.size==0 → continue (line 464)."""
    # Condition "a" only has hue "x"; condition "b" only has hue "y".
    # The cross (a, y) and (b, x) will be empty → vals.size == 0 → continue.
    df = pd.DataFrame(
        {
            "cond": ["a", "a", "b", "b"],
            "hue": ["x", "x", "y", "y"],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fig, ax = plt.subplots()
    condition_strip(df, "cond", "val", hue_col="hue", ax=ax)
    assert len(ax.collections) == 2  # (a, y) and (b, x) are not drawn
    plt.close(fig)


# Regression: >8 conditions reused Set2 colours (Set2 has 8), so two conditions
# shared a colour in every per-condition plot.
def test_plots_many_conditions_get_distinct_colors(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **k: saved.append(fig))
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        [
            {"condition": f"C{c:02d}", "volume_fraction": float(rng.random())}
            for c in range(10)
            for _ in range(3)
        ]
    )
    plots.bar_mean_sem(df, "volume_fraction", tmp_path / "bar.png")
    plots.boxplot_by_condition(df, "volume_fraction", tmp_path / "box.png")
    plots.summary_panel(df, ["volume_fraction"], tmp_path / "panel.png")
    for fig in saved:
        colors = {p.get_facecolor()[:3] for p in fig.axes[0].patches}
        assert len(colors) == 10
    ax = condition_strip(
        df.assign(h=df["condition"]), "condition", "volume_fraction", hue_col="h", show_median=False
    )
    assert len({tuple(c.get_facecolor()[0][:3]) for c in ax.collections}) == 10
    plt.close(ax.figure)


def test_summary_stars_follow_the_corrected_table(tmp_path):
    # A vs C: completely separated, n=4 each -> raw MWU p = 2/70 = 0.029 ("*"
    # uncorrected), but BH across the three pairs gives q = 0.086 ("ns"). The
    # figure must agree with compute_pvalues, which stars the corrected q.
    import matplotlib.pyplot as plt

    from fluorostats import plots, stats

    df = pd.DataFrame(
        {
            "condition": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "m": [1, 2, 3, 4, 1, 6, 9, 14, 11, 12, 13, 14],
        }
    )
    table = stats.compute_pvalues(df, ["m"])
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
        sig_table = stats.compute_pvalues(sig, ["m"])
        plots.summary_panel(sig, ["m"], tmp_path / "s3.png", show_pvalues=True, pvalues=sig_table)
        assert drawn["stars"] == [s for s in sig_table.significance if s != "ns"] == ["*", "*"]
    finally:
        plt.Figure.savefig = real_savefig


def test_group_stats_leave_undefined_mean_and_sem_nan():
    """An empty condition has no mean and a single replicate no SEM: NaN, not 0."""
    df = pd.DataFrame({"condition": ["A", "A", "B"], "m": [1.0, 3.0, 5.0]})
    data, means, sems = plots._group_stats(df, "m", ["A", "B", "C"])
    assert means[:2] == [2.0, 5.0] and np.isnan(means[2])
    assert sems[0] == pytest.approx(1.0) and np.isnan(sems[1]) and np.isnan(sems[2])


@pytest.fixture
def kept_figures(monkeypatch):
    """Keep every figure a plot function saves open; yield a getter for the new ones."""
    before = set(plt.get_fignums())
    monkeypatch.setattr(plt, "close", lambda fig=None: None)
    yield lambda: [plt.figure(n) for n in plt.get_fignums() if n not in before]
    monkeypatch.undo()
    plt.close("all")


def test_bar_figures_share_one_bar_and_jitter_style(tmp_path, kept_figures):
    # The bar + jittered-points block was copied into bar_mean_sem and summary_panel
    # and drifted (jitter 0.12 vs 0.1, capsize 5 vs 4); one helper now draws both.
    df = _replicate_df()
    bar_mean_sem(df, "volume_fraction", tmp_path / "bar.png")
    summary_panel(df, ["volume_fraction"], tmp_path / "panel.png")
    boxplot_by_condition(df, "volume_fraction", tmp_path / "box.png")
    bar_ax, panel_ax, box_ax = (f.axes[0] for f in kept_figures())

    def points(ax):
        from matplotlib.collections import PathCollection

        return np.vstack([c.get_offsets() for c in ax.collections if isinstance(c, PathCollection)])

    def caps(ax):
        err = next(c for c in ax.containers if hasattr(c, "has_yerr"))
        return [line.get_markersize() for line in err.lines[1]]

    np.testing.assert_array_equal(points(bar_ax), points(panel_ax))
    np.testing.assert_allclose(points(box_ax)[:, 0] - 1, points(bar_ax)[:, 0], atol=1e-12)
    assert np.abs(points(bar_ax)[:, 0] - np.repeat([0, 1], 6)).max() <= 0.05
    assert caps(bar_ax) == caps(panel_ax) == [8.0, 8.0]  # capsize 4 -> 2 * 4 pt


def test_plots_compute_pvalues_is_the_stats_function():
    """Moved to stats; the plots name stays for existing callers, with a warning."""
    from fluorostats import plots, stats

    with pytest.warns(DeprecationWarning, match="plots.compute_pvalues is deprecated"):
        assert plots.compute_pvalues is stats.compute_pvalues
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        plots.nope


@pytest.fixture
def kept_figure(monkeypatch):
    """Keep the figure a plot function saves open; yield a getter for it."""
    monkeypatch.setattr(plt, "close", lambda fig=None: None)
    yield plt.gcf
    monkeypatch.undo()
    plt.close("all")


def test_plots_take_colours_and_type_from_the_theme(tmp_path, kept_figure):
    # M6 (SIN-1): plots hard-coded Set2/hsv colours and bold font sizes, so CLI plots
    # did not share the one figure style.
    from fluorostats import pubfig

    pubfig.apply_style("house")
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert list(plots._condition_colors(2)) == cycle[:2]
    assert len(set(map(tuple, plots._condition_colors(len(cycle) + 2)))) == len(cycle) + 2
    bar_mean_sem(_replicate_df(), "volume_fraction", tmp_path / "b.png")
    ax = kept_figure().axes[0]
    assert ax.yaxis.label.get_fontsize() == plt.rcParams["axes.labelsize"]
    assert ax.title.get_fontweight() == plt.rcParams["axes.titleweight"]
    plt.rcdefaults()
