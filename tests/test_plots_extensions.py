"""Smoke tests for the new plot helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fluorostats.plots import (
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
