"""Smoke tests for the new plot helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fluorostats.plots import (
    effect_size_heatmap,
    forest_plot,
    condition_strip,
    modality_panel,
)


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
