"""Tests for fluorostats.stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fluorostats.stats import (
    cliffs_delta,
    bh_fdr,
    mann_whitney,
    stratified_mann_whitney,
    bootstrap_fold_change_ci,
    stouffer_combine,
    scheirer_ray_hare,
)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def test_cliffs_delta_full_separation():
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)


def test_cliffs_delta_identical_samples_zero():
    a = [1, 2, 3, 4]
    assert cliffs_delta(a, a) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------

def test_bh_fdr_matches_known_values():
    # p = [0.001, 0.008, 0.039, 0.041, 0.042], N=5
    # raw q_i = p_i * N / rank_i = [0.005, 0.020, 0.0650, 0.05125, 0.042]
    # Monotone-from-right gives [0.005, 0.020, 0.042, 0.042, 0.042].
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    q = bh_fdr(p)
    assert q[0] == pytest.approx(0.005, abs=1e-3)
    assert q[1] == pytest.approx(0.020, abs=1e-3)
    assert q[2] == pytest.approx(0.042, abs=1e-3)
    assert q[3] == pytest.approx(0.042, abs=1e-3)
    assert q[4] == pytest.approx(0.042, abs=1e-3)


def test_bh_fdr_monotone():
    rng = np.random.default_rng(0)
    p = np.sort(rng.uniform(size=20))
    q = bh_fdr(p)
    # q must be non-decreasing when p is sorted ascending
    assert np.all(np.diff(q) >= -1e-12)


def test_bh_fdr_preserves_nans():
    q = bh_fdr(np.array([0.01, np.nan, 0.05]))
    assert np.isnan(q[1])
    assert q[0] <= q[2]


# ---------------------------------------------------------------------------
# Mann-Whitney + stratified
# ---------------------------------------------------------------------------

def test_mann_whitney_returns_sample_sizes():
    res = mann_whitney([1, 2, 3], [10, 20, 30])
    assert res["n_x"] == 3 and res["n_y"] == 3
    assert res["p"] < 0.2
    assert res["cliffs_delta"] == pytest.approx(-1.0)


def test_stratified_mann_whitney_fdr_across_grid():
    rng = np.random.default_rng(42)
    rows = []
    for region in ["top", "mid"]:
        for cond in ["A", "B"]:
            shift = 10 if (region == "mid" and cond == "B") else 0
            for _ in range(12):
                rows.append({
                    "region": region,
                    "condition": cond,
                    "v": rng.normal(loc=shift),
                })
    df = pd.DataFrame(rows)
    out = stratified_mann_whitney(
        df, value_cols=["v"], group_col="condition",
        group_a="A", group_b="B", strata=["region"],
    )
    assert {"region", "metric", "p", "q", "sig_q05"}.issubset(out.columns)
    assert len(out) == 2
    mid = out[out["region"] == "mid"].iloc[0]
    top = out[out["region"] == "top"].iloc[0]
    # Real effect in mid; null effect in top → mid much more significant
    assert mid["q"] < 0.05
    assert mid["q"] < top["q"]
    assert abs(mid["cliffs_delta"]) > abs(top["cliffs_delta"])


# ---------------------------------------------------------------------------
# Bootstrap fold-change
# ---------------------------------------------------------------------------

def test_bootstrap_fold_change_ci_brackets_truth():
    rng = np.random.default_rng(0)
    a = rng.lognormal(mean=0.0, sigma=0.2, size=30)
    b = rng.lognormal(mean=1.0, sigma=0.2, size=30)
    res = bootstrap_fold_change_ci(a, b, n_boot=2000, ci=0.95, seed=0)
    # True ratio of medians ≈ e^1 ≈ 2.72; CI must include it
    assert res["ci_low"] < 2.7 < res["ci_high"]
    assert res["fold_change_median"] == pytest.approx(2.7, rel=0.4)


# ---------------------------------------------------------------------------
# Stouffer
# ---------------------------------------------------------------------------

def test_stouffer_combine_amplifies_consistent_signal():
    res = stouffer_combine([0.04, 0.05, 0.06])
    assert res["p"] < 0.04
    assert res["n"] == 3


def test_stouffer_combine_handles_single_p():
    res = stouffer_combine([0.5])
    assert res["p"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Scheirer-Ray-Hare
# ---------------------------------------------------------------------------

def test_scheirer_ray_hare_detects_interaction():
    rng = np.random.default_rng(0)
    rows = []
    for mat in ["A", "B"]:
        for day in [1, 7, 14]:
            shift = 5 if (mat == "B" and day == 14) else 0
            for _ in range(10):
                rows.append({"mat": mat, "day": day, "v": rng.normal(loc=shift)})
    df = pd.DataFrame(rows)
    out = scheirer_ray_hare(df, value_col="v", factor_a="mat", factor_b="day")
    assert len(out) == 3
    inter = out[out["source"] == "mat:day"].iloc[0]
    assert inter["p"] < 0.05
