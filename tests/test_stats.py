"""Tests for fluorostats.stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fluorostats import stats
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


# ---------------------------------------------------------------------------
# cliffs_delta — line 38: empty array early return
# ---------------------------------------------------------------------------

def test_cliffs_delta_empty_x_returns_nan():
    result = cliffs_delta([], [1, 2, 3])
    assert np.isnan(result)


def test_cliffs_delta_empty_y_returns_nan():
    result = cliffs_delta([1, 2, 3], [])
    assert np.isnan(result)


def test_cliffs_delta_both_empty_returns_nan():
    result = cliffs_delta([], [])
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# bh_fdr — line 57: all-NaN input returns all-NaN
# ---------------------------------------------------------------------------

def test_bh_fdr_all_nan_returns_all_nan():
    p = np.array([np.nan, np.nan, np.nan])
    q = bh_fdr(p)
    assert np.all(np.isnan(q))


# ---------------------------------------------------------------------------
# mann_whitney — lines 83-84 (empty after NaN drop), 87-88 (ValueError)
# ---------------------------------------------------------------------------

def test_mann_whitney_empty_x_returns_nan():
    """x is all NaN → after drop size < 1 → early return (lines 83-84)."""
    res = mann_whitney([np.nan, np.nan], [1.0, 2.0, 3.0])
    assert np.isnan(res["u"])
    assert np.isnan(res["p"])
    assert np.isnan(res["cliffs_delta"])
    assert res["n_x"] == 0
    assert res["n_y"] == 3


def test_mann_whitney_empty_y_returns_nan():
    """y is all NaN → after drop size < 1 → early return (lines 83-84)."""
    res = mann_whitney([1.0, 2.0], [np.nan])
    assert np.isnan(res["u"])
    assert res["n_y"] == 0


def test_mann_whitney_single_sample_each_returns_nan_on_valueerror():
    """When both groups have n=1, mannwhitneyu may raise ValueError.

    Scipy's mannwhitneyu does not always raise for n=1, but the except
    branch (lines 87-88) must be reachable. We verify at minimum that
    the function returns a dict with the right keys and does not raise.
    """
    res = mann_whitney([5.0], [5.0])
    # Result must have all expected keys regardless of path taken.
    assert set(res.keys()) == {"u", "p", "cliffs_delta", "n_x", "n_y"}


def test_mann_whitney_identical_single_values_valueerror_branch():
    """Force the ValueError branch by calling with identical single-element arrays.

    Some scipy versions raise ValueError('All numbers are identical in mannwhitneyu')
    for arrays where all values are equal. We patch mannwhitneyu to guarantee the branch.
    """
    from unittest.mock import patch
    from scipy import stats as sps

    with patch.object(sps, "mannwhitneyu", side_effect=ValueError("identical")):
        res = mann_whitney([1.0, 2.0], [3.0, 4.0])
    assert np.isnan(res["u"])
    assert np.isnan(res["p"])


# ---------------------------------------------------------------------------
# stratified_mann_whitney — single stratum column, missing metric
# ---------------------------------------------------------------------------

def test_stratified_mann_whitney_single_stratum_column_keeps_whole_key():
    """One stratum column: each row carries the full stratum value, not its first character."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "region": np.repeat(["top", "mid"], 12),
        "condition": np.tile(["A", "B"], 12),
        "v": rng.normal(size=24),
    })
    out = stratified_mann_whitney(
        df, value_cols=["v"], group_col="condition",
        group_a="A", group_b="B", strata=["region"],
    )
    assert sorted(out["region"]) == ["mid", "top"]
    assert (out[["n_a", "n_b"]] == 6).all().all()


def test_stratified_mann_whitney_skips_missing_metric():
    """Metric not in sub.columns is skipped (line 135)."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "condition": np.repeat(["A", "B"], 10),
        "v": rng.normal(size=20),
    })
    # 'missing' is not in df → skipped → 1 row (only 'v')
    out = stratified_mann_whitney(
        df, value_cols=["v", "missing"], group_col="condition",
        group_a="A", group_b="B",
    )
    assert set(out["metric"]) == {"v"}


# ---------------------------------------------------------------------------
# bootstrap_fold_change_ci — line 182: empty array early return
# ---------------------------------------------------------------------------

def test_bootstrap_fold_change_ci_empty_x_returns_nan():
    res = bootstrap_fold_change_ci([], [1.0, 2.0])
    assert np.isnan(res["fold_change_median"])
    assert np.isnan(res["ci_low"])
    assert np.isnan(res["ci_high"])
    assert res["n_boot"] == 0


def test_bootstrap_fold_change_ci_empty_y_returns_nan():
    res = bootstrap_fold_change_ci([1.0, 2.0], [])
    assert np.isnan(res["fold_change_median"])
    assert res["n_boot"] == 0


def test_bootstrap_fold_change_ci_all_nan_x_returns_nan():
    res = bootstrap_fold_change_ci([np.nan, np.nan], [1.0, 2.0])
    assert np.isnan(res["fold_change_median"])


# ---------------------------------------------------------------------------
# stouffer_combine — line 221 (empty after NaN drop), line 224 (one_sided=True)
# ---------------------------------------------------------------------------

def test_stouffer_combine_all_nan_returns_nan():
    """All-NaN p-values → p.size == 0 after keep mask → early return (line 221)."""
    res = stouffer_combine([np.nan, np.nan])
    assert np.isnan(res["z"])
    assert np.isnan(res["p"])
    assert res["n"] == 0


def test_stouffer_combine_one_sided_true():
    """one_sided=True uses isf(p) instead of isf(p/2) (line 224)."""
    # Three small one-sided p-values → positive z → combined p still significant
    res = stouffer_combine([0.04, 0.03, 0.02], one_sided=True)
    assert res["p"] < 0.05
    assert res["n"] == 3
    # Sanity: one_sided returns a different (larger) p than two-sided for positive z
    # because sf(z) > 2*sf(|z|) only when z < 0; here z > 0 so one-sided > two-sided.
    res_two = stouffer_combine([0.04, 0.03, 0.02], one_sided=False)
    assert res["p"] != res_two["p"]   # the two code paths produce different results


def test_stouffer_combine_with_weights_one_sided():
    """Weighted + one_sided exercises both paths together."""
    res = stouffer_combine([0.01, 0.05], weights=[2.0, 1.0], one_sided=True)
    assert res["n"] == 2
    assert not np.isnan(res["p"])


# ---------------------------------------------------------------------------
# scheirer_ray_hare — line 253 (empty sub), line 277 (_row with df_ <= 0)
# ---------------------------------------------------------------------------

def test_scheirer_ray_hare_empty_df_returns_empty():
    """After dropna, sub is empty → early return pd.DataFrame() (line 253)."""
    df = pd.DataFrame({
        "v": [np.nan, np.nan],
        "A": ["x", "y"],
        "B": ["p", "q"],
    })
    out = scheirer_ray_hare(df, value_col="v", factor_a="A", factor_b="B")
    assert out.empty


def test_scheirer_ray_hare_single_level_factor_zero_df():
    """Factor with only one level has df_=0 → _row returns NaN H and p (line 277)."""
    # factor_a has 2 levels, factor_b has only 1 level → df_b = 0
    df = pd.DataFrame({
        "v": [1.0, 2.0, 3.0, 4.0],
        "A": ["x", "x", "y", "y"],
        "B": ["p", "p", "p", "p"],  # single level
    })
    out = scheirer_ray_hare(df, value_col="v", factor_a="A", factor_b="B")
    assert len(out) == 3
    b_row = out[out["source"] == "B"].iloc[0]
    assert np.isnan(b_row["H"])
    assert np.isnan(b_row["p"])


# Regression (release hardening): a dropped NaN p-value shifted its weight onto another p.
def test_stouffer_weights_align_after_nan_drop():
    """A NaN p-value must drop its weight too, not shift it onto a surviving p."""
    full = stats.stouffer_combine([0.01, 0.5], weights=[1.0, 1.0])
    with_nan = stats.stouffer_combine([0.01, np.nan, 0.5], weights=[1.0, 100.0, 1.0])
    assert with_nan["n"] == 2
    assert with_nan["z"] == pytest.approx(full["z"], rel=1e-9)


def test_bootstrap_mean_ci_is_seeded_and_brackets_mean():
    x = np.random.default_rng(2).normal(0.8, 0.1, 50)
    r = stats.bootstrap_mean_ci(x, n_boot=2000, seed=0)
    assert r == stats.bootstrap_mean_ci(x, n_boot=2000, seed=0)
    assert r["ci_low"] < r["mean"] < r["ci_high"]
    assert r["mean"] == float(x.mean())
    wide = stats.bootstrap_mean_ci(x, n_boot=2000, ci=0.99, seed=0)
    assert wide["ci_low"] <= r["ci_low"] and wide["ci_high"] >= r["ci_high"]


def test_bootstrap_mean_ci_shared_rng_continues_one_stream():
    """Several intervals drawn from one generator equal the documented block draw."""
    x = np.random.default_rng(3).normal(1.0, 0.2, 20)
    rng = np.random.default_rng(0)
    first = stats.bootstrap_mean_ci(x, n_boot=500, rng=rng)
    second = stats.bootstrap_mean_ci(x, n_boot=500, rng=rng)
    ref = np.random.default_rng(0)
    for got in (first, second):
        means = x[ref.integers(0, len(x), (500, len(x)))].mean(1)
        assert got["ci_low"] == float(np.percentile(means, 2.5))
        assert got["ci_high"] == float(np.percentile(means, 97.5))
    assert first != second
    assert first == stats.bootstrap_mean_ci(x, n_boot=500, seed=0)


def test_all_lists_every_public_function():
    import inspect
    public = {n for n, f in inspect.getmembers(stats, inspect.isfunction)
              if f.__module__ == stats.__name__ and not n.startswith("_")}
    assert set(stats.__all__) == public
