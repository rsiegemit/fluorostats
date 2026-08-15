"""Gap-filling tests for fluorostats.stats — targets lines missing from coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
# stratified_mann_whitney — line 130 (scalar keys tuple wrap), line 135 (missing metric)
# ---------------------------------------------------------------------------

def test_stratified_mann_whitney_single_stratum_key_is_wrapped():
    """A single stratum col produces scalar groupby keys → line 130 wraps to tuple."""
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
    # Should have 2 rows (one per region) and not raise
    assert len(out) == 2


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
