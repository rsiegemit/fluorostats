"""Tests for fluorostats.stats.compute_pvalues and significance_stars (the per-metric
condition-pair p-value table and its star labels)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from fluorostats import stats


def _two_condition_df():
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


def test_compute_pvalues_flags_significance():
    pvals = stats.compute_pvalues(_two_condition_df(), ["volume_fraction", "missing_metric"])
    assert set(pvals["metric"]) == {"volume_fraction"}
    assert (pvals["significance"] != "ns").all()


def test_compute_pvalues_names_test_correction_and_effect_size():
    # pvalues.csv named neither the test nor the correction and had no effect size.
    df = _two_condition_df()
    df.loc[df.index[:3], "n_components"] = np.nan
    pvals = stats.compute_pvalues(df, ["volume_fraction", "n_components"])
    assert list(pvals.columns) == [
        "metric",
        "condition_a",
        "condition_b",
        "n_a",
        "n_b",
        "p_value",
        "q_value",
        "significance",
        "cliffs_delta",
        "test",
        "correction",
        "reason",
    ]
    assert (pvals["reason"] == "").all()
    assert (pvals["test"] == "Mann-Whitney U, two-sided").all()
    assert (pvals["correction"] == "Benjamini-Hochberg FDR across all tested rows").all()
    for row in pvals.itertuples():
        a = df.loc[df["condition"] == row.condition_a, row.metric].dropna()
        b = df.loc[df["condition"] == row.condition_b, row.metric].dropna()
        ref = stats.mann_whitney(a, b)
        assert (row.n_a, row.n_b) == (len(a), len(b))
        assert row.p_value == ref["p"] and row.cliffs_delta == ref["cliffs_delta"] == -1.0
    assert list(pvals["q_value"]) == list(stats.bh_fdr(pvals["p_value"].to_numpy()))


def test_significance_stars_triple_and_double_star():
    # p < 0.001 → "***"; p in [0.001, 0.01) → "**"
    assert stats.significance_stars(0.0009) == "***"
    assert stats.significance_stars(0.001) == "**"  # exactly at threshold: not < 0.001
    assert stats.significance_stars(0.005) == "**"
    assert stats.significance_stars(0.009) == "**"


def test_significance_stars_single_star():
    # p in [0.01, 0.05) → "*"  (line 258)
    assert stats.significance_stars(0.01) == "*"
    assert stats.significance_stars(0.04) == "*"
    assert stats.significance_stars(0.049) == "*"


def test_significance_stars_ns():
    # p >= 0.05 → "ns"
    assert stats.significance_stars(0.05) == "ns"
    assert stats.significance_stars(0.99) == "ns"


def test_compute_pvalues_keeps_untestable_pairs_with_a_reason():
    # Regression: pairs that could not be tested were dropped silently, so pvalues.csv
    # did not show that a comparison had been attempted.
    df = pd.DataFrame(
        {
            "condition": ["a", "b", "b", "b", "c", "c", "c", "d", "d"],
            "metric": [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 6.0],
        }
    )
    result = stats.compute_pvalues(df, ["metric"])
    rows = {(r.condition_a, r.condition_b): r for r in result.itertuples()}
    assert len(rows) == 6  # every pair is listed
    assert rows[("a", "b")].reason == "n<2" and rows[("a", "b")].n_a == 1
    assert rows[("b", "c")].reason == "all values tied"
    for pair in (("a", "b"), ("b", "c")):
        assert np.isnan(rows[pair].p_value) and np.isnan(rows[pair].q_value)
        assert rows[pair].significance == ""
    assert rows[("b", "d")].reason == "" and rows[("b", "d")].p_value > 0
    tested = result[result["reason"] == ""]
    assert list(tested["q_value"]) == list(stats.bh_fdr(tested["p_value"].to_numpy()))


def test_compute_pvalues_skips_missing_metric():
    """Metrics not in df are skipped."""
    df = pd.DataFrame({"condition": ["a", "b"], "metric": [1.0, 2.0]})
    result = stats.compute_pvalues(df, ["does_not_exist"])
    assert result.empty


def test_compute_pvalues_mannwhitney_valueerror_via_mock():
    """Mock mannwhitneyu to raise ValueError → except continue (lines 294-295)."""
    df = pd.DataFrame(
        {
            "condition": ["a", "a", "a", "b", "b", "b"],
            "metric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    with patch("scipy.stats.mannwhitneyu", side_effect=ValueError("identical")):
        result = stats.compute_pvalues(df, ["metric"])
    # ValueError was caught: the pair is kept, untested, with the reason
    assert result["reason"].tolist() == ["test not computable"]
    assert np.isnan(result["p_value"]).all()
