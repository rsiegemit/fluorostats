"""Tests for fluorostats.power."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats.power import bootstrap_power, power_curve, fdr_power_curve


def test_bootstrap_power_increases_with_n():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, size=12)
    b = rng.normal(loc=2.0, size=12)
    low = bootstrap_power(a, b, n=3, n_sims=200, seed=1)
    high = bootstrap_power(a, b, n=12, n_sims=200, seed=1)
    assert high >= low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_bootstrap_power_no_signal_stays_near_alpha():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20)
    b = rng.normal(size=20)
    power = bootstrap_power(a, b, n=10, n_sims=200, alpha=0.05, seed=1)
    assert power < 0.25


def test_power_curve_monotone_for_signal():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, size=20)
    b = rng.normal(loc=2.0, size=20)
    df = power_curve(a, b, ns=[3, 6, 12], n_sims=200, seed=1)
    assert list(df["n"]) == [3, 6, 12]
    assert df["power"].iloc[-1] >= df["power"].iloc[0]


def test_fdr_power_curve_returns_per_metric_rows():
    rng = np.random.default_rng(0)
    a_metrics = {"m1": rng.normal(size=10), "m2": rng.normal(size=10)}
    b_metrics = {"m1": rng.normal(loc=3, size=10),
                 "m2": rng.normal(loc=0, size=10)}
    df = fdr_power_curve(a_metrics, b_metrics, ns=[6, 12], n_sims=150, seed=1)
    assert set(df["metric"]) == {"m1", "m2"}
    assert set(df["n"]) == {6, 12}
    # m1 (real effect) at n=12 should outperform m2 (null)
    m1_high = df[(df.metric == "m1") & (df.n == 12)].iloc[0]["power"]
    m2_high = df[(df.metric == "m2") & (df.n == 12)].iloc[0]["power"]
    assert m1_high > m2_high
