"""Tests for fluorostats.power."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats import power
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
    b_metrics = {"m1": rng.normal(loc=3, size=10), "m2": rng.normal(loc=0, size=10)}
    df = fdr_power_curve(a_metrics, b_metrics, ns=[6, 12], n_sims=150, seed=1)
    assert set(df["metric"]) == {"m1", "m2"}
    assert set(df["n"]) == {6, 12}
    # m1 (real effect) at n=12 should outperform m2 (null)
    m1_high = df[(df.metric == "m1") & (df.n == 12)].iloc[0]["power"]
    m2_high = df[(df.metric == "m2") & (df.n == 12)].iloc[0]["power"]
    assert m1_high > m2_high


def test_power_empty_group_is_nan_everywhere():
    # A metric with no measured value in a group has no power estimate: NaN from
    # every entry point (fdr_power_curve used to report 0.0).
    assert np.isnan(power.bootstrap_power(np.array([]), np.array([1.0, 2.0]), n=5, n_sims=10))
    r = power.fdr_power_curve(
        {"m": [], "k": [1.0, 2.0, 3.0]},
        {"m": [1.0, 2.0, 3.0], "k": [4.0, 5.0, 6.0]},
        ns=[4],
        n_sims=10,
    )
    assert np.isnan(r.set_index("metric").loc["m", "power"])
    assert 0.0 <= r.set_index("metric").loc["k", "power"] <= 1.0


def test_one_metric_has_one_power_whichever_entry_point():
    # Same metric, same seed: bootstrap_power, power_curve and a one-metric
    # fdr_power_curve (BH over one test is the raw p) give the same power at each n.
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 15), rng.normal(0.8, 1, 15)
    ns = [4, 8, 12]
    curve = power_curve(a, b, ns=ns, n_sims=120, seed=5)
    fdr = fdr_power_curve({"m": a}, {"m": b}, ns=ns, n_sims=120, seed=5)
    for i, n in enumerate(ns):
        single = bootstrap_power(a, b, n=n, n_sims=120, seed=5)
        assert curve["power"].iloc[i] == single == fdr["power"].iloc[i]


@pytest.mark.parametrize(
    "call",
    [
        lambda: bootstrap_power([1, 2], [3, 4], n=3, n_sims=0),
        lambda: fdr_power_curve({"a": [1, 2]}, {"a": [3, 4]}, [3], n_sims=0),
    ],
)
def test_power_rejects_zero_simulations(call):
    with pytest.raises(ValueError, match="n_sims"):
        call()


def test_fdr_power_curve_rejects_an_empty_metric_set():
    with pytest.raises(ValueError, match="metric"):
        fdr_power_curve({}, {}, [3], n_sims=5)


def test_fdr_power_curve_rejects_metrics_missing_from_one_group():
    with pytest.raises(ValueError, match="'a'"):
        fdr_power_curve({"a": [1, 2]}, {"b": [3, 4]}, [3], n_sims=5)
