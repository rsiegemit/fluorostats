"""Tests for fluorostats.agreement."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats.agreement import bland_altman, lins_ccc, icc, agreement_report


def test_identical_series_perfect_agreement():
    a = np.array([1.0, 2, 3, 4, 5])
    assert lins_ccc(a, a) == pytest.approx(1.0)
    assert icc(a, a) == pytest.approx(1.0, abs=1e-6)
    rep = agreement_report(a, a)
    assert rep["exact_match"] is True
    assert rep["bias"] == pytest.approx(0.0)


def test_constant_offset_reduces_ccc_but_keeps_correlation():
    a = np.array([1.0, 2, 3, 4, 5])
    b = a + 3
    ba = bland_altman(a, b)
    assert ba["bias"] == pytest.approx(-3.0)
    assert lins_ccc(a, b) < 0.5           # accuracy penalised
    assert agreement_report(a, b)["spearman"] == pytest.approx(1.0)  # precision intact


def test_bland_altman_limits_of_agreement():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.5, size=200)
    ba = bland_altman(a, b)
    assert ba["loa_lower"] < ba["bias"] < ba["loa_upper"]
    assert ba["n"] == 200


def test_handles_nans():
    a = np.array([1.0, np.nan, 3, 4])
    b = np.array([1.0, 2, np.nan, 4])
    rep = agreement_report(a, b)
    assert rep["n"] == 2   # only paired non-nan entries


def test_report_keys():
    rep = agreement_report([1, 2, 3], [1, 2, 4], "fs", "gt")
    for k in ("ccc", "icc", "spearman", "pearson", "bias",
              "loa_lower", "loa_upper", "mape_pct", "n"):
        assert k in rep
