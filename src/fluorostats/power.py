"""Bootstrap-based power analysis from observed samples.

Given two empirical samples (typical "pilot data"), simulate larger
hypothetical experiments by resampling with replacement, run the same
statistical test that will be reported, and count the fraction of
simulations that clear a chosen alpha (or BH-FDR) threshold.

This avoids parametric power formulas that assume normal distributions
with known variance, which rarely hold for microscopy metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import mann_whitney, bh_fdr


def bootstrap_power(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n: int,
    n_sims: int = 1000,
    alpha: float = 0.05,
    n_tests_correction: int = 1,
    seed: int = 0,
) -> float:
    """Estimated power for a Mann-Whitney test at sample size ``n`` per group.

    For each simulation we draw ``n`` observations with replacement from
    each sample, run the test, and count successes after a Bonferroni-
    style alpha correction for ``n_tests_correction`` hypotheses.

    Returns the fraction of simulations that reach ``alpha / n_tests_correction``.
    """
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    threshold = alpha / max(1, n_tests_correction)
    hits = 0
    for _ in range(n_sims):
        sa = rng.choice(a, size=n, replace=True)
        sb = rng.choice(b, size=n, replace=True)
        res = mann_whitney(sa, sb)
        if res["p"] < threshold:
            hits += 1
    return float(hits / n_sims)


def power_curve(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    ns: list[int],
    n_sims: int = 1000,
    alpha: float = 0.05,
    n_tests_correction: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Power as a function of per-group sample size.

    Returns a DataFrame with columns ``n`` and ``power``.
    """
    rows = []
    for i, n in enumerate(ns):
        p = bootstrap_power(
            samples_a, samples_b, n=n, n_sims=n_sims, alpha=alpha,
            n_tests_correction=n_tests_correction, seed=seed + i,
        )
        rows.append({"n": int(n), "power": float(p)})
    return pd.DataFrame(rows)


def fdr_power_curve(
    samples_per_metric_a: dict[str, np.ndarray],
    samples_per_metric_b: dict[str, np.ndarray],
    ns: list[int],
    n_sims: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-metric power under joint BH-FDR correction.

    Each simulation draws ``n`` observations per group for every metric,
    runs Mann-Whitney on each, applies BH-FDR across the metric set,
    and records which metrics passed q < alpha.

    Returns a long DataFrame with columns ``n``, ``metric``, ``power``.
    """
    metrics = list(samples_per_metric_a.keys())
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        passes = {m: 0 for m in metrics}
        for _ in range(n_sims):
            p_vals = []
            for m in metrics:
                a = np.asarray(samples_per_metric_a[m], dtype=float)
                b = np.asarray(samples_per_metric_b[m], dtype=float)
                a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
                if a.size == 0 or b.size == 0:
                    p_vals.append(float("nan"))
                    continue
                sa = rng.choice(a, size=n, replace=True)
                sb = rng.choice(b, size=n, replace=True)
                p_vals.append(mann_whitney(sa, sb)["p"])
            q = bh_fdr(np.array(p_vals))
            for m, qm in zip(metrics, q):
                if qm < alpha:
                    passes[m] += 1
        for m in metrics:
            rows.append({"n": int(n), "metric": m, "power": passes[m] / n_sims})
    return pd.DataFrame(rows)


__all__ = ["bootstrap_power", "power_curve", "fdr_power_curve"]
