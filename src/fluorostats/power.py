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


def _sim_pvalues(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n: int,
    n_sims: int,
    seed: int,
) -> np.ndarray | None:
    """Mann-Whitney p-values of ``n_sims`` simulated experiments of ``n`` per group,
    each group resampled with replacement from its non-NaN values; None when a
    group has no measured value.

    The one resampling and seeding rule of this module: every power estimate (one
    metric at one ``n``) draws from a fresh ``np.random.default_rng(seed)``, so a
    metric's power at ``n`` is the same from every entry point. Raises ValueError
    for ``n_sims < 1``.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be at least 1, got {n_sims}")
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return None
    rng = np.random.default_rng(seed)
    p = np.empty(n_sims)
    for i in range(n_sims):
        sa = rng.choice(a, size=n, replace=True)
        sb = rng.choice(b, size=n, replace=True)
        p[i] = mann_whitney(sa, sb)["p"]
    return p


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

    Returns the fraction of simulations that reach ``alpha / n_tests_correction``;
    NaN if a sample has no measured (non-NaN) value.
    """
    p = _sim_pvalues(samples_a, samples_b, n, n_sims, seed)
    if p is None:
        return float("nan")
    threshold = alpha / max(1, n_tests_correction)
    return float((p < threshold).sum() / n_sims)


def power_curve(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    ns: list[int],
    n_sims: int = 1000,
    alpha: float = 0.05,
    n_tests_correction: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Power as a function of per-group sample size: `bootstrap_power` at each
    ``n`` with the same ``seed``.

    Returns a DataFrame with columns ``n`` and ``power``.
    """
    rows = [
        {
            "n": int(n),
            "power": bootstrap_power(
                samples_a,
                samples_b,
                n=n,
                n_sims=n_sims,
                alpha=alpha,
                n_tests_correction=n_tests_correction,
                seed=seed,
            ),
        }
        for n in ns
    ]
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
    and records which metrics passed q < alpha. Each metric is resampled as in
    `bootstrap_power` (same seed), so with one metric the two agree. A metric
    with no measured value in a group has NaN power and takes no part in the
    correction.

    Returns a long DataFrame with columns ``n``, ``metric``, ``power``. Raises
    ValueError for an empty metric set, a metric missing from
    ``samples_per_metric_b``, or ``n_sims < 1``.
    """
    metrics = list(samples_per_metric_a.keys())
    if not metrics:
        raise ValueError("fdr_power_curve: no metric given (empty samples_per_metric_a)")
    missing = [m for m in metrics if m not in samples_per_metric_b]
    if missing:
        raise ValueError(f"fdr_power_curve: metrics {missing} have no samples_per_metric_b")
    rows = []
    for n in ns:
        sims = {
            m: _sim_pvalues(samples_per_metric_a[m], samples_per_metric_b[m], n, n_sims, seed)
            for m in metrics
        }
        p = np.array(
            [sims[m] if sims[m] is not None else np.full(n_sims, np.nan) for m in metrics]
        )  # (metric, simulation)
        q = np.column_stack([bh_fdr(p[:, i]) for i in range(n_sims)])
        for m, qm in zip(metrics, q):
            power = float((qm < alpha).sum() / n_sims) if sims[m] is not None else float("nan")
            rows.append({"n": int(n), "metric": m, "power": power})
    return pd.DataFrame(rows)


__all__ = ["bootstrap_power", "power_curve", "fdr_power_curve"]
