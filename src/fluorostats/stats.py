"""Statistical comparisons for multi-condition experiments.

A pragmatic non-parametric toolkit:

  - `bh_fdr` — Benjamini-Hochberg q-values from p-values.
  - `cliffs_delta` — non-parametric effect size in [-1, 1].
  - `mann_whitney` — two-group two-sided test with effect size + n.
  - `welch_t` — Welch's t-test, for contrasts that explicitly ask for it.
  - `stratified_mann_whitney` — per-stratum tests + BH-FDR across strata.
  - `bootstrap_fold_change_ci` — distribution-free fold-change interval.
  - `bootstrap_mean_ci` — percentile-bootstrap interval for a mean.
  - `stouffer_combine` — meta-analytic Z-pooling across independent tests
    (defined in `stats_combine`, as is `scheirer_ray_hare`).
  - `scheirer_ray_hare` — non-parametric 2-way ANOVA on ranks.
  - `gini`, `coefficient_of_variation` — dispersion summaries used by the
    homogeneity metrics (`morphometry`, `objects`, `spatial`, `ring`).

All functions return plain dicts or :class:`pandas.DataFrame` — no
domain-specific column names baked in.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------


def describe(values: ArrayLike, axis: int | None = None) -> dict:
    """``{"n", "mean", "sd", "sem"}`` of the non-NaN values; NaN where undefined.

    NaN entries are dropped and ``n`` counts what remains. The mean is NaN for
    n = 0; the sample SD (ddof=1) and SEM are NaN for n < 2, because a single value
    has no measured spread (0.0 would claim a precision nobody measured).

    With ``axis`` the same summary is taken along that axis of an array (e.g. per
    depth over a ``(stacks, depths)`` array with ``axis=0``) and each entry is an
    array; without it, of all values, as scalars.
    """
    if axis is not None:
        return _describe_along(np.asarray(values, dtype=float), axis)
    a = np.asarray(values, dtype=float).ravel()
    a = a[~np.isnan(a)]
    n = int(a.size)
    nan = float("nan")
    sd = float(a.std(ddof=1)) if n > 1 else nan
    return {
        "n": n,
        "mean": float(a.mean()) if n else nan,
        "sd": sd,
        "sem": sd / float(np.sqrt(n)) if n > 1 else nan,
    }


def _describe_along(a: np.ndarray, axis: int) -> dict:
    """`describe` along ``axis``: the same rules per position, as arrays, without warnings."""
    valid = ~np.isnan(a)
    n = valid.sum(axis=axis)
    undefined = np.full(n.shape, np.nan)
    mean = np.divide(np.where(valid, a, 0.0).sum(axis=axis), n, out=undefined.copy(), where=n > 0)
    dev = np.where(valid, a - np.expand_dims(mean, axis), 0.0)
    sd = np.sqrt(np.divide((dev**2).sum(axis=axis), n - 1, out=undefined.copy(), where=n > 1))
    sem = np.divide(sd, np.sqrt(n), out=undefined.copy(), where=n > 1)
    return {"n": n, "mean": mean, "sd": sd, "sem": sem}


# Location statistics selectable by name: `depth`'s per-slice reducer and the
# bootstrap fold-change aggregate. One registry, so both accept the same names.
REDUCERS: dict[str, Callable[..., Any]] = {"mean": np.mean, "median": np.median}


def reducer(name: str, what: str = "reducer") -> Callable[..., Any]:
    """The :data:`REDUCERS` function called ``name``; ValueError naming the parameter
    ``what`` and the valid names otherwise."""
    if not isinstance(name, str) or name not in REDUCERS:
        names = " or ".join(repr(k) for k in REDUCERS)
        raise ValueError(f"{what} must be {names}, got {name!r}")
    return REDUCERS[name]


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def gini(values: ArrayLike) -> float:
    """Gini coefficient of a 1D array (0 = perfectly even, -> 1 = concentrated).

    Negative inputs are shifted to be non-negative; an all-zero array is 0.0 and
    an empty one NaN.
    """
    v: np.ndarray = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return float("nan")
    v = v - v.min() if v.min() < 0 else v
    if v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def coefficient_of_variation(values: np.ndarray, ddof: int = 0) -> float:
    """Coefficient of variation ``std(ddof) / mean``.

    Every caller in the package (the homogeneity metrics in `morphometry`,
    `objects`, `ring` and `spatial`) takes the CV over every tile of the image:
    the whole population of tiles, not a sample drawn from one, so they use
    ``ddof=0``. ``ddof`` stays a parameter for a caller whose values are a sample
    (``ddof=1``). NaN when there are ``<= ddof`` values or the mean is 0.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size <= ddof or v.mean() == 0:
        return float("nan")
    return float(v.std(ddof=ddof) / v.mean())


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta = P(X > Y) - P(X < Y), in [-1, 1].

    Positive values mean ``x`` tends to exceed ``y``. Robust to outliers,
    distribution-free, and meaningful even for tiny samples. NaN values are
    dropped; NaN when either sample is then empty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (x.size * y.size))


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------


def bh_fdr(p_values: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values from a 1D array of p-values.

    NaNs are preserved and excluded from the rank denominator.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    if not mask.any():
        return out
    pv = p[mask]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from largest to smallest p
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(n)
    out[mask] = q[inverse]
    return out


# ---------------------------------------------------------------------------
# Two-group comparisons
# ---------------------------------------------------------------------------


def welch_t(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> dict:
    """Welch's unequal-variance t-test, two-sided.

    Parametric, so not this package's default (see `mann_whitney`); provided for
    descriptive contrasts that name it. Returns ``t``, ``p``, ``n_x``, ``n_y`` and
    ``reason``: empty when tested, ``"n<2"`` when a group has fewer than
    :data:`MIN_N` values (``t`` and ``p`` NaN, not tested). Values are passed to
    scipy unchanged, so NaN propagates into the result.
    """
    n = {"n_x": len(x), "n_y": len(y)}
    if min(len(x), len(y)) < MIN_N:
        return {"t": float("nan"), "p": float("nan"), **n, "reason": f"n<{MIN_N}"}
    t, p = sps.ttest_ind(x, y, equal_var=False)
    return {"t": float(t), "p": float(p), **n, "reason": ""}


# The fewest values per group a two-group test is run on: a group of one has no spread.
# Every two-group entry point (`mann_whitney`, the grids built on it, and `welch_t`)
# reports such a pair as untested (p NaN, reason "n<2"), so it never enters a BH
# denominator.
MIN_N = 2
_ALTERNATIVES = ("two-sided", "less", "greater")


def mann_whitney(x: ArrayLike, y: ArrayLike, alternative: str = "two-sided") -> dict:
    """Mann-Whitney U test with effect size and sample sizes.

    NaN values are dropped. Returns ``u``, ``p``, ``cliffs_delta``, ``n_x``, ``n_y``
    and ``reason``: empty when tested, else why ``p`` is NaN. ``"n<2"``: a group has
    fewer than :data:`MIN_N` values (``u`` and ``cliffs_delta`` are NaN too).
    ``"all values tied"``: the test has no information, whatever the installed
    scipy says (scipy < 1.18 returned 1.0 there, later versions NaN).
    ``"test not computable"``: scipy returned NaN. ``alternative`` is
    ``"two-sided"``, ``"less"`` or ``"greater"``; anything else is a ValueError.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")
    xa: np.ndarray = np.asarray(x, dtype=float)
    ya: np.ndarray = np.asarray(y, dtype=float)
    xa = xa[~np.isnan(xa)]
    ya = ya[~np.isnan(ya)]
    nan = float("nan")
    n = {"n_x": int(xa.size), "n_y": int(ya.size)}
    if min(xa.size, ya.size) < MIN_N:
        return {"u": nan, "p": nan, "cliffs_delta": nan, **n, "reason": f"n<{MIN_N}"}
    u, p = sps.mannwhitneyu(xa, ya, alternative=alternative)
    reason = ""
    if np.ptp(np.concatenate([xa, ya])) == 0:  # all tied: no information
        p, reason = nan, "all values tied"
    elif np.isnan(p):
        reason = "test not computable"
    delta = cliffs_delta(xa, ya)
    return {"u": float(u), "p": float(p), "cliffs_delta": delta, **n, "reason": reason}


def stratified_mann_whitney(
    df: pd.DataFrame,
    value_cols: list[str],
    group_col: str,
    group_a: Any,
    group_b: Any,
    strata: list[str] | None = None,
) -> pd.DataFrame:
    """Per-stratum two-group test plus BH-FDR across the entire grid.

    Parameters
    ----------
    df : DataFrame
    value_cols : list of metric column names
    group_col : column distinguishing the two groups being compared
    group_a, group_b : values of `group_col` for the two groups
    strata : list of additional columns to stratify by. If None, a single
        stratum is used. The within-stratum two-group test is unaffected
        by the stratification; BH-FDR is computed across every
        (stratum × metric) cell.

    Returns
    -------
    DataFrame with one row per (stratum × metric): stratum keys,
    `metric`, `n_a`, `n_b`, `median_a`, `median_b`, `cliffs_delta`,
    `u`, `p`, BH-corrected `q`, `sig_q05` and `reason` (`mann_whitney`'s: empty
    when tested, else why `p` is NaN).
    """
    strata = list(strata or [])
    grouped = df.groupby(strata, dropna=False) if strata else [((), df)]
    rows = []
    for keys, sub in grouped:
        a_df = sub[sub[group_col] == group_a]
        b_df = sub[sub[group_col] == group_b]
        for metric in value_cols:
            if metric not in sub.columns:
                continue
            a = a_df[metric].dropna().values
            b = b_df[metric].dropna().values
            res = mann_whitney(a, b)
            row = {k: v for k, v in zip(strata, keys)}
            row.update(
                {
                    "metric": metric,
                    "n_a": res["n_x"],
                    "n_b": res["n_y"],
                    "median_a": float(np.median(a)) if a.size else float("nan"),
                    "median_b": float(np.median(b)) if b.size else float("nan"),
                    "cliffs_delta": res["cliffs_delta"],
                    "u": res["u"],
                    "p": res["p"],
                    "reason": res["reason"],
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = bh_fdr(out["p"].values)
        out["sig_q05"] = out["q"] < 0.05
    return out


def significance_stars(p: float) -> str:
    """``***`` p < 0.001, ``**`` < 0.01, ``*`` < 0.05, else ``ns``."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


_PVALUE_COLUMNS = (
    "metric condition_a condition_b n_a n_b p_value q_value significance cliffs_delta test "
    "correction reason"
).split()


def compute_pvalues(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Compute pairwise Mann-Whitney U p-values for all condition pairs.

    Multiple-comparison corrected: a Benjamini-Hochberg ``q_value`` is computed
    across every tested row of the grid (metric × condition-pair), and the
    ``significance`` stars reflect the corrected q, matching the library's default
    correction (``stats.bh_fdr``). The raw ``p_value`` is retained for transparency.

    A pair that cannot be tested is kept with ``p_value``/``q_value`` NaN, empty
    ``significance`` and `mann_whitney`'s ``reason`` ("n<2", "all values tied",
    "test not computable"); ``reason`` is empty for tested rows. Rows with no
    condition label are left out, with a warning; labels mixing numbers and
    strings sort numbers first.

    Returns a DataFrame with columns: metric, condition_a, condition_b, n_a, n_b,
    p_value, q_value, significance, cliffs_delta (``stats.cliffs_delta`` of a vs b),
    test and correction (naming both), reason; empty if no metric is in ``df``.
    """

    if unlabelled := int(df["condition"].isna().sum()):
        msg = f"compute_pvalues: left out {unlabelled} row(s) with no condition label"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    labels = df["condition"].dropna().unique()
    conditions = sorted(labels, key=lambda c: (isinstance(c, str), c))
    rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue
        for i, j in combinations(range(len(conditions)), 2):
            a = df.loc[df["condition"] == conditions[i], metric].dropna().values
            b = df.loc[df["condition"] == conditions[j], metric].dropna().values
            rows.append(
                {
                    "metric": metric,
                    "condition_a": conditions[i],
                    "condition_b": conditions[j],
                    "n_a": len(a),
                    "n_b": len(b),
                    **_pair_test(a, b),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = bh_fdr(out["p_value"].values)
        # stars on the CORRECTED q; none for an untested pair
        out["significance"] = [
            significance_stars(q) if not np.isnan(q) else "" for q in out["q_value"]
        ]
        out["test"] = "Mann-Whitney U, two-sided"
        out["correction"] = "Benjamini-Hochberg FDR across all tested rows"
        # existing columns first, in their earlier order; the added ones last
        out = out[_PVALUE_COLUMNS]
    return out


def _pair_test(a: np.ndarray, b: np.ndarray) -> dict:
    """``p_value``, ``cliffs_delta`` and ``reason`` (empty when tested) for one pair."""
    res = mann_whitney(a, b)
    return {"p_value": res["p"], "cliffs_delta": res["cliffs_delta"], "reason": res["reason"]}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_fold_change_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 5000,
    ci: float = 0.95,
    agg: str = "median",
    seed: int = 0,
) -> dict:
    """Distribution-free CI for the ``agg(y) / agg(x)`` ratio.

    Bootstraps both groups with replacement, recomputes the ratio, and
    returns the median and central-CI quantiles.

    ``agg`` is a :data:`REDUCERS` name. Returns dict with keys: fold_change_median,
    ci_low, ci_high, n_boot.
    """
    func = reducer(agg, "agg")
    _check_bootstrap(n_boot, ci)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return {
            "fold_change_median": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": 0,
        }
    rng = np.random.default_rng(seed)
    bx = rng.choice(x, size=(n_boot, x.size), replace=True)
    by = rng.choice(y, size=(n_boot, y.size), replace=True)
    ax = func(bx, axis=1)
    ay = func(by, axis=1)
    eps = np.finfo(float).tiny
    n_zero = int((ax == 0).sum())
    if n_zero:
        warnings.warn(
            f"bootstrap_fold_change_ci: agg(x) was 0 in {n_zero} of {n_boot} resamples; "
            "those ratios divide by the smallest float, so the interval is not "
            "meaningful.",
            RuntimeWarning,
            stacklevel=2,
        )
    fc = ay / np.where(ax == 0, eps, ax)
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    return {
        "fold_change_median": float(np.median(fc)),
        "ci_low": float(np.quantile(fc, lo_q)),
        "ci_high": float(np.quantile(fc, hi_q)),
        "n_boot": int(n_boot),
    }


def bootstrap_mean_ci(
    x: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Mean with a percentile-bootstrap confidence interval.

    Resamples ``x`` with replacement ``n_boot`` times (all draws in one block, so
    the result is fixed by the generator) and returns ``{"mean", "ci_low",
    "ci_high"}``. The generator is ``default_rng(seed)`` unless ``rng`` is given;
    pass one ``rng`` to draw several intervals from a single stream. NaNs are
    dropped first; raises ValueError when no value remains.
    """
    _check_bootstrap(n_boot, ci)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("bootstrap_mean_ci needs at least one non-NaN value")
    rng = np.random.default_rng(seed) if rng is None else rng
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(1)
    lo = (100 - 100 * ci) / 2  # exactly 2.5 for ci=0.95
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.percentile(means, lo)),
        "ci_high": float(np.percentile(means, 100 - lo)),
    }


def _check_bootstrap(n_boot: int, ci: float) -> None:
    """ValueError naming the parameter unless ``n_boot >= 1`` and ``0 < ci < 1``."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot!r}")
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")


from fluorostats.stats_combine import scheirer_ray_hare, stouffer_combine  # noqa: E402


__all__ = [
    "compute_pvalues",
    "significance_stars",
    "describe",
    "REDUCERS",
    "reducer",
    "gini",
    "coefficient_of_variation",
    "cliffs_delta",
    "bh_fdr",
    "welch_t",
    "MIN_N",
    "mann_whitney",
    "stratified_mann_whitney",
    "bootstrap_fold_change_ci",
    "bootstrap_mean_ci",
    "stouffer_combine",
    "scheirer_ray_hare",
]
