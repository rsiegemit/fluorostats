"""Statistical comparisons for multi-condition experiments.

A pragmatic non-parametric toolkit:

  - `bh_fdr` — Benjamini-Hochberg q-values from p-values.
  - `cliffs_delta` — non-parametric effect size in [-1, 1].
  - `mann_whitney` — two-group two-sided test with effect size + n.
  - `stratified_mann_whitney` — per-stratum tests + BH-FDR across strata.
  - `bootstrap_fold_change_ci` — distribution-free fold-change interval.
  - `stouffer_combine` — meta-analytic Z-pooling across independent tests.
  - `scheirer_ray_hare` — non-parametric 2-way ANOVA on ranks.

All functions return plain dicts or :class:`pandas.DataFrame` — no
domain-specific column names baked in.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta = P(X > Y) - P(X < Y), in [-1, 1].

    Positive values mean ``x`` tends to exceed ``y``. Robust to outliers,
    distribution-free, and meaningful even for tiny samples.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (x.size * y.size))


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def bh_fdr(p_values: np.ndarray) -> np.ndarray:
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

def mann_whitney(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> dict:
    """Mann-Whitney U test with effect size and sample sizes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size < 1 or y.size < 1:
        return {"u": float("nan"), "p": float("nan"),
                "cliffs_delta": float("nan"), "n_x": int(x.size), "n_y": int(y.size)}
    try:
        u, p = sps.mannwhitneyu(x, y, alternative=alternative)
    except ValueError:
        u, p = float("nan"), float("nan")
    return {
        "u": float(u),
        "p": float(p),
        "cliffs_delta": cliffs_delta(x, y),
        "n_x": int(x.size),
        "n_y": int(y.size),
    }


def stratified_mann_whitney(
    df: pd.DataFrame,
    value_cols: list[str],
    group_col: str,
    group_a,
    group_b,
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
    `u`, `p`, and BH-corrected `q`.
    """
    strata = list(strata or [])
    grouped = df.groupby(strata, dropna=False) if strata else [((), df)]
    rows = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        a_df = sub[sub[group_col] == group_a]
        b_df = sub[sub[group_col] == group_b]
        for metric in value_cols:
            if metric not in sub.columns:
                continue
            a = a_df[metric].dropna().values
            b = b_df[metric].dropna().values
            res = mann_whitney(a, b)
            row = {k: v for k, v in zip(strata, keys)}
            row.update({
                "metric": metric,
                "n_a": res["n_x"],
                "n_b": res["n_y"],
                "median_a": float(np.median(a)) if a.size else float("nan"),
                "median_b": float(np.median(b)) if b.size else float("nan"),
                "cliffs_delta": res["cliffs_delta"],
                "u": res["u"],
                "p": res["p"],
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = bh_fdr(out["p"].values)
        out["sig_q05"] = out["q"] < 0.05
    return out


# ---------------------------------------------------------------------------
# Bootstrap fold-change CI
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

    Returns dict with keys: fold_change_median, ci_low, ci_high, n_boot.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return {"fold_change_median": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot": 0}
    func = {"median": np.median, "mean": np.mean}[agg]
    rng = np.random.default_rng(seed)
    bx = rng.choice(x, size=(n_boot, x.size), replace=True)
    by = rng.choice(y, size=(n_boot, y.size), replace=True)
    ax = func(bx, axis=1)
    ay = func(by, axis=1)
    eps = np.finfo(float).tiny
    fc = ay / np.where(ax == 0, eps, ax)
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    return {
        "fold_change_median": float(np.median(fc)),
        "ci_low": float(np.quantile(fc, lo_q)),
        "ci_high": float(np.quantile(fc, hi_q)),
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# Stouffer Z combination
# ---------------------------------------------------------------------------

def stouffer_combine(
    p_values: np.ndarray,
    weights: np.ndarray | None = None,
    one_sided: bool = False,
) -> dict:
    """Meta-analytic Z-combination of independent p-values.

    Useful for pooling evidence across strata or modalities. By default
    treats p-values as two-sided; pass `one_sided=True` if directional
    p's were supplied.
    """
    p = np.asarray(p_values, dtype=float)
    p = p[~np.isnan(p)]
    if p.size == 0:
        return {"z": float("nan"), "p": float("nan"), "n": 0}
    p = np.clip(p, np.finfo(float).tiny, 1 - 1e-15)
    if one_sided:
        z_individual = sps.norm.isf(p)
    else:
        z_individual = sps.norm.isf(p / 2.0)
    if weights is None:
        z = z_individual.sum() / np.sqrt(p.size)
    else:
        w = np.asarray(weights, dtype=float)[: p.size]
        z = (w * z_individual).sum() / np.sqrt((w ** 2).sum())
    p_combined = float(sps.norm.sf(z) if one_sided else 2 * sps.norm.sf(abs(z)))
    return {"z": float(z), "p": p_combined, "n": int(p.size)}


# ---------------------------------------------------------------------------
# Scheirer-Ray-Hare (non-parametric 2-way ANOVA)
# ---------------------------------------------------------------------------

def scheirer_ray_hare(
    df: pd.DataFrame,
    value_col: str,
    factor_a: str,
    factor_b: str,
) -> pd.DataFrame:
    """Non-parametric 2-way ANOVA on ranks (Scheirer-Ray-Hare).

    Returns one row per source: factor_a, factor_b, interaction, residual.
    Columns: source, df, H (sum-of-squares / MS_total), p (chi² survival).
    """
    sub = df[[value_col, factor_a, factor_b]].dropna().copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_rank"] = sps.rankdata(sub[value_col].values)
    N = len(sub)
    ms_total = sub["_rank"].var(ddof=0) * N / (N - 1) if N > 1 else 1.0  # ≈ N(N+1)/12

    def _ss(grouper):
        means = sub.groupby(grouper)["_rank"].mean()
        sizes = sub.groupby(grouper)["_rank"].size()
        grand = sub["_rank"].mean()
        return float((sizes * (means - grand) ** 2).sum())

    cell_means = sub.groupby([factor_a, factor_b])["_rank"].mean()
    cell_sizes = sub.groupby([factor_a, factor_b])["_rank"].size()
    grand = sub["_rank"].mean()
    ss_cells = float((cell_sizes * (cell_means - grand) ** 2).sum())
    ss_a = _ss(factor_a)
    ss_b = _ss(factor_b)
    ss_ab = ss_cells - ss_a - ss_b
    df_a = sub[factor_a].nunique() - 1
    df_b = sub[factor_b].nunique() - 1
    df_ab = df_a * df_b

    def _row(name, ss, df_):
        if df_ <= 0 or ms_total <= 0:
            return {"source": name, "df": df_, "H": float("nan"), "p": float("nan")}
        H = ss / ms_total
        return {"source": name, "df": int(df_), "H": float(H),
                "p": float(sps.chi2.sf(H, df_))}

    return pd.DataFrame([
        _row(factor_a, ss_a, df_a),
        _row(factor_b, ss_b, df_b),
        _row(f"{factor_a}:{factor_b}", ss_ab, df_ab),
    ])


__all__ = [
    "cliffs_delta",
    "bh_fdr",
    "mann_whitney",
    "stratified_mann_whitney",
    "bootstrap_fold_change_ci",
    "stouffer_combine",
    "scheirer_ray_hare",
]
