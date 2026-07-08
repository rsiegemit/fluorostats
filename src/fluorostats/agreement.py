"""Method-comparison / agreement statistics.

For validating one measurement method against another (or against ground
truth): Bland-Altman limits of agreement, Lin's concordance correlation
coefficient, ICC(A,1) absolute agreement, plus a combined report. Kept as its
own module (not folded into ``stats``) since it answers a different question —
*do two methods agree?* rather than *do two groups differ?*
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sps


def _paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]


def bland_altman(a, b) -> dict:
    """Bland-Altman agreement between two paired measurement vectors.

    Returns bias (mean difference), SD of differences, 95% limits of
    agreement, and the per-point means/differences for plotting.
    """
    a, b = _paired(a, b)
    diff = a - b
    bias = float(diff.mean()) if diff.size else float("nan")
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return {
        "n": int(a.size),
        "bias": bias,
        "sd_diff": sd,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "mean_vals": (a + b) / 2,
        "diff_vals": diff,
    }


def lins_ccc(a, b) -> float:
    """Lin's concordance correlation coefficient (agreement with the identity
    line): combines precision (correlation) and accuracy (bias)."""
    a, b = _paired(a, b)
    if a.size < 2:
        return float("nan")
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    denom = va + vb + (ma - mb) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def icc(a, b) -> float:
    """ICC(A,1), two-way random, absolute agreement, single measure — treats
    the two methods as two raters of the same targets."""
    a, b = _paired(a, b)
    n = a.size
    if n < 2:
        return float("nan")
    Y = np.column_stack([a, b])
    k = 2
    grand = Y.mean()
    ms_rows = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom > 0 else float("nan")


def agreement_report(a, b, name_a: str = "method A", name_b: str = "method B") -> dict:
    """Full agreement summary between two paired measurement vectors:
    Bland-Altman bias + limits of agreement, Lin's CCC, ICC, Spearman,
    Pearson, MAPE (using ``b`` as reference), and an exact-match flag."""
    a, b = _paired(a, b)
    ba = bland_altman(a, b)
    spearman = _sps.spearmanr(a, b).statistic if a.size > 1 else float("nan")
    pearson = _sps.pearsonr(a, b).statistic if a.size > 1 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs((a - b) / np.where(b == 0, np.nan, b)))) * 100
    return {
        "name_a": name_a, "name_b": name_b, "n": int(a.size),
        "bias": ba["bias"], "loa_lower": ba["loa_lower"], "loa_upper": ba["loa_upper"],
        "ccc": lins_ccc(a, b), "icc": icc(a, b),
        "spearman": float(spearman), "pearson": float(pearson),
        "mape_pct": mape,
        "exact_match": bool(a.size and np.array_equal(a, b)),
    }


__all__ = ["bland_altman", "lins_ccc", "icc", "agreement_report"]
