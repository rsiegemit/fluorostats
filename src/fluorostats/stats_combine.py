"""Pooled and two-factor tests: Stouffer Z-combination and Scheirer-Ray-Hare.

Split from :mod:`fluorostats.stats`, which re-exports both names.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps


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
    p's were supplied. Two-sided p-values carry no direction, so effects in
    opposite directions add up as if they agreed; when direction matters, pool
    one-sided p-values that all test the same direction.
    """
    p = np.asarray(p_values, dtype=float)
    keep = ~np.isnan(p)
    p = p[keep]
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
        w = np.asarray(weights, dtype=float)[keep]  # aligned with the non-NaN p-values
        z = (w * z_individual).sum() / np.sqrt((w**2).sum())
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

    Returns one row per source: factor_a, factor_b, interaction.
    Columns: source, df, H (sum-of-squares / MS_total), p (chi² survival).

    The interaction sum of squares is obtained by subtraction, which is exact only
    for a balanced design (equal observations in every factor_a × factor_b cell);
    an unbalanced design is warned about. Ranks are not tie-corrected.
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
    n_cells = sub[factor_a].nunique() * sub[factor_b].nunique()
    if cell_sizes.nunique() > 1 or len(cell_sizes) < n_cells:
        warnings.warn(
            f"scheirer_ray_hare: the {factor_a} x {factor_b} design is unbalanced "
            f"({len(cell_sizes)} of {n_cells} cells observed, cell sizes "
            f"{cell_sizes.min()}-{cell_sizes.max()}); the interaction term assumes "
            "a balanced design.",
            RuntimeWarning,
            stacklevel=2,
        )
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
        return {"source": name, "df": int(df_), "H": float(H), "p": float(sps.chi2.sf(H, df_))}

    return pd.DataFrame(
        [
            _row(factor_a, ss_a, df_a),
            _row(factor_b, ss_b, df_b),
            _row(f"{factor_a}:{factor_b}", ss_ab, df_ab),
        ]
    )


__all__ = ["stouffer_combine", "scheirer_ray_hare"]
