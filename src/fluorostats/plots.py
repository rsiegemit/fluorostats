"""Comparison plots grouped by experimental condition, with replicate support."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-metric plots
# ---------------------------------------------------------------------------

def boxplot_by_condition(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    ylabel: str | None = None,
) -> None:
    """Boxplot of *metric* grouped by condition, with individual data points."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = sorted(df["condition"].unique())
    data = [df.loc[df["condition"] == c, metric].dropna().values for c in conditions]

    fig, ax = plt.subplots(figsize=(max(4, len(conditions) * 1.5), 5))

    bp = ax.boxplot(
        data, labels=conditions, patch_artist=True, widths=0.5, showfliers=False,
    )

    colors = plt.cm.Set2(range(len(conditions)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, d in enumerate(data):
        jitter = 0.1 * (np.random.default_rng(42).random(len(d)) - 0.5)
        ax.scatter(np.full(len(d), i + 1) + jitter, d, color="k", s=20, alpha=0.7, zorder=3)

    ax.set_ylabel(ylabel or _label_for(metric))
    ax.set_title(f"{_label_for(metric)} by Condition")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def bar_mean_sem(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    ylabel: str | None = None,
    show_pvalues: bool = False,
) -> None:
    """Bar chart with mean +/- SEM and individual replicate points.

    Publication-standard format for n=3-5 replicates.
    P-value brackets are off by default (use pvalues.csv instead).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = sorted(df["condition"].unique())
    means = []
    sems = []
    data = []
    for c in conditions:
        vals = df.loc[df["condition"] == c, metric].dropna().values
        data.append(vals)
        means.append(vals.mean() if len(vals) > 0 else 0)
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

    x = np.arange(len(conditions))
    colors = plt.cm.Set2(range(len(conditions)))

    fig, ax = plt.subplots(figsize=(max(4, len(conditions) * 1.8), 5))

    bars = ax.bar(x, means, yerr=sems, width=0.6, color=colors, edgecolor="black",
                  linewidth=0.8, capsize=5, error_kw={"linewidth": 1.5})

    # Overlay individual replicate points
    for i, d in enumerate(data):
        jitter = 0.12 * (np.random.default_rng(42).random(len(d)) - 0.5)
        ax.scatter(np.full(len(d), x[i]) + jitter, d, color="k", s=30, alpha=0.7, zorder=3)

    # P-value annotations
    if show_pvalues and len(conditions) > 1:
        _add_pvalue_annotations(ax, data, x, conditions)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20 if len(conditions) > 3 else 0)
    ax.set_ylabel(ylabel or _label_for(metric), fontsize=11, fontweight="bold")
    ax.set_title(f"{_label_for(metric)} by Condition", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-metric summary figure
# ---------------------------------------------------------------------------

def summary_panel(
    df: pd.DataFrame,
    metrics: list[str],
    out_path: Path,
    title: str = "Quantification Summary",
    ncols: int = 4,
    show_pvalues: bool = False,
) -> None:
    """Multi-panel figure: one bar+SEM subplot per metric.

    Generates a publication-ready composite figure with all metrics.
    P-value brackets are off by default (use pvalues.csv instead).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    available = [m for m in metrics if m in df.columns and df[m].notna().any()]
    n = len(available)
    if n == 0:
        return

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flat

    conditions = sorted(df["condition"].unique())
    colors = plt.cm.Set2(range(len(conditions)))
    x = np.arange(len(conditions))

    for idx, metric in enumerate(available):
        ax = axes[idx]
        data = []
        means = []
        sems = []
        for c in conditions:
            vals = df.loc[df["condition"] == c, metric].dropna().values
            data.append(vals)
            means.append(vals.mean() if len(vals) > 0 else 0)
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        ax.bar(x, means, yerr=sems, width=0.6, color=colors, edgecolor="black",
               linewidth=0.8, capsize=4, error_kw={"linewidth": 1.2})

        for i, d in enumerate(data):
            jitter = 0.1 * (np.random.default_rng(42).random(len(d)) - 0.5)
            ax.scatter(np.full(len(d), x[i]) + jitter, d, color="k", s=20, alpha=0.7, zorder=3)

        if show_pvalues and len(conditions) > 1:
            _add_pvalue_annotations(ax, data, x, conditions)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20 if len(conditions) > 3 else 0, fontsize=9)
        ax.set_ylabel(_label_for(metric), fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _add_pvalue_annotations(ax, data, x, conditions, max_brackets: int = 6):
    """Add p-value brackets — only the most meaningful comparisons.

    For <= 4 conditions: show all significant pairs.
    For > 4 conditions: show only adjacent pairs (dose-response neighbors).
    Limits total brackets to max_brackets to keep plots clean.
    """
    from scipy.stats import mannwhitneyu

    n = len(conditions)

    # Choose which pairs to test
    if n <= 4:
        pairs = list(combinations(range(n), 2))
    else:
        # Adjacent pairs only (i, i+1) — natural for dose-response
        pairs = [(i, i + 1) for i in range(n - 1)]

    # Compute p-values and filter to significant
    sig_pairs = []
    for i, j in pairs:
        if len(data[i]) < 2 or len(data[j]) < 2:
            continue
        try:
            _, p = mannwhitneyu(data[i], data[j], alternative="two-sided")
        except ValueError:
            continue
        stars = _pvalue_stars(p)
        if stars != "ns":
            sig_pairs.append((i, j, stars, p))

    # Sort by significance (most significant first) and limit
    sig_pairs.sort(key=lambda t: t[3])
    sig_pairs = sig_pairs[:max_brackets]
    # Re-sort by position for clean stacking
    sig_pairs.sort(key=lambda t: (t[0], t[1]))

    if not sig_pairs:
        return

    y_max = ax.get_ylim()[1]
    step = y_max * 0.07
    offset = y_max * 0.05

    for drawn_idx, (i, j, stars, _) in enumerate(sig_pairs):
        y = y_max + offset + drawn_idx * step
        ax.plot([x[i], x[i], x[j], x[j]], [y - step * 0.2, y, y, y - step * 0.2],
                color="black", linewidth=0.8)
        ax.text((x[i] + x[j]) / 2, y, stars, ha="center", va="bottom", fontsize=9)

    # Extend y-axis just enough for drawn brackets
    new_max = y_max + offset + len(sig_pairs) * step + step * 0.5
    ax.set_ylim(ax.get_ylim()[0], new_max)


def _pvalue_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compute_pvalues(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Compute pairwise Mann-Whitney U p-values for all condition pairs.

    Returns a DataFrame with columns: metric, condition_a, condition_b, p_value, significance.
    """
    from scipy.stats import mannwhitneyu

    conditions = sorted(df["condition"].unique())
    rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue
        for i, j in combinations(range(len(conditions)), 2):
            a = df.loc[df["condition"] == conditions[i], metric].dropna().values
            b = df.loc[df["condition"] == conditions[j], metric].dropna().values
            if len(a) < 2 or len(b) < 2:
                continue
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
                rows.append({
                    "metric": metric,
                    "condition_a": conditions[i],
                    "condition_b": conditions[j],
                    "n_a": len(a),
                    "n_b": len(b),
                    "p_value": p,
                    "significance": _pvalue_stars(p),
                })
            except ValueError:
                continue

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

_LABELS = {
    "volume_fraction": "Volume Fraction",
    "area_fraction": "Area Fraction (%)",
    "n_components": "Connected Components",
    "euler_number": "Euler Number",
    "largest_component_fraction": "Largest Component Fraction",
    "total_length_um": "Total Skeleton Length (µm)",
    "n_branches": "Number of Branches",
    "n_junctions": "Number of Junctions",
    "mean_branch_length_um": "Mean Branch Length (µm)",
    "mean_cluster_area_px": "Mean Cluster Area (px)",
    "median_cluster_area_px": "Median Cluster Area (px)",
}


def _label_for(metric: str) -> str:
    return _LABELS.get(metric, metric.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Effect-size grids and forest plots
# ---------------------------------------------------------------------------

def effect_size_heatmap(
    stats_df: pd.DataFrame,
    out_path: Path | None = None,
    *,
    row_col: str = "metric",
    col_col: str = "stratum",
    value_col: str = "cliffs_delta",
    sig_col: str | None = "sig_q05",
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Heatmap of an effect-size column across (metric × stratum).

    Cells flagged by `sig_col` (boolean column, e.g. BH-FDR sig) are
    annotated with a star. Pass a long-format DataFrame produced by
    :func:`fluorostats.stats.stratified_mann_whitney` (or similar);
    if `col_col` does not exist, all rows are collapsed into a single
    "all" column.
    """
    df = stats_df.copy()
    if col_col not in df.columns:
        df[col_col] = "all"
    pivot = df.pivot_table(index=row_col, columns=col_col, values=value_col,
                           aggfunc="first")
    sig = (df.pivot_table(index=row_col, columns=col_col, values=sig_col,
                          aggfunc="first").fillna(False).astype(bool)
           if sig_col and sig_col in df.columns else None)

    nrows, ncols = pivot.shape
    figsize = figsize or (max(4, ncols * 1.1), max(2, nrows * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(ncols)); ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(nrows)); ax.set_yticklabels([_label_for(r) for r in pivot.index])
    for i in range(nrows):
        for j in range(ncols):
            v = pivot.values[i, j]
            if not np.isnan(v):
                txt = f"{v:.2f}"
                if sig is not None and sig.values[i, j]:
                    txt += "*"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(value_col, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def forest_plot(
    df: pd.DataFrame,
    out_path: Path | None = None,
    *,
    label_col: str = "label",
    center_col: str = "fold_change_median",
    lo_col: str = "ci_low",
    hi_col: str = "ci_high",
    reference: float = 1.0,
    log_scale: bool = True,
    color: str = "#d62728",
    title: str | None = None,
    xlabel: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Forest plot — one row per measurement, center ± [lo, hi]."""
    df = df.copy()
    df = df.sort_values(center_col).reset_index(drop=True)
    y = np.arange(len(df))
    err = np.array([df[center_col] - df[lo_col], df[hi_col] - df[center_col]])
    figsize = figsize or (8, max(2, 0.4 * len(df)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, df[center_col], xerr=err, color=color, alpha=0.65,
            edgecolor="black", capsize=4)
    if reference is not None:
        ax.axvline(reference, color="black", lw=0.7)
    if log_scale:
        ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(df[label_col].tolist(), fontsize=9)
    ax.set_xlabel(xlabel or f"{center_col} (95% CI)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def condition_strip(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    hue_col: str | None = None,
    marker_col: str | None = None,
    ax=None,
    palette: dict | None = None,
    markers: dict | None = None,
    show_median: bool = True,
):
    """Strip plot with optional hue and per-point marker differentiation.

    `marker_col` is handy to distinguish e.g. "original" vs "new" batches
    in the same condition strip (▲/○) without duplicating the x ticks.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, df[x_col].nunique() * 0.8), 4))
    rng = np.random.default_rng(0)
    x_levels = sorted(df[x_col].dropna().unique())
    hue_levels = sorted(df[hue_col].dropna().unique()) if hue_col else [None]
    marker_levels = sorted(df[marker_col].dropna().unique()) if marker_col else [None]
    palette = palette or {h: c for h, c in zip(
        hue_levels, plt.cm.Set2(np.linspace(0, 1, max(1, len(hue_levels))))
    )}
    markers = markers or {m: ("^" if i % 2 else "o") for i, m in enumerate(marker_levels)}
    n_hue = max(1, len(hue_levels))
    for xi, xv in enumerate(x_levels):
        for hi, hv in enumerate(hue_levels):
            for mv in marker_levels:
                sel = df[x_col] == xv
                if hue_col:
                    sel &= df[hue_col] == hv
                if marker_col:
                    sel &= df[marker_col] == mv
                vals = df.loc[sel, y_col].dropna().values
                if vals.size == 0:
                    continue
                offset = (hi - (n_hue - 1) / 2) * 0.25
                x = xi + offset + rng.uniform(-0.05, 0.05, vals.size)
                ax.scatter(x, vals, color=palette.get(hv, "k"),
                           marker=markers.get(mv, "o"),
                           s=55, alpha=0.8, edgecolors="black", linewidths=0.4)
                if show_median:
                    ax.plot([xi + offset - 0.13, xi + offset + 0.13],
                            [np.median(vals)] * 2,
                            color=palette.get(hv, "k"), linewidth=3)
    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels(x_levels)
    ax.set_ylabel(_label_for(y_col))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if hue_col and len(hue_levels) > 1:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=palette[h], markersize=8, label=str(h))
                   for h in hue_levels]
        ax.legend(handles=handles, loc="best", frameon=False)
    return ax


def modality_panel(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    modality_col: str = "modality",
    group_col: str = "condition",
    out_path: Path | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Compare a set of metrics across staining modalities.

    Produces a grid with rows = modalities, cols = metrics. Each cell is
    a condition strip with median markers.
    """
    modalities = sorted(df[modality_col].dropna().unique())
    nrows, ncols = len(modalities), len(metrics)
    figsize = figsize or (ncols * 3.5, nrows * 3.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, mod in enumerate(modalities):
        sub = df[df[modality_col] == mod]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            if metric not in sub.columns:
                ax.set_visible(False)
                continue
            condition_strip(sub, group_col, metric, ax=ax)
            if j == 0:
                ax.set_ylabel(f"{mod}\n{_label_for(metric)}",
                              fontsize=9, fontweight="bold")
            else:
                ax.set_ylabel(_label_for(metric), fontsize=9)
            if i < nrows - 1:
                ax.set_xticklabels([])
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig
