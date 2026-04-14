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
    show_pvalues: bool = True,
) -> None:
    """Bar chart with mean +/- SEM, individual replicate points, and p-values.

    Publication-standard format for n=3-5 replicates.
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
    show_pvalues: bool = True,
) -> None:
    """Multi-panel figure: one bar+SEM subplot per metric.

    Generates a publication-ready composite figure with all metrics.
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

def _add_pvalue_annotations(ax, data, x, conditions):
    """Add p-value brackets between all condition pairs."""
    from scipy.stats import mannwhitneyu

    pairs = list(combinations(range(len(conditions)), 2))
    y_max = ax.get_ylim()[1]
    step = y_max * 0.08
    offset = y_max * 0.05

    for pair_idx, (i, j) in enumerate(pairs):
        if len(data[i]) < 2 or len(data[j]) < 2:
            continue

        try:
            _, p = mannwhitneyu(data[i], data[j], alternative="two-sided")
        except ValueError:
            continue

        stars = _pvalue_stars(p)
        if stars == "ns" and len(pairs) > 3:
            continue  # skip non-significant for busy plots

        y = y_max + offset + pair_idx * step
        ax.plot([x[i], x[i], x[j], x[j]], [y - step * 0.2, y, y, y - step * 0.2],
                color="black", linewidth=0.8)
        ax.text((x[i] + x[j]) / 2, y, stars, ha="center", va="bottom", fontsize=9)

    # Extend y-axis to fit annotations
    new_max = y_max + offset + len(pairs) * step + step
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
