"""Figures for the depth-penetration batch driver (`depth_batch`).

Depth-vs-intensity curves (group mean ± SEM over faint individual stacks, with
the group λ overlay), per-window AUC dot/bar plots, and the multi-window
retention summary. Each writes ``<out_base>.png`` (300 dpi) and ``.pdf``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .depth import window_label, window_title
from .style import PALETTE, lighten, material_color


def group_color(name: str, override: str | None) -> str:
    """A group's plot colour: the manifest override, else the house palette."""
    return override or material_color(name)


def plot_depth_curves(rows: list[dict], aggregates: dict[str, dict | None],
                      colors: dict[str, str], ykey: str, ylabel: str, title: str,
                      out_base: Path, auc_window: tuple[float, float] | None = None) -> None:
    """One line per group (mean ± SEM band) + faint individual stacks."""
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    if auc_window is not None:
        ax.axvspan(auc_window[0], auc_window[1], color=PALETTE["highlight"],
                   alpha=0.08, lw=0, zorder=0)
        ax.axvline(auc_window[1], color=PALETTE["muted"], lw=0.8, ls=":", zorder=0)
    for g, agg in aggregates.items():
        c = colors[g]
        # individual stacks
        for r in rows:
            if r["group"] != g:
                continue
            res = r["result"]
            faint = r["role"] != "primary"
            ax.plot(res["depth_um"], res[ykey], color=c,
                    lw=0.8, alpha=0.35,
                    ls="--" if faint else "-", zorder=1)
        if agg is None:
            continue
        mean = agg["sub_mean"] if ykey == "bg_subtracted" else agg["norm_mean"]
        sem = agg["sub_sem"] if ykey == "bg_subtracted" else agg["norm_sem"]
        ax.fill_between(agg["depth_um"], mean - sem, mean + sem,
                        color=lighten(c, 0.25), alpha=0.35, lw=0, zorder=2)
        lam = agg.get("lambda_mean_um", float("nan"))
        lab = f"{g} (n={agg['n']})"
        if ykey == "bg_subtracted" and np.isfinite(lam):
            # Overlay the single-exponential with the group λ (fit on absolute
            # profile). I0 = near-surface group mean so the curve is anchored.
            lam_sem = agg.get("lambda_sem_um", 0.0)
            r2 = agg.get("r_squared_mean", float("nan"))
            n_ok = agg.get("n_lambda_ok", 0)
            i0 = float(np.mean(mean[:3]))
            ax.plot(agg["depth_um"], i0 * np.exp(-agg["depth_um"] / lam),
                    color=c, lw=1.1, ls=(0, (4, 2)), alpha=0.9, zorder=4)
            # Show how many stacks yielded an in-range λ, so a single-stack or
            # partly-degenerate group λ is never read as a confident group value.
            lab += f"\n  λ={lam:.0f}±{lam_sem:.0f} µm, R²={r2:.2f} ({n_ok}/{agg['n']} in range)"
        ax.plot(agg["depth_um"], mean, color=c, lw=2.2, zorder=3, label=lab)
    ax.set_xlabel("Depth (µm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.margins(x=0)
    _save(fig, out_base)


def plot_auc(rows: list[dict], colors: dict[str, str], window: tuple, out_base: Path,
             use_normalized: bool = True) -> None:
    """Dot + mean/SEM bar of AUC per group for one window (primary stacks)."""
    metric = "normalized" if use_normalized else "absolute"
    lab = window_label(window)
    groups = list(colors.keys())
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    xpos = {g: i for i, g in enumerate(groups)}
    for g in groups:
        vals = [r["result"]["auc"][lab][metric] for r in rows
                if r["group"] == g and r["role"] == "primary"]
        if not vals:
            continue
        c = colors[g]
        x = xpos[g]
        m = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.bar(x, m, width=0.55, color=lighten(c, 0.28), edgecolor=c, lw=1.4, zorder=1)
        if sem:
            ax.errorbar(x, m, yerr=sem, color=c, capsize=4, lw=1.4, zorder=2)
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, color=c,
                   edgecolor="white", s=42, lw=0.8, zorder=3)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(groups)
    unit = "normalised" if use_normalized else "absolute"
    ax.set_ylabel(f"AUC {window_title(window)} ({unit})")
    ax.set_title(f"Signal retained over {window_title(window)}")
    ax.margins(x=0.25)
    _save(fig, out_base)


def plot_retention_multiwindow(rows: list[dict], colors: dict[str, str], windows: list[tuple],
                               out_base: Path,
                               title: str = "Signal retained vs depth window") -> None:
    """Grouped bars: mean retained fraction (AUC_norm / window width) per window.

    Dividing normalised AUC by window width gives the average fraction of
    surface signal retained across that depth window (0-1), so windows of
    different sizes sit on one comparable axis and a gap between groups that
    widens with depth is visible at a glance.
    """
    groups = list(colors.keys())
    labels = [window_label(w) for w in windows]
    titles = [window_title(w) for w in windows]
    nW = len(windows)
    x = np.arange(nW)
    width = 0.8 / max(len(groups), 1)
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * nW, 3.8))
    for gi, g in enumerate(groups):
        c = colors[g]
        means, sems, ptsx, ptsy = [], [], [], []
        for wi, lab in enumerate(labels):
            fr = [r["result"]["auc"][lab]["normalized"] / r["result"]["auc"][lab]["width_um"]
                  for r in rows if r["group"] == g and r["role"] == "primary"]
            xc = x[wi] + (gi - (len(groups) - 1) / 2) * width
            means.append(np.mean(fr))
            sems.append(np.std(fr, ddof=1) / np.sqrt(len(fr)) if len(fr) > 1 else 0.0)
            ptsx += list(np.full(len(fr), xc) + np.linspace(-0.06, 0.06, len(fr)))
            ptsy += list(fr)
        xcs = x + (gi - (len(groups) - 1) / 2) * width
        ax.bar(xcs, means, width=width, color=lighten(c, 0.28), edgecolor=c,
               lw=1.3, label=g, zorder=1)
        ax.errorbar(xcs, means, yerr=sems, fmt="none", ecolor=c, capsize=3, lw=1.2, zorder=2)
        ax.scatter(ptsx, ptsy, color=c, edgecolor="white", s=30, lw=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_ylabel("Mean retained fraction (surface = 1)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    _save(fig, out_base)


def _save(fig, out_base: Path) -> None:
    # Append, not with_suffix: a fractional window label ("0.5_100um") has a dot.
    fig.tight_layout()
    fig.savefig(out_base.parent / f"{out_base.name}.png", dpi=300)
    fig.savefig(out_base.parent / f"{out_base.name}.pdf")
    plt.close(fig)
