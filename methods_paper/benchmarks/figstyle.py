"""Shared Nature-level figure style for the fluorostats methods paper.

One visual system for every panel: Okabe-Ito colourblind-safe palette, fluorostats
always blue, vector PDF + 300-dpi PNG export, bold lowercase panel labels, scale
bars, greyscale-surviving markers/linestyles. Import from figure scripts:

    from figstyle import (OKABE, TOOL, apply_style, save, panel, scalebar,
                          caption, boot_ci, MAIN, EXT)
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- Okabe-Ito palette (colourblind-safe) ----
OKABE = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#7F7F7F", "lgrey": "#B9BEC6",
}
# fixed tool identities (colour + marker + linestyle) so series survive greyscale
TOOL = {
    "fluorostats": dict(c=OKABE["blue"],       m="o", ls="-"),
    "StarDist":    dict(c=OKABE["orange"],     m="s", ls="--"),
    "Cellpose":    dict(c=OKABE["vermillion"], m="^", ls="-."),
    "Omnipose":    dict(c=OKABE["purple"],     m="D", ls=":"),
    "DL":          dict(c=OKABE["black"],      m="*", ls="--"),
}

MAIN = Path(__file__).resolve().parent / "figures" / "main"
EXT = Path(__file__).resolve().parent / "figures" / "extended"
MAIN.mkdir(parents=True, exist_ok=True)
EXT.mkdir(parents=True, exist_ok=True)


def apply_style():
    """Set global rcParams for a clean Nature-style sans-serif look, white bg."""
    fam = "Helvetica"
    if not any(f.name == fam for f in font_manager.fontManager.ttflist):
        fam = "Arial"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [fam, "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in vector output
        "svg.fonttype": "none",
    })


def tool_style(name: str) -> dict:
    for k, v in TOOL.items():
        if k.lower() in str(name).lower():
            return v
    return dict(c=OKABE["grey"], m="o", ls="-")


def color_for(name: str) -> str:
    return tool_style(name)["c"]


def panel(ax, label, dx=-0.02, dy=1.0, fs=11):
    """Bold lowercase panel label (a, b, c...) at top-left of an axis."""
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=fs,
            fontweight="bold", va="bottom", ha="right")


# ---- image-panel helpers (shared so every figure renders images identically) ----
def imnorm(a, lo=1, hi=99.5):
    """Percentile-normalise an intensity image to [0,1] for display."""
    a = np.asarray(a, float)
    p0, p1 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p0) / (p1 - p0 + 1e-9), 0, 1)


def image_axes(ax, spine="#999999", lw=0.5):
    """Strip ticks and apply the standard light-grey frame to an image axis."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(spine); sp.set_linewidth(lw)


def outline(ax, labels, rgb, width=2):
    """Overlay thick instance/mask outlines (RGBA) on an image axis. rgb in 0-1."""
    from skimage.segmentation import find_boundaries
    from scipy import ndimage as _ndi
    b = find_boundaries(np.asarray(labels) > 0, mode="outer")
    if width > 1:
        b = _ndi.binary_dilation(b, iterations=width - 1)
    ov = np.zeros((*b.shape, 4)); ov[b] = (*rgb, 1.0)
    ax.imshow(ov, interpolation="nearest")


def composite2ch(live, dead, live_gain=1.0, dead_gain=1.0):
    """Green(live)/magenta(dead) 2-channel composite (colourblind-safe pairing)."""
    g = imnorm(live) * live_gain; r = imnorm(dead) * dead_gain
    return np.clip(np.dstack([r + 0.1 * g, g, r]), 0, 1)


def scalebar(ax, length_px, label, loc="lower right", color="white", pad=0.06):
    """Draw a scale bar on an image axis. length_px in data (pixel) units."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    W = abs(x1 - x0); H = abs(y0 - y1)
    xr = min(x0, x1) + (1 - pad) * W - length_px if "right" in loc else min(x0, x1) + pad * W
    yb = max(y0, y1) - pad * H if "lower" in loc else min(y0, y1) + pad * H
    ax.plot([xr, xr + length_px], [yb, yb], color=color, lw=2.4, solid_capstyle="butt")
    ax.text(xr + length_px / 2, yb - 0.02 * H, label, color=color, ha="center",
            va="bottom", fontsize=6, fontweight="bold")


def save(fig, name, folder=None, tight=True):
    """Export vector PDF + 300-dpi PNG to figures/main (default) or a given folder."""
    folder = folder or MAIN
    if tight:
        fig.tight_layout()
    stem = Path(folder) / name
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", f"{stem}.pdf / .png")


def caption(name, text, folder=None):
    folder = folder or MAIN
    (Path(folder) / f"{name}.txt").write_text(text.strip() + "\n")


def ranked_barh(ax, labels, means, los=None, his=None, *, colors=None,
                highlight="fluorostats", ascending=True, fmt=None, label_fs=6.3):
    """Ranked horizontal bars (worst->best bottom->top) with optional CI whiskers.
    fluorostats is auto-highlighted blue, others grey unless `colors` given.
    Returns the plotted order (list of labels)."""
    labels = list(labels); means = list(means)
    idx = sorted(range(len(means)), key=lambda i: means[i], reverse=not ascending)
    order = [labels[i] for i in idx]
    for row, i in enumerate(idx):
        c = (colors[i] if colors else (OKABE["blue"] if highlight and highlight in str(labels[i]).lower()
             else color_for(labels[i]) if color_for(labels[i]) != OKABE["grey"] else OKABE["grey"]))
        ax.barh(row, means[i], color=c, edgecolor="black", lw=0.4, height=0.7, zorder=2)
        if los is not None and his is not None:
            ax.plot([los[i], his[i]], [row, row], color="black", lw=0.9, zorder=3)
            for xb in (los[i], his[i]):
                ax.plot([xb, xb], [row-0.14, row+0.14], color="black", lw=0.9, zorder=3)
        if fmt:
            ax.text(means[i], row, fmt(means[i]), va="center", ha="left", fontsize=5.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=label_fs)
    return order


def bland_altman(ax, a, b, *, color=None, loa_k=1.96, label="", annotate=True):
    """Bland-Altman agreement panel. Draws points, bias line, +/- limits.
    Returns (bias, loa). a,b are paired arrays (difference = a - b)."""
    color = color or OKABE["blue"]
    a = np.asarray(a, float); b = np.asarray(b, float)
    mean_, diff = (a + b) / 2, a - b
    bias = float(diff.mean()); loa = float(loa_k * diff.std(ddof=1))
    ax.scatter(mean_, diff, s=20, color=color, edgecolor="none", alpha=0.75, zorder=3)
    ax.axhline(bias, color=OKABE["vermillion"], lw=1.1, zorder=2)
    ax.axhline(bias + loa, color="#888", ls="--", lw=0.8); ax.axhline(bias - loa, color="#888", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    if annotate:
        ax.text(0.02, 0.96, f"bias {bias:+.3f}\n±{loa:.3f} (95% LoA){label}",
                transform=ax.transAxes, va="top", fontsize=5.6, color="#333")
    return bias, loa


def identity(ax, color="black", lw=0.7):
    """Draw a y=x identity line spanning the current axis limits."""
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], ls="--", color=color, lw=lw, zorder=1)


def boot_ci(x, reps=10000, seed=0, ci=95):
    """Bootstrap mean + CI of a 1D array."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (reps, len(x)))
    means = x[idx].mean(1)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return float(x.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))
