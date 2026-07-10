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
# fluorostats is ALWAYS blue (#0072B2) and only fluorostats. DL tools: StarDist
# orange, Cellpose vermillion, Omnipose purple. Classical / SciPy / scikit-image
# baselines: neutral grey #999999 (locked by FIGURE_OVERHAUL_BRIEFS §0).
OKABE = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999", "lgrey": "#C4C9D0",
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

# Nature Methods column widths (inches). Use these as the figure WIDTH.
COL1 = 3.46      # single column  (88 mm)
COL15 = 4.72     # 1.5 column     (120 mm)
COL2 = 7.20      # double column  (183 mm)

# One font scale for every figure (points), Nature's 5-7 pt window for body text
# with 8 pt bold panel letters (the one documented exception). Reference these,
# never hardcode sizes.
FS = {"title": 7, "label": 7, "tick": 6, "legend": 6, "annot": 6, "small": 5, "panel": 8}


def apply_style():
    """Set global rcParams for a clean Nature-style sans-serif look, white bg.

    Font is DejaVu Sans: a single bundled sans-serif family with full coverage of
    the glyphs used on-figure (arrows, checkmarks, ±, µ, Greek, minus), so no panel
    text renders as a missing-glyph box regardless of the host machine's fonts."""
    fam = "DejaVu Sans"
    plt.rcParams.update({
        # dpi: preview at 200, export raster fallback at 600 (Nature-safe for insets)
        "figure.dpi": 200, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, no Type-3, text stays selectable
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": [fam, "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        # Nature 5-7 pt window: ticks 6, axis/panel titles 7, legend/annot 6
        "font.size": FS["tick"],
        "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"],
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.linewidth": 0.6, "axes.edgecolor": "#333333", "axes.titlepad": 4.0,
        "axes.titlelocation": "left", "axes.titleweight": "regular",
        "axes.spines.top": False, "axes.spines.right": False,   # despine everywhere
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False, "lines.solid_capstyle": "round",
    })


def tool_style(name: str) -> dict:
    for k, v in TOOL.items():
        if k.lower() in str(name).lower():
            return v
    return dict(c=OKABE["grey"], m="o", ls="-")


def color_for(name: str) -> str:
    return tool_style(name)["c"]


def panel(ax, label, title=None, dx=None, dy=None, pad=(-16, 5)):
    """Nature-style panel label: bold lowercase letter at a fixed offset (points)
    from the axis top-left, plus an optional short roman descriptor beside it.

    Replaces chart-style titles. Offset-points keep labels identical across figures
    and never expand the saved bbox. dx/dy accepted for back-compat but ignored."""
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction",
                xytext=pad, textcoords="offset points",
                fontsize=FS["panel"], fontweight="bold", va="baseline", ha="left")
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                    xytext=(pad[0] + 9, pad[1]), textcoords="offset points",
                    fontsize=FS["label"], fontweight="regular", color="#222",
                    va="baseline", ha="left")


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


def save(fig, name, folder=None, tight=True, pad=0.4):
    """Export editable vector PDF + 600-dpi PNG at the figure's EXACT authored size.

    Never crops with bbox_inches='tight' (that silently rescales the physical width
    and breaks the "author at column width" contract, §0A rule 7). For figures built
    with layout='constrained', pass tight=False — constrained-layout already spaces
    everything; calling tight_layout on top of it fights the engine. If a stray
    element clips, fix the layout, don't crop."""
    folder = folder or MAIN
    engine = fig.get_layout_engine()
    is_constrained = engine is not None and type(engine).__name__.lower().startswith("constrained")
    if tight and not is_constrained:
        fig.tight_layout(pad=pad)
    stem = Path(folder) / name
    fig.savefig(f"{stem}.pdf")                 # vector, exact figsize -> uniform column width
    fig.savefig(f"{stem}.png", dpi=600)        # 600-dpi raster fallback / preview
    plt.close(fig)
    print("saved", f"{stem}.pdf / .png (%.2fx%.2f in)" % tuple(fig.get_size_inches()))


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


def lollipop(ax, labels, values, *, colors=None, floor=None, los=None, his=None,
             ascending=False, fmt="{:.3f}", label_fs=6, value_fs=5.6, highlight="fluorostats"):
    """Horizontal dot/lollipop plot with a ZOOMED x-axis for near-max values
    (Cleveland: encode magnitude by position, not bar area). Use this instead of
    bars whenever every value hugs the axis max (correlation / accuracy panels, §0B.3).

    floor: left x-limit (justified zoom floor, e.g. 0.90). Defaults just below min.
    Draws a thin stem from floor to the value + a dot; optional CI whiskers; value
    labels in clear space. fluorostats auto-highlighted blue, others grey."""
    labels = list(labels); values = list(values)
    idx = sorted(range(len(values)), key=lambda i: values[i], reverse=not ascending)
    order = [labels[i] for i in idx]
    lo_x = floor if floor is not None else max(0.0, min(values) - 0.03)
    hi_x = max(values) + (max(values) - lo_x) * 0.14
    for row, i in enumerate(idx):
        c = (colors[i] if colors else (OKABE["blue"]
             if highlight and highlight in str(labels[i]).lower() else OKABE["grey"]))
        ax.plot([lo_x, values[i]], [row, row], color=c, lw=1.0, alpha=0.55, zorder=2,
                solid_capstyle="butt")
        ax.plot(values[i], row, "o", color=c, ms=5, mec="white", mew=0.5, zorder=4)
        if los is not None and his is not None:
            ax.plot([los[i], his[i]], [row, row], color=c, lw=1.4, alpha=0.9, zorder=3)
        ax.text(values[i] + (hi_x - lo_x) * 0.015, row, fmt.format(values[i]),
                va="center", ha="left", fontsize=value_fs, color="#333")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=label_fs)
    ax.set_xlim(lo_x, hi_x); ax.set_ylim(-0.6, len(order) - 0.4)
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
