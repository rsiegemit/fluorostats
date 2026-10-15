"""Figure kit: the one figure style of fluorostats, for every figure it draws.

Two rcParams themes chosen by name (:func:`apply_style`): ``"compact"`` — a
5-7 pt type scale on white for multi-panel figures, with the Okabe-Ito
colourblind-safe palette — and ``"house"`` — the 11 pt look with a light grid
and the :data:`PALETTE` colours that the CLI, depth and render figures use
(`fluorostats.style` keeps its names as a thin layer over this). Plus bold
lowercase panel letters, image-panel helpers (percentile normalisation,
outlines, fills, scale bars), dot/lollipop and Bland-Altman panels,
:func:`save_figure` (the one writer every figure goes through) and :func:`save`,
which writes a vector PDF plus a 600-dpi PNG at the figure's exact authored size.

Dataset-specific settings (text width, data roots, tool colour identities)
belong to the scripts that use this kit, not here. Usage::

    from fluorostats import pubfig as F
    F.apply_style()
    fig = plt.figure(figsize=(5.15, 3.0))
    ...
    F.panel(ax, "a", "ground-truth recovery")
    F.save(fig, "fig1", out_dir)
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.rcsetup import cycler
from numpy.typing import ArrayLike

from . import agreement

# ---- Okabe-Ito palette (colourblind-safe) ----
OKABE = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999", "lgrey": "#C4C9D0",
}

# One font scale for every compact figure (points): 5-7 pt for body text, with 8 pt
# bold panel letters. Reference these, never hardcode sizes.
FS = {"title": 7, "label": 7, "tick": 6, "legend": 6, "annot": 6, "small": 5, "panel": 8}

# The house palette (the "house" theme and the CLI / depth / render figures).
PALETTE = {
    "primary":   "#3F6AB3",  # cool slate blue
    "accent":    "#E25C5C",  # warm coral red
    "highlight": "#E8A946",  # soft amber
    "muted":     "#9AA4B0",  # cool grey
    "ink":       "#1F2937",  # deep blue-grey for text
    "paper":     "#FCFCFA",  # off-white background
    "grid":      "#E5E7EB",  # very light grey gridlines
    "background": "#FCFCFA",
    "panel":     "#FFFFFF",
}

MATERIAL_COLORS = {
    "GelMA": PALETTE["primary"],
    "Hybrid": PALETTE["accent"],
    "Control": PALETTE["muted"],
}

# Dark render palette (presentation slides)
DARK_PALETTE = {
    "background": "#000000",
    "mesh":       "#F5C518",   # rich yellow
    "grid":       "#3A3A3A",
    "ink":        "#FFFFFF",
    "scalebar":   "#FFFFFF",
}

# Full text width in inches (372 pt = 130.74 mm). A figure authored at the width
# it is placed at is not rescaled, so the FS point sizes print at their size.
TEXT_WIDTH_IN = 5.147


def _compact_rc(font_scale: float) -> dict:
    """The "compact" theme: small sans-serif type on white, despined, no grid.

    Font is DejaVu Sans: one bundled family with full coverage of the glyphs used
    on figures (arrows, checkmarks, ±, µ, Greek, minus), so no text renders as a
    missing-glyph box whatever fonts the host machine has.
    """
    fam = "DejaVu Sans"
    fs = {k: v * font_scale for k, v in FS.items()}
    return {
        # dpi: preview at 200, export raster fallback at 600 (safe for small insets)
        "figure.dpi": 200, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, no Type-3, text stays selectable
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": [fam, "Arial", "Helvetica"],
        # a real U+2212 minus on negative tick labels, matching body text
        "axes.unicode_minus": True,
        # 5-7 pt window: ticks 6, axis/panel titles 7, legend/annot 6
        "font.size": fs["tick"],
        "axes.titlesize": fs["title"], "axes.labelsize": fs["label"],
        "xtick.labelsize": fs["tick"], "ytick.labelsize": fs["tick"], "legend.fontsize": fs["legend"],
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
    }


# The "house" theme's settings other than font sizes.
_HOUSE_RC = {
    # Figure
    "figure.facecolor": PALETTE["paper"], "figure.edgecolor": PALETTE["paper"],
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "savefig.facecolor": PALETTE["paper"], "savefig.edgecolor": "none",
    # Axes
    "axes.facecolor": PALETTE["panel"], "axes.edgecolor": PALETTE["ink"],
    "axes.labelcolor": PALETTE["ink"], "axes.titlecolor": PALETTE["ink"],
    "axes.titleweight": "semibold", "axes.titlepad": 10, "axes.labelpad": 6,
    "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=[PALETTE["primary"], PALETTE["accent"],
                                     PALETTE["highlight"], PALETTE["muted"],
                                     "#6BAA75", "#9A6FB0"]),
    # Grid
    "grid.color": PALETTE["grid"], "grid.linewidth": 0.7, "grid.linestyle": "-",
    # Ticks
    "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"],
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    # Typography (sizes come from the font scale)
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.titleweight": "semibold",
    # Legend
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.6,
    "legend.borderpad": 0.4,
    # Lines / markers
    "lines.linewidth": 1.8, "lines.markersize": 6, "lines.markeredgewidth": 0.6,
    "patch.linewidth": 0.6, "patch.edgecolor": PALETTE["ink"],
    # Image
    "image.cmap": "viridis",
}


def _house_rc(font_scale: float) -> dict:
    """The "house" theme: 11 pt type (times ``font_scale``), PALETTE colours, y grid."""
    base = 11.0 * font_scale
    return {**_HOUSE_RC,
            "font.size": base, "axes.titlesize": base * 1.15, "axes.labelsize": base * 1.0,
            "xtick.labelsize": base * 0.85, "ytick.labelsize": base * 0.85,
            "legend.fontsize": base * 0.9, "figure.titlesize": base * 1.4}


# Figure themes by name: each maps a font scale to the rcParams it sets.
THEMES: dict[str, Callable[[float], dict]] = {"compact": _compact_rc, "house": _house_rc}


def apply_style(theme: str = "compact", font_scale: float = 1.0) -> None:
    """Set the global rcParams of a figure theme (see :data:`THEMES`).

    ``font_scale`` multiplies every font size (e.g. talks vs print). Idempotent —
    safe to call multiple times.
    """
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {tuple(THEMES)}, got {theme!r}")
    plt.rcParams.update(THEMES[theme](font_scale))


def material_color(name: str, fallback: str | None = None) -> str:
    """Look up the canonical color for a condition/material label."""
    if name in MATERIAL_COLORS:
        return MATERIAL_COLORS[name]
    for key, color in MATERIAL_COLORS.items():
        if key.lower() in str(name).lower():
            return color
    return fallback or PALETTE["muted"]


def darken(color: str, amount: float = 0.2) -> str:
    """Return a slightly darker version of a hex color."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((max(0, r - amount), max(0, g - amount), max(0, b - amount)))


def lighten(color: str, amount: float = 0.2) -> str:
    """Return a slightly lighter version of a hex color."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((min(1, r + amount), min(1, g + amount), min(1, b + amount)))


def panel(ax: plt.Axes, label: str, title: str | None = None, dx: float | None = None,
          dy: float | None = None, pad: tuple[float, float] = (-16, 5),
          xanchor: float | None = None) -> None:
    """Panel label: a bold lowercase letter at a fixed offset (points) from the
    axis top-left, plus an optional short roman descriptor beside it.

    Offset points keep labels identical across figures and never expand the saved
    bbox. ``dx``/``dy`` are accepted for backward compatibility and ignored.

    ``xanchor``: None (default) anchors the letter to the axes frame. A float
    places the letter x at that FIGURE-fraction position (y still from the axes'
    top edge and ``pad[1]``), so row-leading letters across panels with different
    tick-label widths share one page x instead of drifting with each axes frame.
    """
    if xanchor is None:
        ax.annotate(label, xy=(0, 1), xycoords="axes fraction",
                    xytext=pad, textcoords="offset points",
                    fontsize=FS["panel"], fontweight="bold", va="baseline", ha="left")
        if title:
            ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                        xytext=(pad[0] + 9, pad[1]), textcoords="offset points",
                        fontsize=FS["label"], fontweight="regular", color="#222",
                        va="baseline", ha="left")
        return
    fig = ax.get_figure()
    assert fig is not None  # an Axes is always created inside a figure
    trans = mtransforms.blended_transform_factory(fig.transFigure, ax.transAxes)
    ax.annotate(label, xy=(xanchor, 1), xycoords=trans,
                xytext=(0, pad[1]), textcoords="offset points",
                fontsize=FS["panel"], fontweight="bold", va="baseline", ha="left")
    if title:
        ax.annotate(title, xy=(xanchor, 1), xycoords=trans,
                    xytext=(9, pad[1]), textcoords="offset points",
                    fontsize=FS["label"], fontweight="regular", color="#222",
                    va="baseline", ha="left")


# ---- image-panel helpers (shared so every figure renders images identically) ----

def imnorm(a: ArrayLike, lo: float = 1, hi: float = 99.5) -> np.ndarray:
    """Percentile-normalise an intensity image to [0, 1] for display."""
    a = np.asarray(a, float)
    p0, p1 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p0) / (p1 - p0 + 1e-9), 0, 1)


def image_axes(ax: plt.Axes, spine: str = "#999999", lw: float = 0.5) -> None:
    """Strip ticks and apply the standard light-grey frame to an image axis."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(spine)
        sp.set_linewidth(lw)


def outline(ax: plt.Axes, labels: np.ndarray, rgb: Sequence[float], width: int = 2) -> None:
    """Overlay thick instance/mask outlines (RGBA) on an image axis. rgb in 0-1."""
    from scipy import ndimage as ndi
    from skimage.segmentation import find_boundaries

    b = find_boundaries(np.asarray(labels) > 0, mode="outer")
    if width > 1:
        b = ndi.binary_dilation(b, iterations=width - 1)
    ov = np.zeros((*b.shape, 4))
    ov[b] = (*rgb, 1.0)
    ax.imshow(ov, interpolation="nearest")


def composite2ch(live: np.ndarray, dead: np.ndarray, live_gain: float = 1.0,
                 dead_gain: float = 1.0) -> np.ndarray:
    """Green (live) / magenta (dead) two-channel RGB composite, colourblind-safe."""
    g = imnorm(live) * live_gain
    r = imnorm(dead) * dead_gain
    return np.clip(np.dstack([r + 0.1 * g, g, r]), 0, 1)


def fill_mask(ax: plt.Axes, mask: np.ndarray, rgb: Sequence[float], alpha: float = 0.85) -> None:
    """Overlay a flat semi-transparent RGBA fill wherever mask > 0 (rgb in 0-1)."""
    ov = np.zeros((*mask.shape, 4))
    ov[mask > 0] = (*rgb, alpha)
    ax.imshow(ov, interpolation="nearest")


def scalebar(ax: plt.Axes, length_px: float, label: str, loc: str = "lower right",
             color: str = "white", pad: float = 0.07, center_label: bool = False) -> None:
    """Draw a scale bar on an image axis; ``length_px`` in data (pixel) units.

    The label is anchored to the bar end nearest the image edge (right-aligned for
    a right ``loc``, left for a left one), so a label wider than the bar can never
    overflow the axes. ``center_label`` centres it over the bar instead; use it only
    when the bar is long enough for a centred label to stay inside the axes.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    W = abs(x1 - x0)
    H = abs(y0 - y1)
    right = "right" in loc
    xr = min(x0, x1) + (1 - pad) * W - length_px if right else min(x0, x1) + pad * W
    yb = max(y0, y1) - pad * H if "lower" in loc else min(y0, y1) + pad * H
    ax.plot([xr, xr + length_px], [yb, yb], color=color, lw=2.4, solid_capstyle="butt")
    if center_label:
        tx, tha = xr + length_px / 2, "center"
    else:
        tx, tha = (xr + length_px if right else xr), ("right" if right else "left")
    ax.text(tx, yb - 0.02 * H, label, color=color, ha=tha, va="bottom",
            fontsize=6, fontweight="bold")


# ---- export ----

def save_figure(fig: plt.Figure, outputs: dict, bbox_inches: str | None = None) -> None:
    """Write ``fig`` to each path of ``outputs`` (``{path: dpi}``; dpi None = the
    ``savefig.dpi`` rcParam), creating parent folders, then close it.

    The one writer behind every figure the library saves; layout (tight_layout or a
    constrained engine) is the caller's, so each keeps its own.
    """
    for path, dpi in outputs.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def save(fig: plt.Figure, name: str, folder: str | Path, tight: bool = True, pad: float = 0.4) -> None:
    """Export an editable vector PDF + 600-dpi PNG at the figure's exact size.

    Never crops with ``bbox_inches='tight'``, which silently rescales the physical
    width and breaks authoring at column width. For ``layout='constrained'``
    figures pass ``tight=False``: the constrained engine already spaces everything.
    If an element clips, fix the layout rather than cropping.
    """
    engine = fig.get_layout_engine()
    is_constrained = engine is not None and type(engine).__name__.lower().startswith("constrained")
    if tight and not is_constrained:
        fig.tight_layout(pad=pad)
    stem = Path(folder) / name
    # vector PDF at the exact figsize (uniform column width) + 600-dpi raster fallback
    save_figure(fig, {f"{stem}.pdf": None, f"{stem}.png": 600})
    print("saved", f"{stem}.pdf / .png (%.2fx%.2f in)" % tuple(fig.get_size_inches()))


def caption(name: str, text: str, folder: str | Path) -> None:
    """Write a figure's caption next to it as ``<folder>/<name>.txt``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.txt").write_text(text.strip() + "\n")


# ---- data panels ----

def lollipop(ax: plt.Axes, labels: Sequence[str], values: Sequence[float], *,
             colors: Sequence | None = None, floor: float | None = None,
             los: Sequence[float] | None = None, his: Sequence[float] | None = None,
             ascending: bool = False, fmt: str = "{:.3f}", label_fs: float = 6,
             value_fs: float = 5.6, highlight: str | None = "fluorostats") -> list[str]:
    """Horizontal dot/lollipop plot with a zoomed x-axis for near-max values
    (encode magnitude by position, not bar area). Use instead of bars when every
    value hugs the axis maximum (correlation / accuracy panels).

    ``floor`` is the left x-limit (a justified zoom floor, e.g. 0.90; default just
    below the minimum). Draws a stem from floor to value plus a dot, optional CI
    whiskers and value labels. Labels containing ``highlight`` are drawn blue,
    others grey. Returns the plotted order of labels.
    """
    labels = list(labels)
    values = list(values)
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
        # offset the value label a few points off the dot so it never strikes
        # through the marker; offset points are axis-scale robust
        ax.annotate(fmt.format(values[i]), (values[i], row), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=value_fs, color="#333")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=label_fs)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-0.6, len(order) - 0.4)
    return order


def bland_altman(ax: plt.Axes, a: ArrayLike, b: ArrayLike, *, color: str | None = None,
                 loa_k: float = 1.96, label: str = "", annotate: bool = True) -> tuple[float, float]:
    """Bland-Altman agreement panel: points, bias line and +/- limits of agreement.

    Numbers come from `fluorostats.agreement.bland_altman` (difference = a - b,
    NaN pairs dropped); ``loa_k`` scales the SD of the differences. Returns
    ``(bias, loa)``.
    """
    color = color or OKABE["blue"]
    ba = agreement.bland_altman(a, b)
    bias = ba["bias"]
    loa = float(loa_k * ba["sd_diff"])
    ax.scatter(ba["mean_vals"], ba["diff_vals"], s=20, color=color, edgecolor="none",
               alpha=0.75, zorder=3)
    ax.axhline(bias, color=OKABE["vermillion"], lw=1.1, zorder=2)
    ax.axhline(bias + loa, color="#888", ls="--", lw=0.8)
    ax.axhline(bias - loa, color="#888", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    if annotate:
        ax.text(0.02, 0.96, f"bias {bias:+.3f}\n±{loa:.3f} (95% LoA){label}",
                transform=ax.transAxes, va="top", fontsize=5.6, color="#333")
    return bias, loa


def round_sig(x: float, n: int = 2) -> float:
    """Round ``x`` to ``n`` significant figures (382.6 -> 380, 14.8 -> 15), so an
    annotation can be derived from data yet match the rounding of reported text."""
    from math import floor, log10

    if not x:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))
