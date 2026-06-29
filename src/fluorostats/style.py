"""Publication-style matplotlib defaults and palette helpers.

Call :func:`apply_style` at the top of a script (or in the matplotlib
rc file) to opt every figure into the FluoroStats look:

    from fluorostats.style import apply_style, PALETTE
    apply_style()
    ax.set_facecolor(PALETTE["background"])

The palette is intentionally small and bioscience-friendly:

    PALETTE["primary"]     muted slate blue   — GelMA / baseline
    PALETTE["accent"]      coral red          — Hybrid / treatment
    PALETTE["highlight"]   warm amber         — flagged metrics
    PALETTE["muted"]       soft grey          — n.s. / supporting bars
    PALETTE["ink"]         deep blue-grey     — body text, spines
    PALETTE["paper"]       off-white          — figure background

`MATERIAL_COLORS` is a default mapping for the GelMA-vs-Hybrid project
that other code can override.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


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

# Dark/PPTX-style render palette
DARK_PALETTE = {
    "background": "#000000",
    "mesh":       "#F5C518",   # rich yellow, slide-2 style
    "mesh_alt":   "#FF8C42",   # alt warm orange for second material
    "grid":       "#3A3A3A",
    "ink":        "#FFFFFF",
    "scalebar":   "#FFFFFF",
}


def apply_style(font_scale: float = 1.0) -> None:
    """Configure matplotlib for clean, modern publication figures.

    Idempotent — safe to call multiple times. ``font_scale`` multiplies
    every font size (useful for talks vs print).
    """
    base = 11.0 * font_scale
    mpl.rcParams.update({
        # Figure
        "figure.facecolor": PALETTE["paper"],
        "figure.edgecolor": PALETTE["paper"],
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.facecolor": PALETTE["paper"],
        "savefig.edgecolor": "none",
        # Axes
        "axes.facecolor": PALETTE["panel"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
        "axes.labelpad": 6,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=[
            PALETTE["primary"], PALETTE["accent"], PALETTE["highlight"],
            PALETTE["muted"], "#6BAA75", "#9A6FB0",
        ]),
        # Grid
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        # Ticks
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": base,
        "axes.titlesize": base * 1.15,
        "axes.labelsize": base * 1.0,
        "xtick.labelsize": base * 0.85,
        "ytick.labelsize": base * 0.85,
        "legend.fontsize": base * 0.9,
        "figure.titlesize": base * 1.4,
        "figure.titleweight": "semibold",
        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.6,
        "legend.borderpad": 0.4,
        # Lines / markers
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.6,
        "patch.edgecolor": PALETTE["ink"],
        # Image
        "image.cmap": "viridis",
    })


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
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    return mcolors.to_hex(tuple(max(0, c - amount) for c in rgb))


def lighten(color: str, amount: float = 0.2) -> str:
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    return mcolors.to_hex(tuple(min(1, c + amount) for c in rgb))


__all__ = ["apply_style", "PALETTE", "MATERIAL_COLORS", "DARK_PALETTE",
           "material_color", "darken", "lighten"]
