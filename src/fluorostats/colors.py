"""Colours: the Okabe-Ito and house palettes and per-group colour choice.

Split from :mod:`fluorostats.pubfig`, which re-exports every name here.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ---- Okabe-Ito palette (colourblind-safe) ----
OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
    "lgrey": "#C4C9D0",
}

# The house palette (the "house" theme and the CLI / depth / render figures).
PALETTE = {
    "primary": "#3F6AB3",  # cool slate blue
    "accent": "#E25C5C",  # warm coral red
    "highlight": "#E8A946",  # soft amber
    "muted": "#9AA4B0",  # cool grey
    "ink": "#1F2937",  # deep blue-grey for text
    "paper": "#FCFCFA",  # off-white background
    "grid": "#E5E7EB",  # very light grey gridlines
    "background": "#FCFCFA",
    "panel": "#FFFFFF",
}

# Colours kept for these exact condition names (existing figures rely on them).
MATERIAL_COLORS = {
    "GelMA": PALETTE["primary"],
    "Hybrid": PALETTE["accent"],
    "Control": PALETTE["muted"],
}

# Colours for every other group, in order: the house colours, then Okabe-Ito
# (colourblind-safe). Past the end, evenly spaced hues.
GROUP_CYCLE = (
    PALETTE["primary"],
    PALETTE["accent"],
    PALETTE["highlight"],
    "#009E73",
    "#56B4E9",
    "#CC79A7",
    "#D55E00",
    "#0072B2",
    "#F0E442",
    "#000000",
)

# Dark render palette (presentation slides)
DARK_PALETTE = {
    "background": "#000000",
    "mesh": "#F5C518",  # rich yellow
    "grid": "#3A3A3A",
    "ink": "#FFFFFF",
    "scalebar": "#FFFFFF",
}


def material_color(name: str, fallback: str | None = None) -> str:
    """The ``MATERIAL_COLORS`` colour of exactly this name, else ``fallback`` (or grey).

    For a set of groups use :func:`group_colors`, which keeps them distinct.
    """
    return MATERIAL_COLORS.get(name, fallback or PALETTE["muted"])


def group_colors(names, overrides: dict | None = None) -> dict[str, str]:
    """One distinct colour per group name, in order.

    A non-empty ``overrides[name]`` wins; an exact ``MATERIAL_COLORS`` name keeps its
    colour; every other group takes the next ``GROUP_CYCLE`` colour no group uses
    yet, and evenly spaced hues once the cycle is used up.
    """
    overrides = overrides or {}
    fixed = {n: overrides.get(n) or MATERIAL_COLORS.get(n) for n in names}
    used = {mcolors.to_hex(c) for c in fixed.values() if c}
    spare = [c for c in GROUP_CYCLE if mcolors.to_hex(c) not in used]
    need = sum(1 for c in fixed.values() if not c)
    if need > len(spare):
        hues = plt.get_cmap("hsv")(np.linspace(0, 1, need - len(spare), endpoint=False))
        spare += [mcolors.to_hex(h) for h in hues]
    it = iter(spare)
    return {n: c or next(it) for n, c in fixed.items()}


def darken(color: str, amount: float = 0.2) -> str:
    """Return a slightly darker version of a hex color."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((max(0, r - amount), max(0, g - amount), max(0, b - amount)))


def lighten(color: str, amount: float = 0.2) -> str:
    """Return a slightly lighter version of a hex color."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((min(1, r + amount), min(1, g + amount), min(1, b + amount)))
