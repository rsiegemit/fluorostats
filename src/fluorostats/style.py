"""The house figure theme and palette: a compatibility layer over `pubfig`.

`fluorostats.pubfig` is the one figure style module; this module keeps the names
earlier code imports from here::

    from fluorostats.style import apply_style, PALETTE
    apply_style()            # pubfig.apply_style("house")
    ax.set_facecolor(PALETTE["background"])

The palette is intentionally small and bioscience-friendly:

    PALETTE["primary"]     muted slate blue   — baseline / control
    PALETTE["accent"]      coral red          — treatment
    PALETTE["highlight"]   warm amber         — flagged metrics
    PALETTE["muted"]       soft grey          — n.s. / supporting bars
    PALETTE["ink"]         deep blue-grey     — body text, spines
    PALETTE["paper"]       off-white          — figure background

`MATERIAL_COLORS` is a default mapping for the GelMA-vs-Hybrid project
that other code can override.
"""

from __future__ import annotations

from . import pubfig
from .pubfig import DARK_PALETTE, MATERIAL_COLORS, PALETTE, darken, lighten, material_color


def apply_style(font_scale: float = 1.0) -> None:
    """Configure matplotlib for the house figure look: ``pubfig.apply_style("house")``.

    Idempotent — safe to call multiple times. ``font_scale`` multiplies
    every font size (useful for talks vs print).
    """
    pubfig.apply_style("house", font_scale=font_scale)


__all__ = [
    "apply_style",
    "PALETTE",
    "MATERIAL_COLORS",
    "DARK_PALETTE",
    "material_color",
    "darken",
    "lighten",
]
