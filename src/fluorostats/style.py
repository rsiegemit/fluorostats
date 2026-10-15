"""Deprecated: the house figure theme and palette live in `fluorostats.pubfig`.

Earlier code imported them from here::

    from fluorostats.style import apply_style, PALETTE   # deprecated
    from fluorostats.pubfig import apply_style, PALETTE  # use this
    apply_style("house")

Every name still resolves for one release, with a DeprecationWarning (STY-4).
"""

from __future__ import annotations

import warnings

from . import pubfig
from .deprecation import moved_names


def apply_style(font_scale: float = 1.0) -> None:
    """Deprecated: ``pubfig.apply_style("house", font_scale=font_scale)``."""
    warnings.warn(
        "style.apply_style is deprecated; use pubfig.apply_style('house', font_scale=...)",
        DeprecationWarning,
        stacklevel=2,
    )
    pubfig.apply_style("house", font_scale=font_scale)


__getattr__ = moved_names(
    __name__,
    ("PALETTE", "MATERIAL_COLORS", "DARK_PALETTE", "material_color", "darken", "lighten"),
    pubfig,
)

__all__ = ["apply_style"]
