"""Tests for fluorostats.style: the deprecated layer over `pubfig` (STY-4)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from fluorostats import pubfig, style


def test_apply_style_is_the_house_theme_and_warns():
    with pytest.warns(DeprecationWarning, match="use pubfig.apply_style"):
        style.apply_style(font_scale=1.5)
    via_style = dict(plt.rcParams)
    plt.rcdefaults()
    pubfig.apply_style("house", font_scale=1.5)
    assert dict(plt.rcParams) == via_style
    assert plt.rcParams["font.size"] == 16.5 and plt.rcParams["axes.grid"] is True
    plt.rcdefaults()


@pytest.mark.parametrize(
    "name", ["PALETTE", "MATERIAL_COLORS", "DARK_PALETTE", "material_color", "darken", "lighten"]
)
def test_palette_names_resolve_to_pubfig_with_a_warning(name):
    with pytest.warns(DeprecationWarning, match=f"style.{name} is deprecated; use"):
        assert getattr(style, name) is getattr(pubfig, name)
