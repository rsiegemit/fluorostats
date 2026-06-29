"""Tests for fluorostats.style."""

from __future__ import annotations

import matplotlib as mpl

from fluorostats.style import (
    apply_style, PALETTE, MATERIAL_COLORS, DARK_PALETTE,
    material_color, darken, lighten,
)


def test_apply_style_updates_rcparams():
    apply_style()
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["legend.frameon"] is False


def test_apply_style_font_scale_doubles_size():
    apply_style(font_scale=1.0)
    base = mpl.rcParams["font.size"]
    apply_style(font_scale=2.0)
    assert mpl.rcParams["font.size"] == base * 2


def test_palette_keys():
    for k in ["primary", "accent", "ink", "paper", "grid"]:
        assert k in PALETTE
        assert PALETTE[k].startswith("#")


def test_material_color_known_and_fallback():
    assert material_color("GelMA") == MATERIAL_COLORS["GelMA"]
    assert material_color("Hybrid") == MATERIAL_COLORS["Hybrid"]
    assert material_color("UnknownThing") == PALETTE["muted"]


def test_darken_lighten_are_inverses_ish():
    base = "#888888"
    assert darken(base, 0.2) != base
    assert lighten(base, 0.2) != base


def test_dark_palette_has_mesh_and_background():
    assert "mesh" in DARK_PALETTE
    assert "background" in DARK_PALETTE
