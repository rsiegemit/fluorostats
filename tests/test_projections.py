"""Tests for fluorostats.projections (2D MIP panels)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
import pytest

from fluorostats.projections import (
    mip_overlay,
    live_dead_mip,
    mip_grid,
    depth_coded_mip,
    layer_split_mip,
)
from fluorostats.pubfig import PALETTE


def test_mip_overlay_produces_rgb_image():
    intensity = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32) * 100
    mask = intensity[0] > 50
    mask3 = np.broadcast_to(mask, intensity.shape)
    fig, ax = plt.subplots()
    out = mip_overlay(intensity, mask=mask3, ax=ax)
    assert out is ax
    plt.close(fig)


def test_live_dead_mip_returns_axis():
    live = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32) * 100
    dead = np.random.RandomState(1).rand(4, 16, 16).astype(np.float32) * 50
    fig, ax = plt.subplots()
    out = live_dead_mip(live, dead, ax=ax)
    assert out is ax
    plt.close(fig)


def test_live_dead_mip_handles_zero_dead():
    live = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32)
    fig, ax = plt.subplots()
    live_dead_mip(live, dead=None, ax=ax)
    rgb = ax.images[0].get_array()
    assert rgb.shape == (16, 16, 3)
    # live only: every pixel is a shade of the live colour (40, 220, 60) on black
    np.testing.assert_allclose(rgb[..., 0] / (40 / 255), rgb[..., 1] / (220 / 255), atol=1e-6)
    assert rgb[..., 1].max() == pytest.approx(220 / 255)
    plt.close(fig)


def test_mip_grid_writes_cells(tmp_path):
    live = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32)
    cells = {
        ("D1", "top"): {"live": live},
        ("D1", "middle"): {"live": live * 0.5},
        ("D7", "top"): {"live": live * 0.8},
        ("D7", "middle"): {"live": live * 1.2},
    }
    fig = mip_grid(cells, rows=["D1", "D7"], cols=["top", "middle"], title="test grid")
    out = tmp_path / "grid.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 1000


def test_depth_coded_mip_runs():
    vol = np.random.RandomState(0).rand(8, 16, 16).astype(np.float32)
    fig, ax = plt.subplots()
    depth_coded_mip(vol, ax=ax)
    rgb = ax.images[0].get_array()
    assert rgb.shape == (16, 16, 3) and rgb.min() >= 0 and rgb.max() <= 1
    assert len(ax.get_xticks()) == 0 and len(ax.get_yticks()) == 0
    plt.close(fig)


def test_layer_split_mip_runs():
    vol = np.random.RandomState(0).rand(10, 16, 16).astype(np.float32)
    fig, (a, b, ref_top, ref_bot) = plt.subplots(1, 4)
    layer_split_mip(vol, ax_top=a, ax_bot=b, split=0.5)
    live_dead_mip(vol[:5], ax=ref_top)  # the top half is slices 0-4
    live_dead_mip(vol[5:], ax=ref_bot)
    np.testing.assert_array_equal(a.images[0].get_array(), ref_top.images[0].get_array())
    np.testing.assert_array_equal(b.images[0].get_array(), ref_bot.images[0].get_array())
    plt.close(fig)


def test_mip_overlay_without_mask():
    intensity = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32) * 100
    fig, ax = plt.subplots()
    out = mip_overlay(intensity, mask=None, ax=ax)
    assert out is ax
    plt.close(fig)


def test_live_dead_mip_all_zero_live_no_border():
    """All-zero live hits the m.max()==0 early return in _to_norm; border=None."""
    live = np.zeros((4, 16, 16), dtype=np.float32)
    fig, ax = plt.subplots()
    default_edge = ax.spines["left"].get_edgecolor()
    live_dead_mip(live, dead=None, ax=ax, border=None)
    assert not ax.images[0].get_array().any()  # black: nothing to show
    assert ax.spines["left"].get_edgecolor() == default_edge
    plt.close(fig)


def test_depth_coded_mip_z_axis_last():
    vol = np.random.RandomState(0).rand(16, 16, 8).astype(np.float32)
    fig, ax = plt.subplots()
    depth_coded_mip(vol, ax=ax, z_axis=2)
    assert ax.images[0].get_array().shape == (16, 16, 3)  # projected along the last axis
    plt.close(fig)


def test_mip_grid_missing_cell_placeholder(tmp_path):
    live = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32)
    cells = {("D1", "top"): {"live": live}}  # ("D1","bot") missing -> placeholder
    fig = mip_grid(
        cells, rows=["D1"], cols=["top", "bot"], row_labels=["Day1"], col_labels=["Top", "Bottom"]
    )
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert texts == ["Top", "Bottom", "Day1", "—"]
    rendered, placeholder = fig.axes[3], fig.axes[4]
    assert len(rendered.images) == 1 and len(placeholder.images) == 0
    assert to_hex(placeholder.get_facecolor()) == "#f0f0f0"
    plt.close(fig)


def test_layer_split_mip_only_top_axis():
    vol = np.random.RandomState(0).rand(10, 16, 16).astype(np.float32)
    fig, ax = plt.subplots()
    top, bot = layer_split_mip(vol, ax_top=ax, ax_bot=None, split=0.5)
    assert top is ax and bot is None
    plt.close(fig)


def test_mip_overlay_creates_own_axes_when_none():
    """ax=None → fig, ax = plt.subplots() fires (line 285)."""
    intensity = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32) * 100
    ax = mip_overlay(intensity)  # no ax passed
    assert tuple(ax.figure.get_size_inches()) == (4.0, 4.0)
    assert ax.images[0].get_array().shape == (16, 16, 3)
    plt.close(ax.figure)


def test_live_dead_mip_creates_own_axes_when_none():
    """ax=None → fig, ax = plt.subplots() fires (line 325)."""
    live = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32)
    ax = live_dead_mip(live)  # no ax passed
    assert tuple(ax.figure.get_size_inches()) == (4.0, 4.0)
    assert len(ax.images) == 1
    assert to_hex(ax.spines["left"].get_edgecolor()) == PALETTE["ink"].lower()  # default border
    plt.close(ax.figure)


def test_depth_coded_mip_creates_own_axes_when_none():
    """ax=None → fig, ax = plt.subplots() fires (line 452)."""
    vol = np.random.RandomState(0).rand(8, 16, 16).astype(np.float32)
    ax = depth_coded_mip(vol)  # no ax passed
    assert tuple(ax.figure.get_size_inches()) == (4.0, 4.0)
    assert ax.images[0].get_array().shape == (16, 16, 3)
    plt.close(ax.figure)


def test_mip_overlay_composites_with_the_qc_rule():
    """One overlay rule: mip_overlay shows qc.overlay_rgb of the MIP (min-max stretch)."""
    from fluorostats import qc

    intensity = np.random.RandomState(1).rand(4, 16, 16) * 100 + 50  # min well above 0
    mask = intensity > 120
    fig, ax = plt.subplots()
    mip_overlay(intensity, mask=mask, ax=ax)
    expected = qc.overlay_rgb(intensity.max(0), mask.any(0), (220, 60, 220), 0.6)
    np.testing.assert_array_equal(ax.images[0].get_array(), expected)
    plt.close(fig)
