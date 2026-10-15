"""Tests for fluorostats.render3d and metrics_3d FOV helpers."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from fluorostats.metrics_3d import (
    fov_volume_mm3,
    density_per_mm3,
    normalise_skeleton_metrics,
)
from fluorostats.render3d import (
    block_reduce_max,
    render_isosurface,
    render_voxel_cloud,
    mip_overlay,
    save_isosurface,
    live_dead_mip,
    mip_grid,
    depth_coded_mip,
    layer_split_mip,
)
from fluorostats.pubfig import DARK_PALETTE, PALETTE


def _solid_mask() -> np.ndarray:
    """A mask large enough to clear render_isosurface's signal guard.

    downsample=(1,4,4) on this 8x64x64 solid block leaves ~8*12*12 blocks,
    well above the ds.sum() < 100 threshold, so the marching-cubes body runs.
    """
    mask = np.zeros((8, 64, 64), dtype=bool)
    mask[1:7, 8:56, 8:56] = True
    return mask


# ---------------------------------------------------------------------------
# FOV helpers
# ---------------------------------------------------------------------------

def test_fov_volume_mm3_matches_manual_calc():
    # 10 x 100 x 100 voxels at 2 x 1 x 1 µm = 2e5 µm³ = 2e-4 mm³
    v = fov_volume_mm3((10, 100, 100), (2.0, 1.0, 1.0))
    assert v == pytest.approx(2e-4, rel=1e-9)


def test_density_per_mm3_simple():
    v = fov_volume_mm3((10, 100, 100), (1.0, 1.0, 1.0))
    assert density_per_mm3(50, (10, 100, 100), (1.0, 1.0, 1.0)) == pytest.approx(50 / v)


def test_normalise_skeleton_metrics_adds_density_columns():
    base = {"total_length_um": 1000.0, "n_branches": 50, "n_junctions": 10}
    out = normalise_skeleton_metrics(base, (10, 100, 100), (1.0, 1.0, 1.0))
    assert "length_density_um_per_mm3" in out
    assert "junction_density_per_mm3" in out
    assert "fov_volume_mm3" in out
    # Original dict unchanged
    assert "length_density_um_per_mm3" not in base


# ---------------------------------------------------------------------------
# block_reduce_max
# ---------------------------------------------------------------------------

def test_block_reduce_max_shape_and_value():
    mask = np.zeros((4, 8, 8), dtype=bool)
    mask[0, 0, 0] = True
    ds = block_reduce_max(mask, (2, 2, 2))
    assert ds.shape == (2, 4, 4)
    assert bool(ds[0, 0, 0])
    assert ds.sum() == 1


# ---------------------------------------------------------------------------
# Renderers — assert they produce axes without crashing
# ---------------------------------------------------------------------------

def test_render_isosurface_returns_axes():
    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[2:6, 8:24, 8:24] = True
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    out = render_isosurface(mask, voxel_size_um=(2.0, 1.0, 1.0), ax=ax,
                            downsample=(1, 4, 4), scalebar_um=10.0)
    assert out is ax
    plt.close(fig)


def test_render_isosurface_handles_empty_mask():
    mask = np.zeros((4, 16, 16), dtype=bool)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")
    out = render_isosurface(mask, ax=ax, title="empty")
    assert out is ax  # does not raise
    plt.close(fig)


def test_mip_overlay_produces_rgb_image():
    intensity = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32) * 100
    mask = intensity[0] > 50
    mask3 = np.broadcast_to(mask, intensity.shape)
    fig, ax = plt.subplots()
    out = mip_overlay(intensity, mask=mask3, ax=ax)
    assert out is ax
    plt.close(fig)


def test_save_isosurface_writes_file(tmp_path):
    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[2:6, 8:24, 8:24] = True
    out_path = tmp_path / "iso.png"
    save_isosurface(mask, out_path, voxel_size_um=(2.0, 1.0, 1.0),
                    downsample=(1, 4, 4), scalebar_um=20.0)
    assert out_path.exists() and out_path.stat().st_size > 1000


def test_render_isosurface_dark_style():
    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[2:6, 8:24, 8:24] = True
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    render_isosurface(mask, voxel_size_um=(2.0, 1.0, 1.0), ax=ax,
                      style="dark", smooth_iter=2, color="#F5C518")
    # under the signal guard: a dark empty panel with the message in the dark ink
    assert to_hex(ax.get_facecolor()) == DARK_PALETTE["background"]
    assert [t.get_text() for t in ax.texts] == ["(insufficient signal)"]
    assert to_hex(ax.texts[0].get_color()) == DARK_PALETTE["ink"].lower()
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
    fig = mip_grid(cells, rows=["D1", "D7"], cols=["top", "middle"],
                   title="test grid")
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
    live_dead_mip(vol[:5], ax=ref_top)          # the top half is slices 0-4
    live_dead_mip(vol[5:], ax=ref_bot)
    np.testing.assert_array_equal(a.images[0].get_array(), ref_top.images[0].get_array())
    np.testing.assert_array_equal(b.images[0].get_array(), ref_bot.images[0].get_array())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full marching-cubes body — mask large enough to clear the signal guard
# ---------------------------------------------------------------------------

def test_render_isosurface_light_shaded_runs_mesh_body():
    """Light style, shade=True — exercises the Lambertian shading path."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    out = render_isosurface(_solid_mask(), voxel_size_um=(2.0, 1.0, 1.0), ax=ax,
                            downsample=(1, 4, 4), shade=True, scalebar_um=20.0,
                            title="light shaded")
    assert out is ax
    # Mesh body ran: something was drawn into the 3D collection list.
    assert len(ax.collections) > 0
    plt.close(fig)


def test_render_isosurface_flat_fill_no_shade():
    """shade=False takes the flat-fill Poly3DCollection branch."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    render_isosurface(_solid_mask(), ax=ax, downsample=(1, 4, 4), shade=False)
    assert len(ax.collections) > 0
    plt.close(fig)


def test_render_isosurface_dark_smoothed_mesh_body():
    """Dark style + smoothing on a real mesh (draws box edges + scalebar)."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    render_isosurface(_solid_mask(), ax=ax, downsample=(1, 4, 4),
                      style="dark", smooth_iter=2, shade=False)
    assert any(isinstance(c, Poly3DCollection) for c in ax.collections)
    assert to_hex(ax.get_facecolor()) == DARK_PALETTE["background"]
    assert len(ax.get_xticks()) == len(ax.get_yticks()) == len(ax.get_zticks()) == 0
    assert ax.get_xlabel() == ""                          # bare axes: no µm labels
    assert len(ax.lines) == 12 + 1                        # bounding box edges + scale bar
    assert [t.get_text() for t in ax.texts] == ["100 µm"]
    assert to_hex(ax.texts[0].get_color()) == DARK_PALETTE["scalebar"].lower()
    plt.close(fig)


def test_render_isosurface_creates_own_axes_when_none():
    """ax=None branch builds its own 3D figure."""
    ax = render_isosurface(_solid_mask(), downsample=(1, 4, 4))
    assert ax.name == "3d"
    assert tuple(ax.figure.get_size_inches()) == (6.0, 5.0)
    assert any(isinstance(c, Poly3DCollection) for c in ax.collections)
    assert ax.get_xlabel() == ax.get_ylabel() == ax.get_zlabel() == "µm"   # light style
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Voxel cloud
# ---------------------------------------------------------------------------

def test_render_voxel_cloud_returns_axis():
    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[2:6, 8:24, 8:24] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    out = render_voxel_cloud(mask, voxel_size_um=(2.0, 1.0, 1.0), ax=ax,
                             downsample=(1, 4, 4), title="cloud")
    assert out is ax
    plt.close(fig)


def test_render_voxel_cloud_no_signal_guard():
    mask = np.zeros((4, 16, 16), dtype=bool)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    out = render_voxel_cloud(mask, ax=ax, title="empty")
    assert out is ax  # hit the "(no signal)" branch without raising
    plt.close(fig)


def test_render_voxel_cloud_creates_own_axes():
    mask = np.zeros((6, 24, 24), dtype=bool)
    mask[1:5, 4:20, 4:20] = True
    ax = render_voxel_cloud(mask, downsample=(1, 4, 4))
    assert ax.name == "3d"
    assert len(ax.collections) > 0                        # the voxel faces
    assert len(ax.get_xticks()) == len(ax.get_yticks()) == len(ax.get_zticks()) == 0
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# mip_overlay without mask, live_dead border, depth_coded z_axis, grid gaps
# ---------------------------------------------------------------------------

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
    assert not ax.images[0].get_array().any()            # black: nothing to show
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
    fig = mip_grid(cells, rows=["D1"], cols=["top", "bot"],
                   row_labels=["Day1"], col_labels=["Top", "Bottom"])
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


def test_block_reduce_max_degenerate_factor_returns_min_shape():
    mask = np.ones((2, 2, 2), dtype=bool)
    ds = block_reduce_max(mask, (4, 4, 4))  # factors exceed size
    assert ds.shape == (1, 1, 1)
    assert ds.sum() == 0


# ---------------------------------------------------------------------------
# mip_overlay — line 285: ax=None creates its own figure
# ---------------------------------------------------------------------------

def test_mip_overlay_creates_own_axes_when_none():
    """ax=None → fig, ax = plt.subplots() fires (line 285)."""
    intensity = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32) * 100
    ax = mip_overlay(intensity)   # no ax passed
    assert tuple(ax.figure.get_size_inches()) == (4.0, 4.0)
    assert ax.images[0].get_array().shape == (16, 16, 3)
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# live_dead_mip — line 325: ax=None creates its own figure
# ---------------------------------------------------------------------------

def test_live_dead_mip_creates_own_axes_when_none():
    """ax=None → fig, ax = plt.subplots() fires (line 325)."""
    live = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32)
    ax = live_dead_mip(live)   # no ax passed
    assert tuple(ax.figure.get_size_inches()) == (4.0, 4.0)
    assert len(ax.images) == 1
    assert to_hex(ax.spines["left"].get_edgecolor()) == PALETTE["ink"].lower()   # default border
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# depth_coded_mip — line 452: ax=None creates its own figure
# ---------------------------------------------------------------------------

def test_depth_coded_mip_creates_own_axes_when_none():
    """ax=None → fig, ax = plt.subplots() fires (line 452)."""
    vol = np.random.RandomState(0).rand(8, 16, 16).astype(np.float32)
    ax = depth_coded_mip(vol)   # no ax passed
    assert tuple(ax.figure.get_size_inches()) == (4.0, 4.0)
    assert ax.images[0].get_array().shape == (16, 16, 3)
    plt.close(ax.figure)



def test_unknown_style_is_rejected():
    mask = np.zeros((4, 16, 16), bool)
    with pytest.raises(ValueError, match="'light', 'dark'"):
        render_isosurface(mask, style="Dark")
    plt.close("all")


def test_import_does_not_switch_the_matplotlib_backend(monkeypatch):
    """A library module must leave the user's backend alone."""
    import importlib

    from fluorostats import render3d
    calls = []
    monkeypatch.setattr(matplotlib, "use", lambda *a, **k: calls.append(a))
    importlib.reload(render3d)
    assert calls == []


def test_mip_overlay_composites_with_the_qc_rule():
    """One overlay rule: mip_overlay shows qc.overlay_rgb of the MIP (min-max stretch)."""
    from fluorostats import qc
    intensity = np.random.RandomState(1).rand(4, 16, 16) * 100 + 50     # min well above 0
    mask = intensity > 120
    fig, ax = plt.subplots()
    mip_overlay(intensity, mask=mask, ax=ax)
    expected = qc.overlay_rgb(intensity.max(0), mask.any(0), (220, 60, 220), 0.6)
    np.testing.assert_array_equal(ax.images[0].get_array(), expected)
    plt.close(fig)
