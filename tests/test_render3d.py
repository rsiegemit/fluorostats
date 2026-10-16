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
    save_isosurface,
)
from fluorostats.pubfig import DARK_PALETTE


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
    out = render_isosurface(
        mask, voxel_size_um=(2.0, 1.0, 1.0), ax=ax, downsample=(1, 4, 4), scalebar_um=10.0
    )
    assert out is ax
    plt.close(fig)


def test_render_isosurface_handles_empty_mask():
    mask = np.zeros((4, 16, 16), dtype=bool)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")
    out = render_isosurface(mask, ax=ax, title="empty")
    assert out is ax  # does not raise
    plt.close(fig)


def test_save_isosurface_writes_file(tmp_path):
    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[2:6, 8:24, 8:24] = True
    out_path = tmp_path / "iso.png"
    save_isosurface(
        mask, out_path, voxel_size_um=(2.0, 1.0, 1.0), downsample=(1, 4, 4), scalebar_um=20.0
    )
    assert out_path.exists() and out_path.stat().st_size > 1000


def test_render_isosurface_dark_style():
    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[2:6, 8:24, 8:24] = True
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    render_isosurface(
        mask, voxel_size_um=(2.0, 1.0, 1.0), ax=ax, style="dark", smooth_iter=2, color="#F5C518"
    )
    # under the signal guard: a dark empty panel with the message in the dark ink
    assert to_hex(ax.get_facecolor()) == DARK_PALETTE["background"]
    assert [t.get_text() for t in ax.texts] == ["(insufficient signal)"]
    assert to_hex(ax.texts[0].get_color()) == DARK_PALETTE["ink"].lower()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full marching-cubes body — mask large enough to clear the signal guard
# ---------------------------------------------------------------------------


def test_render_isosurface_light_shaded_runs_mesh_body():
    """Light style, shade=True — exercises the Lambertian shading path."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    out = render_isosurface(
        _solid_mask(),
        voxel_size_um=(2.0, 1.0, 1.0),
        ax=ax,
        downsample=(1, 4, 4),
        shade=True,
        scalebar_um=20.0,
        title="light shaded",
    )
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
    render_isosurface(
        _solid_mask(), ax=ax, downsample=(1, 4, 4), style="dark", smooth_iter=2, shade=False
    )
    assert any(isinstance(c, Poly3DCollection) for c in ax.collections)
    assert to_hex(ax.get_facecolor()) == DARK_PALETTE["background"]
    assert len(ax.get_xticks()) == len(ax.get_yticks()) == len(ax.get_zticks()) == 0
    assert ax.get_xlabel() == ""  # bare axes: no µm labels
    assert len(ax.lines) == 12 + 1  # bounding box edges + scale bar
    assert [t.get_text() for t in ax.texts] == ["100 µm"]
    assert to_hex(ax.texts[0].get_color()) == DARK_PALETTE["scalebar"].lower()
    plt.close(fig)


def test_render_isosurface_creates_own_axes_when_none():
    """ax=None branch builds its own 3D figure."""
    ax = render_isosurface(_solid_mask(), downsample=(1, 4, 4))
    assert ax.name == "3d"
    assert tuple(ax.figure.get_size_inches()) == (6.0, 5.0)
    assert any(isinstance(c, Poly3DCollection) for c in ax.collections)
    assert ax.get_xlabel() == ax.get_ylabel() == ax.get_zlabel() == "µm"  # light style
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Voxel cloud
# ---------------------------------------------------------------------------


def test_render_voxel_cloud_returns_axis():
    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[2:6, 8:24, 8:24] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    out = render_voxel_cloud(
        mask, voxel_size_um=(2.0, 1.0, 1.0), ax=ax, downsample=(1, 4, 4), title="cloud"
    )
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
    assert len(ax.collections) > 0  # the voxel faces
    assert len(ax.get_xticks()) == len(ax.get_yticks()) == len(ax.get_zticks()) == 0
    plt.close(ax.figure)


def test_block_reduce_max_degenerate_factor_returns_min_shape():
    mask = np.ones((2, 2, 2), dtype=bool)
    ds = block_reduce_max(mask, (4, 4, 4))  # factors exceed size
    assert ds.shape == (1, 1, 1)
    assert ds.sum() == 0


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
