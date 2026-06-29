"""Tests for fluorostats.render3d and metrics_3d FOV helpers."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fluorostats.metrics_3d import (
    fov_volume_mm3,
    density_per_mm3,
    normalise_skeleton_metrics,
)
from fluorostats.render3d import (
    block_reduce_max,
    render_isosurface,
    mip_overlay,
    save_isosurface,
    live_dead_mip,
    mip_grid,
    depth_coded_mip,
    layer_split_mip,
)


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
    plt.close(fig)


def test_layer_split_mip_runs():
    vol = np.random.RandomState(0).rand(10, 16, 16).astype(np.float32)
    fig, (a, b) = plt.subplots(1, 2)
    layer_split_mip(vol, ax_top=a, ax_bot=b, split=0.5)
    plt.close(fig)
