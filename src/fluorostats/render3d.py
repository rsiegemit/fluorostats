"""3D mask visualisation: isosurface meshes and voxel clouds.

Produces publication-style reconstructions on a physical-micrometre
grid, suitable for side-by-side material/condition comparisons.

Both renderers downsample the mask first (configurable per axis) to
keep mesh sizes manageable; pass ``downsample=(1, 1, 1)`` for full
resolution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage.measure import marching_cubes


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def block_reduce_max(mask: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    """Max-pool a 3D mask by integer factors per axis.

    Trailing voxels that don't fill a block are dropped (matches the
    behaviour of `skimage.measure.block_reduce` with `func=np.max` but
    avoids the float upcast).
    """
    fz, fy, fx = factors
    nz, ny, nx = mask.shape
    nz2, ny2, nx2 = nz // fz, ny // fy, nx // fx
    if nz2 == 0 or ny2 == 0 or nx2 == 0:
        return np.zeros((max(1, nz2), max(1, ny2), max(1, nx2)), dtype=mask.dtype)
    m = mask[: nz2 * fz, : ny2 * fy, : nx2 * fx]
    return m.reshape(nz2, fz, ny2, fy, nx2, fx).max(axis=(1, 3, 5))


# ---------------------------------------------------------------------------
# Isosurface mesh
# ---------------------------------------------------------------------------

def render_isosurface(
    mask: np.ndarray,
    voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ax=None,
    color: str = "#d62728",
    alpha: float = 0.9,
    downsample: tuple[int, int, int] = (1, 4, 4),
    scalebar_um: float | None = 100.0,
    elev: float = 22.0,
    azim: float = -58.0,
    grid: bool = True,
    title: str | None = None,
):
    """Marching-cubes isosurface render on a physical-micrometre grid.

    Returns the matplotlib ``Axes3D`` so the caller can compose multiple
    renders into a panel. Pass ``ax=None`` to create a new figure.
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    ds = block_reduce_max(mask.astype(np.uint8), downsample)
    if ds.sum() < 100:
        ax.text2D(0.5, 0.5, "(insufficient signal)",
                  ha="center", va="center", transform=ax.transAxes)
        if title:
            ax.set_title(title, fontsize=11)
        return ax
    padded = np.pad(ds, 1, mode="constant", constant_values=0)
    spacing = (
        voxel_size_um[0] * downsample[0],
        voxel_size_um[1] * downsample[1],
        voxel_size_um[2] * downsample[2],
    )
    verts, faces, _, _ = marching_cubes(padded.astype(np.float32),
                                         level=0.5, spacing=spacing)
    verts_xyz = verts[:, [2, 1, 0]]
    mesh = Poly3DCollection(verts_xyz[faces], alpha=alpha, facecolor=color,
                            edgecolor="none", linewidth=0)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    nz, ny, nx = ds.shape
    box = (nx * spacing[2], ny * spacing[1], nz * spacing[0])
    ax.set_xlim(0, box[0]); ax.set_ylim(0, box[1]); ax.set_zlim(0, box[2])
    ax.set_box_aspect(box)
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.grid(grid, color="#cccccc", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=7, length=2)
    ax.set_xlabel("µm", fontsize=8)
    ax.set_ylabel("µm", fontsize=8)
    ax.set_zlabel("µm", fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    if scalebar_um:
        ax.plot([10, 10 + scalebar_um], [10, 10], [-2, -2],
                color="black", linewidth=2.5, zorder=10)
        ax.text(10 + scalebar_um / 2, 10, -8, f"{int(scalebar_um)} µm",
                color="black", ha="center", fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
    return ax


# ---------------------------------------------------------------------------
# Voxel cloud
# ---------------------------------------------------------------------------

def render_voxel_cloud(
    mask: np.ndarray,
    voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ax=None,
    color: str = "#d62728",
    alpha: float = 0.4,
    downsample: tuple[int, int, int] = (1, 8, 8),
    elev: float = 22.0,
    azim: float = -58.0,
    title: str | None = None,
):
    """Voxel-cloud render — chunky but clear "is there material here?"."""
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    ds = block_reduce_max(mask.astype(bool), downsample)
    if ds.sum() == 0:
        ax.text2D(0.5, 0.5, "(no signal)", ha="center", va="center",
                  transform=ax.transAxes)
        if title:
            ax.set_title(title, fontsize=11)
        return ax
    # voxels() expects shape (X, Y, Z) per axis aspect; we adopt it.
    ax.voxels(ds.transpose(2, 1, 0), facecolors=color, edgecolor=None,
              alpha=alpha)
    nz, ny, nx = ds.shape
    spacing = (
        voxel_size_um[0] * downsample[0],
        voxel_size_um[1] * downsample[1],
        voxel_size_um[2] * downsample[2],
    )
    box = (nx * spacing[2], ny * spacing[1], nz * spacing[0])
    ax.set_box_aspect((nx, ny, nz))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if title:
        ax.set_title(title, fontsize=11)
    return ax


# ---------------------------------------------------------------------------
# 2D MIP overlay (useful inside grid figures)
# ---------------------------------------------------------------------------

def mip_overlay(
    intensity: np.ndarray,
    mask: np.ndarray | None = None,
    ax=None,
    z_axis: int = 0,
    overlay_color: tuple[int, int, int] = (220, 60, 220),
    overlay_alpha: float = 0.6,
):
    """Grey-scale maximum-intensity projection with optional magenta mask.

    Returns the axis. Convenient cell renderer for a slide-3-style grid
    of representative stacks (one cell per condition × region × day).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    mip = intensity.max(axis=z_axis)
    mip_norm = mip / max(1, mip.max()) * 255
    rgb = np.stack([mip_norm, mip_norm, mip_norm], axis=-1).astype(np.uint8)
    if mask is not None:
        mip_mask = mask.any(axis=z_axis)
        overlay = np.array(overlay_color, dtype=float)
        rgb[mip_mask] = (
            rgb[mip_mask] * (1 - overlay_alpha) + overlay * overlay_alpha
        ).astype(np.uint8)
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


# ---------------------------------------------------------------------------
# Save helper for one-stop rendering
# ---------------------------------------------------------------------------

def save_isosurface(
    mask: np.ndarray,
    out_path: str | Path,
    voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    **kwargs,
) -> None:
    """Render a single isosurface to disk."""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    render_isosurface(mask, voxel_size_um=voxel_size_um, ax=ax, **kwargs)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "block_reduce_max",
    "render_isosurface",
    "render_voxel_cloud",
    "mip_overlay",
    "save_isosurface",
]
