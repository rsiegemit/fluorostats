"""3D mask visualisation: isosurface meshes and voxel clouds.

Produces publication-style reconstructions on a physical-micrometre
grid, suitable for side-by-side material/condition comparisons.

Both renderers downsample the mask first (configurable per axis) to
keep mesh sizes manageable; pass ``downsample=(1, 1, 1)`` for full
resolution.

The µm axis labels and scale bars assume a calibrated ``voxel_size_um``
(``(z, y, x)`` in µm, e.g. ``meta["voxel_size_um"]`` from `fluorostats.io`). The
default ``(1, 1, 1)`` means voxel units: the axes then count voxels and a
"100 µm" bar is 100 voxels long.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes

from . import projections
from .deprecation import moved_names
from .pubfig import DARK_PALETTE, PALETTE, save_figure

# The 2D MIP panels live in `projections`; the names still resolve here, where they
# were first defined, for one release with a DeprecationWarning (STY-4).
__getattr__ = moved_names(
    __name__,
    ("mip_overlay", "live_dead_mip", "mip_grid", "depth_coded_mip", "layer_split_mip"),
    projections,
)


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
    ax: Axes3D | None = None,
    color: str = "#d62728",
    alpha: float = 0.95,
    downsample: tuple[int, int, int] = (1, 4, 4),
    scalebar_um: float | None = 100.0,
    elev: float = 22.0,
    azim: float = -58.0,
    grid: bool = True,
    title: str | None = None,
    style: str = "light",
    smooth_iter: int = 0,
    shade: bool = True,
) -> Axes3D:
    """Marching-cubes isosurface render on a physical-micrometre grid.

    Parameters
    ----------
    style : "light" | "dark"
        ``"light"`` keeps the white-on-grey publication look. ``"dark"``
        renders on a black background with bright mesh and faint grey
        grid, for dark presentation slides.
    smooth_iter : int
        Number of Laplacian smoothing passes applied to the vertex
        positions before rendering. 0 = raw marching-cubes (slightly
        faceted); 1–3 = smoother tubes. Each pass averages every vertex
        with the centroid of its face-neighbours.
    shade : bool
        When True, lets matplotlib apply per-face shading based on the
        light source instead of a flat fill — gives the mesh visible
        depth.

    Returns the matplotlib ``Axes3D``.
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    ds = block_reduce_max(mask.astype(np.uint8), downsample)
    theme = _theme(style)
    bg, text_color, scalebar_color = theme["background"], theme["text"], theme["scalebar"]

    if ds.sum() < 100:
        ax.set_facecolor(bg)
        ax.text2D(
            0.5,
            0.5,
            "(insufficient signal)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=text_color,
        )
        if title:
            ax.set_title(title, color=text_color)
        return ax
    spacing = (
        voxel_size_um[0] * downsample[0],
        voxel_size_um[1] * downsample[1],
        voxel_size_um[2] * downsample[2],
    )
    ax.add_collection3d(_isosurface_mesh(ds, spacing, smooth_iter, shade, color, alpha))
    nz, ny, nx = ds.shape
    box = (nx * spacing[2], ny * spacing[1], nz * spacing[0])
    _style_mesh_axes(ax, box, theme, grid)
    ax.view_init(elev=elev, azim=azim)
    if scalebar_um:
        ax.plot(
            [10, 10 + scalebar_um], [10, 10], [0, 0], color=scalebar_color, linewidth=3.0, zorder=10
        )
        ax.text(
            10 + scalebar_um / 2,
            10,
            -box[2] * 0.05,
            f"{int(scalebar_um)} µm",
            color=scalebar_color,
            ha="center",
            fontsize=10,
            fontweight="semibold",
        )
    if title:
        ax.set_title(title, color=text_color)
    return ax


# Isosurface themes by ``style`` name: colours, and whether the axes are bare (no
# ticks or µm labels, bounding box drawn instead) as on dark slides.
_THEMES = {
    "light": {
        "background": "white",
        "grid": "#cccccc",
        "text": PALETTE["ink"],
        "scalebar": PALETTE["ink"],
        "bare_axes": False,
    },
    "dark": {
        "background": DARK_PALETTE["background"],
        "grid": DARK_PALETTE["grid"],
        "text": DARK_PALETTE["ink"],
        "scalebar": DARK_PALETTE["scalebar"],
        "bare_axes": True,
    },
}


def _theme(style: str) -> dict:
    """The :data:`_THEMES` entry for ``style``; ValueError naming the valid styles."""
    if style not in _THEMES:
        raise ValueError(f"style must be one of {tuple(_THEMES)}, got {style!r}")
    return _THEMES[style]


def _isosurface_mesh(ds, spacing, smooth_iter: int, shade: bool, color: str, alpha: float):
    """Marching-cubes mesh of a binary volume as a Poly3DCollection.

    With ``shade``, each face is lit by a Lambertian dot product with a fixed
    light; otherwise it is a flat fill.
    """
    padded = np.pad(ds, 1, mode="constant", constant_values=0)
    verts, faces, _, _ = marching_cubes(padded.astype(np.float32), level=0.5, spacing=spacing)
    if smooth_iter > 0:
        verts = _laplacian_smooth(verts, faces, n_iter=smooth_iter)
    verts_xyz = verts[:, [2, 1, 0]]
    polys = verts_xyz[faces]
    if not shade:
        mesh = Poly3DCollection(polys, alpha=alpha, facecolor=color, edgecolor="none", linewidth=0)
        mesh.set_facecolor(color)
        return mesh
    from matplotlib.colors import to_rgb

    rgb = np.array(to_rgb(color))
    e1 = polys[:, 1] - polys[:, 0]
    e2 = polys[:, 2] - polys[:, 0]
    normals = np.cross(e1, e2)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.where(norm == 0, 1, norm))
    light = np.array(
        [
            np.cos(np.radians(45)) * np.cos(np.radians(315)),
            np.cos(np.radians(45)) * np.sin(np.radians(315)),
            np.sin(np.radians(45)),
        ]
    )
    shading = np.clip(normals @ light, 0.0, 1.0)
    intensities = 0.55 + 0.45 * shading
    face_colors = rgb[None, :] * intensities[:, None]
    face_colors = np.clip(face_colors, 0, 1)
    face_rgba = np.concatenate([face_colors, np.full((face_colors.shape[0], 1), alpha)], axis=1)
    return Poly3DCollection(polys, facecolors=face_rgba, edgecolor="none", linewidth=0)


def _style_mesh_axes(ax, box, theme: dict, grid: bool) -> None:
    """Physical-µm limits, aspect, panes, grid and labels for an isosurface axis."""
    bg, grid_color, text_color = theme["background"], theme["grid"], theme["text"]
    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.set_zlim(0, box[2])
    ax.set_box_aspect(box)
    ax.set_facecolor(bg)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(grid_color)
    ax.grid(grid, color=grid_color, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=7, length=2, colors=text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.zaxis.label.set_color(text_color)
    if theme["bare_axes"]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Draw the bounding-box edges explicitly for the dark grid look
        for edge in _box_edges(box):
            ax.plot(*edge, color=grid_color, linewidth=0.6, alpha=0.7, zorder=0)
    else:
        ax.set_xlabel("µm")
        ax.set_ylabel("µm")
        ax.set_zlabel("µm")


# ---------------------------------------------------------------------------
# Smoothing + box helpers
# ---------------------------------------------------------------------------


def _laplacian_smooth(verts: np.ndarray, faces: np.ndarray, n_iter: int = 1) -> np.ndarray:
    """Move each vertex halfway toward the centroid of its face-neighbours.

    Plain Laplacian smoothing (no Taubin inflation step, so every pass shrinks the
    mesh slightly) — softens the marching-cubes staircase without melting away
    fine branches when ``n_iter`` is small (1–3).
    """
    v = verts.astype(np.float64).copy()
    n_v = v.shape[0]
    for _ in range(n_iter):
        sums = np.zeros_like(v)
        counts = np.zeros(n_v, dtype=np.int64)
        for col_a, col_b in ((0, 1), (1, 2), (2, 0)):
            a = faces[:, col_a]
            b = faces[:, col_b]
            np.add.at(sums, a, v[b])
            np.add.at(counts, a, 1)
            np.add.at(sums, b, v[a])
            np.add.at(counts, b, 1)
        mask = counts > 0
        v_new = v.copy()
        v_new[mask] = 0.5 * v[mask] + 0.5 * (sums[mask] / counts[mask, None])
        v = v_new
    return v


def _box_edges(box: tuple[float, float, float]):
    """12 edges of the bounding box for explicit grid rendering."""
    x, y, z = box
    corners = [
        (0, 0, 0),
        (x, 0, 0),
        (x, y, 0),
        (0, y, 0),
        (0, 0, z),
        (x, 0, z),
        (x, y, z),
        (0, y, z),
    ]
    pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in pairs:
        a, b = corners[i], corners[j]
        yield ([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])


# ---------------------------------------------------------------------------
# Voxel cloud
# ---------------------------------------------------------------------------


def render_voxel_cloud(
    mask: np.ndarray,
    voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ax: Axes3D | None = None,
    color: str = "#d62728",
    alpha: float = 0.4,
    downsample: tuple[int, int, int] = (1, 8, 8),
    elev: float = 22.0,
    azim: float = -58.0,
    title: str | None = None,
) -> Axes3D:
    """Voxel-cloud render — chunky but clear "is there material here?"."""
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    ds = block_reduce_max(mask.astype(bool), downsample)
    if ds.sum() == 0:
        ax.text2D(0.5, 0.5, "(no signal)", ha="center", va="center", transform=ax.transAxes)
        if title:
            ax.set_title(title, fontsize=11)
        return ax
    # matplotlib voxels() indexes as (X, Y, Z), so we transpose from (Z, Y, X).
    ax.voxels(ds.transpose(2, 1, 0), facecolors=color, edgecolor=None, alpha=alpha)
    nz, ny, nx = ds.shape
    # physical box aspect (accounts for anisotropic voxels + per-axis downsample)
    spacing = (
        voxel_size_um[0] * downsample[0],
        voxel_size_um[1] * downsample[1],
        voxel_size_um[2] * downsample[2],
    )
    ax.set_box_aspect((nx * spacing[2], ny * spacing[1], nz * spacing[0]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if title:
        ax.set_title(title, fontsize=11)
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
    fig.tight_layout()
    save_figure(fig, {out_path: 180}, bbox_inches="tight")


__all__ = [
    "block_reduce_max",
    "render_isosurface",
    "render_voxel_cloud",
    "save_isosurface",
]
