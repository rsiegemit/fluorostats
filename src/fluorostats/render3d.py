"""3D mask visualisation: isosurface meshes and voxel clouds.

Produces publication-style reconstructions on a physical-micrometre
grid, suitable for side-by-side material/condition comparisons.

Both renderers downsample the mask first (configurable per axis) to
keep mesh sizes manageable; pass ``downsample=(1, 1, 1)`` for full
resolution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
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
):
    """Marching-cubes isosurface render on a physical-micrometre grid.

    Parameters
    ----------
    style : "light" | "dark"
        ``"light"`` keeps the white-on-grey publication look. ``"dark"``
        renders on a black background with bright mesh and faint grey
        grid — matches the PPTX slide-2 reference (Matrigel/rCOL style).
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

    if style == "dark":
        from .style import DARK_PALETTE
        bg = DARK_PALETTE["background"]
        grid_color = DARK_PALETTE["grid"]
        text_color = DARK_PALETTE["ink"]
        scalebar_color = DARK_PALETTE["scalebar"]
    else:
        bg = "white"
        grid_color = "#cccccc"
        text_color = "#1F2937"
        scalebar_color = "#1F2937"

    if ds.sum() < 100:
        ax.set_facecolor(bg)
        ax.text2D(0.5, 0.5, "(insufficient signal)",
                  ha="center", va="center", transform=ax.transAxes,
                  color=text_color)
        if title:
            ax.set_title(title, color=text_color)
        return ax
    padded = np.pad(ds, 1, mode="constant", constant_values=0)
    spacing = (
        voxel_size_um[0] * downsample[0],
        voxel_size_um[1] * downsample[1],
        voxel_size_um[2] * downsample[2],
    )
    verts, faces, _, _ = marching_cubes(padded.astype(np.float32),
                                         level=0.5, spacing=spacing)
    if smooth_iter > 0:
        verts = _laplacian_smooth(verts, faces, n_iter=smooth_iter)
    verts_xyz = verts[:, [2, 1, 0]]
    polys = verts_xyz[faces]
    if shade:
        # Per-face shading via Lambertian dot product with a light source.
        from matplotlib.colors import to_rgb
        rgb = np.array(to_rgb(color))
        e1 = polys[:, 1] - polys[:, 0]
        e2 = polys[:, 2] - polys[:, 0]
        normals = np.cross(e1, e2)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, np.where(norm == 0, 1, norm))
        light = np.array([np.cos(np.radians(45)) * np.cos(np.radians(315)),
                          np.cos(np.radians(45)) * np.sin(np.radians(315)),
                          np.sin(np.radians(45))])
        shading = np.clip(normals @ light, 0.0, 1.0)
        intensities = 0.55 + 0.45 * shading
        face_colors = rgb[None, :] * intensities[:, None]
        face_colors = np.clip(face_colors, 0, 1)
        face_rgba = np.concatenate([face_colors,
                                    np.full((face_colors.shape[0], 1), alpha)],
                                   axis=1)
        mesh = Poly3DCollection(polys, facecolors=face_rgba,
                                 edgecolor="none", linewidth=0)
    else:
        mesh = Poly3DCollection(polys, alpha=alpha, facecolor=color,
                                 edgecolor="none", linewidth=0)
        mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    nz, ny, nx = ds.shape
    box = (nx * spacing[2], ny * spacing[1], nz * spacing[0])
    ax.set_xlim(0, box[0]); ax.set_ylim(0, box[1]); ax.set_zlim(0, box[2])
    ax.set_box_aspect(box)
    ax.set_facecolor(bg)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(grid_color)
    ax.grid(grid, color=grid_color, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=7, length=2,
                   colors=text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.zaxis.label.set_color(text_color)
    if style == "dark":
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        # Manually draw box edges for the slide-2 grid look
        for edge in _box_edges(box):
            ax.plot(*edge, color=grid_color, linewidth=0.6, alpha=0.7, zorder=0)
    else:
        ax.set_xlabel("µm"); ax.set_ylabel("µm"); ax.set_zlabel("µm")
    ax.view_init(elev=elev, azim=azim)
    if scalebar_um:
        ax.plot([10, 10 + scalebar_um], [10, 10], [0, 0],
                color=scalebar_color, linewidth=3.0, zorder=10)
        ax.text(10 + scalebar_um / 2, 10, -box[2] * 0.05,
                f"{int(scalebar_um)} µm",
                color=scalebar_color, ha="center", fontsize=10,
                fontweight="semibold")
    if title:
        ax.set_title(title, color=text_color)
    return ax


# ---------------------------------------------------------------------------
# Smoothing + box helpers
# ---------------------------------------------------------------------------

def _laplacian_smooth(verts: np.ndarray, faces: np.ndarray,
                      n_iter: int = 1) -> np.ndarray:
    """Average each vertex toward the centroid of its face-neighbours.

    Cheap Taubin-like smoothing — softens the marching-cubes staircase
    without melting away fine branches when ``n_iter`` is small (1–3).
    """
    v = verts.astype(np.float64).copy()
    n_v = v.shape[0]
    for _ in range(n_iter):
        sums = np.zeros_like(v)
        counts = np.zeros(n_v, dtype=np.int64)
        for col_a, col_b in ((0, 1), (1, 2), (2, 0)):
            a = faces[:, col_a]; b = faces[:, col_b]
            np.add.at(sums, a, v[b]); np.add.at(counts, a, 1)
            np.add.at(sums, b, v[a]); np.add.at(counts, b, 1)
        mask = counts > 0
        v_new = v.copy()
        v_new[mask] = 0.5 * v[mask] + 0.5 * (sums[mask] / counts[mask, None])
        v = v_new
    return v


def _box_edges(box: tuple[float, float, float]):
    """12 edges of the bounding box for explicit grid rendering."""
    x, y, z = box
    corners = [(0, 0, 0), (x, 0, 0), (x, y, 0), (0, y, 0),
               (0, 0, z), (x, 0, z), (x, y, z), (0, y, z)]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in pairs:
        a, b = corners[i], corners[j]
        yield ([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])




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


def live_dead_mip(
    live: np.ndarray,
    dead: np.ndarray | None = None,
    ax=None,
    z_axis: int = 0,
    *,
    live_color: tuple[int, int, int] = (40, 220, 60),
    dead_color: tuple[int, int, int] = (240, 30, 40),
    percentile_clip: tuple[float, float] = (1, 99.5),
    gamma: float = 0.75,
    background: tuple[int, int, int] = (0, 0, 0),
    border: str | None = "#1F2937",
    border_lw: float = 1.0,
):
    """Two-channel maximum-intensity projection (slide-3 style).

    ``live`` is the live-cell channel (typically Alexa-488 / Calcein /
    GFP), ``dead`` is the dead-cell channel (typically PI / Alexa-568).
    Each channel is independently percentile-clipped, gamma-corrected,
    then composited additively into an RGB image on a black background.

    Returns the matplotlib axis. The signature mirrors :func:`mip_overlay`
    so the two helpers can be swapped inside grid layouts.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    lo, hi = percentile_clip
    lc = np.array(live_color, dtype=float) / 255.0
    bg = np.array(background, dtype=float) / 255.0

    def _to_norm(vol: np.ndarray) -> np.ndarray:
        m = vol.max(axis=z_axis).astype(float)
        if m.max() == 0:
            return m
        v_lo, v_hi = np.percentile(m, [lo, hi])
        m = np.clip((m - v_lo) / max(1e-9, v_hi - v_lo), 0, 1)
        return m ** gamma

    live_n = _to_norm(live)
    rgb = (live_n[..., None] * lc[None, None, :] +
           (1 - live_n[..., None]) * bg[None, None, :])

    if dead is not None:
        dc = np.array(dead_color, dtype=float) / 255.0
        dead_n = _to_norm(dead)
        # Additive blend on top of the live composite
        rgb = rgb + dead_n[..., None] * dc[None, None, :]

    rgb = np.clip(rgb, 0, 1)
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    if border:
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(border_lw)
    return ax


def mip_grid(
    cells: dict,
    *,
    rows: list,
    cols: list,
    render_func=live_dead_mip,
    figsize: tuple[float, float] | None = None,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    title: str | None = None,
    cell_kwargs: dict | None = None,
    label_color: str = "#1F2937",
    background: str = "white",
):
    """Render a (row × col) grid of MIPs.

    Parameters
    ----------
    cells : dict
        Mapping ``(row_key, col_key) -> dict``. Each value's dict is
        spread into the cell renderer (so e.g. ``{"live": vol_g}`` for
        a single-channel MIP, or ``{"live": vol_g, "dead": vol_r}``
        for two-channel). Missing keys render a placeholder cell.
    rows, cols : list
        Ordered keys for the grid axes.
    render_func : callable
        Called as ``render_func(ax=ax, **cells[(r, c)])``. Defaults to
        :func:`live_dead_mip`.
    row_labels, col_labels : list of str
        Display labels (one per row/col). Defaults to ``str(key)``.
    cell_kwargs : dict
        Extra kwargs forwarded to every ``render_func`` call.
    """
    nrows, ncols = len(rows), len(cols)
    figsize = figsize or (2.8 * ncols + 1.0, 2.8 * nrows + 0.8)
    cell_kwargs = cell_kwargs or {}

    fig = plt.figure(figsize=figsize, facecolor=background)
    gs = fig.add_gridspec(nrows + 1, ncols + 1,
                          width_ratios=[0.18] + [1] * ncols,
                          height_ratios=[0.18] + [1] * nrows,
                          wspace=0.06, hspace=0.06)

    col_labels = col_labels or [str(c) for c in cols]
    row_labels = row_labels or [str(r) for r in rows]

    for ci, lab in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, ci + 1])
        ax.text(0.5, 0.4, lab, ha="center", va="center",
                fontsize=13, fontweight="semibold", color=label_color)
        ax.axis("off")
    for ri, lab in enumerate(row_labels):
        ax = fig.add_subplot(gs[ri + 1, 0])
        ax.text(0.5, 0.5, lab, ha="center", va="center",
                fontsize=13, fontweight="semibold", color=label_color,
                rotation=0)
        ax.axis("off")

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            ax = fig.add_subplot(gs[ri + 1, ci + 1])
            cell = cells.get((r, c))
            if cell is None:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color=label_color,
                        fontsize=14)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                render_func(ax=ax, **cell, **cell_kwargs)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="semibold",
                     color=label_color)
    return fig


def depth_coded_mip(
    volume: np.ndarray,
    ax=None,
    z_axis: int = 0,
    *,
    cmap: str = "viridis",
    percentile_clip: tuple[float, float] = (1, 99.5),
    gamma: float = 0.8,
    background: tuple[int, int, int] = (0, 0, 0),
):
    """Depth-coded MIP — argmax-of-z coloured by a colormap.

    For each XY pixel, takes the intensity-weighted depth and maps it
    through ``cmap``. Conveys 3D structure in a single 2D panel with
    less abstraction than an isosurface; cousin of the slide-4 layered
    look.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    vol = volume.astype(np.float32)
    lo, hi = np.percentile(vol, percentile_clip)
    vol_norm = np.clip((vol - lo) / max(1e-9, hi - lo), 0, 1) ** gamma
    mip = vol_norm.max(axis=z_axis)
    nz = vol.shape[z_axis]
    # intensity-weighted depth
    z_idx = np.arange(nz)
    if z_axis == 0:
        weights = vol_norm.sum(axis=(1, 2))[:, None, None] if False else vol_norm
        depth = (vol_norm * z_idx[:, None, None]).sum(axis=0) / np.where(
            vol_norm.sum(axis=0) == 0, 1, vol_norm.sum(axis=0))
    else:
        depth = (vol_norm * np.arange(nz).reshape(
            [1] * z_axis + [-1] + [1] * (vol.ndim - z_axis - 1))).sum(
            axis=z_axis) / np.where(vol_norm.sum(axis=z_axis) == 0, 1,
                                    vol_norm.sum(axis=z_axis))
    depth_norm = depth / max(1, nz - 1)
    cmap_obj = plt.get_cmap(cmap)
    color = cmap_obj(depth_norm)[..., :3]
    bg = np.array(background, dtype=float) / 255.0
    rgb = color * mip[..., None] + bg[None, None, :] * (1 - mip[..., None])
    ax.imshow(np.clip(rgb, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def layer_split_mip(
    volume: np.ndarray,
    ax_top=None, ax_bot=None,
    z_axis: int = 0,
    *,
    split: float = 0.5,
    render_func=live_dead_mip,
    channel_key: str = "live",
    **kwargs,
):
    """Render top half and bottom half of a volume on two axes.

    Mirrors slide-4 layout (top layers vs middle layers). ``split`` is
    the fractional Z position; ``render_func`` is called for each half
    with the half-volume passed under ``channel_key``.
    """
    nz = volume.shape[z_axis]
    cut = int(nz * split)
    top = np.take(volume, range(cut), axis=z_axis)
    bot = np.take(volume, range(cut, nz), axis=z_axis)
    if ax_top is not None:
        render_func(ax=ax_top, **{channel_key: top}, **kwargs)
    if ax_bot is not None:
        render_func(ax=ax_bot, **{channel_key: bot}, **kwargs)
    return ax_top, ax_bot


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
