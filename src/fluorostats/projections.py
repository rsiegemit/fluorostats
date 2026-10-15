"""2D maximum-intensity projection (MIP) panels of 3D stacks.

The single-axis renderers (grey MIP with a mask overlay, two-channel live/dead
MIP, depth-coded MIP) take ``ax=None, z_axis=0``, so any of them can be the
``render_func`` of `mip_grid` (a labelled row × col figure) or of
`layer_split_mip` (top vs bottom half of a stack).
"""

from __future__ import annotations

from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import pubfig
from .pubfig import PALETTE
from .qc import overlay_rgb


def mip_overlay(
    intensity: np.ndarray,
    mask: np.ndarray | None = None,
    ax: Axes | None = None,
    z_axis: int = 0,
    overlay_color: tuple[int, int, int] = (220, 60, 220),
    overlay_alpha: float = 0.6,
) -> Axes:
    """Grey-scale maximum-intensity projection with optional magenta mask.

    Composited with `qc.overlay_rgb` (min-max grey stretch), the same rule as
    the CLI's QC overlays. Returns the axis. Convenient cell renderer for a
    condition × time grid of representative stacks (one cell per condition ×
    region × day).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    mip = intensity.max(axis=z_axis)
    # numpy < 2.3 stubs type any(axis=...) as possibly a scalar; with an axis it is an array.
    mip_mask = (
        np.zeros(mip.shape, dtype=bool) if mask is None else cast(np.ndarray, mask.any(axis=z_axis))
    )
    ax.imshow(overlay_rgb(mip, mip_mask, overlay_color, overlay_alpha))
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def live_dead_mip(
    live: np.ndarray,
    dead: np.ndarray | None = None,
    ax: Axes | None = None,
    z_axis: int = 0,
    *,
    live_color: tuple[int, int, int] = (40, 220, 60),
    dead_color: tuple[int, int, int] = (240, 30, 40),
    percentile_clip: tuple[float, float] = (1, 99.5),
    gamma: float = 0.75,
    background: tuple[int, int, int] = (0, 0, 0),
    border: str | None = PALETTE["ink"],
    border_lw: float = 1.0,
) -> Axes:
    """Two-channel (live/dead) maximum-intensity projection.

    ``live`` is the live-cell channel (typically Alexa-488 / Calcein /
    GFP), ``dead`` is the dead-cell channel (typically PI / Alexa-568).
    Each channel's MIP is independently percentile-clipped, gamma-corrected,
    then composited additively into an RGB image on ``background``
    (`pubfig.composite2ch`, the one live/dead composite).

    Returns the matplotlib axis. The signature mirrors :func:`mip_overlay`
    so the two helpers can be swapped inside grid layouts.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    rgb = pubfig.composite2ch(
        live.max(axis=z_axis),
        None if dead is None else dead.max(axis=z_axis),
        live_rgb=np.array(live_color, dtype=float) / 255.0,
        dead_rgb=np.array(dead_color, dtype=float) / 255.0,
        clip=percentile_clip,
        gamma=gamma,
        background=np.array(background, dtype=float) / 255.0,
    )
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
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
    render_func: Callable[..., Axes] = live_dead_mip,
    figsize: tuple[float, float] | None = None,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    title: str | None = None,
    cell_kwargs: dict | None = None,
    label_color: str = PALETTE["ink"],
    background: str = "white",
) -> Figure:
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
    gs = fig.add_gridspec(
        nrows + 1,
        ncols + 1,
        width_ratios=[0.18] + [1] * ncols,
        height_ratios=[0.18] + [1] * nrows,
        wspace=0.06,
        hspace=0.06,
    )

    col_labels = col_labels or [str(c) for c in cols]
    row_labels = row_labels or [str(r) for r in rows]

    _grid_labels(fig, gs, col_labels, row_labels, label_color)

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            ax = fig.add_subplot(gs[ri + 1, ci + 1])
            cell = cells.get((r, c))
            if cell is None:
                _placeholder_cell(ax, label_color)
            else:
                render_func(ax=ax, **cell, **cell_kwargs)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="semibold", color=label_color)
    return fig


def _grid_labels(fig: Figure, gs, col_labels: list[str], row_labels: list[str], color: str) -> None:
    """Column labels along the top row and row labels down the first column of ``gs``."""
    for ci, lab in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, ci + 1])
        ax.text(
            0.5,
            0.4,
            lab,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="semibold",
            color=color,
        )
        ax.axis("off")
    for ri, lab in enumerate(row_labels):
        ax = fig.add_subplot(gs[ri + 1, 0])
        ax.text(
            0.5,
            0.5,
            lab,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="semibold",
            color=color,
            rotation=0,
        )
        ax.axis("off")


def _placeholder_cell(ax: Axes, color: str) -> None:
    """Grey cell with a dash, for a (row, col) that has no stack."""
    ax.set_facecolor("#f0f0f0")
    ax.text(
        0.5,
        0.5,
        "—",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=color,
        fontsize=14,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def depth_coded_mip(
    volume: np.ndarray,
    ax: Axes | None = None,
    z_axis: int = 0,
    *,
    cmap: str = "viridis",
    percentile_clip: tuple[float, float] = (1, 99.5),
    gamma: float = 0.8,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Axes:
    """Depth-coded MIP — intensity-weighted mean depth coloured by a colormap.

    For each XY pixel, takes the intensity-weighted depth and maps it
    through ``cmap``. Conveys 3D structure in a single 2D panel with
    less abstraction than an isosurface; cousin of the layered
    look.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    vol_norm = pubfig.imnorm(volume, *percentile_clip) ** gamma
    mip = vol_norm.max(axis=z_axis)
    nz = volume.shape[z_axis]
    # intensity-weighted depth
    z_idx = np.arange(nz)
    if z_axis == 0:
        depth = (vol_norm * z_idx[:, None, None]).sum(axis=0) / np.where(
            vol_norm.sum(axis=0) == 0, 1, vol_norm.sum(axis=0)
        )
    else:
        depth = (
            vol_norm * np.arange(nz).reshape([1] * z_axis + [-1] + [1] * (volume.ndim - z_axis - 1))
        ).sum(axis=z_axis) / np.where(vol_norm.sum(axis=z_axis) == 0, 1, vol_norm.sum(axis=z_axis))
    depth_norm = depth / max(1, nz - 1)
    cmap_obj = plt.get_cmap(cmap)
    color = cmap_obj(depth_norm)[..., :3]
    bg = np.array(background, dtype=float) / 255.0
    rgb = color * mip[..., None] + bg[None, None, :] * (1 - mip[..., None])
    ax.imshow(np.clip(rgb, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def layer_split_mip(
    volume: np.ndarray,
    ax_top: Axes | None = None,
    ax_bot: Axes | None = None,
    z_axis: int = 0,
    *,
    split: float = 0.5,
    render_func: Callable[..., Axes] = live_dead_mip,
    channel_key: str = "live",
    **kwargs,
) -> tuple[Axes | None, Axes | None]:
    """Render top half and bottom half of a volume on two axes.

    Top layers vs deeper layers side by side. ``split`` is
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


__all__ = [
    "mip_overlay",
    "live_dead_mip",
    "mip_grid",
    "depth_coded_mip",
    "layer_split_mip",
]
