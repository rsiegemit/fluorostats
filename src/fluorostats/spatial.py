"""Spatial sampling & heterogeneity — tile a field, slab a volume, quantify uniformity.

A large field of view or z-stack holds far more information than a single summary
number: how *uniform* a measurement is across space is itself a readout (e.g. print
quality, seeding evenness). These helpers sample a field at sub-region scale with any
reducer and quantify how patchy or clustered the result is. All assay-agnostic —
they take an array/points and a callable, nothing domain-specific.

  - `tile_bounds` / `tile_reduce` — partition a 2D field into a grid and reduce each cell.
  - `tile_point_density` — bin a point set (e.g. object centroids) into a grid of counts.
  - `slab_reduce` — split a volume into axial slabs and reduce each (through-depth profiling).
  - `morans_i` — spatial autocorrelation of a value grid (clustered vs random vs dispersed).
  - `spatial_heterogeneity` — CV + Moran's I summary of a value grid.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def tile_bounds(shape: tuple[int, int], grid: tuple[int, int] = (5, 5)):
    """Yield ``(y0, y1, x0, x1)`` pixel bounds for each cell of a ``grid`` partition.

    Cells tile the full 2D ``shape`` with near-equal size (last cell absorbs any
    remainder). Iterates row-major: ``(iy, ix)`` with ``ix`` fastest.
    """
    ny, nx = shape
    gy, gx = grid
    ys = np.linspace(0, ny, gy + 1).astype(int)
    xs = np.linspace(0, nx, gx + 1).astype(int)
    for iy in range(gy):
        for ix in range(gx):
            yield int(ys[iy]), int(ys[iy + 1]), int(xs[ix]), int(xs[ix + 1])


def tile_reduce(
    field: np.ndarray,
    func: Callable[[np.ndarray], float],
    grid: tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Apply ``func`` to each grid tile of a 2D ``field``; return a ``grid`` array.

    ``func`` maps a tile (2D sub-array) to a scalar. A tile whose ``func`` returns
    ``None`` or ``NaN`` becomes ``NaN`` in the output — so downstream stats can skip
    empty/degenerate regions. Handy with a lambda, e.g.
    ``tile_reduce(mask, lambda t: 100 * t.mean())`` for per-tile coverage.
    """
    gy, gx = grid
    out = np.full((gy, gx), np.nan, dtype=float)
    for k, (y0, y1, x0, x1) in enumerate(tile_bounds(field.shape[:2], grid)):
        val = func(field[y0:y1, x0:x1])
        out[k // gx, k % gx] = np.nan if val is None else float(val)
    return out


def tile_point_density(
    points_yx: np.ndarray,
    shape: tuple[int, int],
    grid: tuple[int, int] = (5, 5),
    per_area: bool = False,
    pixel_area: float = 1.0,
) -> np.ndarray:
    """Bin a point set into a grid of counts (or densities).

    ``points_yx`` is ``(N, >=2)``; the last two columns are taken as ``(y, x)`` (so
    3D ``(z, y, x)`` centroids work directly). With ``per_area=True`` each cell is
    divided by its area (cell pixel count × ``pixel_area``) to give a density.
    """
    gy, gx = grid
    out = np.zeros((gy, gx), dtype=float)
    pts = np.asarray(points_yx, dtype=float)
    for k, (y0, y1, x0, x1) in enumerate(tile_bounds(shape, grid)):
        if pts.shape[0]:
            inb = ((pts[:, -2] >= y0) & (pts[:, -2] < y1)
                   & (pts[:, -1] >= x0) & (pts[:, -1] < x1))
            c = float(inb.sum())
        else:
            c = 0.0
        if per_area:
            area = (y1 - y0) * (x1 - x0) * pixel_area
            c = c / area if area > 0 else np.nan
        out[k // gx, k % gx] = c
    return out


def slab_reduce(
    volume: np.ndarray,
    func: Callable[[np.ndarray], float],
    n_slabs: int = 10,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``volume`` into ``n_slabs`` along ``axis`` and reduce each slab.

    Returns ``(centers, values)`` where ``centers`` are the fractional-depth
    midpoints (0–1) of each slab and ``values[i] = func(slab_i)``. ``func`` receives
    the slab as a sub-volume (same ndim as ``volume``), so it can project, threshold,
    or measure however it likes — e.g. network coverage vs depth through a wall.
    A ``func`` returning ``None``/``NaN`` yields ``NaN`` for that slab.
    """
    n = volume.shape[axis]
    edges = np.linspace(0, n, n_slabs + 1).astype(int)
    centers = np.empty(n_slabs, dtype=float)
    values = np.full(n_slabs, np.nan, dtype=float)
    for i in range(n_slabs):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            centers[i] = (lo + 0.5) / n
            continue
        sl = [slice(None)] * volume.ndim
        sl[axis] = slice(lo, hi)
        val = func(volume[tuple(sl)])
        centers[i] = (lo + hi) / 2.0 / n
        values[i] = np.nan if val is None else float(val)
    return centers, values


def morans_i(grid: np.ndarray, connectivity: str = "rook") -> float:
    """Spatial autocorrelation (Moran's I) of a 2D value grid.

    ``+1`` neighbouring cells are similar (clustered/patchy), ``0`` random, ``-1``
    checkerboard/dispersed. ``NaN`` cells are ignored (their edges drop out).
    ``connectivity`` is ``"rook"`` (4-neighbour) or ``"queen"`` (8-neighbour).
    Returns ``NaN`` for <3 valid cells or zero variance.
    """
    g = np.asarray(grid, dtype=float)
    mask = ~np.isnan(g)
    if mask.sum() < 3:
        return float("nan")
    mu = g[mask].mean()
    z = np.where(mask, g - mu, 0.0)
    if connectivity == "queen":
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        offs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    nr, nc = g.shape
    num = 0.0
    s0 = 0.0
    for i in range(nr):
        for j in range(nc):
            if not mask[i, j]:
                continue
            for di, dj in offs:
                ni, nj = i + di, j + dj
                if 0 <= ni < nr and 0 <= nj < nc and mask[ni, nj]:
                    num += z[i, j] * z[ni, nj]
                    s0 += 1
    denom = float((z[mask] ** 2).sum())
    if s0 == 0 or denom == 0:
        return float("nan")
    return float((mask.sum() / s0) * (num / denom))


def spatial_heterogeneity(grid: np.ndarray, connectivity: str = "rook") -> dict:
    """Summarise how uniform a value grid is.

    Returns dict with keys:
        cv — coefficient of variation across valid cells (0 = uniform).
        morans_i — spatial autocorrelation (see `morans_i`).
        n — number of valid (non-NaN) cells.
    """
    g = np.asarray(grid, dtype=float)
    v = g[~np.isnan(g)]
    cv = float(v.std(ddof=1) / v.mean()) if v.size > 1 and v.mean() != 0 else float("nan")
    return {"cv": cv, "morans_i": morans_i(g, connectivity), "n": int(v.size)}


__all__ = [
    "tile_bounds", "tile_reduce", "tile_point_density", "slab_reduce",
    "morans_i", "spatial_heterogeneity",
]
