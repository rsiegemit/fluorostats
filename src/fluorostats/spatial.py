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

from typing import Callable, Iterator

import numpy as np

from .stats import coefficient_of_variation


def tile_bounds(
    shape: tuple[int, int],
    grid: tuple[int, int] = (5, 5),
) -> Iterator[tuple[int, int, int, int]]:
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
    3D ``(z, y, x)`` centroids work directly). Points are continuous coordinates, so
    they are binned into half-open cells between the exact edges ``k · n / grid``
    (a point on an interior edge belongs to the next cell; not the whole-pixel
    edges `tile_bounds` gives an image); a point on or past the field border counts
    in the nearest edge cell; a NaN or infinite point raises ValueError. With
    ``per_area=True`` each count is divided by the cell area
    (``(ny / gy) · (nx / gx) · pixel_area``) to give a density.
    """
    gy, gx = grid
    ny, nx = shape
    out: np.ndarray = np.zeros((gy, gx), dtype=float)
    pts = np.asarray(points_yx, dtype=float)
    if not np.isfinite(pts).all():
        raise ValueError("tile_point_density: points must be finite (got NaN or inf)")
    if pts.shape[0]:
        # multiply before dividing: exact on the edges for integer coordinates
        iy = np.clip(np.floor(pts[:, -2] * gy / ny).astype(int), 0, gy - 1)
        ix = np.clip(np.floor(pts[:, -1] * gx / nx).astype(int), 0, gx - 1)
        np.add.at(out, (iy, ix), 1.0)
    if per_area:
        out = out / ((ny / gy) * (nx / gx) * pixel_area)
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


# Neighbour offsets for `morans_i`, by connectivity name.
_NEIGHBOUR_OFFSETS = {
    "rook": [(0, 1), (1, 0), (0, -1), (-1, 0)],
    "queen": [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
}


def morans_i(grid: np.ndarray, connectivity: str = "rook") -> float:
    """Spatial autocorrelation (Moran's I) of a 2D value grid.

    ``+1`` neighbouring cells are similar (clustered/patchy), ``0`` random, ``-1``
    checkerboard/dispersed. ``NaN`` cells are ignored (their edges drop out).
    ``connectivity`` is ``"rook"`` (4-neighbour) or ``"queen"`` (8-neighbour).
    Returns ``NaN`` for <3 valid cells or zero variance.
    """
    if connectivity not in _NEIGHBOUR_OFFSETS:
        raise ValueError(
            f"connectivity must be one of {tuple(_NEIGHBOUR_OFFSETS)}, got {connectivity!r}"
        )
    g = np.asarray(grid, dtype=float)
    mask = ~np.isnan(g)
    if mask.sum() < 3:
        return float("nan")
    mu = g[mask].mean()
    z = np.where(mask, g - mu, 0.0)
    offs = _NEIGHBOUR_OFFSETS[connectivity]
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
        cv — coefficient of variation across valid cells (0 = uniform), in the
            population form (ddof=0): the cells are every tile of the image, not
            a sample of them, as for `morphometry` lateral_cv and `objects`
            centroid_cv.
        morans_i — spatial autocorrelation (see `morans_i`).
        n — number of valid (non-NaN) cells.
    """
    g = np.asarray(grid, dtype=float)
    v = g[~np.isnan(g)]
    cv = coefficient_of_variation(v)
    return {"cv": cv, "morans_i": morans_i(g, connectivity), "n": int(v.size)}


def coverage_heterogeneity(mask: np.ndarray, grid: tuple[int, int] = (5, 5)) -> dict:
    """Spatial uniformity of a binary mask's coverage across a tile grid.

    For a 3D ``(Z, Y, X)`` mask the through-z coverage map is used. Returns
    ``coverage_tile_cv`` (tile-to-tile CV, 0 = evenly seeded) and
    ``coverage_morans_i`` (spatial autocorrelation; >0 = patchy/clustered).
    """
    cov = mask.mean(axis=0) if mask.ndim == 3 else mask.astype(float)
    het = spatial_heterogeneity(tile_reduce(cov, lambda t: float(t.mean()), grid=grid))
    return {"coverage_tile_cv": het["cv"], "coverage_morans_i": het["morans_i"]}


__all__ = [
    "tile_bounds",
    "tile_reduce",
    "tile_point_density",
    "slab_reduce",
    "coverage_heterogeneity",
    "morans_i",
    "spatial_heterogeneity",
]
