"""Intensity-based spatial morphometry — no segmentation required.

These metrics quantify how signal is distributed in space without
committing to a binary mask. Useful when:

  - Segmentation is fragile (low SNR, sparse signal, viability stains).
  - You want a per-stack number that summarises *uniformity* or *depth
    penetration* directly from the raw intensity volume.
  - You want a complementary measurement to cross-check a mask-based
    metric (e.g. "is the homogeneity claim robust to thresholding?").

All public functions are pure and accept a 3D volume `(z, y, x)`.
"""

from __future__ import annotations

import numpy as np


def lateral_homogeneity(
    volume: np.ndarray,
    tiles: int = 8,
    z_axis: int = 0,
) -> dict:
    """Spatial uniformity of signal across an XY tile grid.

    The volume is projected to a single XY image (mean along `z_axis`),
    then split into a `tiles x tiles` grid. Per-tile mean intensity is
    summarised by Gini coefficient and coefficient of variation — both
    are 0 when the signal is perfectly uniform across tiles and rise
    as the distribution becomes more concentrated in a few tiles.

    Parameters
    ----------
    volume : np.ndarray
        3D intensity volume.
    tiles : int, default 8
        Grid resolution per spatial axis. 8x8 is a sensible default
        for ~512px images.
    z_axis : int, default 0
        Axis to project along.

    Returns
    -------
    dict with keys: lateral_gini, lateral_cv, n_tiles.
    """
    xy = volume.mean(axis=z_axis)
    ny, nx = xy.shape
    ty = ny // tiles
    tx = nx // tiles
    if ty == 0 or tx == 0:
        return {"lateral_gini": float("nan"), "lateral_cv": float("nan"), "n_tiles": 0}
    cropped = xy[: ty * tiles, : tx * tiles]
    tile_means = cropped.reshape(tiles, ty, tiles, tx).mean(axis=(1, 3)).ravel()
    return {
        "lateral_gini": float(_gini(tile_means)),
        "lateral_cv": float(_cv(tile_means)),
        "n_tiles": int(tile_means.size),
    }


def depth_profile(volume: np.ndarray, z_axis: int = 0) -> np.ndarray:
    """Mean intensity per z slice — a 1D depth profile."""
    return volume.mean(axis=tuple(a for a in range(volume.ndim) if a != z_axis))


def depth_span(
    volume: np.ndarray,
    voxel_size_um: tuple[float, float, float] | None = None,
    relative_threshold: float = 0.1,
    z_axis: int = 0,
) -> dict:
    """Axial extent of signal above a fractional threshold.

    Builds a depth profile, finds the slices whose mean intensity is
    above `relative_threshold * max(profile)`, and returns the span
    between the first and last such slices.

    Parameters
    ----------
    volume : np.ndarray
        3D intensity volume.
    voxel_size_um : (vz, vy, vx) or None
        If provided, span is returned in micrometres; otherwise in slices.
    relative_threshold : float
        Slices with mean intensity below this fraction of the peak are
        considered signal-free.

    Returns
    -------
    dict with keys:
        z_lo, z_hi : int — first and last suprathreshold slice indices.
        span_slices : int — z_hi - z_lo + 1, or 0 if no slice qualifies.
        span_um : float | None — span in micrometres if voxel_size_um given.
    """
    profile = depth_profile(volume, z_axis=z_axis)
    if profile.max() == 0:
        return {"z_lo": 0, "z_hi": 0, "span_slices": 0, "span_um": 0.0 if voxel_size_um else None}
    cutoff = relative_threshold * profile.max()
    above = np.where(profile >= cutoff)[0]
    if above.size == 0:
        return {"z_lo": 0, "z_hi": 0, "span_slices": 0, "span_um": 0.0 if voxel_size_um else None}
    z_lo, z_hi = int(above[0]), int(above[-1])
    span_slices = z_hi - z_lo + 1
    span_um = float(span_slices * voxel_size_um[z_axis]) if voxel_size_um else None
    return {"z_lo": z_lo, "z_hi": z_hi, "span_slices": span_slices, "span_um": span_um}


def depth_centroid(
    volume: np.ndarray,
    voxel_size_um: tuple[float, float, float] | None = None,
    z_axis: int = 0,
) -> dict:
    """Intensity-weighted depth centroid and percentile depths.

    Parameters
    ----------
    volume : np.ndarray
        3D intensity volume.
    voxel_size_um : tuple or None
        If provided, results are in micrometres; otherwise in slices.

    Returns
    -------
    dict with keys:
        z_centroid : float — intensity-weighted mean z position.
        z_p25, z_p50, z_p75 : float — depths above which 25/50/75 % of
            cumulative intensity lies.
    """
    profile = depth_profile(volume, z_axis=z_axis)
    total = profile.sum()
    if total == 0:
        return {"z_centroid": 0.0, "z_p25": 0.0, "z_p50": 0.0, "z_p75": 0.0}
    z = np.arange(profile.size)
    scale = voxel_size_um[z_axis] if voxel_size_um else 1.0
    centroid = float((z * profile).sum() / total) * scale
    cum = np.cumsum(profile) / total

    def _percentile(q: float) -> float:
        idx = int(np.searchsorted(cum, q / 100.0))
        return float(min(idx, profile.size - 1) * scale)

    return {
        "z_centroid": centroid,
        "z_p25": _percentile(25),
        "z_p50": _percentile(50),
        "z_p75": _percentile(75),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative 1D array.

    0 = perfect uniformity, approaching 1 = perfect concentration.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return float("nan")
    v = v - v.min() if v.min() < 0 else v  # shift to non-negative
    if v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def _cv(values: np.ndarray) -> float:
    """Coefficient of variation = std / mean."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0 or v.mean() == 0:
        return float("nan")
    return float(v.std(ddof=0) / v.mean())
