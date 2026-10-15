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

from .spatial import tile_reduce
from .stats import coefficient_of_variation, gini


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
    if ny < tiles or nx < tiles:
        return {"lateral_gini": float("nan"), "lateral_cv": float("nan"), "n_tiles": 0}
    # spatial.tile_bounds cells: every pixel counts, the last cell absorbing any
    # remainder; means accumulate in float64 so float32 input does not round them
    tile_means = tile_reduce(xy, lambda t: float(t.mean(dtype=np.float64)),
                             grid=(tiles, tiles)).ravel()
    return {
        "lateral_gini": float(gini(tile_means)),
        "lateral_cv": float(coefficient_of_variation(tile_means)),
        "n_tiles": int(tile_means.size),
    }


def depth_profile(volume: np.ndarray, z_axis: int = 0) -> np.ndarray:
    """Mean intensity per z slice — a 1D depth profile."""
    return volume.mean(axis=tuple(a for a in range(volume.ndim) if a != z_axis))


def _z_um(voxel_size_um: tuple[float, float, float] | None) -> float:
    """The z voxel size (the first entry of ``(vz, vy, vx)``), NaN when uncalibrated."""
    return float(voxel_size_um[0]) if voxel_size_um else float("nan")


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
        Voxel size; its z entry (the first, whatever ``z_axis`` is) converts the
        span to micrometres.
    relative_threshold : float
        Slices with mean intensity below this fraction of the peak are
        considered signal-free.
    z_axis : int
        Axis of ``volume`` that is depth.

    Returns
    -------
    dict with keys:
        z_lo, z_hi : int — first and last suprathreshold slice indices.
        span_slices : int — z_hi - z_lo + 1, or 0 if no slice qualifies.
        span_um : float — span in micrometres; NaN without ``voxel_size_um``.
    """
    profile = depth_profile(volume, z_axis=z_axis)
    vz = _z_um(voxel_size_um)
    if profile.max() == 0:
        return {"z_lo": 0, "z_hi": 0, "span_slices": 0, "span_um": 0.0 * vz}
    cutoff = relative_threshold * profile.max()
    above = np.where(profile >= cutoff)[0]
    if above.size == 0:
        return {"z_lo": 0, "z_hi": 0, "span_slices": 0, "span_um": 0.0 * vz}
    z_lo, z_hi = int(above[0]), int(above[-1])
    span_slices = z_hi - z_lo + 1
    return {"z_lo": z_lo, "z_hi": z_hi, "span_slices": span_slices,
            "span_um": float(span_slices * vz)}


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
    voxel_size_um : (vz, vy, vx) or None
        Voxel size; its z entry (the first, whatever ``z_axis`` is) gives the
        ``_um`` keys.
    z_axis : int
        Axis of ``volume`` that is depth.

    Returns
    -------
    dict with keys:
        z_centroid : float — intensity-weighted mean z position, in slices.
        z_p25, z_p50, z_p75 : float — depths (slices) above which 25/50/75 % of
            cumulative intensity lies.
        z_centroid_um, z_p25_um, z_p50_um, z_p75_um : float — the same in
            micrometres; NaN without ``voxel_size_um``.
    """
    profile = depth_profile(volume, z_axis=z_axis)
    total = profile.sum()
    if total == 0:
        slices = {"z_centroid": 0.0, "z_p25": 0.0, "z_p50": 0.0, "z_p75": 0.0}
    else:
        z = np.arange(profile.size)
        cum = np.cumsum(profile) / total

        def _percentile(q: float) -> float:
            idx = int(np.searchsorted(cum, q / 100.0))
            return float(min(idx, profile.size - 1))

        slices = {"z_centroid": float((z * profile).sum() / total),
                  "z_p25": _percentile(25), "z_p50": _percentile(50),
                  "z_p75": _percentile(75)}
    vz = _z_um(voxel_size_um)
    return {**slices, **{f"{k}_um": v * vz for k, v in slices.items()}}
