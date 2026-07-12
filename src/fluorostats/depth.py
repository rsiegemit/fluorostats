"""Axial (depth) intensity profiling for confocal z-stacks.

Generic, acquisition-agnostic helpers for turning a z-stack into a
mean-intensity-versus-depth profile, subtracting a matched blank/no-fluo
control, normalising to the near-surface signal, and integrating the
profile over a physical depth window (area under the curve).

Everything here is pure: functions take arrays and return arrays or
dataclasses, so they compose freely and are trivial to unit-test. Nothing
is specific to any single experiment — the caller supplies the volume, the
voxel spacing, and the analysis window.

Typical use::

    from fluorostats.io import load_volume
    from fluorostats import depth

    vol, meta = load_volume(path)
    prof = depth.intensity_depth_profile(vol, meta["voxel_size_um"][0])
    prof = depth.subtract_background(prof, blank_profile)
    prof = depth.normalize_to_surface(prof, n_surface=3)
    auc = depth.auc_depth(prof.depth_um, prof.normalized, z_max=100.0)

The volume may be 3D ``(Z, Y, X)`` or 4D ``(C, Z, Y, X)``; pass
``channel`` to pick a channel from a 4D stack.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class DepthProfile:
    """Mean intensity as a function of physical depth.

    Attributes
    ----------
    depth_um:
        Depth of each slice from the first slice, in micrometres.
    mean:
        Spatial mean intensity per slice (background-subtracted once
        :func:`subtract_background` has been applied).
    sem:
        Standard error of the per-slice spatial mean (std / sqrt(n_px)).
    normalized:
        ``mean`` divided by the near-surface reference; ``None`` until
        :func:`normalize_to_surface` is called.
    surface_reference:
        The scalar near-surface value used for normalisation, if any.
    n_pixels:
        Number of pixels averaged per slice (for SEM bookkeeping).
    """

    depth_um: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    n_pixels: int
    normalized: np.ndarray | None = None
    surface_reference: float | None = None


def _select_plane_stack(volume: np.ndarray, channel: int) -> np.ndarray:
    """Return a ``(Z, Y, X)`` float array from a 3D or 4D volume."""
    arr = np.asarray(volume)
    if arr.ndim == 4:
        arr = arr[channel]
    elif arr.ndim != 3:
        raise ValueError(
            f"expected 3D (Z,Y,X) or 4D (C,Z,Y,X) volume, got shape {arr.shape}"
        )
    return arr.astype(np.float64, copy=False)


def intensity_depth_profile(
    volume: np.ndarray,
    voxel_size_z_um: float,
    *,
    channel: int = 0,
    reducer: str = "mean",
) -> DepthProfile:
    """Collapse each z-slice to a single intensity value.

    Parameters
    ----------
    volume:
        3D ``(Z, Y, X)`` or 4D ``(C, Z, Y, X)`` array.
    voxel_size_z_um:
        Physical spacing between slices, in micrometres. Depth of slice
        ``i`` is ``i * voxel_size_z_um`` (slice 0 = surface).
    channel:
        Channel index to use when ``volume`` is 4D.
    reducer:
        ``"mean"`` (default) or ``"median"`` for the per-slice spatial
        statistic. ``"median"`` is more robust to bright debris/bubbles.

    Returns
    -------
    DepthProfile with ``normalized`` unset.
    """
    if voxel_size_z_um <= 0:
        raise ValueError(f"voxel_size_z_um must be > 0, got {voxel_size_z_um}")

    stack = _select_plane_stack(volume, channel)
    n_slices = stack.shape[0]
    flat = stack.reshape(n_slices, -1)
    n_px = flat.shape[1]

    if reducer == "mean":
        per_slice = flat.mean(axis=1)
    elif reducer == "median":
        per_slice = np.median(flat, axis=1)
    else:
        raise ValueError(f"reducer must be 'mean' or 'median', got {reducer!r}")

    std = flat.std(axis=1, ddof=0)
    sem = std / np.sqrt(n_px)
    depth = np.arange(n_slices, dtype=np.float64) * float(voxel_size_z_um)

    return DepthProfile(
        depth_um=depth,
        mean=per_slice,
        sem=sem,
        n_pixels=n_px,
    )


def resample_to_depth(
    source: DepthProfile,
    target_depth_um: np.ndarray,
) -> np.ndarray:
    """Interpolate a profile's ``mean`` onto another depth axis.

    Depths outside the source range are held at the nearest endpoint
    value (``np.interp`` default), which is appropriate for flat blanks.
    """
    return np.interp(
        np.asarray(target_depth_um, dtype=np.float64),
        source.depth_um,
        source.mean,
    )


def subtract_background(
    signal: DepthProfile,
    background: DepthProfile | float,
    *,
    clip_negative: bool = True,
) -> DepthProfile:
    """Subtract a blank/no-fluo control from a signal profile.

    ``background`` may be a scalar (single offset) or another
    :class:`DepthProfile`, which is resampled depth-for-depth onto the
    signal's axis before subtraction — so blank and signal stacks need
    not share slice count or spacing.

    ``clip_negative`` floors the result at 0 (intensities below the blank
    are noise, not negative fluorescence).
    """
    if isinstance(background, DepthProfile):
        bg = resample_to_depth(background, signal.depth_um)
    else:
        bg = float(background)

    corrected = signal.mean - bg
    if clip_negative:
        corrected = np.clip(corrected, 0.0, None)

    return replace(
        signal,
        mean=corrected,
        normalized=None,
        surface_reference=None,
    )


def normalize_to_surface(
    profile: DepthProfile,
    *,
    n_surface: int = 3,
) -> DepthProfile:
    """Normalise a profile to its mean near-surface intensity.

    The reference is the average of the first ``n_surface`` slices. This
    removes per-stack gain/laser differences so that decay *shape* can be
    compared across acquisitions. Requires a positive surface reference.
    """
    if n_surface < 1:
        raise ValueError(f"n_surface must be >= 1, got {n_surface}")
    n = min(n_surface, profile.mean.shape[0])
    reference = float(np.mean(profile.mean[:n]))
    if reference <= 0:
        raise ValueError(
            "near-surface reference is non-positive; cannot normalise "
            "(check background subtraction / surface orientation)"
        )
    return replace(
        profile,
        normalized=profile.mean / reference,
        surface_reference=reference,
    )


def auc_depth(
    depth_um: np.ndarray,
    values: np.ndarray,
    *,
    z_min: float = 0.0,
    z_max: float = 100.0,
) -> float:
    """Trapezoidal area under a depth profile over ``[z_min, z_max]``.

    Endpoints falling between slices are linearly interpolated so the
    window is exactly ``[z_min, z_max]`` regardless of slice spacing.
    Units are ``value * micrometre`` (e.g. for a surface-normalised
    curve, a perfectly non-decaying signal over 0-100 um gives 100).
    """
    depth = np.asarray(depth_um, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if depth.shape != vals.shape:
        raise ValueError("depth_um and values must have the same shape")
    if z_max <= z_min:
        raise ValueError(f"z_max ({z_max}) must exceed z_min ({z_min})")

    order = np.argsort(depth)
    depth, vals = depth[order], vals[order]

    if z_min < depth[0] or z_max > depth[-1]:
        # Window exceeds the acquired range; clamp and let the caller
        # decide via coverage (reported separately by the driver).
        z_min = max(z_min, float(depth[0]))
        z_max = min(z_max, float(depth[-1]))
        if z_max <= z_min:
            return 0.0

    # Build a grid that includes the interpolated window endpoints.
    interior = depth[(depth > z_min) & (depth < z_max)]
    grid = np.concatenate(([z_min], interior, [z_max]))
    grid_vals = np.interp(grid, depth, vals)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 rename
    return float(trapezoid(grid_vals, grid))


__all__ = [
    "DepthProfile",
    "intensity_depth_profile",
    "resample_to_depth",
    "subtract_background",
    "normalize_to_surface",
    "auc_depth",
]
