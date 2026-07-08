"""Live/Dead viability quantification from two-channel fluorescence.

The classic Calcein-AM (live) / ethidium-or-PI (dead) assay reports two
signals. In thick 3D samples, confocal intensity attenuates with depth, so a
single plane or a maximum-intensity projection *overestimates* viability and
misses depth-dependent core death. This module quantifies viability in a
depth-aware way:

    - ``live_dead_fractions``     — whole-volume live / dead area or volume
                                    fractions and a viability ratio.
    - ``viability_depth_profile`` — per-z live / dead fraction (reveals a
                                    depth gradient a 2D method cannot see).
    - ``viability_2d_vs_3d``      — mid-plane vs MIP vs full-3D live fraction,
                                    quantifying the 2D overestimation.
    - ``attenuation_correct``     — per-z intensity normalisation to separate a
                                    biological death gradient from optical decay.

Segmentation is delegated to :mod:`fluorostats.segment`; pass already-segmented
boolean masks, or raw channels plus a ``threshold``/``method`` to segment here.
"""

from __future__ import annotations

import numpy as np

from .segment import binarize


def _mask(channel, mask=None, method="otsu", threshold_scale=1.0, min_size=32):
    if mask is not None:
        return mask > 0
    return binarize(channel, method=method, threshold_scale=threshold_scale,
                    min_size=min_size)


def live_dead_fractions(
    live: np.ndarray,
    dead: np.ndarray | None = None,
    *,
    live_mask=None,
    dead_mask=None,
    **seg,
) -> dict:
    """Whole-volume live / dead fractions and viability ratio.

    Returns dict: live_fraction, dead_fraction (of all voxels), and
    viability = live / (live + dead) area (NaN if no cells detected).
    """
    lm = _mask(live, live_mask, **seg)
    lf = float(lm.mean())
    out = {"live_fraction": lf}
    if dead is not None or dead_mask is not None:
        dm = _mask(dead, dead_mask, **seg)
        df = float(dm.mean())
        denom = lm.sum() + dm.sum()
        out["dead_fraction"] = df
        out["viability"] = float(lm.sum() / denom) if denom > 0 else float("nan")
    return out


def viability_depth_profile(
    live: np.ndarray,
    dead: np.ndarray | None = None,
    z_axis: int = 0,
    *,
    live_mask=None,
    dead_mask=None,
    **seg,
) -> dict:
    """Per-z live (and dead) area fraction — the depth-resolved viability.

    Returns dict of 1D arrays (length = n z slices): live_by_z, and if a dead
    channel is given, dead_by_z and viability_by_z.
    """
    lm = _mask(live, live_mask, **seg)
    axes = tuple(a for a in range(lm.ndim) if a != z_axis)
    live_by_z = lm.mean(axis=axes)
    out = {"live_by_z": live_by_z}
    if dead is not None or dead_mask is not None:
        dm = _mask(dead, dead_mask, **seg)
        dead_by_z = dm.mean(axis=axes)
        la = lm.sum(axis=axes).astype(float)
        da = dm.sum(axis=axes).astype(float)
        denom = la + da
        out["dead_by_z"] = dead_by_z
        out["viability_by_z"] = np.divide(la, denom, out=np.full_like(la, np.nan),
                                          where=denom > 0)
    return out


def viability_2d_vs_3d(
    live: np.ndarray,
    z_axis: int = 0,
    *,
    live_mask=None,
    **seg,
) -> dict:
    """Compare live fraction from a single mid plane, a MIP, and the full 3D
    volume — quantifying how much a 2D readout overestimates viability.

    Returns dict: live_fraction_midplane, live_fraction_mip, live_fraction_3d,
    and overestimate_mip = mip / 3d (>1 means the 2D MIP overstates coverage).
    """
    lm = _mask(live, live_mask, **seg)
    nz = lm.shape[z_axis]
    mid = np.take(lm, nz // 2, axis=z_axis)
    mip = lm.max(axis=z_axis)
    f3d = float(lm.mean())
    f_mid = float(mid.mean())
    f_mip = float(mip.mean())
    return {
        "live_fraction_midplane": f_mid,
        "live_fraction_mip": f_mip,
        "live_fraction_3d": f3d,
        "overestimate_mip": float(f_mip / f3d) if f3d > 0 else float("nan"),
        "overestimate_midplane": float(f_mid / f3d) if f3d > 0 else float("nan"),
    }


def attenuation_correct(
    volume: np.ndarray,
    z_axis: int = 0,
    *,
    reference: str = "max",
) -> np.ndarray:
    """Per-z intensity normalisation to compensate depth attenuation.

    Each z slice is scaled so its mean matches a reference slice (the brightest
    slice by default, i.e. shallow, least-attenuated). Signal that then *stays*
    low with depth reflects genuine biology, not optical decay — use before
    :func:`viability_depth_profile` to check a death gradient is real.

    Returns a new float array; the input is not modified.
    """
    vol = volume.astype(np.float32)
    axes = tuple(a for a in range(vol.ndim) if a != z_axis)
    slice_means = vol.mean(axis=axes)
    if reference == "max":
        ref = slice_means.max()
    elif reference == "first":
        ref = slice_means.flat[0]
    else:
        ref = slice_means.mean()
    scale = np.divide(ref, slice_means, out=np.ones_like(slice_means),
                      where=slice_means > 0)
    shape = [1] * vol.ndim
    shape[z_axis] = -1
    return vol * scale.reshape(shape)


__all__ = [
    "live_dead_fractions",
    "viability_depth_profile",
    "viability_2d_vs_3d",
    "attenuation_correct",
]
