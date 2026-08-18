"""Oriented-texture and open-space descriptors for network-like signal.

Two complementary, assay-agnostic readouts of how a fibrous / mesh-like
structure (F-actin, collagen, microvascular network, ...) is organised:

  - `orientation_anisotropy` — structure-tensor coherence: is the signal
    aligned along a preferred direction or isotropic? (intensity-based, no
    segmentation needed).
  - `mesh_size` — the characteristic open-gap scale between network strands
    (mask-based).

Both operate on a single 2D image/mask so they compose with any projection.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def orientation_anisotropy(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    sigma: float = 2.0,
) -> dict:
    """Structure-tensor orientation coherence of a 2D intensity image.

    Coherence is 0 for isotropic texture (no preferred direction) and rises
    toward 1 as the signal aligns along one orientation. Computed from the
    gradient structure tensor averaged over the (optionally masked) field.

    Parameters
    ----------
    image : 2D array — intensity (not a binary mask).
    mask : optional 2D bool — restrict the tensor average to this region.
    sigma : gradient/tensor smoothing scale in pixels.

    Returns
    -------
    dict with keys:
        coherence — 0 (isotropic) … 1 (fully aligned).
        dominant_orientation_deg — preferred axis in [0, 180).
        n_pixels — pixels contributing to the average.
    """
    from skimage.feature import structure_tensor

    img = np.asarray(image, dtype=float)
    if mask is not None:
        sel = np.asarray(mask) > 0
    else:
        sel = np.ones(img.shape, bool)
    if not sel.any() or np.ptp(img) == 0:
        return {"coherence": 0.0, "dominant_orientation_deg": float("nan"),
                "n_pixels": int(sel.sum())}

    axx, axy, ayy = structure_tensor(img, sigma=sigma, order="rc")
    # 'rc' order returns (Arr, Arc, Acc) = (Ayy, Axy, Axx) in image (y, x) axes.
    jyy, jxy, jxx = axx.mean(where=sel), axy.mean(where=sel), ayy.mean(where=sel)
    denom = jxx + jyy
    coherence = float(np.hypot(jxx - jyy, 2.0 * jxy) / denom) if denom > 0 else 0.0
    orient = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)          # radians, principal axis
    orient_deg = float(np.degrees(orient) % 180.0)
    return {"coherence": min(coherence, 1.0),
            "dominant_orientation_deg": orient_deg,
            "n_pixels": int(sel.sum())}


def mesh_size(mask: np.ndarray, spacing: float | tuple[float, ...] = 1.0) -> float:
    """Characteristic open-gap (pore / mesh) size of a network mask.

    Twice the mean distance from background pixels to the nearest foreground
    strand — the typical gap between strands. Returns physical units when
    ``spacing`` is given (scalar or per-axis), else pixels. NaN if the mask is
    empty or fully foreground (no gaps to measure).
    """
    m = np.asarray(mask) > 0
    bg = ~m
    if not m.any() or not bg.any():
        return float("nan")
    dt = ndi.distance_transform_edt(bg, sampling=spacing)
    return float(2.0 * dt[bg].mean())


__all__ = ["orientation_anisotropy", "mesh_size"]
