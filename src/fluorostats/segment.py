"""Thresholding and binary segmentation for 2D and 3D arrays."""

from __future__ import annotations

import numpy as np
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.morphology import (
    ball,
    disk,
    opening,
    remove_small_objects,
)

# Global thresholding algorithms fluorostats implements (all in-library).
THRESHOLD_METHODS = {
    "otsu": threshold_otsu,
    "li": threshold_li,
    "isodata": threshold_isodata,
    "triangle": threshold_triangle,
    "yen": threshold_yen,
    "mean": threshold_mean,
}


def choose_threshold_method(arr: np.ndarray) -> dict:
    """Pick a thresholding algorithm from the intensity histogram shape.

    Transparent heuristic (returns its reasoning; always overridable). It targets
    the one failure mode that is detectable from image statistics — Otsu missing a
    dim, sparse foreground (thin/dim vessels, low label): Otsu's bimodal assumption
    then puts the threshold above most signal and keeps almost nothing, while ``li``
    finds the dim foreground. When Otsu keeps an implausibly tiny fraction yet other
    methods find several times more signal, ``li`` is chosen; otherwise ``otsu``.

    This is a best-effort heuristic, NOT an oracle — Otsu can also *over*-segment
    noisy backgrounds, which histogram statistics cannot cleanly separate from a
    real dim foreground. When in doubt use ``binarize(method="consensus")`` (a
    majority vote of all six algorithms), which needs no regime guess.

    Returns dict: method, reason, skew, and the foreground fraction each method
    would select (so the caller can see the spread).
    """
    a = arr.astype(np.float64).ravel()
    mu, sd = a.mean(), a.std() + 1e-12
    skew = float(((a - mu) ** 3).mean() / sd**3)
    fracs = {}
    for name, fn in THRESHOLD_METHODS.items():
        try:
            fracs[name] = float((a > fn(a)).mean())
        except Exception:
            fracs[name] = float("nan")
    otsu_f = fracs.get("otsu", 0.0)
    li_f = fracs.get("li", 0.0)
    if otsu_f < 0.01 and li_f > 3 * otsu_f:
        method, reason = "li", (
            f"Otsu keeps only {otsu_f:.4f} of pixels while Li finds {li_f:.4f} "
            f"— dim/sparse foreground Otsu misses")
    else:
        method, reason = "otsu", (
            f"Otsu foreground {otsu_f:.4f} plausible (skew {skew:.1f})")
    return {"method": method, "reason": reason, "skew": skew, "fractions": fracs}


def binarize(
    arr: np.ndarray,
    method: str = "otsu",
    min_size: int = 64,
    threshold_scale: float = 1.0,
) -> np.ndarray:
    """Segment fluorescence signal into a binary mask.

    Parameters
    ----------
    arr : ndarray (2D or 3D, float or uint)
        Single-channel intensity array.
    method : {"otsu", "li", "isodata", "triangle", "yen", "mean", "auto", "consensus"}
        Global thresholding algorithm. ``"auto"`` selects one from the histogram
        shape via :func:`choose_threshold_method` (no single algorithm is best on
        all data — sparse/dim signal breaks Otsu). ``"consensus"`` majority-votes
        the six algorithms per pixel (robust when you cannot pick one).
    min_size : int
        Minimum object size in pixels/voxels. Smaller objects are removed.
    threshold_scale : float
        Multiply the computed threshold by this factor. Values < 1.0
        capture more dim signal; values > 1.0 are more conservative.

    Returns
    -------
    ndarray[bool]
        Binary mask, same shape as input.
    """
    arr_float = arr.astype(np.float64)

    # Global threshold
    if method == "consensus":
        votes = np.zeros(arr_float.shape, dtype=np.uint8)
        for fn in THRESHOLD_METHODS.values():
            try:
                votes += (arr_float > fn(arr_float) * threshold_scale).astype(np.uint8)
            except Exception:
                pass
        mask = votes >= (len(THRESHOLD_METHODS) // 2 + 1)
    else:
        if method == "auto":
            method = choose_threshold_method(arr_float)["method"]
        if method not in THRESHOLD_METHODS:
            raise ValueError(f"Unknown threshold method: {method!r}")
        thresh = THRESHOLD_METHODS[method](arr_float) * threshold_scale
        mask = arr_float > thresh

    # Morphological opening to remove salt noise
    if arr.ndim == 3:
        selem = ball(1)
    else:
        selem = disk(1)
    mask = opening(mask, footprint=selem)

    # Remove small connected components. skimage renamed `min_size` -> `max_size`
    # in 0.26 (same "remove objects <= threshold" semantic); support both versions.
    if min_size > 0:
        try:
            mask = remove_small_objects(mask, max_size=min_size)  # skimage >= 0.26
        except TypeError:
            mask = remove_small_objects(mask, min_size=min_size)  # skimage < 0.26

    return mask
