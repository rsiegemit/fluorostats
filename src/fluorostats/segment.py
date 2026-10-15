"""Thresholding and binary segmentation for 2D and 3D arrays."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

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
    remove_small_objects,
)
from skimage.morphology import opening as binary_opening

from .preprocess import background_subtract, denoise

# Global thresholding algorithms fluorostats implements (all in-library).
THRESHOLD_METHODS: dict[str, Callable[..., Any]] = {
    "otsu": threshold_otsu,
    "li": threshold_li,
    "isodata": threshold_isodata,
    "triangle": threshold_triangle,
    "yen": threshold_yen,
    "mean": threshold_mean,
}

# Every accepted ``binarize(method=...)``: the algorithms above plus the two
# meta-methods that choose among them. The CLI builds its --threshold choices here.
METHODS = (*THRESHOLD_METHODS, "auto", "consensus")

# Salt-noise opening applied after thresholding, by name: the footprint for a mask
# of the given ndim, or None for no opening. "ball" (3x3x3 cross in 3D, disk(1) in
# 2D) removes any structure thinner than 3 z-slices; "disk" opens each z-slice in
# 2D only, so thin sheets/vessels that span 1-2 slices survive.
OPENINGS: dict[str, Callable[[int], np.ndarray | None]] = {
    "ball": lambda ndim: ball(1) if ndim == 3 else disk(1),
    "disk": lambda ndim: disk(1)[None] if ndim == 3 else disk(1),
    "none": lambda ndim: None,
}


def segment_channel(
    img: np.ndarray,
    *,
    method: str = "otsu",
    min_size: int = 64,
    threshold_scale: float = 1.0,
    sigma: float = 1.0,
    bg_radius: int = 0,
    opening: str = "ball",
) -> tuple[np.ndarray, np.ndarray]:
    """The full single-channel segmentation the CLI runs: Gaussian denoise,
    optional white-top-hat background subtraction (``bg_radius > 0``), then
    `binarize` (``opening`` as there: the default 3D opening erases structures
    thinner than 3 z-slices). Returns ``(preprocessed_image, mask)``.
    """
    img = denoise(img, sigma=sigma)
    if bg_radius > 0:
        img = background_subtract(img, radius=bg_radius)
    return img, binarize(img, method=method, min_size=min_size,
                         threshold_scale=threshold_scale, opening=opening)


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
    opening: str = "ball",
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
    opening : {"ball", "disk", "none"}
        Salt-noise opening after thresholding (see `OPENINGS`). The default
        ``"ball"`` opens 3D masks with a 3x3x3 cross, which erases any structure
        thinner than 3 z-slices (thin sheets, monolayers, vessels on a coarse
        z-step); ``"disk"`` opens each z-slice in 2D and keeps them; ``"none"``
        skips the opening. 2D masks: ``"ball"`` and ``"disk"`` are both disk(1).

    Returns
    -------
    ndarray[bool]
        Binary mask, same shape as input.
    """
    arr_float = arr.astype(np.float64)

    # Global threshold
    if method == "consensus":
        votes = np.zeros(arr_float.shape, dtype=np.uint8)
        for name, fn in THRESHOLD_METHODS.items():
            try:
                votes += (arr_float > fn(arr_float) * threshold_scale).astype(np.uint8)
            except Exception as exc:
                # A failed method still counts toward the majority, i.e. it votes
                # background everywhere; say so rather than skipping it silently.
                warnings.warn(f"consensus threshold: {name} failed ({type(exc).__name__}: "
                              f"{exc}); it counts as a background vote.",
                              RuntimeWarning, stacklevel=2)
        mask = votes >= (len(THRESHOLD_METHODS) // 2 + 1)
    else:
        if method == "auto":
            method = choose_threshold_method(arr_float)["method"]
        if method not in THRESHOLD_METHODS:
            raise ValueError(f"Unknown threshold method: {method!r}")
        thresh = THRESHOLD_METHODS[method](arr_float) * threshold_scale
        mask = arr_float > thresh

    if opening not in OPENINGS:
        raise ValueError(f"Unknown opening: {opening!r}; choose from {sorted(OPENINGS)}")
    footprint = OPENINGS[opening](arr.ndim)
    if footprint is not None:
        mask = binary_opening(mask, footprint=footprint)

    return remove_small(mask, min_size)


def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected objects smaller than ``min_size`` pixels/voxels (objects of
    exactly ``min_size`` are kept), identically on every scikit-image version.

    The one small-object rule of the package (`binarize`, `objects.label_3d`);
    ``min_size <= 0`` returns the mask unchanged.
    """
    if min_size <= 0:
        return mask
    try:
        # skimage >= 0.26 renamed min_size -> max_size, which removes objects of that
        # size *or fewer*; min_size - 1 keeps "smaller objects are removed".
        # (its < 0.26 annotations lack max_size: mypy flags it there only)
        return remove_small_objects(mask, max_size=min_size - 1)  # type: ignore[call-arg, unused-ignore]
    except TypeError:  # pragma: no cover - environment-gated: runs only on skimage < 0.26
        return remove_small_objects(mask, min_size=min_size)
