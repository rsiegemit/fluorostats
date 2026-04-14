"""Thresholding and binary segmentation for 2D and 3D arrays."""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_li, threshold_otsu
from skimage.morphology import (
    ball,
    disk,
    opening,
    remove_small_objects,
)


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
    method : {"otsu", "li"}
        Global thresholding method.
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
    if method == "otsu":
        thresh = threshold_otsu(arr_float)
    elif method == "li":
        thresh = threshold_li(arr_float)
    else:
        raise ValueError(f"Unknown threshold method: {method!r}")

    thresh *= threshold_scale
    mask = arr_float > thresh

    # Morphological opening to remove salt noise
    if arr.ndim == 3:
        selem = ball(1)
    else:
        selem = disk(1)
    mask = opening(mask, footprint=selem)

    # Remove small connected components
    if min_size > 0:
        mask = remove_small_objects(mask, max_size=min_size)

    return mask
