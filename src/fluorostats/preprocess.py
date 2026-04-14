"""Preprocessing: channel selection, denoising, background subtraction, auto-crop."""

from __future__ import annotations

import re

import numpy as np
from skimage.filters import gaussian
from skimage.morphology import disk, white_tophat


# Patterns that indicate the green / F-actin channel
_GREEN_PATTERNS = re.compile(
    r"(alexa.*488|fitc|gfp|green|factin|f-actin|ch2|488)",
    re.IGNORECASE,
)


def select_green_channel(
    arr: np.ndarray,
    channel_names: list[str],
    override: int | str | None = None,
) -> np.ndarray:
    """Select the green / F-actin channel from a multi-channel array.

    Parameters
    ----------
    arr : ndarray
        Array with leading channel axis: (C, ...).
    channel_names : list[str]
        Names for each channel.
    override : int or str, optional
        Force a specific channel by index or name substring.

    Returns
    -------
    ndarray
        Single-channel array with channel axis removed.
    """
    n_ch = arr.shape[0]

    if override is not None:
        idx = _resolve_override(override, channel_names, n_ch)
        return arr[idx]

    # Try matching channel names against known green patterns
    for i, name in enumerate(channel_names):
        if _GREEN_PATTERNS.search(name):
            return arr[i]

    # Fallback heuristics
    if n_ch == 1:
        return arr[0]
    if n_ch == 3:
        # RGB — green is index 1
        return arr[1]
    if n_ch == 2:
        # Common confocal layout: Ch1=red/transmitted, Ch2=green/fluorescence
        return arr[1]

    # Default to first channel
    return arr[0]


def _resolve_override(
    override: int | str, channel_names: list[str], n_ch: int
) -> int:
    if isinstance(override, int):
        if 0 <= override < n_ch:
            return override
        raise ValueError(f"Channel index {override} out of range (0..{n_ch - 1})")
    # String match
    for i, name in enumerate(channel_names):
        if override.lower() in name.lower():
            return i
    raise ValueError(f"No channel matching '{override}' in {channel_names}")


def denoise(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian denoise. Works for both 2D and 3D arrays.

    For 3D, applies per-slice to keep memory low and match confocal
    PSF (XY blur >> Z blur at typical confocal sampling).
    """
    if arr.ndim == 2:
        return gaussian(arr, sigma=sigma, preserve_range=True)

    # 3D: per-slice
    out = np.empty_like(arr, dtype=np.float64)
    for z in range(arr.shape[0]):
        out[z] = gaussian(arr[z], sigma=sigma, preserve_range=True)
    return out


def background_subtract(arr: np.ndarray, radius: int = 15) -> np.ndarray:
    """White top-hat background subtraction.

    Removes slowly varying background illumination. Applied per-slice
    for 3D data to handle confocal z-dependent intensity.
    """
    selem = disk(radius)

    if arr.ndim == 2:
        return white_tophat(arr, footprint=selem)

    out = np.empty_like(arr)
    for z in range(arr.shape[0]):
        out[z] = white_tophat(arr[z], footprint=selem)
    return out


def auto_crop(arr: np.ndarray, margin: int = 5) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop away dark/gray borders added by microscope software.

    Detects uniform-intensity border strips (scale bars, gray padding)
    by checking if edge rows/columns have very low variance compared
    to the image interior.

    Parameters
    ----------
    arr : ndarray (Y, X) or (C, Y, X)
        Image array. If multi-channel, uses the mean across channels.
    margin : int
        Extra pixels to trim inward after detecting the border.

    Returns
    -------
    (cropped_array, (y0, y1, x0, x1))
        Cropped array and the crop coordinates applied.
    """
    has_channel = arr.ndim == 3
    if has_channel:
        work = arr.mean(axis=0)
    else:
        work = arr.copy()

    h, w = work.shape

    # Compute per-row and per-column variance
    row_var = np.var(work, axis=1)
    col_var = np.var(work, axis=0)

    # Threshold: rows/columns with variance < 5% of median variance are borders
    median_row_var = np.median(row_var[row_var > 0]) if np.any(row_var > 0) else 1.0
    median_col_var = np.median(col_var[col_var > 0]) if np.any(col_var > 0) else 1.0
    row_thresh = median_row_var * 0.05
    col_thresh = median_col_var * 0.05

    # Find first/last non-border row and column
    active_rows = np.where(row_var > row_thresh)[0]
    active_cols = np.where(col_var > col_thresh)[0]

    if len(active_rows) == 0 or len(active_cols) == 0:
        # No clear border detected, return unchanged
        return arr, (0, h, 0, w)

    y0 = max(0, active_rows[0] + margin)
    y1 = min(h, active_rows[-1] - margin + 1)
    x0 = max(0, active_cols[0] + margin)
    x1 = min(w, active_cols[-1] - margin + 1)

    if y1 <= y0 or x1 <= x0:
        return arr, (0, h, 0, w)

    if has_channel:
        return arr[:, y0:y1, x0:x1], (y0, y1, x0, x1)
    return arr[y0:y1, x0:x1], (y0, y1, x0, x1)
