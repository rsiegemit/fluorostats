"""2D image metrics: area coverage, connectivity, cluster analysis."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def area_fraction(mask: np.ndarray) -> float:
    """Fraction of pixels that are foreground (cell-covered)."""
    return float(mask.sum() / mask.size)


def coverage_metrics(
    mask: np.ndarray,
    pixel_size_um: float = 1.0,
) -> dict:
    """Connectivity and cluster analysis for 2D binary masks.

    Parameters
    ----------
    mask : ndarray (Y, X), bool
        Binary segmentation mask.
    pixel_size_um : float
        Pixel size in µm (for calibrated area). Default 1.0 (uncalibrated).

    Returns
    -------
    dict with keys:
        area_fraction : float
        n_components : int
            Number of disconnected cell clusters.
        largest_component_fraction : float
            Fraction of total foreground in the largest cluster.
            Near 1.0 = one confluent sheet; low = scattered patches.
        mean_cluster_area_px : float
            Average cluster size in pixels.
        median_cluster_area_px : float
            Median cluster size in pixels (robust to outliers).
    """
    af = area_fraction(mask)
    labeled, n_comp = label(mask)

    total_fg = mask.sum()

    if n_comp == 0 or total_fg == 0:
        return {
            "area_fraction": af,
            "n_components": 0,
            "largest_component_fraction": 0.0,
            "mean_cluster_area_px": 0.0,
            "median_cluster_area_px": 0.0,
        }

    component_sizes = np.bincount(labeled.ravel())
    # Index 0 is background, skip it
    fg_sizes = component_sizes[1:]
    largest = int(fg_sizes.max())
    lcf = largest / total_fg

    return {
        "area_fraction": af,
        "n_components": n_comp,
        "largest_component_fraction": float(lcf),
        "mean_cluster_area_px": float(fg_sizes.mean()),
        "median_cluster_area_px": float(np.median(fg_sizes)),
    }
