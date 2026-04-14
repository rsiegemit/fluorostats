"""3D volumetric metrics: volume fraction, connectivity, skeleton analysis."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from skimage.measure import euler_number
from skimage.morphology import skeletonize


def volume_fraction(mask: np.ndarray) -> float:
    """Fraction of voxels that are foreground."""
    return float(mask.sum() / mask.size)


def connectivity_metrics(mask: np.ndarray) -> dict:
    """Connected component and topological metrics.

    Returns
    -------
    dict with keys:
        n_components : int
            Number of connected components.
        euler_number : int
            Euler characteristic (connectivity=3 for 3D).
            More negative = more loops/handles = more interconnected.
        largest_component_fraction : float
            Fraction of total foreground volume in the largest component.
            Near 1.0 = one dominant connected network; low = fragmented.
    """
    labeled, n_comp = label(mask)
    euler = int(euler_number(mask, connectivity=3))

    # Largest component fraction
    total_fg = mask.sum()
    if total_fg == 0 or n_comp == 0:
        lcf = 0.0
    else:
        component_sizes = np.bincount(labeled.ravel())
        # Index 0 is background, skip it
        largest = int(component_sizes[1:].max()) if len(component_sizes) > 1 else 0
        lcf = largest / total_fg

    return {
        "n_components": n_comp,
        "euler_number": euler,
        "largest_component_fraction": float(lcf),
    }


def skeleton_metrics(
    mask: np.ndarray,
    voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Skeletonize the mask and compute branch statistics.

    Returns
    -------
    dict with keys:
        total_length_um : float
        n_branches : int
        n_junctions : int
    """
    try:
        skel = skeletonize(mask)

        if skel.sum() == 0:
            return {"total_length_um": 0.0, "n_branches": 0, "n_junctions": 0}

        import skan

        skeleton_obj = skan.Skeleton(skel, spacing=voxel_size_um)
        branch_data = skan.summarize(skeleton_obj, separator="_")

        total_length = float(branch_data["branch_distance"].sum())
        n_branches = len(branch_data)

        # Count junctions (degree > 2 endpoints)
        n_junctions = int(
            (branch_data["branch_type"] == 2).sum()  # junction-to-junction
        )

        mean_branch_length = total_length / n_branches if n_branches > 0 else 0.0

        return {
            "total_length_um": total_length,
            "n_branches": n_branches,
            "n_junctions": n_junctions,
            "mean_branch_length_um": mean_branch_length,
        }
    except Exception:
        # Degenerate skeleton — return zeros rather than crashing
        return {
            "total_length_um": 0.0,
            "n_branches": 0,
            "n_junctions": 0,
            "mean_branch_length_um": 0.0,
        }
