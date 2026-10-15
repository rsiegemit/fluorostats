"""3D volumetric metrics: volume fraction, connectivity, skeleton analysis.

Also exposes field-of-view normalisation helpers (``fov_volume_mm3`` and
``*_per_mm3``). Use these whenever stacks have different voxel sizes
(digital-zoom heterogeneity); they convert per-FOV counts to absolute
densities that can be compared across acquisition settings.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import label
from skimage.measure import euler_number

# Skeleton morphometry lives in its own module (fluorostats.skeleton); it is
# re-exported here for backwards compatibility.
from .skeleton import skeleton_metrics, prune_skeleton, n_junction_nodes  # noqa: F401


def volume_fraction(mask: np.ndarray) -> float:
    """Fraction of voxels that are foreground, in [0, 1] (not a percent)."""
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
            NaN for an empty mask.
    """
    # Dual-connectivity pairing (intentional, not a mismatch): foreground components
    # are counted under 6-connectivity (scipy.ndimage.label default, face-adjacency),
    # and the Euler number under 26-connectivity. In 3D these are the DUAL pair required
    # by the Jordan-Brouwer separation theorem — using the same connectivity for both
    # gives topologically inconsistent χ. This pairing is what reproduces the analytic
    # χ on the topology phantoms to zero error (see benchmarks/results/b1_topology_phantoms.csv).
    labeled, n_comp = label(mask)
    euler = int(euler_number(mask, connectivity=3))

    # Largest component fraction
    total_fg = mask.sum()
    if total_fg == 0 or n_comp == 0:
        lcf = float("nan")                  # no component: undefined, not 0
    else:
        component_sizes: np.ndarray = np.bincount(labeled.ravel())
        # Index 0 is background, skip it
        largest = int(component_sizes[1:].max()) if len(component_sizes) > 1 else 0
        lcf = float(largest / total_fg)

    return {
        "n_components": n_comp,
        "euler_number": euler,
        "largest_component_fraction": float(lcf),
    }


# ---------------------------------------------------------------------------
# FOV-normalised densities (voxel-size invariant)
# ---------------------------------------------------------------------------

def fov_volume_mm3(
    shape_zyx: tuple[int, int, int],
    voxel_size_um: tuple[float, float, float],
) -> float:
    """Imaged volume in mm³ from voxel count and physical voxel size.

    Use to normalise count- or length-style metrics into densities that
    are comparable across stacks with different digital zoom.
    """
    return float(np.prod(shape_zyx)) * float(np.prod(voxel_size_um)) * 1e-9


def _per_unit(count: float, volume: float) -> float:
    """``count / volume``; NaN, with a warning, for a non-positive volume."""
    if volume > 0:
        return float(count / volume)
    warnings.warn(f"non-positive imaged volume ({volume} mm³); density is NaN",
                  RuntimeWarning, stacklevel=3)
    return float("nan")


def density_per_mm3(count: float, shape_zyx: tuple[int, int, int],
                    voxel_size_um: tuple[float, float, float]) -> float:
    """Convert a per-FOV count (or length in µm) to per-mm³ density.

    NaN, with a RuntimeWarning, when the imaged volume is zero (an empty shape
    or a zero voxel size).
    """
    return _per_unit(count, fov_volume_mm3(shape_zyx, voxel_size_um))


def normalise_skeleton_metrics(
    metrics: dict,
    shape_zyx: tuple[int, int, int],
    voxel_size_um: tuple[float, float, float],
) -> dict:
    """Append `length_density_um_per_mm3`, `junction_density_per_mm3`,
    `branch_density_per_mm3`, and `fov_volume_mm3`.

    Returns a new dict; the input is not mutated.
    """
    v = fov_volume_mm3(shape_zyx, voxel_size_um)
    out = dict(metrics)
    out["fov_volume_mm3"] = v
    out["length_density_um_per_mm3"] = _per_unit(metrics.get("total_length_um", 0.0), v)
    # junction DENSITY uses the same field-standard branchpoint definition as the
    # reported junction COUNT (n_junction_nodes, degree>=3 nodes — AngioTool/REAVER),
    # not skan's n_junctions (junction-to-junction branch segments), so count and
    # density describe the same quantity per unit volume.
    out["junction_density_per_mm3"] = _per_unit(metrics.get("n_junction_nodes", 0), v)
    out["branch_density_per_mm3"] = _per_unit(metrics.get("n_branches", 0), v)
    return out
