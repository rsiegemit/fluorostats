"""Annular cross-section morphometry — lumen + wall geometry from a 2D mask.

For a tubular / vessel-like construct imaged in cross-section: a wall of
material (cells) enclosing a central lumen (void). Operates on one 2D
foreground mask (e.g. a cellular MIP thresholded to the wall) and is robust
to an off-centre or partially open ring. Pure geometry — nothing assay-specific,
so it serves any annular structure (printed tubes, vessels, spheroid shells).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .deprecation import renamed_keyword
from .stats import coefficient_of_variation


@renamed_keyword("spacing", "voxel_size_um")
def ring_morphometry(
    mask: np.ndarray,
    voxel_size_um: float | tuple[float, float] = (1.0, 1.0),
    min_lumen_frac: float = 0.005,
) -> dict:
    """Lumen + wall geometry of an annular (tube cross-section) mask.

    Parameters
    ----------
    mask : 2D bool array
        Foreground = wall material.
    voxel_size_um : (dy, dx) or float
        Physical pixel size (one number for square pixels). Areas are returned in ``dy*dx`` units and lengths
        in the same linear unit; pass ``(1, 1)`` for pixels.
    min_lumen_frac : float
        Minimum enclosed-void area, as a fraction of the outer object area, to
        count as a lumen — rejects tiny interior specks.

    Returns
    -------
    dict with keys:
        outer_area, lumen_area, wall_area   — areas
        wall_coverage_frac  — wall material / wall-annulus area, in [0, 1]
        outer_diam, inner_diam  — equivalent-circle diameters
        wall_thickness_mean, wall_thickness_cv  — local wall thickness (medial axis)
        lumen_circularity   — 4π·A/P² of the lumen (1 = perfect circle)
        concentricity       — 1 - |lumen − outer centroid| / outer radius, in [0, 1]
        lumen_present       — bool
    Geometry keys are NaN when the mask has no foreground.
    """
    m = np.asarray(mask) > 0
    if m.ndim != 2:
        raise ValueError(
            f"ring_morphometry expects a 2D (Y, X) cross-section mask, got shape {m.shape}"
        )
    dy, dx = np.broadcast_to(np.asarray(voxel_size_um, dtype=float), (2,)).tolist()
    pix_area = dy * dx
    nan = float("nan")
    empty = {
        "outer_area": nan,
        "lumen_area": nan,
        "wall_area": nan,
        "wall_coverage_frac": nan,
        "outer_diam": nan,
        "inner_diam": nan,
        "wall_thickness_mean": nan,
        "wall_thickness_cv": nan,
        "lumen_circularity": nan,
        "concentricity": nan,
        "lumen_present": False,
    }
    if not m.any():
        return empty

    outer = _outer_region(m)
    wall = m & outer
    outer_area = float(outer.sum()) * pix_area

    lumen = _lumen(m, outer, pix_area, min_lumen_frac * outer_area)
    lumen_area = float(lumen.sum()) * pix_area
    lumen_present = bool(lumen.any())

    wall_area = outer_area - lumen_area
    wall_coverage = (
        float(np.clip(wall.sum() * pix_area / wall_area, 0.0, 1.0)) if wall_area > 0 else nan
    )
    outer_diam = 2.0 * np.sqrt(outer_area / np.pi) if outer_area > 0 else nan
    inner_diam = 2.0 * np.sqrt(lumen_area / np.pi) if lumen_area > 0 else 0.0

    wt_mean, wt_cv = _wall_thickness(outer & ~lumen, dy, dx)
    lumen_circularity = _circularity(lumen) if lumen_present else nan
    concentricity = nan
    if lumen_present and outer_diam and outer_diam == outer_diam:
        concentricity = _concentricity(outer, lumen, outer_diam, dy, dx)

    return {
        "outer_area": outer_area,
        "lumen_area": lumen_area,
        "wall_area": wall_area,
        "wall_coverage_frac": wall_coverage,
        "outer_diam": float(outer_diam) if outer_diam == outer_diam else nan,
        "inner_diam": float(inner_diam),
        "wall_thickness_mean": wt_mean,
        "wall_thickness_cv": wt_cv,
        "lumen_circularity": lumen_circularity,
        "concentricity": concentricity,
        "lumen_present": lumen_present,
    }


def _outer_region(m: np.ndarray) -> np.ndarray:
    """Outer boundary: the largest filled connected component."""
    filled = ndi.binary_fill_holes(m)
    lab, n = ndi.label(filled)
    if n > 1:
        biggest = 1 + int(np.argmax(ndi.sum(np.ones_like(lab), lab, range(1, n + 1))))
        return lab == biggest
    return filled


def _lumen(m: np.ndarray, outer: np.ndarray, pix_area: float, min_area: float) -> np.ndarray:
    """Lumen: the largest enclosed void, if its area reaches ``min_area``."""
    hlab, hn = ndi.label(outer & ~m)
    lumen = np.zeros_like(m)
    if hn:
        sizes = ndi.sum(np.ones_like(hlab), hlab, range(1, hn + 1))
        idx = 1 + int(np.argmax(sizes))
        if float(sizes.max()) * pix_area >= min_area:
            lumen = hlab == idx
    return lumen


def _wall_thickness(band: np.ndarray, dy: float, dx: float) -> tuple[float, float]:
    """Local wall thickness (mean, CV): 2× the distance transform of the solid
    wall band (outer minus lumen), sampled on its medial axis."""
    from skimage.morphology import skeletonize

    dt = ndi.distance_transform_edt(band, sampling=(dy, dx))
    skel = skeletonize(band)
    if skel.any():
        th = 2.0 * dt[skel]
        cv = coefficient_of_variation(th)  # thickness >= 0: NaN only for a zero mean
        return float(th.mean()), 0.0 if np.isnan(cv) else cv
    return float("nan"), float(
        "nan"
    )  # pragma: no cover - a non-empty wall band always has a skeleton


def _circularity(lumen: np.ndarray) -> float:
    """Lumen circularity 4πA/P² (pixel space; spacing-independent for isotropic pixels)."""
    from skimage.measure import regionprops

    props = regionprops(lumen.astype(np.uint8))[0]
    perim = props.perimeter
    return float(4.0 * np.pi * props.area / (perim**2)) if perim > 0 else float("nan")


def _concentricity(outer, lumen, outer_diam: float, dy: float, dx: float) -> float:
    """1 − (lumen-centroid offset / outer radius), clipped to [0, 1]."""
    cy_o, cx_o = ndi.center_of_mass(outer)
    cy_l, cx_l = ndi.center_of_mass(lumen)
    offset = np.hypot((cy_l - cy_o) * dy, (cx_l - cx_o) * dx)
    return float(np.clip(1.0 - offset / (outer_diam / 2.0), 0.0, 1.0))


__all__ = ["ring_morphometry"]
