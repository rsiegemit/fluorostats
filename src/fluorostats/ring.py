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


def ring_morphometry(
    mask: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
    min_lumen_frac: float = 0.005,
) -> dict:
    """Lumen + wall geometry of an annular (tube cross-section) mask.

    Parameters
    ----------
    mask : 2D bool array
        Foreground = wall material.
    spacing : (dy, dx)
        Physical pixel size. Areas are returned in ``dy*dx`` units and lengths
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
    from skimage.measure import regionprops
    from skimage.morphology import skeletonize

    m = np.asarray(mask) > 0
    dy, dx = float(spacing[0]), float(spacing[1])
    pix_area = dy * dx
    nan = float("nan")
    empty = {
        "outer_area": nan, "lumen_area": nan, "wall_area": nan,
        "wall_coverage_frac": nan, "outer_diam": nan, "inner_diam": nan,
        "wall_thickness_mean": nan, "wall_thickness_cv": nan,
        "lumen_circularity": nan, "concentricity": nan, "lumen_present": False,
    }
    if not m.any():
        return empty

    # Outer boundary = largest filled connected component.
    filled = ndi.binary_fill_holes(m)
    lab, n = ndi.label(filled)
    if n > 1:
        biggest = 1 + int(np.argmax(ndi.sum(np.ones_like(lab), lab, range(1, n + 1))))
        outer = lab == biggest
    else:
        outer = filled
    wall = m & outer
    outer_area = float(outer.sum()) * pix_area

    # Lumen = largest enclosed void above the size floor.
    holes = outer & ~m
    hlab, hn = ndi.label(holes)
    lumen = np.zeros_like(m)
    if hn:
        sizes = ndi.sum(np.ones_like(hlab), hlab, range(1, hn + 1))
        idx = 1 + int(np.argmax(sizes))
        if float(sizes.max()) * pix_area >= min_lumen_frac * outer_area:
            lumen = hlab == idx
    lumen_area = float(lumen.sum()) * pix_area
    lumen_present = bool(lumen.any())

    wall_area = outer_area - lumen_area
    wall_coverage = float(np.clip(wall.sum() * pix_area / wall_area, 0.0, 1.0)) \
        if wall_area > 0 else nan
    outer_diam = 2.0 * np.sqrt(outer_area / np.pi) if outer_area > 0 else nan
    inner_diam = 2.0 * np.sqrt(lumen_area / np.pi) if lumen_area > 0 else 0.0

    # Local wall thickness: 2× the distance transform of the solid wall band
    # (outer minus lumen), sampled on its medial axis.
    band = outer & ~lumen
    dt = ndi.distance_transform_edt(band, sampling=(dy, dx))
    skel = skeletonize(band)
    if skel.any():
        th = 2.0 * dt[skel]
        wt_mean = float(th.mean())
        wt_cv = float(th.std(ddof=0) / th.mean()) if th.mean() > 0 else 0.0
    else:  # pragma: no cover - a non-empty wall band always yields a skeleton
        wt_mean, wt_cv = nan, nan

    # Lumen circularity (pixel-space, spacing-independent for isotropic pixels).
    lumen_circularity = nan
    if lumen_present:
        props = regionprops(lumen.astype(np.uint8))[0]
        perim = props.perimeter
        lumen_circularity = float(4.0 * np.pi * props.area / (perim ** 2)) \
            if perim > 0 else nan

    # Concentricity: lumen centroid offset relative to the outer radius.
    concentricity = nan
    if lumen_present and outer_diam and outer_diam == outer_diam:
        cy_o, cx_o = ndi.center_of_mass(outer)
        cy_l, cx_l = ndi.center_of_mass(lumen)
        offset = np.hypot((cy_l - cy_o) * dy, (cx_l - cx_o) * dx)
        concentricity = float(np.clip(1.0 - offset / (outer_diam / 2.0), 0.0, 1.0))

    return {
        "outer_area": outer_area, "lumen_area": lumen_area, "wall_area": wall_area,
        "wall_coverage_frac": wall_coverage,
        "outer_diam": float(outer_diam) if outer_diam == outer_diam else nan,
        "inner_diam": float(inner_diam),
        "wall_thickness_mean": wt_mean, "wall_thickness_cv": wt_cv,
        "lumen_circularity": lumen_circularity, "concentricity": concentricity,
        "lumen_present": lumen_present,
    }


__all__ = ["ring_morphometry"]
