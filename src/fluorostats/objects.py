"""Per-object measurements: counting, sizing, and spatial distribution.

These helpers operate on a 3D binary mask, label connected components,
and report per-object size/position summaries. Useful for nuclei counts
from DAPI, cluster-size distributions, and centroid-based homogeneity
checks (complementary to intensity-based `morphometry.lateral_homogeneity`).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def label_3d(mask: np.ndarray, min_size: int = 0) -> tuple[np.ndarray, int]:
    """Connected-component label with optional size filtering.

    Parameters
    ----------
    mask : np.ndarray of bool
        Foreground volume.
    min_size : int
        Drop components with fewer than `min_size` voxels. Use 0 to keep all.

    Returns
    -------
    labels : np.ndarray of int
        Label image with relabelled (1..N) surviving components.
    n : int
        Number of components after filtering.
    """
    labels, n = ndi.label(mask)
    if min_size <= 0 or n == 0:
        return labels, int(n)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False  # background
    if not keep.any():
        return np.zeros_like(labels), 0
    remap = np.zeros(sizes.size, dtype=labels.dtype)
    remap[keep] = np.arange(1, keep.sum() + 1, dtype=labels.dtype)
    return remap[labels], int(keep.sum())


def object_volumes_voxels(labels: np.ndarray) -> np.ndarray:
    """Per-object volume in voxels (excluding background label 0)."""
    if labels.max() == 0:
        return np.array([], dtype=np.int64)
    sizes = np.bincount(labels.ravel())
    return sizes[1:].astype(np.int64)


def equivalent_diameters_um(
    labels: np.ndarray,
    voxel_size_um: tuple[float, float, float],
) -> np.ndarray:
    """Equivalent spherical diameter per object in micrometres.

    diameter = 2 * (3V / 4π)^(1/3) where V is per-object volume in µm³.
    """
    sizes_vox = object_volumes_voxels(labels)
    if sizes_vox.size == 0:
        return np.array([], dtype=float)
    voxel_um3 = float(np.prod(voxel_size_um))
    volumes_um3 = sizes_vox.astype(float) * voxel_um3
    return 2.0 * np.cbrt(3.0 * volumes_um3 / (4.0 * np.pi))


def object_centroids(labels: np.ndarray) -> np.ndarray:
    """Centroid (z, y, x) per object, shape (N, 3) in voxel units."""
    n = int(labels.max())
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    centers = ndi.center_of_mass(labels > 0, labels, np.arange(1, n + 1))
    return np.asarray(centers, dtype=float)


def object_density_per_mm3(
    n_objects: int,
    shape_zyx: tuple[int, int, int],
    voxel_size_um: tuple[float, float, float],
) -> float:
    """Object count divided by the imaged volume in mm³."""
    vol_mm3 = float(np.prod(shape_zyx)) * float(np.prod(voxel_size_um)) * 1e-9
    if vol_mm3 == 0:
        return 0.0
    return float(n_objects / vol_mm3)


def centroid_homogeneity(
    centroids: np.ndarray,
    shape_zyx: tuple[int, int, int],
    tiles: int = 8,
) -> dict:
    """Spatial uniformity of object centroids over an XY tile grid.

    Same Gini/CV summary as `morphometry.lateral_homogeneity`, but
    computed on object *counts per tile* rather than intensity per tile.

    Returns dict with keys: centroid_gini, centroid_cv, n_objects.
    """
    n_obj = centroids.shape[0]
    if n_obj == 0:
        return {"centroid_gini": float("nan"), "centroid_cv": float("nan"), "n_objects": 0}
    _, ny, nx = shape_zyx
    ty = ny / tiles
    tx = nx / tiles
    yi = np.clip((centroids[:, 1] / ty).astype(int), 0, tiles - 1)
    xi = np.clip((centroids[:, 2] / tx).astype(int), 0, tiles - 1)
    counts = np.zeros(tiles * tiles, dtype=int)
    np.add.at(counts, yi * tiles + xi, 1)
    return {
        "centroid_gini": float(_gini(counts)),
        "centroid_cv": float(_cv(counts)),
        "n_objects": int(n_obj),
    }


# ---------------------------------------------------------------------------
# Local helpers (duplicated from morphometry to keep modules independent)
# ---------------------------------------------------------------------------

def _gini(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return float("nan")
    v = v - v.min() if v.min() < 0 else v
    if v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def _cv(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0 or v.mean() == 0:
        return float("nan")
    return float(v.std(ddof=0) / v.mean())
