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


def watershed_split(
    mask: np.ndarray,
    min_size: int = 0,
    min_distance: int = 5,
    footprint_size: int = 3,
) -> tuple[np.ndarray, int]:
    """Split touching convex objects via a distance-transform watershed.

    Connected-component labeling merges touching nuclei/cells into one object
    (fluorostats' main under-count failure mode in crowded fields). This seeds
    a watershed at local maxima of the distance transform to separate them —
    the classical fix, no training required.

    Parameters
    ----------
    mask : np.ndarray of bool
    min_size : int
        Drop objects below this voxel count after splitting.
    min_distance : int
        Distance-transform value floor for seeds: only voxels with
        ``dist > min_distance`` can seed (≈ min object radius, voxels);
        larger → fewer seeds.
    footprint_size : int
        Size of the local-maximum neighbourhood.

    Returns (labels, n).
    """
    from scipy.ndimage import distance_transform_edt, maximum_filter
    from skimage.segmentation import watershed as _watershed

    m = mask > 0
    if not m.any():
        return np.zeros(m.shape, int), 0
    dist = distance_transform_edt(m)
    fp = np.ones((footprint_size,) * m.ndim, dtype=bool)
    local_max = (maximum_filter(dist, footprint=fp) == dist) & (dist > min_distance)
    markers, _ = ndi.label(local_max)
    labels = _watershed(-dist, markers, mask=m)
    if min_size > 0:
        labels, n = label_3d(labels > 0, min_size=0)  # relabel contiguous
        # re-apply size filter on the watershed labels
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_size
        keep[0] = False
        if not keep.any():
            return np.zeros_like(labels), 0
        remap = np.zeros(sizes.size, dtype=labels.dtype)
        remap[keep] = np.arange(1, keep.sum() + 1, dtype=labels.dtype)
        return remap[labels], int(keep.sum())
    return labels, int(labels.max())


def count_local_maxima(
    intensity: np.ndarray,
    min_distance: int = 3,
    threshold_rel: float = 0.15,
    mask: np.ndarray | None = None,
    smooth_sigma: float = 0.0,
) -> dict:
    """Count intensity local maxima — one peak per cell (Fiji "Find Maxima").

    Prominence-based peak counting excels at counting **crowded, overlapping**
    cells that each present a single intensity peak (realistic gaussian-like
    staining): area fraction and connected-component labeling under-count when
    cells touch (the merged-blob failure mode), while one peak still marks each
    cell. No training required.

    NOT a universal winner — choose by regime (see ``objects.label_3d`` for
    connected components and ``watershed_split`` for touching convex cells):

    - Over-counts **flat-topped / saturated** cells (a plateau is many equal
      maxima) and **noisy** images (every bump is a peak). Set ``smooth_sigma``
      (~cell radius / 1.5) to collapse each cell to one peak first.
    - For **well-separated** cells, connected-component counting is exact and far
      more noise-robust.
    - Answers "how many objects", NOT "what area fraction" — use
      ``metrics_2d.area_fraction`` for coverage questions.

    Parameters
    ----------
    intensity : np.ndarray
        Raw (or denoised) intensity image/volume — NOT a binary mask.
    min_distance : int
        Minimum separation (voxels) between peaks; larger = fewer, merges close cells.
    threshold_rel : float
        Peaks must exceed this fraction of the image maximum (prominence floor).
    mask : np.ndarray, optional
        Restrict peak search to this foreground region.
    smooth_sigma : float
        If > 0, gaussian-smooth before peak finding (suppresses plateau/noise
        over-counting). Recommended ~ cell radius / 1.5 for flat or noisy cells.

    Returns dict: {'count', 'coords'} where coords is (n, ndim).
    """
    from skimage.feature import peak_local_max

    img = np.asarray(intensity, dtype=float)
    if smooth_sigma > 0:
        img = ndi.gaussian_filter(img, smooth_sigma)
    if mask is not None:
        img = np.where(np.asarray(mask) > 0, img, 0.0)
    if img.size == 0 or img.max() <= 0:
        return {"count": 0, "coords": np.empty((0, img.ndim), dtype=int)}
    coords = peak_local_max(
        img, min_distance=min_distance, threshold_abs=img.max() * threshold_rel
    )
    return {"count": int(len(coords)), "coords": coords}


def clear_border_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove objects touching any image border and relabel.

    Useful for count endpoints where partial edge objects bias the total
    (e.g. BBBC039 ground truth excludes border nuclei). Returns (labels, n).
    """
    from skimage.segmentation import clear_border
    cleared = clear_border(labels)
    return label_3d(cleared > 0, min_size=0)


def object_volumes_voxels(labels: np.ndarray) -> np.ndarray:
    """Per-object volume in voxels (excluding background label 0)."""
    if labels.max() == 0:
        return np.array([], dtype=np.int64)
    sizes = np.bincount(labels.ravel())[1:]
    return sizes[sizes > 0].astype(np.int64)   # drop gaps from non-contiguous label images


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

    Expects centroids as (N, 3) in (z, y, x) voxel order (as from
    ``object_centroids``). Same Gini/CV summary as
    `morphometry.lateral_homogeneity`, but computed on object *counts per
    tile* rather than intensity per tile.

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


def angular_homogeneity(
    centroids: np.ndarray,
    center: tuple[float, float],
    bins: int = 36,
) -> dict:
    """Circumferential uniformity of points around a centre (polar angle).

    Complements `centroid_homogeneity` (Cartesian tiles) for annular geometry:
    are objects spread evenly around the ring or clustered on one side? Uses
    the last two columns of ``centroids`` as (y, x); ``center`` is (cy, cx).

    Returns dict with keys:
        angular_gini — 0 when counts are even across all angle bins, → 1 when
            concentrated in a narrow arc.
        resultant_length — circular concentration |mean(e^{iθ})|, 0 (uniform)
            to 1 (all one direction).
        dominant_angle_deg — direction of the mean vector, in [0, 360).
        n — number of points.
    """
    pts = np.asarray(centroids, dtype=float)
    if pts.shape[0] == 0:
        return {"angular_gini": float("nan"), "resultant_length": float("nan"),
                "dominant_angle_deg": float("nan"), "n": 0}
    dy = pts[:, -2] - center[0]
    dx = pts[:, -1] - center[1]
    ang = np.arctan2(dy, dx)
    counts, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi))
    resultant = np.mean(np.exp(1j * ang))
    return {
        "angular_gini": float(_gini(counts)),
        "resultant_length": float(np.abs(resultant)),
        "dominant_angle_deg": float(np.degrees(np.angle(resultant)) % 360.0),
        "n": int(pts.shape[0]),
    }


def radial_distribution(
    centroids: np.ndarray,
    center: tuple[float, float],
    n_bins: int = 8,
    r_max: float | None = None,
) -> dict:
    """Fraction of points per normalised-radius shell from a centre.

    Answers "do objects sit near the lumen edge, the outer edge, or evenly
    through the wall?" Uses the last two columns of ``centroids`` as (y, x).

    Returns dict with keys:
        fractions — length-``n_bins`` array summing to 1 (inner→outer shells of
            equal normalised-radius width).
        mean_norm_radius — mean radius normalised to ``r_max`` (0 = centre).
        r_max — the radius used for normalisation.
        n — number of points.
    """
    pts = np.asarray(centroids, dtype=float)
    if pts.shape[0] == 0:
        return {"fractions": np.zeros(n_bins), "mean_norm_radius": float("nan"),
                "r_max": float("nan"), "n": 0}
    r = np.hypot(pts[:, -2] - center[0], pts[:, -1] - center[1])
    rmax = float(r_max) if r_max is not None else float(r.max())
    if rmax <= 0:
        return {"fractions": np.zeros(n_bins), "mean_norm_radius": 0.0,
                "r_max": rmax, "n": int(pts.shape[0])}
    rn = np.clip(r / rmax, 0.0, 1.0)
    counts, _ = np.histogram(rn, bins=n_bins, range=(0.0, 1.0))
    return {
        "fractions": counts / counts.sum(),
        "mean_norm_radius": float(rn.mean()),
        "r_max": rmax,
        "n": int(pts.shape[0]),
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
