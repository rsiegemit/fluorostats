"""Per-object measurements: counting, sizing, and spatial distribution.

These helpers operate on a 3D binary mask, label connected components,
and report per-object size/position summaries. Useful for nuclei counts
from DAPI, cluster-size distributions, and centroid-based homogeneity
checks (complementary to intensity-based `morphometry.lateral_homogeneity`).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import local_maxima, reconstruction
from skimage.segmentation import clear_border, watershed

from .metrics_3d import density_per_mm3
from .segment import remove_small_labels
from .spatial import tile_point_density
from .stats import coefficient_of_variation, gini


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
    return remove_small_labels(labels, min_size)


def watershed_split(
    mask: np.ndarray,
    min_size: int = 0,
    min_distance: int = 5,
    footprint_size: int | None = None,
) -> tuple[np.ndarray, int]:
    """Split touching convex objects via a distance-transform watershed.

    Connected-component labeling merges touching nuclei/cells into one object
    (fluorostats' main under-count failure mode in crowded fields). This seeds
    a watershed at the h-maxima of the Euclidean distance transform (EDT), with
    ``h = min_distance / 2``: a seed is a distance peak that rises more than
    ``min_distance / 2`` above the saddle joining it to any higher peak. Bumps
    of an irregular outline and the ridge of an elongated object are shallower
    than that, so they do not split one object.

    Geometrically, two touching objects whose inscribed radius is ``r`` and
    whose neck has half-width ``w`` (both in pixels/voxels) are split when
    ``2r - 2w > min_distance``: the constriction is more than ``min_distance``
    narrower than the objects' inscribed diameter (a seed must rise *more* than
    ``min_distance / 2``; at exactly ``min_distance / 2`` the peaks merge).
    Every connected component keeps at least one seed, so objects too small to
    seed stay single objects.

    Connectivity is the same as `label_3d`'s: objects are face-connected (4 in
    2D, 6 in 3D), so objects touching only at a corner or edge never share a
    label, and with ``min_size=0`` the count is never below `label_3d`'s.

    Parameters
    ----------
    mask : np.ndarray of bool
        2D or 3D foreground. The EDT is in voxel units on every axis, so
        isotropic voxels are assumed.
    min_size : int
        Drop objects below this voxel count after splitting.
    min_distance : int
        About the smallest object radius, in pixels/voxels: the neck-depth
        criterion above (seed dynamics > ``min_distance / 2``). Larger means
        fewer splits.
    footprint_size : deprecated
        Retired (the seeding uses no footprint); accepted, by keyword or as the
        v0.8.2 fourth positional argument, with a DeprecationWarning and ignored.

    Returns (labels, n).
    """
    if footprint_size is not None:
        warnings.warn(
            "watershed_split(): 'footprint_size' no longer has an effect and is ignored",
            DeprecationWarning,
            stacklevel=2,
        )
    m = mask > 0
    if not m.any():
        return np.zeros(m.shape, int), 0
    dist = ndi.distance_transform_edt(m)
    # h-maxima as regional maxima of the reconstruction of (EDT - h) under the EDT;
    # peaks joined above their saddle by less than h merge into one plateau, one seed.
    rec = reconstruction(dist - min_distance / 2, dist, method="dilation")
    peaks = local_maxima(rec, connectivity=m.ndim) & m
    markers, _ = ndi.label(peaks, structure=np.ones((3,) * m.ndim))
    components, n_comp = ndi.label(m)
    # A plateau joined across a corner can span two face-connected components:
    # one seed per (plateau, component) piece, so no label crosses a corner.
    pieces = markers[peaks] * (n_comp + 1) + components[peaks]
    markers[peaks] = np.unique(pieces, return_inverse=True)[1] + 1
    n_seeds = int(markers.max())
    unseeded = np.setdiff1d(np.arange(1, n_comp + 1), components[markers > 0])
    if unseeded.size:  # components too small to seed: one marker each
        pos = ndi.maximum_position(dist, components, unseeded)
        markers[tuple(np.array(pos).T)] = np.arange(n_seeds + 1, n_seeds + unseeded.size + 1)
    labels = watershed(-dist, markers, mask=m)
    if min_size > 0:
        return remove_small_labels(labels, min_size)
    return labels, int(labels.max())


def count_local_maxima(
    intensity: np.ndarray,
    min_distance: int = 3,
    threshold_rel: float = 0.15,
    mask: np.ndarray | None = None,
    smooth_sigma: float = 0.0,
) -> dict:
    """Count intensity local maxima — one peak per cell (like Fiji "Find Maxima").

    Peak counting excels at counting **crowded, overlapping**
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
        Peaks must exceed this fraction of the image maximum: an absolute height
        floor, not a prominence (Fiji's Find Maxima uses prominence instead).
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
    coords = peak_local_max(img, min_distance=min_distance, threshold_abs=img.max() * threshold_rel)
    return {"count": int(len(coords)), "coords": coords}


def clear_border_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove objects touching any image border and relabel the rest 1..n.

    Useful for count endpoints where partial edge objects bias the total
    (e.g. BBBC039 ground truth excludes border nuclei). Touching objects keep
    their own labels, so a `watershed_split` result stays split. Returns (labels, n).
    """
    return remove_small_labels(clear_border(labels), 1)


def object_volumes_voxels(labels: np.ndarray) -> np.ndarray:
    """Per-object volume in voxels (excluding background label 0)."""
    if labels.max() == 0:
        return np.array([], dtype=np.int64)
    sizes = np.bincount(labels.ravel())[1:]
    return sizes[sizes > 0].astype(np.int64)  # drop gaps from non-contiguous label images


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
    """Centroid per object in voxel units, shape (N, labels.ndim): (z, y, x) for a
    3D label image, (y, x) for 2D. One row per label present, in label order;
    ids missing from a non-contiguous label image give no row."""
    ids = np.unique(labels[labels > 0])
    if ids.size == 0:
        return np.zeros((0, labels.ndim), dtype=float)
    centers = ndi.center_of_mass(labels > 0, labels, ids)
    return np.asarray(centers, dtype=float)


def object_shape_metrics(
    labels_2d: np.ndarray,
    voxel_size_um: tuple[float, float] | None = None,
) -> dict:
    """Per-object 2D shape descriptors from a labelled image.

    Quantifies object *shape* (as opposed to size/position): how elongated, how
    convex, and which way it points — e.g. whether nuclei are rounded or spread,
    and whether they share a common orientation (alignment).

    Parameters
    ----------
    labels_2d : 2D int label image (0 = background).
    voxel_size_um : optional (dy, dx); if given, axis lengths are returned in
        micrometres (mean pixel pitch; exact for isotropic pixels), else in pixels.

    Returns
    -------
    dict of 1D arrays in label order (empty arrays if no objects):
        elongation — major/minor axis ratio (1 = round, larger = more elongated).
        orientation_deg — major-axis angle in [0, 180).
        solidity — area / convex-hull area (1 = convex, lower = ragged/concave).
        major_axis, minor_axis — axis lengths (µm if voxel given, else pixels).
    """
    from skimage.measure import regionprops

    lab = np.asarray(labels_2d)
    if lab.ndim != 2:
        raise ValueError("object_shape_metrics expects a 2D label image")
    scale = float(np.mean(voxel_size_um)) if voxel_size_um is not None else 1.0
    elong, orient, solid, maj, minr = [], [], [], [], []
    for p in regionprops(lab):
        mn = p.axis_minor_length
        mj = p.axis_major_length
        elong.append(mj / mn if mn > 0 else float("nan"))
        orient.append(np.degrees(p.orientation) % 180.0)
        solid.append(float(p.solidity))
        maj.append(mj * scale)
        minr.append(mn * scale)
    return {
        "elongation": np.asarray(elong, dtype=float),
        "orientation_deg": np.asarray(orient, dtype=float),
        "solidity": np.asarray(solid, dtype=float),
        "major_axis": np.asarray(maj, dtype=float),
        "minor_axis": np.asarray(minr, dtype=float),
    }


def nearest_neighbor_stats(
    centroids: np.ndarray,
    voxel_size_um: tuple[float, ...] | None = None,
) -> dict:
    """Nearest-neighbour spacing + Clark-Evans clustering index for a point set.

    Characterises the spatial *pattern* of objects: are they clustered, randomly
    (Poisson) placed, or regularly spaced? Uses each object's distance to its
    nearest neighbour, compared to the value expected under complete spatial
    randomness (CSR) at the same density in the point bounding box.

    Parameters
    ----------
    centroids : (N, D) array of object positions (voxel coords); D is 2 or 3.
    voxel_size_um : optional per-axis size for physical distances (else pixels).

    Returns
    -------
    dict with keys:
        mean_nn_dist, median_nn_dist — nearest-neighbour distance.
        clark_evans_R — observed / CSR-expected mean NN (<1 clustered, ≈1 random,
            >1 dispersed/regular), using the 2D or 3D CSR formula for the number
            of axes with non-zero range (a constant axis is dropped); NaN if that
            number is not 2 or 3 (with a warning when it is below 2).
        pattern — "clustered" | "random" | "dispersed" from ``clark_evans_R``.
        n — number of points.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(centroids, dtype=float)
    n, d = pts.shape if pts.ndim == 2 else (0, 0)
    if n < 2:
        return {
            "mean_nn_dist": float("nan"),
            "median_nn_dist": float("nan"),
            "clark_evans_R": float("nan"),
            "pattern": "n/a",
            "n": int(n),
        }
    scale = np.asarray(voxel_size_um if voxel_size_um is not None else (1.0,) * d, float)
    xy = pts * scale
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]  # distance to nearest other point
    mean_nn = float(nn.mean())
    ranges = xy.max(0) - xy.min(0)
    # The pattern's dimension is the number of axes the points actually span: 3D
    # centroids in one z-plane are a 2D pattern and take the 2D CSR constant.
    d_eff = int((ranges > 0).sum())
    lam = n / float(np.prod(ranges[ranges > 0]))
    if d_eff == 2:
        expected = 0.5 / np.sqrt(lam)
    elif d_eff == 3:
        expected = 0.55396 / lam ** (1.0 / 3.0)
    else:
        if d_eff < 2:
            warnings.warn(
                f"nearest_neighbor_stats: the points span fewer than 2 axes "
                f"({d_eff}); clark_evans_R is NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
        expected = float("nan")
    R = mean_nn / expected if expected and expected == expected else float("nan")
    pattern = (
        "n/a" if R != R else ("clustered" if R < 0.9 else "dispersed" if R > 1.1 else "random")
    )
    return {
        "mean_nn_dist": mean_nn,
        "median_nn_dist": float(np.median(nn)),
        "clark_evans_R": float(R) if R == R else float("nan"),
        "pattern": pattern,
        "n": int(n),
    }


def object_mask_association(
    centroids: np.ndarray,
    mask: np.ndarray,
    voxel_size_um: tuple[float, ...] | None = None,
    max_dist_um: float | None = None,
) -> dict:
    """How close objects sit to a reference structure (e.g. nuclei to a network).

    Distance from each object centroid to the nearest ``True`` voxel of ``mask``.
    The last ``mask.ndim`` columns of ``centroids`` are used, so 3D ``(z, y, x)``
    centroids work against a 2D projection mask (uses ``y, x``) or a 3D mask.

    Parameters
    ----------
    centroids : (N, >=mask.ndim) array of object positions in voxel coordinates.
    mask : bool array — the reference structure to measure distance to.
    voxel_size_um : optional per-axis size for physical distances (else pixels);
        its last ``mask.ndim`` entries are used, so a (z, y, x) size suits a 2D mask.
    max_dist_um : "within" threshold for the co-localised fraction; defaults to
        one voxel step (objects effectively on the structure).

    Returns
    -------
    dict with keys: median_distance, mean_distance, frac_within, n.
    NaN / 0 counts if there are no objects or an empty mask.
    """
    m = np.asarray(mask) > 0
    pts = np.asarray(centroids, dtype=float)
    if pts.shape[0] == 0 or not m.any():
        return {
            "median_distance": float("nan"),
            "mean_distance": float("nan"),
            "frac_within": float("nan"),
            "n": int(pts.shape[0]),
        }
    sampling = tuple(voxel_size_um)[-m.ndim :] if voxel_size_um is not None else (1.0,) * m.ndim
    dt = ndi.distance_transform_edt(~m, sampling=sampling)
    idx = np.round(pts[:, -m.ndim :]).astype(int)
    for ax in range(m.ndim):
        idx[:, ax] = np.clip(idx[:, ax], 0, m.shape[ax] - 1)
    dist = dt[tuple(idx.T)]
    thr = max_dist_um if max_dist_um is not None else float(max(sampling))
    return {
        "median_distance": float(np.median(dist)),
        "mean_distance": float(np.mean(dist)),
        "frac_within": float(np.mean(dist <= thr)),
        "n": int(pts.shape[0]),
    }


def object_density_per_mm3(
    n_objects: int,
    shape_zyx: tuple[int, int, int],
    voxel_size_um: tuple[float, float, float],
) -> float:
    """Object count divided by the imaged volume in mm³ (see
    `metrics_3d.density_per_mm3`; NaN with a warning for a zero volume)."""
    return density_per_mm3(n_objects, shape_zyx, voxel_size_um)


def centroid_homogeneity(
    centroids: np.ndarray,
    shape_zyx: tuple[int, int, int],
    tiles: int = 8,
) -> dict:
    """Spatial uniformity of object centroids over an XY tile grid.

    Accepts centroids as (N, 3) in (z, y, x) voxel order (as from
    ``object_centroids``) or (N, 2) in (y, x), with ``shape_zyx`` the matching
    (Z, Y, X) or (Y, X) shape; only the last two axes are tiled. Same Gini/CV
    summary as
    `morphometry.lateral_homogeneity`, but computed on object *counts per
    tile* rather than intensity per tile.

    Returns dict with keys: centroid_gini, centroid_cv, n_objects.
    """
    n_obj = centroids.shape[0]
    if n_obj == 0:
        return {"centroid_gini": float("nan"), "centroid_cv": float("nan"), "n_objects": 0}
    counts = tile_point_density(
        centroids, (shape_zyx[-2], shape_zyx[-1]), grid=(tiles, tiles)
    ).ravel()
    return {
        "centroid_gini": float(gini(counts)),
        "centroid_cv": float(coefficient_of_variation(counts)),
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
        return {
            "angular_gini": float("nan"),
            "resultant_length": float("nan"),
            "dominant_angle_deg": float("nan"),
            "n": 0,
        }
    dy = pts[:, -2] - center[0]
    dx = pts[:, -1] - center[1]
    ang = np.arctan2(dy, dx)
    counts, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi))
    resultant = np.mean(np.exp(1j * ang))
    return {
        "angular_gini": float(gini(counts)),
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
    Points all at the centre (``r_max`` 0 from the data) are in the inner shell;
    an explicit ``r_max <= 0`` raises ValueError.
    """
    if r_max is not None and r_max <= 0:
        raise ValueError(f"radial_distribution: r_max must be positive, got {r_max}")
    pts = np.asarray(centroids, dtype=float)
    if pts.shape[0] == 0:
        return {
            "fractions": np.zeros(n_bins),
            "mean_norm_radius": float("nan"),
            "r_max": float("nan"),
            "n": 0,
        }
    r = np.hypot(pts[:, -2] - center[0], pts[:, -1] - center[1])
    rmax = float(r_max) if r_max is not None else float(r.max())
    rn = np.clip(r / rmax, 0.0, 1.0) if rmax > 0 else np.zeros_like(r)  # all at the centre
    counts, _ = np.histogram(rn, bins=n_bins, range=(0.0, 1.0))
    return {
        "fractions": counts / counts.sum(),
        "mean_norm_radius": float(rn.mean()),
        "r_max": rmax,
        "n": int(pts.shape[0]),
    }
