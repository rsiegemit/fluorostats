"""Skeleton morphometry: thinning, spur-pruning, branch/junction statistics.

Separated from ``metrics_3d`` for modularity. Built on scikit-image
``skeletonize`` (Lee-1994 thinning in 3D) and the ``skan`` branch graph, so
counts are directly comparable to Fiji's AnalyzeSkeleton.

Two junction conventions are exposed deliberately:
    - ``n_junctions``       — junction-to-junction *branches* (skan branch_type 2)
    - ``n_junction_nodes``  — degree>=3 *nodes* (AngioTool/REAVER/AnalyzeSkeleton
                              branchpoint definition)
"""

from __future__ import annotations

import warnings

import numpy as np
import skan
from scipy.ndimage import label, convolve
from skimage.morphology import skeletonize

from .deprecation import renamed_keyword


# The result of `skeleton_metrics` for an empty skeleton: the one definition of
# its keys, also used by the CLI when skeletons are skipped. Counts and length are
# a measured zero; the mean length of zero branches is undefined (NaN).
EMPTY_METRICS = {
    "total_length_um": 0.0,
    "n_branches": 0,
    "n_junctions": 0,
    "n_junction_nodes": 0,
    "mean_branch_length_um": float("nan"),
}
# The result when the skeleton could not be measured: NaN, not a measured zero.
UNMEASURED_METRICS = {k: float("nan") for k in EMPTY_METRICS}


def _neighbour_counts(s: np.ndarray) -> np.ndarray:
    """Number of foreground neighbours (full connectivity) of every voxel of ``s``."""
    kernel = np.ones((3,) * s.ndim, dtype=int)
    kernel[(1,) * s.ndim] = 0
    return convolve(s.astype(int), kernel, mode="constant")


def _has_branches(s: np.ndarray) -> bool:
    """True if some skeleton voxel has a neighbour; isolated points form no branch
    (and skan cannot build a graph without one)."""
    return bool(_neighbour_counts(s)[s].any())


def n_junction_nodes(skel: np.ndarray) -> int:
    """Number of skeleton junction NODES (skeleton voxels with >= 3 skeleton
    neighbours), with adjacent junction voxels (full connectivity: faces, edges
    and corners, as for the neighbour count) merged into a single node.

    Matches the "branchpoint / junction" definition used by AngioTool, REAVER,
    and Fiji's AnalyzeSkeleton — distinct from ``skeleton_metrics``'
    ``n_junctions`` (junction-to-junction *branches*). Prefer this when
    comparing against those tools or when a node count is expected.

    Note: on an unpruned skeleton of a rough mask this over-counts; run
    :func:`prune_skeleton` first (or ``skeleton_metrics(prune=True)``).
    """
    s = skel > 0
    junction_vox = s & (_neighbour_counts(s) >= 3)
    # merge with the same full connectivity the neighbour count uses
    return int(label(junction_vox, structure=np.ones((3,) * s.ndim))[1])


def _spurs(skeleton_obj, branch_data, junctions: np.ndarray, min_length: float) -> list:
    """Indices of the spurs (junction-to-endpoint branches shorter than
    ``min_length``) to remove: at each junction the shortest ones, but never so
    many that the junction keeps fewer than two branches, so pruning always leaves
    a through-path and never deletes a whole structure."""
    cluster = label(junctions, structure=np.ones((3,) * junctions.ndim))[0]
    coords = skeleton_obj.coordinates.astype(int)
    src = cluster[tuple(coords[branch_data["node_id_src"]].T)]
    dst = cluster[tuple(coords[branch_data["node_id_dst"]].T)]
    joining = src != dst  # a branch inside one junction cluster joins nothing
    n_at = np.bincount(np.r_[src[joining], dst[joining]], minlength=cluster.max() + 1)
    junction_end = np.where(src > 0, src, dst)
    length = branch_data["branch_distance"]
    spur = ((branch_data["branch_type"] == 1) & (length < min_length)).to_numpy()
    remove: list = []
    for j in np.unique(junction_end[spur]):
        shortest_first = length[spur & (junction_end == j)].sort_values()
        remove += list(shortest_first.index[: max(int(n_at[j]) - 2, 0)])
    return remove


@renamed_keyword("min_branch_length_px", "min_branch_length")
def prune_skeleton(
    skel: np.ndarray,
    min_branch_length: float = 10.0,
    max_iter: int = 5,
    *,
    voxel_size_um: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Remove short spur branches from a skeleton.

    Iteratively drops junction-to-endpoint branches shorter than
    ``min_branch_length`` and re-thins, until none remain or ``max_iter``
    is reached. A junction always keeps two branches (its longest short ones
    when it has fewer than two long ones), so pruning never removes the path
    through a junction or a whole structure. Branch length is measured along the path in voxels, or, when
    ``voxel_size_um`` (per-axis voxel size, one entry per ``skel`` axis) is given, in
    µm, in which case ``min_branch_length`` is a
    length in those units. This is the standard clean-up step that skeleton
    analysers (AnalyzeSkeleton, AngioTool, REAVER) apply before counting
    branches/junctions; without it a rough segmentation yields many spurious
    short branches and junction nodes.

    Returns a new boolean skeleton; the input is not modified.
    """
    s = np.array(skel > 0)
    for _ in range(max_iter):
        if s.sum() < 3 or not _has_branches(s):
            break
        # Protect junction voxels so removing a spur never severs a through-path.
        junctions = s & (_neighbour_counts(s) >= 3)
        skeleton_obj = skan.Skeleton(s, spacing=1 if voxel_size_um is None else voxel_size_um)
        branch_data = skan.summarize(skeleton_obj, separator="_")
        spurs = _spurs(skeleton_obj, branch_data, junctions, min_branch_length)
        if len(spurs) == 0:
            break
        remove = np.zeros_like(s)
        for i in spurs:
            coords = skeleton_obj.path_coordinates(i).astype(int)
            remove[tuple(coords.T)] = True
        remove &= ~junctions  # keep junction nodes intact
        if not remove.any():  # pragma: no cover  # spur endpoints are never junctions
            break
        s = s & ~remove
        s = skeletonize(s)
    return s


def skeleton_metrics(
    mask: np.ndarray,
    voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    prune: bool = False,
    min_branch_length_um: float | None = None,
) -> dict:
    """Skeletonize the mask and compute branch statistics.

    Parameters
    ----------
    mask : 2D or 3D bool array
    voxel_size_um : (vz, vy, vx)
        Voxel size in µm (a 2D mask uses the trailing ``(vy, vx)``). The default
        ``(1, 1, 1)`` is uncalibrated: lengths are then in voxel units, not µm.
    prune : bool, default False
        If True, remove short spur branches before measuring (see
        :func:`prune_skeleton`). Default False preserves the original,
        unpruned behaviour so existing results are unchanged.
    min_branch_length_um : float or None
        Spur-length threshold in micrometres, compared with each spur's length
        along its path using the per-axis voxel size. Ignored unless
        ``prune=True``; when None, the threshold is 10 voxels.

    Returns
    -------
    dict with keys:
        total_length_um, n_branches, n_junctions (junction-to-junction
        branches), n_junction_nodes (degree>=3 nodes; field-standard
        branchpoint count), mean_branch_length_um. Zeros for an empty
        skeleton (mean_branch_length_um NaN: there is no branch to average);
        all NaN, with a RuntimeWarning, if the skeleton cannot be measured.
        A ``voxel_size_um`` with fewer entries than ``mask`` has axes raises
        ValueError.
    """
    if np.ndim(voxel_size_um) != 1 or len(voxel_size_um) < np.ndim(mask):
        raise ValueError(
            f"skeleton_metrics: voxel_size_um needs one entry per mask axis "
            f"({np.ndim(mask)}), got {voxel_size_um!r}"
        )
    try:
        skel = skeletonize(mask)
        # Match spacing length to the mask dimensionality so 2D masks work too
        # (voxel_size_um is (z, y, x); a 2D image uses the trailing (y, x)).
        spacing = tuple(voxel_size_um)[-skel.ndim :]

        if prune and skel.sum() > 0:
            if min_branch_length_um is None:
                skel = prune_skeleton(skel)
            else:
                skel = prune_skeleton(skel, min_branch_length_um, voxel_size_um=spacing)

        if not _has_branches(skel):  # empty, or isolated points only
            return dict(EMPTY_METRICS)

        skeleton_obj = skan.Skeleton(skel, spacing=spacing)
        branch_data = skan.summarize(skeleton_obj, separator="_")

        total_length = float(branch_data["branch_distance"].sum())
        n_branches = len(branch_data)
        n_junctions = int((branch_data["branch_type"] == 2).sum())
        mean_branch_length = total_length / n_branches if n_branches > 0 else 0.0

        return {
            "total_length_um": total_length,
            "n_branches": n_branches,
            "n_junctions": n_junctions,
            "n_junction_nodes": n_junction_nodes(skel),
            "mean_branch_length_um": mean_branch_length,
        }
    except Exception as exc:  # any skeletonize/skan failure: surfaced, never a fake zero
        warnings.warn(
            f"skeleton_metrics failed ({type(exc).__name__}: {exc}); returning NaN (not measured).",
            RuntimeWarning,
            stacklevel=2,
        )
        return dict(UNMEASURED_METRICS)


__all__ = ["skeleton_metrics", "prune_skeleton", "n_junction_nodes"]
