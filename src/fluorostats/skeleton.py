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

import numpy as np
from scipy.ndimage import label, convolve
from skimage.morphology import skeletonize


def n_junction_nodes(skel: np.ndarray) -> int:
    """Number of skeleton junction NODES (skeleton voxels with >= 3 skeleton
    neighbours), with adjacent junction voxels merged into a single node.

    Matches the "branchpoint / junction" definition used by AngioTool, REAVER,
    and Fiji's AnalyzeSkeleton — distinct from ``skeleton_metrics``'
    ``n_junctions`` (junction-to-junction *branches*). Prefer this when
    comparing against those tools or when a node count is expected.

    Note: on an unpruned skeleton of a rough mask this over-counts; run
    :func:`prune_skeleton` first (or ``skeleton_metrics(prune=True)``).
    """
    s = skel > 0
    kernel = np.ones((3,) * s.ndim, dtype=int)
    kernel[(1,) * s.ndim] = 0
    neighbours = convolve(s.astype(int), kernel, mode="constant")
    junction_vox = s & (neighbours >= 3)
    return int(label(junction_vox)[1])


def prune_skeleton(
    skel: np.ndarray,
    min_branch_length_px: float = 10.0,
    max_iter: int = 5,
) -> np.ndarray:
    """Remove short spur branches from a skeleton.

    Iteratively drops junction-to-endpoint branches shorter than
    ``min_branch_length_px`` (in voxels) and re-thins, until none remain or
    ``max_iter`` is reached. This is the standard clean-up step that skeleton
    analysers (AnalyzeSkeleton, AngioTool, REAVER) apply before counting
    branches/junctions; without it a rough segmentation yields many spurious
    short branches and junction nodes.

    Returns a new boolean skeleton; the input is not modified.
    """
    s = np.array(skel > 0)
    try:
        import skan
    except Exception:
        return s
    kernel = np.ones((3,) * s.ndim, dtype=int)
    kernel[(1,) * s.ndim] = 0
    for _ in range(max_iter):
        if s.sum() < 3:
            break
        # Protect junction voxels so removing a spur never severs a through-path.
        neighbours = convolve(s.astype(int), kernel, mode="constant")
        junctions = s & (neighbours >= 3)
        try:
            skeleton_obj = skan.Skeleton(s)
            branch_data = skan.summarize(skeleton_obj, separator="_")
        except Exception:
            break
        spurs = branch_data.index[
            (branch_data["branch_type"] == 1)
            & (branch_data["branch_distance"] < min_branch_length_px)
        ]
        if len(spurs) == 0:
            break
        remove = np.zeros_like(s)
        for i in spurs:
            coords = skeleton_obj.path_coordinates(i).astype(int)
            remove[tuple(coords.T)] = True
        remove &= ~junctions          # keep junction nodes intact
        if not remove.any():
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
    prune : bool, default False
        If True, remove short spur branches before measuring (see
        :func:`prune_skeleton`). Default False preserves the original,
        unpruned behaviour so existing results are unchanged.
    min_branch_length_um : float or None
        Spur-length threshold in micrometres (converted to voxels via the mean
        voxel size). Ignored unless ``prune=True``; defaults to ~10 voxels.

    Returns
    -------
    dict with keys:
        total_length_um, n_branches, n_junctions (junction-to-junction
        branches), n_junction_nodes (degree>=3 nodes; field-standard
        branchpoint count), mean_branch_length_um.
    """
    empty = {"total_length_um": 0.0, "n_branches": 0, "n_junctions": 0,
             "n_junction_nodes": 0, "mean_branch_length_um": 0.0}
    try:
        skel = skeletonize(mask)

        if prune and skel.sum() > 0:
            if min_branch_length_um is None:
                min_px = 10.0
            else:
                mean_vox = float(np.mean(voxel_size_um)) or 1.0
                min_px = max(1.0, min_branch_length_um / mean_vox)
            skel = prune_skeleton(skel, min_branch_length_px=min_px)

        if skel.sum() == 0:
            return dict(empty)

        import skan

        # Match spacing length to the mask dimensionality so 2D masks work too
        # (voxel_size_um is (z, y, x); a 2D image uses the trailing (y, x)).
        spacing = tuple(voxel_size_um)[-skel.ndim:]
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
    except Exception:
        return dict(empty)


__all__ = ["skeleton_metrics", "prune_skeleton", "n_junction_nodes"]
