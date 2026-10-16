"""Link per-plane 2D labels into 3D objects (StarDist 3D in ``b_clustering_curve_dl``).

``stitch_planes`` follows Cellpose's ``stitch3D`` rule; ``drop_short`` removes
objects spanning fewer than ``min_planes`` z planes (the short fragments a 2D model
leaves at the defocused top and bottom of a nucleus).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def stitch_planes(planes, iou_min: float) -> np.ndarray:
    """Link per-plane 2D labels (a z stack) into 3D objects, as Cellpose's ``stitch3D``.

    Walking down z, each label takes the id of the previous plane's label it
    overlaps best when that IoU is at least ``iou_min``; otherwise it starts a new
    object. Overlaps come from one pair count per plane.
    """
    planes = np.asarray(planes)
    out = np.zeros(planes.shape, np.int32)
    next_id = 1
    for z, lab in enumerate(planes):
        lab = lab.astype(np.int64)
        ids = np.unique(lab[lab > 0])
        if ids.size == 0:
            continue
        mapping = np.zeros(int(lab.max()) + 1, np.int32)
        prev = out[z - 1].astype(np.int64) if z else np.zeros_like(lab)
        both = (lab > 0) & (prev > 0)
        if both.any():
            pairs, inter = np.unique(lab[both] * (int(prev.max()) + 1) + prev[both],
                                     return_counts=True)
            cur, old = np.divmod(pairs, int(prev.max()) + 1)
            iou = inter / (np.bincount(lab.ravel())[cur] + np.bincount(prev.ravel())[old] - inter)
            order = np.lexsort((-iou, cur))           # best overlap first for each label
            first = np.r_[True, cur[order][1:] != cur[order][:-1]]
            best = order[first]
            ok = iou[best] >= iou_min
            mapping[cur[best][ok]] = old[best][ok]
        for i in ids:
            if mapping[i] == 0:
                mapping[i] = next_id
                next_id += 1
        out[z] = mapping[lab]
    return out


def drop_short(labels: np.ndarray, min_planes: int) -> np.ndarray:
    """``labels`` with every object spanning fewer than ``min_planes`` z planes set to 0."""
    out = labels.copy()
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is not None and sl[0].stop - sl[0].start < min_planes:
            out[sl][labels[sl] == i] = 0
    return out
