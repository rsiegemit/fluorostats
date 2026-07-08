"""Instance-segmentation validation metrics.

For scoring a labeled prediction against a labeled ground truth by matching
objects at an IoU threshold: precision, recall, F1 (the DSB2018 / Cell Tracking
Challenge convention), plus mean matched IoU. Complements
:mod:`fluorostats.agreement` (which compares scalar measurements) by operating
on instance label images.
"""

from __future__ import annotations

import numpy as np


def match_instances(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one match of predicted to GT instances at an IoU threshold.

    Returns dict: tp, fp, fn, n_pred, n_gt, matched_ious (list).
    """
    pids = [p for p in np.unique(pred) if p != 0]
    gids = [g for g in np.unique(gt) if g != 0]
    if not gids:
        return {"tp": 0, "fp": len(pids), "fn": 0,
                "n_pred": len(pids), "n_gt": 0, "matched_ious": []}
    gt_masks = {g: (gt == g) for g in gids}
    matched_g = set()
    tp = 0
    ious = []
    # order predictions by size (larger first) for stable greedy matching
    pids.sort(key=lambda p: int((pred == p).sum()), reverse=True)
    for p in pids:
        pm = pred == p
        cand = np.unique(gt[pm])
        cand = cand[cand != 0]
        best, bg = 0.0, None
        for g in cand:
            if g in matched_g:
                continue
            gm = gt_masks[g]
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou = inter / union if union else 0.0
            if iou > best:
                best, bg = iou, g
        if best >= iou_threshold and bg is not None:
            matched_g.add(bg)
            tp += 1
            ious.append(float(best))
    fp = len(pids) - tp
    fn = len(gids) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "n_pred": len(pids),
            "n_gt": len(gids), "matched_ious": ious}


def instance_f1(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Precision, recall, F1, and mean matched IoU at a single IoU threshold."""
    m = match_instances(pred, gt, iou_threshold)
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom else 1.0
    return {
        "f1": float(f1), "precision": float(precision), "recall": float(recall),
        "mean_iou": float(np.mean(m["matched_ious"])) if m["matched_ious"] else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "n_pred": m["n_pred"], "n_gt": m["n_gt"],
    }


def average_precision(pred: np.ndarray, gt: np.ndarray,
                      thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)) -> dict:
    """Mean AP over IoU thresholds (Kaggle DSB2018 AP = TP / (TP+FP+FN))."""
    aps = []
    per = {}
    for t in thresholds:
        m = match_instances(pred, gt, t)
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        ap = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
        aps.append(ap)
        per[round(t, 2)] = float(ap)
    return {"mAP": float(np.mean(aps)), "per_threshold": per}


__all__ = ["match_instances", "instance_f1", "average_precision"]
