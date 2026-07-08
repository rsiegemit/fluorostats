"""Tests for fluorostats.validate and objects watershed/border helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fluorostats.validate import instance_f1, match_instances, average_precision
from fluorostats.objects import watershed_split, clear_border_labels, label_3d


def _two_labeled_blobs():
    lab = np.zeros((40, 40), int)
    lab[5:15, 5:15] = 1
    lab[25:35, 25:35] = 2
    return lab


def test_instance_f1_perfect_match():
    gt = _two_labeled_blobs()
    r = instance_f1(gt, gt)
    assert r["f1"] == pytest.approx(1.0)
    assert r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0


def test_instance_f1_missing_one_object():
    gt = _two_labeled_blobs()
    pred = gt.copy(); pred[pred == 2] = 0   # miss object 2
    r = instance_f1(pred, gt)
    assert r["tp"] == 1 and r["fn"] == 1
    assert r["recall"] == pytest.approx(0.5)


def test_instance_f1_false_positive():
    gt = _two_labeled_blobs()
    pred = gt.copy(); pred[0:3, 0:3] = 3    # spurious extra object
    r = instance_f1(pred, gt)
    assert r["fp"] >= 1
    assert r["precision"] < 1.0


def test_average_precision_perfect():
    gt = _two_labeled_blobs()
    ap = average_precision(gt, gt)
    assert ap["mAP"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# objects: watershed split + clear border
# ---------------------------------------------------------------------------

def test_watershed_splits_touching_blobs():
    # two overlapping disks forming a dumbbell -> CC labeling sees 1, watershed 2
    mask = np.zeros((40, 70), bool)
    yy, xx = np.ogrid[:40, :70]
    mask |= (yy - 20) ** 2 + (xx - 25) ** 2 <= 13 ** 2
    mask |= (yy - 20) ** 2 + (xx - 45) ** 2 <= 13 ** 2
    _, n_cc = label_3d(mask)
    _, n_ws = watershed_split(mask, min_distance=5)
    assert n_cc == 1
    assert n_ws == 2


def test_clear_border_removes_edge_objects():
    lab = np.zeros((30, 30), int)
    lab[0:5, 10:15] = 1     # touches top border
    lab[12:18, 12:18] = 2   # interior
    _, n = clear_border_labels(lab)
    assert n == 1           # only the interior object survives
