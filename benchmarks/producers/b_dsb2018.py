"""Produce results/b_dsb2018.csv — fluorostats vs global thresholds on the DSB2018 test split.

StarDist's curated fluorescence subset of the 2018 Data Science Bowl (50 test
images, instance masks). fluorostats ``segment.binarize`` (Otsu or Li) +
``objects.label_3d`` against plain skimage thresholds + CC labelling; instance
F1/precision/recall@0.5 by ``validate.instance_f1``, count CCC by
``agreement.lins_ccc``. Data: ``$FLUOROSTATS_DATA/dsb2018/test/{images,masks}``.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_isodata, threshold_li, threshold_otsu, threshold_triangle, threshold_yen
from skimage.morphology import remove_small_objects

from common import DATA, RESULTS, require
from fluorostats.agreement import lins_ccc
from fluorostats.objects import label_3d
from fluorostats.segment import binarize
from fluorostats.validate import instance_f1

MIN_SIZE = 15


def seg_fluorostats(img, method):
    return label_3d(binarize(img, method=method, min_size=MIN_SIZE), min_size=0)[0]


def seg_threshold(img, thresh_fn):
    if img.max() <= img.min():   # constant image: some estimators fail; nothing to segment
        return np.zeros(img.shape, dtype=np.int32)
    # skimage 0.26 positional size arg removes objects <= its value, so MIN_SIZE - 1
    # keeps the "smaller than MIN_SIZE" cleanup used by binarize.
    mask = remove_small_objects(img > thresh_fn(img), MIN_SIZE - 1)
    return label_3d(mask, min_size=0)[0]


METHODS = {
    "fluorostats": lambda im: seg_fluorostats(im, "otsu"),
    "fluorostats-li": lambda im: seg_fluorostats(im, "li"),
    "otsu": lambda im: seg_threshold(im, threshold_otsu),
    "li": lambda im: seg_threshold(im, threshold_li),
    "isodata": lambda im: seg_threshold(im, threshold_isodata),
    "triangle": lambda im: seg_threshold(im, threshold_triangle),
    "yen": lambda im: seg_threshold(im, threshold_yen),
}


def n_instances(labels):
    return int(len(np.unique(labels)) - (1 if 0 in labels else 0))


def load_pairs():
    root = require(DATA / "dsb2018" / "test")
    pairs = []
    for img_path in sorted(glob.glob(str(root / "images" / "*.tif"))):
        mask_path = root / "masks" / Path(img_path).name
        img = tifffile.imread(img_path)
        if img.ndim != 2 or not mask_path.exists():
            print(f"skipped {Path(img_path).name}: shape {img.shape}, mask {mask_path.exists()}")
            continue
        pairs.append((img.astype(np.float64), tifffile.imread(mask_path).astype(np.int32)))
    return pairs


def main():
    acc = {m: {"f1": [], "prec": [], "rec": [], "pred": [], "gt": []} for m in METHODS}
    for img, gt in load_pairs():
        for mname, fn in METHODS.items():
            pred = fn(img)
            res = instance_f1(pred, gt, iou_threshold=0.5)
            for k, v in (("f1", res["f1"]), ("prec", res["precision"]), ("rec", res["recall"]),
                         ("pred", n_instances(pred)), ("gt", n_instances(gt))):
                acc[mname][k].append(v)
    rows = []
    for mname, d in acc.items():
        pred_c = np.array(d["pred"], dtype=float)
        gt_c = np.array(d["gt"], dtype=float)
        rows.append({"method": mname,
                     "mean_f1@0.5": round(float(np.mean(d["f1"])), 4),
                     "mean_precision": round(float(np.mean(d["prec"])), 4),
                     "mean_recall": round(float(np.mean(d["rec"])), 4),
                     "count_mae": round(float(np.mean(np.abs(pred_c - gt_c))), 3),
                     "count_ccc": round(float(lins_ccc(gt_c, pred_c)), 4),
                     "mean_pred_count": round(float(pred_c.mean()), 2),
                     "mean_gt_count": round(float(gt_c.mean()), 2)})
    rows.sort(key=lambda r: r["mean_f1@0.5"], reverse=True)
    df = pd.DataFrame(rows)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(RESULTS / "b_dsb2018.csv", index=False, lineterminator="\r\n")  # csv-module line endings of the committed table
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
