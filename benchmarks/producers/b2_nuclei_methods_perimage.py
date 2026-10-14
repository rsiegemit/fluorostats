"""Produce results/b2_nuclei_methods_perimage.csv — per-image instance F1 on BBBC039.

First 100 BBBC039 images (sorted): seven global thresholds + connected
components, fluorostats raw Otsu+CC (``common.raw_otsu_cc``) and distance-transform
watershed (``objects.watershed_split``), each scored by ``validate.instance_f1``
at IoU 0.5 against the reconstructed instance GT. Figure 2 plots this table.
Data: ``$FLUOROSTATS_DATA/BBBC039``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from skimage import filters
from skimage.morphology import remove_small_objects

from common import (RAW_OTSU_CC, bbbc039_gt_instances, bbbc039_pairs, raw_otsu_cc, threshold_cc, thresholds,
                    write_table)
from fluorostats.objects import watershed_split
from fluorostats.validate import instance_f1

N_IMAGES = 100
MIN_SIZE = 20
THRESHOLDS = thresholds(
    ["otsu", "li", "isodata", "triangle", "yen", "mean", "minimum"],
    {"otsu": "Otsu (1979)", "li": "Li (1993)", "isodata": "Isodata (1978)", "triangle": "Triangle (1977)",
     "yen": "Yen (1995)", "mean": "Mean", "minimum": "Minimum"})


def threshold_f1(im, fn, gt):
    try:
        lab = threshold_cc(im, fn, max_size=MIN_SIZE)
    except RuntimeError:
        # threshold_minimum cannot find two histogram peaks on some images:
        # the method produces no segmentation, scored as F1 = 0 (ERR-2: recorded).
        return 0.0
    return instance_f1(lab, gt)["f1"]


def main():
    rows = []
    for stem, img_path, mask_path in bbbc039_pairs()[:N_IMAGES]:
        gt = bbbc039_gt_instances(mask_path)
        im = np.asarray(Image.open(img_path)).astype(np.float32)
        rec = {"image": stem, **{name: threshold_f1(im, fn, gt) for name, fn in THRESHOLDS.items()}}
        otsu = remove_small_objects(im > filters.threshold_otsu(im), MIN_SIZE)
        rec[RAW_OTSU_CC] = instance_f1(raw_otsu_cc(im, MIN_SIZE), gt)["f1"]
        rec["Watershed (1991)"] = instance_f1(
            watershed_split(otsu, min_size=MIN_SIZE, min_distance=4)[0], gt)["f1"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    write_table(df, "b2_nuclei_methods_perimage", {"n_images": N_IMAGES, "min_size": MIN_SIZE, "iou_threshold": 0.5, "watershed_min_distance": 4})
    print(df.drop(columns=["image"]).mean().round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
