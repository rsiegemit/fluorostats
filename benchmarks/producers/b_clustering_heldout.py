"""Produce results/b_clustering_heldout.csv — the BBBC024 crowding check on held-out volumes.

``objects.watershed_split``'s seeding (h-maxima of the distance transform,
``min_distance`` 6) was chosen on the first 5 volumes of each BBBC024 clustering
level (``b_clustering_curve``) and synthetic phantoms. This table scores the same
two fluorostats pipelines, with ``b_clustering_curve``'s settings, on volumes 6-15
of each level, which played no part in that choice: per level, the mean and the
minimum per-volume instance F1 (IoU 0.5) and the range of object counts against
the ground-truth count.
Data: ``$FLUOROSTATS_DATA/BBBC024{,_c25,_c50,_c75}``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
from skimage import filters

from b_clustering_curve import LEVELS, MIN_SIZE
from common import RAW_OTSU_CC, SMALL_OBJECT_RULE, bbbc024_volumes, raw_otsu_cc, write_table
from fluorostats.objects import watershed_split
from fluorostats.segment import remove_small
from fluorostats.validate import instance_f1

HELD_OUT = slice(5, 15)   # volumes 6-15 (1-based) of each level
WATERSHED_MIN_DISTANCE = 6   # as b_clustering_curve


def labels(vol: np.ndarray) -> dict[str, np.ndarray]:
    otsu = remove_small(vol > filters.threshold_otsu(vol), MIN_SIZE)
    return {RAW_OTSU_CC: raw_otsu_cc(vol, MIN_SIZE),
            "fluorostats_watershed": watershed_split(otsu, min_size=MIN_SIZE,
                                                     min_distance=WATERSHED_MIN_DISTANCE)[0]}


def main():
    rows = []
    for lvl, sub in LEVELS.items():
        for fp, gp in bbbc024_volumes(sub)[HELD_OUT]:
            gt = tifffile.imread(gp).astype(np.int32)
            n_gt = len(np.unique(gt[gt > 0]))
            for method, lab in labels(tifffile.imread(fp).astype(np.float32)).items():
                rows.append({"clustering": lvl, "method": method, "f1": instance_f1(lab, gt)["f1"],
                             "count": len(np.unique(lab[lab > 0])), "gt_count": n_gt})
        print(f"done {lvl}", flush=True)
    g = pd.DataFrame(rows).groupby(["method", "clustering"])
    out = pd.DataFrame({"mean_F1": g["f1"].mean().round(3), "min_F1": g["f1"].min().round(3),
                        "count_min": g["count"].min(), "count_max": g["count"].max(),
                        "gt_count_min": g["gt_count"].min(), "gt_count_max": g["gt_count"].max(),
                        "n_volumes": g.size()}).reset_index()
    write_table(out, "b_clustering_heldout",
                {"levels": LEVELS, "volumes": "6-15 of each level (1-based)", "min_size": MIN_SIZE,
                 "small_object_rule": SMALL_OBJECT_RULE,
                 "watershed_min_distance": WATERSHED_MIN_DISTANCE, "scorer": "validate.instance_f1"})
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
