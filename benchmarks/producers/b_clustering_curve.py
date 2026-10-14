"""Produce results/b_clustering_curve.csv — instance F1 vs nuclear clustering (BBBC024).

Four global thresholds + connected components, and fluorostats CC / watershed,
scored by ``validate.instance_f1`` on the first 5 volumes of each BBBC024
clustering level (c00 = base set, c25/c50/c75 = increasing overlap).
Data: ``$FLUOROSTATS_DATA/BBBC024{,_c25,_c50,_c75}``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
from skimage import filters
from skimage.morphology import remove_small_objects

from common import RESULTS, YEAR_LABELS, bbbc024_volumes, threshold_cc, thresholds
from fluorostats.objects import label_3d, watershed_split
from fluorostats.validate import instance_f1

LEVELS = {"c00": "BBBC024", "c25": "BBBC024_c25", "c50": "BBBC024_c50", "c75": "BBBC024_c75"}
MIN_SIZE = 30
N_VOL = 5
THRESHOLDS = thresholds(["otsu", "li", "isodata", "triangle"], YEAR_LABELS)


def main():
    rows = []
    for lvl, sub in LEVELS.items():
        acc = {m: [] for m in list(THRESHOLDS) + ["fluorostats_CC", "fluorostats_watershed"]}
        for fp, gp in bbbc024_volumes(sub, N_VOL):
            vol = tifffile.imread(fp).astype(np.float32)
            gt = tifffile.imread(gp).astype(np.int32)
            for name, fn in THRESHOLDS.items():
                acc[name].append(instance_f1(threshold_cc(vol, fn, max_size=MIN_SIZE), gt)["f1"])
            otsu = remove_small_objects(vol > filters.threshold_otsu(vol), MIN_SIZE)
            acc["fluorostats_CC"].append(instance_f1(label_3d(otsu, min_size=MIN_SIZE)[0], gt)["f1"])
            ws = watershed_split(otsu, min_size=MIN_SIZE, min_distance=6)[0]
            acc["fluorostats_watershed"].append(instance_f1(ws, gt)["f1"])
        rows += [{"clustering": lvl, "method": m, "mean_F1": round(float(np.mean(v)), 3)}
                 for m, v in acc.items()]
        print(f"done {lvl}", flush=True)
    piv = pd.DataFrame(rows).pivot(index="method", columns="clustering", values="mean_F1")
    piv.to_csv(RESULTS / "b_clustering_curve.csv")
    print(piv.to_string())


if __name__ == "__main__":
    main()
