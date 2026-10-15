"""Produce results/b_nuclei_size.csv — per-nucleus diameter recovery vs BBBC024 GT.

Median equivalent diameter (``objects.equivalent_diameters_um``, unit voxels)
of objects from four global thresholds and fluorostats raw Otsu+CC, against the GT
median over the first 6 BBBC024 volumes. Data: ``$FLUOROSTATS_DATA/BBBC024``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from common import (RAW_OTSU_CC, SMALL_OBJECT_RULE, YEAR_LABELS, bbbc024_volumes, raw_otsu_cc, threshold_cc,
                    thresholds, write_table)
from fluorostats.objects import equivalent_diameters_um

VOX = (1.0, 1.0, 1.0)
MIN_SIZE = 30
THRESHOLDS = thresholds(["otsu", "li", "isodata", "triangle"], YEAR_LABELS)


def main():
    gt_all = []
    per_method = {m: [] for m in list(THRESHOLDS) + [RAW_OTSU_CC]}
    for fp, gp in bbbc024_volumes(n=6):
        vol = tifffile.imread(fp).astype(np.float32)
        gt = tifffile.imread(gp).astype(np.int32)
        gt_all.extend(equivalent_diameters_um(gt, VOX).tolist())
        for name, fn in THRESHOLDS.items():
            per_method[name].extend(equivalent_diameters_um(threshold_cc(vol, fn, MIN_SIZE), VOX).tolist())
        per_method[RAW_OTSU_CC].extend(equivalent_diameters_um(raw_otsu_cc(vol, MIN_SIZE), VOX).tolist())
    gt_med = float(np.median(gt_all))
    rows = []
    for m, v in per_method.items():
        med = float(np.median(v))
        rows.append({"method": m, "median_diam_um": round(med, 2), "GT_median": round(gt_med, 2),
                     "pct_error": round(abs(med - gt_med) / gt_med * 100, 1)})
    df = pd.DataFrame(rows).sort_values("pct_error")
    write_table(df, "b_nuclei_size", {"voxel_um": VOX, "min_size": MIN_SIZE, "small_object_rule": SMALL_OBJECT_RULE, "n_volumes": 6})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
