"""B2 3D nuclei — fluorostats vs threshold methods on BBBC024 (3D synthetic).

Public: BBBC024 (Broad Bioimage Benchmark Collection; Svoboda et al. 2009),
3D synthetic HL60 nuclei volumes with EXACT instance labels. Tests 3D nucleus
counting and instance F1 against ground truth across several citable threshold
methods (Otsu 1979, Li 1993, Isodata 1978, Triangle 1977) plus fluorostats'
connected-component and watershed-split labeling.

Runs on a subset of volumes for tractable 3D instance matching.
"""

from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import filters
from skimage.morphology import remove_small_objects

from fluorostats.objects import label_3d, watershed_split
from fluorostats.validate import instance_f1

import os
BASE = Path(os.environ.get("BBBC024_DIR", "/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/BBBC024"))
TAG = os.environ.get("BBBC024_TAG", "c00")
RES = Path(__file__).resolve().parent / "results"
N_VOL = 8
MIN_SIZE = 30

THRESHOLDS = {"Otsu_1979": filters.threshold_otsu, "Li_1993": filters.threshold_li,
              "Isodata_1978": filters.threshold_isodata,
              "Triangle_1977": filters.threshold_triangle}


def main():
    finals = sorted(glob.glob(str(BASE / "**/image-final_*.tif"), recursive=True))[:N_VOL]
    rows_acc = {m: {"f1": [], "cnt": []} for m in
                list(THRESHOLDS) + ["fluorostats(Otsu+CC)", "fluorostats(Otsu+watershed)"]}
    gt_counts = []
    for i, fp in enumerate(finals):
        idx = Path(fp).stem.split("_")[-1]
        lab_fp = Path(str(fp).replace("image-final_","image-labels_"))
        if not lab_fp.exists():
            continue
        vol = tifffile.imread(fp).astype(np.float32)
        gt = tifffile.imread(lab_fp).astype(np.int32)
        gt_counts.append(int(len(np.unique(gt)) - 1))
        for name, fn in THRESHOLDS.items():
            try:
                mask = remove_small_objects(vol > fn(vol), MIN_SIZE)
                lab, _ = ndi.label(mask)
            except Exception:
                lab = np.zeros(vol.shape, int)
            rows_acc[name]["f1"].append(instance_f1(lab, gt)["f1"])
            rows_acc[name]["cnt"].append(int(len(np.unique(lab)) - 1))
        # fluorostats CC + watershed on Otsu
        mask = remove_small_objects(vol > filters.threshold_otsu(vol), MIN_SIZE)
        cc, _ = label_3d(mask, min_size=MIN_SIZE)
        ws, _ = watershed_split(mask, min_size=MIN_SIZE, min_distance=6)
        rows_acc["fluorostats(Otsu+CC)"]["f1"].append(instance_f1(cc, gt)["f1"])
        rows_acc["fluorostats(Otsu+CC)"]["cnt"].append(int(len(np.unique(cc)) - 1))
        rows_acc["fluorostats(Otsu+watershed)"]["f1"].append(instance_f1(ws, gt)["f1"])
        rows_acc["fluorostats(Otsu+watershed)"]["cnt"].append(int(len(np.unique(ws)) - 1))
        print(f"[{i+1}/{len(finals)}] {idx} gt={gt_counts[-1]}", flush=True)

    gt_counts = np.array(gt_counts)
    rows = []
    for m, d in rows_acc.items():
        f1 = np.array(d["f1"]); cnt = np.array(d["cnt"])
        rows.append({"method": m, "mean_F1": round(f1.mean(), 3),
                     "count_MAE": round(float(np.abs(cnt - gt_counts).mean()), 2),
                     "count_bias": round(float((cnt - gt_counts).mean()), 2)})
    df = pd.DataFrame(rows).sort_values("mean_F1", ascending=False).reset_index(drop=True)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / f"b2_bbbc024_3d_{TAG}.csv", index=False)
    print(f"\n=== BBBC024 3D nuclei [{TAG}] (n={len(gt_counts)} volumes, exact GT) ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"\nGT median count/volume = {np.median(gt_counts):.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
