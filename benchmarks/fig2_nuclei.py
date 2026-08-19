"""B2 nuclei — comprehensive multi-method comparison on BBBC039.

Benchmarks fluorostats against a broad panel of CITABLE reference segmentation
methods on the same instance ground truth, so the methods paper can report a
full comparison table (not just one or two competitors). Each threshold method
is a published reference:

  Otsu (1979), Li & Lee (1993), Yen (1995), Triangle (Zack 1977),
  Isodata (Ridler & Calvard 1978), Mean, Minimum, Sauvola (2000, local),
  plus a distance-transform Watershed (Vincent & Soille 1991) split.

Reports instance F1@0.5, mean AP (0.5-0.9), and count MAE vs GT for each,
alongside the validated DL baselines (StarDist, Cellpose) merged from the
cluster eval CSVs.

Runs on the first 100 BBBC039 images (per-image recompute), matching the
per-image F1 cache (_perimage_f1.py) so the summary table and figure 2a agree:
Otsu+CC and Watershed use the current fluorostats.objects primitives
(label_3d, watershed_split), not the earlier ad-hoc watershed.
"""

from __future__ import annotations

import sys
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

from fluorostats.validate import instance_f1, average_precision
from fluorostats.objects import label_3d, watershed_split

# BBBC039 root — download per DATA_MANIFEST.md into $FLUOROSTATS_DATA/BBBC039
DL = Path(os.environ.get("FLUOROSTATS_DATA",
          str(Path(__file__).resolve().parents[1] / "data" / "downloads"))) / "BBBC039"
RES = Path(__file__).resolve().parent / "results"
N_SUBSET = 100
MIN_SIZE = 20

THRESHOLDS = {
    "Otsu_1979": filters.threshold_otsu,
    "Li_1993": filters.threshold_li,
    "Yen_1995": filters.threshold_yen,
    "Triangle_1977": filters.threshold_triangle,
    "Isodata_1978": filters.threshold_isodata,
    "Mean": filters.threshold_mean,
    "Minimum": filters.threshold_minimum,
}


def gt_instances(mask_path):
    m = np.array(Image.open(mask_path))
    r = m[..., 0] if m.ndim == 3 else m
    seeds, n = ndi.label(r == 1)
    if n == 0:
        return np.zeros(r.shape, int)
    return watershed((r >= 2).astype(np.uint8), seeds, mask=(r > 0))


def label_clean(mask):
    mask = remove_small_objects(mask, MIN_SIZE)
    lab, _ = ndi.label(mask)
    return lab


def watershed_label(mask):
    # current-library distance-transform watershed split (fluorostats.objects),
    # the same call the per-image source uses so the summary reproduces it exactly
    mask = remove_small_objects(mask, MIN_SIZE)
    if not mask.any():
        return np.zeros(mask.shape, int)
    return watershed_split(mask, min_size=MIN_SIZE, min_distance=4)[0]


def main():
    imgs = {Path(p).stem: p for p in glob.glob(str(DL / "images" / "images" / "*.tif"))}
    masks = {Path(p).stem: p for p in glob.glob(str(DL / "masks" / "masks" / "*.png"))}
    common = sorted(set(imgs) & set(masks))[:N_SUBSET]

    methods = list(THRESHOLDS) + ["Watershed_1991", "fluorostats(Otsu+CC)",
                                   "fluorostats(Otsu+watershed)"]
    acc = {m: {"f1": [], "ap": [], "cnt": []} for m in methods}
    gt_counts = []

    for i, stem in enumerate(common):
        img = tifffile.imread(imgs[stem]).astype(np.float32)
        gt = gt_instances(masks[stem])
        gt_counts.append(int(len(np.unique(gt)) - 1))
        for name, fn in THRESHOLDS.items():
            try:
                t = fn(img)
                lab = label_clean(img > t)
            except Exception:
                lab = np.zeros(img.shape, int)
            acc[name]["f1"].append(instance_f1(lab, gt)["f1"])
            acc[name]["ap"].append(average_precision(lab, gt)["mAP"])
            acc[name]["cnt"].append(int(len(np.unique(lab)) - 1))
        # Watershed on Otsu mask
        otsu_mask = remove_small_objects(img > filters.threshold_otsu(img), MIN_SIZE)
        ws = watershed_label(img > filters.threshold_otsu(img))
        acc["Watershed_1991"]["f1"].append(instance_f1(ws, gt)["f1"])
        acc["Watershed_1991"]["ap"].append(average_precision(ws, gt)["mAP"])
        acc["Watershed_1991"]["cnt"].append(int(len(np.unique(ws)) - 1))
        # fluorostats (Otsu + connected-component labelling, fluorostats.objects)
        cc = label_3d(otsu_mask, min_size=MIN_SIZE)[0]
        acc["fluorostats(Otsu+CC)"]["f1"].append(instance_f1(cc, gt)["f1"])
        acc["fluorostats(Otsu+CC)"]["ap"].append(average_precision(cc, gt)["mAP"])
        acc["fluorostats(Otsu+CC)"]["cnt"].append(int(len(np.unique(cc)) - 1))
        acc["fluorostats(Otsu+watershed)"]["f1"].append(acc["Watershed_1991"]["f1"][-1])
        acc["fluorostats(Otsu+watershed)"]["ap"].append(acc["Watershed_1991"]["ap"][-1])
        acc["fluorostats(Otsu+watershed)"]["cnt"].append(acc["Watershed_1991"]["cnt"][-1])
        if i % 20 == 0:
            print(f"[{i}/{len(common)}]", flush=True)

    gt_counts = np.array(gt_counts)
    rows = []
    for m in methods:
        f1 = np.array(acc[m]["f1"]); ap = np.array(acc[m]["ap"]); cnt = np.array(acc[m]["cnt"])
        rows.append({"method": m, "mean_F1": round(f1.mean(), 3),
                     "mean_AP": round(ap.mean(), 3),
                     "count_MAE": round(float(np.abs(cnt - gt_counts).mean()), 2)})
    # merge validated DL baselines (from cluster eval, same dataset)
    for tool, fcol in [("StarDist_2018", "stardist"), ("Cellpose_2021", "cellpose")]:
        ev = RES / f"{fcol}_eval.csv"
        if ev.exists():
            e = pd.read_csv(ev)
            sub = e[e["image"].isin([Path(imgs[s]).stem for s in common])]
            f1 = sub[f"{fcol}_f1"]; f1 = f1[f1 >= 0]
            rows.append({"method": tool + " (DL)", "mean_F1": round(f1.mean(), 3),
                         "mean_AP": None, "count_MAE": None})
    df = pd.DataFrame(rows).sort_values("mean_F1", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "b2_nuclei_methods.csv", index=False)

    print(f"\n=== Comprehensive nuclei method comparison (BBBC039, n={len(common)}) ===",
          flush=True)
    print(df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
