"""B2 nuclei — fluorostats variants using the new v0.5 capabilities.

Compares three fluorostats configurations on BBBC039 against instance GT:
  1. plain connected-component labeling (label_3d)          [baseline]
  2. + watershed_split (separate touching nuclei)           [v0.5]
  3. + clear_border_labels (drop edge objects)              [v0.5]

Reports instance F1@0.5, mean AP (0.5-0.9), and count MAE for each, plus the
validated DL baselines for context (StarDist 0.871, Cellpose 0.862). Measures
whether the new capabilities help on this (mostly well-separated) dataset.
"""

from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from fluorostats.preprocess import denoise
from fluorostats.segment import binarize
from fluorostats.objects import label_3d, watershed_split, clear_border_labels
from fluorostats.validate import instance_f1, average_precision

DL = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/"
          "downloads/BBBC039")
RES = Path(__file__).resolve().parent / "results"
SEG = dict(method="otsu", threshold_scale=1.0, sigma=1.0, min_size=20)


def gt_instances(mask_path):
    m = np.array(Image.open(mask_path))
    r = m[..., 0] if m.ndim == 3 else m
    seeds, n = ndi.label(r == 1)
    if n == 0:
        return np.zeros(r.shape, int)
    return watershed((r >= 2).astype(np.uint8), seeds, mask=(r > 0))


def fs_mask(img):
    sm = denoise(img.astype(np.float32), sigma=SEG["sigma"])
    return binarize(sm, method=SEG["method"],
                    threshold_scale=SEG["threshold_scale"], min_size=SEG["min_size"])


def main():
    imgs = {Path(p).stem: p for p in glob.glob(str(DL / "images" / "images" / "*.tif"))}
    masks = {Path(p).stem: p for p in glob.glob(str(DL / "masks" / "masks" / "*.png"))}
    common = sorted(set(imgs) & set(masks))

    variants = {"cc": [], "cc+watershed": [], "cc+border": []}
    counts = {k: [] for k in variants}
    gt_counts = []
    aps = {k: [] for k in variants}

    for i, stem in enumerate(common):
        img = tifffile.imread(imgs[stem])
        gt = gt_instances(masks[stem])
        gt_counts.append(int(len(np.unique(gt)) - 1))
        mask = fs_mask(img)

        cc, _ = label_3d(mask, min_size=SEG["min_size"])
        ws, _ = watershed_split(mask, min_size=SEG["min_size"], min_distance=4)
        cb, _ = clear_border_labels(cc)

        for name, lab in [("cc", cc), ("cc+watershed", ws), ("cc+border", cb)]:
            variants[name].append(instance_f1(lab, gt)["f1"])
            counts[name].append(int(len(np.unique(lab)) - 1))
            aps[name].append(average_precision(lab, gt)["mAP"])
        if i % 40 == 0:
            print(f"[{i}/{len(common)}] {stem[:16]}", flush=True)

    gt_counts = np.array(gt_counts)
    rows = []
    for name in variants:
        f1 = np.array(variants[name])
        ap = np.array(aps[name])
        cnt = np.array(counts[name])
        rows.append({
            "variant": name,
            "mean_F1": round(float(f1.mean()), 3),
            "mean_AP_0.5-0.9": round(float(ap.mean()), 3),
            "count_MAE": round(float(np.abs(cnt - gt_counts).mean()), 2),
            "count_bias": round(float((cnt - gt_counts).mean()), 2),
        })
    df = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "b2_nuclei_variants.csv", index=False)

    print("\n=== fluorostats variants on BBBC039 (n=200) vs instance GT ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print("\nDL baselines (validated): StarDist F1 0.871 / Cellpose F1 0.862", flush=True)
    print("Published StarDist mAP(0.5) context = 0.864.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
