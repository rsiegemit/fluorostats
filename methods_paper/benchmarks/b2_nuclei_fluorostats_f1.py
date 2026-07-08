"""fluorostats instance-F1 on BBBC039 — apples-to-apples with the DL tools.

Labels fluorostats' segmentation into instances and scores F1@IoU0.5 against
the same reconstructed GT instances used for Cellpose/StarDist on the cluster.
This completes the honest comparison: fluorostats is expected to trail the DL
instance segmenters on F1 (connected-component labeling merges touching nuclei)
while remaining competitive on raw count. Also reports a border-excluded count
to probe whether the DL tools' positive count bias is a border-nuclei artifact.
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
from skimage.segmentation import watershed, clear_border

from fluorostats.preprocess import denoise
from fluorostats.segment import binarize
from fluorostats.objects import label_3d

DL = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/"
          "downloads/BBBC039")
RES = Path(__file__).resolve().parent / "results"
SEG = dict(method="otsu", threshold_scale=1.0, sigma=1.0, min_size=20)


def gt_instances(mask_path: Path):
    m = np.array(Image.open(mask_path))
    r = m[..., 0] if m.ndim == 3 else m
    interior = (r == 1); boundary = (r >= 2)
    seeds, n = ndi.label(interior)
    if n == 0:
        return np.zeros(r.shape, int)
    return watershed(boundary.astype(np.uint8), seeds, mask=(r > 0))


def f1_at(pred, gt, thr=0.5):
    pids = [p for p in np.unique(pred) if p != 0]
    gids = [g for g in np.unique(gt) if g != 0]
    if not gids:
        return (1.0 if not pids else 0.0)
    gt_masks = {g: (gt == g) for g in gids}
    matched = set(); tp = 0
    for p in pids:
        pm = pred == p
        cand = np.unique(gt[pm]); cand = cand[cand != 0]
        best = 0; bg = None
        for g in cand:
            gm = gt_masks[g]
            inter = (pm & gm).sum(); union = (pm | gm).sum()
            iou = inter / union if union else 0
            if iou > best:
                best = iou; bg = g
        if best >= thr and bg not in matched:
            matched.add(bg); tp += 1
    fp = len(pids) - tp; fn = len(gids) - tp
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def fs_labels(img):
    sm = denoise(img.astype(np.float32), sigma=SEG["sigma"])
    mask = binarize(sm, method=SEG["method"],
                    threshold_scale=SEG["threshold_scale"], min_size=SEG["min_size"])
    lbl, _ = label_3d(mask, min_size=SEG["min_size"])
    return lbl


def main():
    imgs = {Path(p).stem: p for p in glob.glob(str(DL / "images" / "images" / "*.tif"))}
    masks = {Path(p).stem: p for p in glob.glob(str(DL / "masks" / "masks" / "*.png"))}
    common = sorted(set(imgs) & set(masks))
    rows = []
    for i, stem in enumerate(common):
        img = tifffile.imread(imgs[stem])
        pred = fs_labels(img)
        gt = gt_instances(Path(masks[stem]))
        f1 = f1_at(pred, gt)
        # border-excluded counts (both pred and GT) to test the DL bias theory
        rows.append({
            "image": stem,
            "fs_f1": round(f1, 4),
            "fs_count": int(len(np.unique(pred)) - 1),
            "gt_count": int(len(np.unique(gt)) - 1),
            "fs_count_noborder": int(len(np.unique(clear_border(pred))) - 1),
            "gt_count_noborder": int(len(np.unique(clear_border(gt))) - 1),
        })
        if i % 40 == 0:
            print(f"[{i}/{len(common)}] {stem[:18]} f1={f1:.3f}", flush=True)
    df = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "b2_nuclei_fluorostats_f1.csv", index=False)

    print(f"\nfluorostats mean instance F1@0.5 = {df['fs_f1'].mean():.3f}", flush=True)
    print("(compare: StarDist 0.871, Cellpose 0.862 — DL wins on instance F1)", flush=True)
    # border-effect check
    bias_full = (df["fs_count"] - df["gt_count"]).mean()
    bias_nb = (df["fs_count_noborder"] - df["gt_count_noborder"]).mean()
    print(f"\nfluorostats count bias vs GT: full={bias_full:+.2f}, "
          f"border-excluded={bias_nb:+.2f}", flush=True)
    print(f"GT count drops {df['gt_count'].mean():.1f} -> "
          f"{df['gt_count_noborder'].mean():.1f} when border nuclei removed "
          f"(BBBC039 GT excludes some border objects — likely source of the DL "
          f"tools' positive count bias).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
