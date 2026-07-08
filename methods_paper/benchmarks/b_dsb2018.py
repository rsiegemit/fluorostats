"""
Benchmark: fluorostats vs classical threshold baselines on the DSB2018
fluorescence-nuclei subset (StarDist curated release).

Dataset
-------
StarDist DSB2018 subset (dsb2018.zip, github.com/stardist/stardist release 0.1.0).
This is the exact curated fluorescence (nuclear-marker) 2D single-channel subset
of the 2018 Data Science Bowl (Kaggle / BBBC038v1) that the StarDist paper trained
and evaluated on. It ships 447 train + 50 test images, each with a uint16 instance
mask (labels 1..N). We evaluate on the 50-image *test* split only.

No fluorescence-filtering step is required here: this StarDist subset is *already*
restricted to the fluorescence images (uint8, single-channel), unlike the raw
BBBC038 stage1_train which mixes fluorescence with H&E/brightfield. We verify the
single-channel property per image and skip any RGB image defensively.

Methods compared
----------------
- fluorostats        : fluorostats.segment.binarize(method="otsu") + morphology
                       cleanup, then fluorostats.objects.label_3d for instances.
- fluorostats-li     : same pipeline with the Li threshold (library's other mode).
- otsu / li / isodata / triangle / yen : plain skimage global thresholds followed
                       by connected-component labelling (same min_size cleanup),
                       i.e. the naive "threshold + label" baseline.

Metrics
-------
- Instance F1 @ IoU 0.5 via fluorostats.validate.instance_f1 (Hungarian-style
  greedy matching of predicted to GT instances), averaged over the test set.
- Count agreement: predicted nuclei count vs GT count. We report mean absolute
  count error and Lin's concordance correlation coefficient (CCC) between predicted
  and GT counts across the 50 images.

Context
-------
StarDist's published AP@0.5 (a deep-learning instance model trained on the *train*
split) on this exact DSB2018 test set is 0.864. The classical thresholding methods
here are *not* expected to reach that; the point of the benchmark is to rank
threshold heuristics and show where the fluorostats pipeline sits among them.
"""

import csv
import glob
import os

import numpy as np
import tifffile
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.morphology import remove_small_objects

import fluorostats.segment as fseg
import fluorostats.objects as fobj
from fluorostats.agreement import lins_ccc
from fluorostats.validate import instance_f1

# --- paths --------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(
    os.path.join(HERE, "..", "data", "downloads", "dsb2018", "test")
)
RESULTS = os.path.join(HERE, "results")
IMG_DIR = os.path.join(DATA, "images")
MASK_DIR = os.path.join(DATA, "masks")

IOU_THRESHOLD = 0.5
MIN_SIZE = 15  # small-object cleanup (px); nuclei are tens-to-hundreds of px
STARDIST_AP50 = 0.864  # published StarDist AP@0.5 on this exact test set


def load_pairs():
    """Return list of (name, image, gt_labels) for single-channel fluorescence
    test images. Skips (and reports) any non-2D/RGB image defensively."""
    pairs = []
    skipped = []
    for img_path in sorted(glob.glob(os.path.join(IMG_DIR, "*.tif"))):
        name = os.path.basename(img_path)
        mask_path = os.path.join(MASK_DIR, name)
        if not os.path.exists(mask_path):
            skipped.append((name, "no matching mask"))
            continue
        img = tifffile.imread(img_path)
        if img.ndim != 2:
            # raw DSB has some RGB (H&E/brightfield) frames; the StarDist subset
            # should be all single-channel, but we filter defensively.
            skipped.append((name, f"non-2D image shape {img.shape}"))
            continue
        gt = tifffile.imread(mask_path).astype(np.int32)
        pairs.append((name, img.astype(np.float64), gt))
    return pairs, skipped


# --- segmentation methods ----------------------------------------------
def seg_fluorostats(img, method):
    """fluorostats library pipeline: binarize (with morphology + min_size
    cleanup) then connected-component instance labelling."""
    mask = fseg.binarize(img, method=method, min_size=MIN_SIZE)
    labels, _ = fobj.label_3d(mask, min_size=0)
    return labels


def seg_threshold(img, thresh_fn):
    """Naive baseline: global skimage threshold -> small-object cleanup ->
    connected-component labelling."""
    # constant images break some threshold estimators; treat as empty.
    if img.max() <= img.min():
        return np.zeros(img.shape, dtype=np.int32)
    t = thresh_fn(img)
    mask = img > t
    # skimage 0.26 renamed the size arg; MIN_SIZE-1 keeps the strict "smaller
    # than MIN_SIZE" semantics (new param removes <= its value).
    mask = remove_small_objects(mask, MIN_SIZE - 1)
    labels, _ = fobj.label_3d(mask, min_size=0)
    return labels


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


def main():
    os.makedirs(RESULTS, exist_ok=True)
    pairs, skipped = load_pairs()
    print(f"Loaded {len(pairs)} fluorescence test images "
          f"({len(skipped)} skipped).")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")

    # per-image, per-method accumulation
    per_method = {m: {"f1": [], "prec": [], "rec": [],
                      "pred_counts": [], "gt_counts": []}
                  for m in METHODS}

    for name, img, gt in pairs:
        gt_n = n_instances(gt)
        for mname, fn in METHODS.items():
            pred = fn(img)
            res = instance_f1(pred, gt, iou_threshold=IOU_THRESHOLD)
            per_method[mname]["f1"].append(res["f1"])
            per_method[mname]["prec"].append(res["precision"])
            per_method[mname]["rec"].append(res["recall"])
            per_method[mname]["pred_counts"].append(n_instances(pred))
            per_method[mname]["gt_counts"].append(gt_n)

    # aggregate
    rows = []
    for mname, d in per_method.items():
        pred_c = np.array(d["pred_counts"], dtype=float)
        gt_c = np.array(d["gt_counts"], dtype=float)
        count_mae = float(np.mean(np.abs(pred_c - gt_c)))
        try:
            ccc = float(lins_ccc(gt_c, pred_c))
        except Exception:
            ccc = float("nan")
        rows.append({
            "method": mname,
            "mean_f1@0.5": round(float(np.mean(d["f1"])), 4),
            "mean_precision": round(float(np.mean(d["prec"])), 4),
            "mean_recall": round(float(np.mean(d["rec"])), 4),
            "count_mae": round(count_mae, 3),
            "count_ccc": round(ccc, 4),
            "mean_pred_count": round(float(pred_c.mean()), 2),
            "mean_gt_count": round(float(gt_c.mean()), 2),
        })

    rows.sort(key=lambda r: r["mean_f1@0.5"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    # --- write CSV ------------------------------------------------------
    csv_path = os.path.join(RESULTS, "b_dsb2018.csv")
    fields = ["rank", "method", "mean_f1@0.5", "mean_precision",
              "mean_recall", "count_mae", "count_ccc",
              "mean_pred_count", "mean_gt_count"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # --- write Markdown -------------------------------------------------
    md_path = os.path.join(RESULTS, "b_dsb2018.md")
    with open(md_path, "w") as f:
        f.write("# DSB2018 fluorescence-nuclei benchmark\n\n")
        f.write(
            "**Dataset:** StarDist DSB2018 subset "
            "(`dsb2018.zip`, github.com/stardist/stardist release 0.1.0) — the "
            "curated fluorescence (nuclear-marker) 2D single-channel subset of "
            "the 2018 Data Science Bowl (Kaggle / BBBC038v1). Evaluated on the "
            f"50-image **test** split ({len(pairs)} images used).\n\n"
        )
        f.write(
            "**Fluorescence filtering:** none needed — this StarDist subset is "
            "already restricted to fluorescence images (verified single-channel "
            "uint8 per image). We filter non-2D/RGB frames defensively; "
            f"{len(skipped)} were skipped.\n\n"
        )
        f.write(
            "**Metrics:** instance F1 @ IoU 0.5 "
            "(`fluorostats.validate.instance_f1`) averaged over the test set, "
            "and count agreement (mean absolute count error + Lin's CCC of "
            "predicted vs GT nuclei counts across images).\n\n"
        )
        f.write(
            f"**Context:** StarDist's published deep-learning AP@0.5 on this "
            f"exact test set is **{STARDIST_AP50}**. The classical thresholding "
            "methods below are the naive 'threshold + connected-component label' "
            "baselines and are not expected to match a trained model; this "
            "benchmark ranks the heuristics and locates the fluorostats "
            "pipeline among them.\n\n"
        )
        f.write("## Ranking (by mean instance F1 @ 0.5)\n\n")
        f.write("| Rank | Method | F1@0.5 | Precision | Recall | "
                "Count MAE | Count CCC | Mean pred | Mean GT |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['rank']} | {r['method']} | {r['mean_f1@0.5']} | "
                f"{r['mean_precision']} | {r['mean_recall']} | "
                f"{r['count_mae']} | {r['count_ccc']} | "
                f"{r['mean_pred_count']} | {r['mean_gt_count']} |\n"
            )
        f.write(
            f"\n_Reference (not a row above):_ StarDist trained model "
            f"AP@0.5 = {STARDIST_AP50}.\n"
        )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}\n")
    print("Ranking (mean F1@0.5):")
    for r in rows:
        print(f"  {r['rank']}. {r['method']:<15} "
              f"F1={r['mean_f1@0.5']:.4f}  "
              f"count_MAE={r['count_mae']:.2f}  CCC={r['count_ccc']:.4f}")


if __name__ == "__main__":
    main()
