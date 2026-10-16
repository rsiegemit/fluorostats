"""Produce results/b_dl_determinism.csv — run-to-run spread of segmentation on BBBC039.

The first 10 BBBC039 images (sorted), segmented ``N_RUNS`` times by each method:
StarDist and Cellpose on the cluster CPU, each run a fresh process that reloads the
model (``cluster/bbbc024_dl_infer.py ... 2d``, see DATA_MANIFEST.md), and the
fluorostats default (raw Otsu+CC, ``b2_nuclei_methods_perimage``'s ``min_size``)
here. Per run: the mean instance F1 (IoU 0.5) and the total object count over the
10 images, and whether every label image equals run 1's exactly. The table gives,
per method, the CV% across runs of both, and the number of runs identical to run 1.
Predictions: ``$FLUOROSTATS_DATA/BBBC039_dl_repeat/<tool>/run_<k>/<image>.tif``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from common import (DATA, RAW_OTSU_CC, bbbc039_gt_instances, bbbc039_pairs, raw_otsu_cc, require,
                    write_table)
from fluorostats.validate import instance_f1

N_IMAGES = 10
N_RUNS = 20
MIN_SIZE = 20   # as b2_nuclei_methods_perimage
DL_TOOLS = {"stardist": "StarDist", "cellpose": "Cellpose"}
PRED = DATA / "BBBC039_dl_repeat"


def cv_percent(v: np.ndarray) -> float:
    return float(100 * v.std(ddof=1) / v.mean()) if v.mean() else float("nan")


def dl_run(tool: str, k: int, stem: str) -> np.ndarray:
    return tifffile.imread(require(PRED / tool / f"run_{k:02d}" / f"{stem}.tif"))


def summarise(method: str, runs: list[list[np.ndarray]], gts: list[np.ndarray]) -> dict:
    """``runs[k][i]``: run k's labels for image i."""
    f1 = np.array([np.mean([instance_f1(lab, gt)["f1"] for lab, gt in zip(run, gts)])
                   for run in runs])
    count = np.array([sum(len(np.unique(lab[lab > 0])) for lab in run) for run in runs], float)
    same = sum(all(np.array_equal(a, b) for a, b in zip(run, runs[0])) for run in runs)
    return {"method": method, "n_runs": len(runs), "n_images": len(gts),
            "mean_F1": round(float(f1.mean()), 4), "F1_cv_percent": cv_percent(f1),
            "count_mean": float(count.mean()), "count_cv_percent": cv_percent(count),
            "runs_identical_to_run1": int(same)}


def main():
    pairs = bbbc039_pairs()[:N_IMAGES]
    gts = [bbbc039_gt_instances(m) for _, _, m in pairs]
    images = [np.array(Image.open(p)).astype(np.float32) for _, p, _ in pairs]
    rows = [summarise(RAW_OTSU_CC, [[raw_otsu_cc(im, MIN_SIZE) for im in images]
                                    for _ in range(N_RUNS)], gts)]
    for tool, name in DL_TOOLS.items():
        runs = [[dl_run(tool, k, s) for s, _, _ in pairs] for k in range(1, N_RUNS + 1)]
        rows.append(summarise(name, runs, gts))
    df = pd.DataFrame(rows)
    write_table(df, "b_dl_determinism",
                {"n_images": N_IMAGES, "n_runs": N_RUNS, "min_size": MIN_SIZE,
                 "images": [s for s, _, _ in pairs], "iou_threshold": 0.5,
                 "dl_hardware": "cluster CPU (Cellpose fails in ROCm hipBLASLt; TensorFlow CPU build)",
                 "scorer": "validate.instance_f1"})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
