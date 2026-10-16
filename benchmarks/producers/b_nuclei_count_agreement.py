"""Produce results/b_nuclei_count_agreement.csv — per-image nucleus count vs ground truth (BBBC039).

The fluorostats count is the number of objects of ``common.denoised_otsu_cc``
(the ``b_dl_ci`` pipeline, min size 20) on every BBBC039 image; the StarDist,
Cellpose and Omnipose counts are read from the committed DL tables (images a DL
table scores negative are left out); the ground-truth count is the number of
instances of ``common.bbbc039_gt_instances``. Per tool, over the images it
scores: Lin's CCC (``agreement.lins_ccc``), Spearman rho, mean absolute error and
mean bias (tool - GT, ``agreement.bland_altman``).
Data: ``$FLUOROSTATS_DATA/BBBC039``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
from scipy import stats as sps

from common import (DENOISED_OTSU_CC, RESULTS, SMALL_OBJECT_RULE, bbbc039_gt_instances, bbbc039_pairs,
                    denoised_otsu_cc, write_table)
from fluorostats.agreement import bland_altman, lins_ccc

MIN_SIZE = 20
DL = {"StarDist": ("stardist_eval.csv", "stardist_count", "stardist_f1"),
      "Cellpose": ("cellpose_eval.csv", "cellpose_count", "cellpose_f1"),
      "Omnipose": ("omnipose_eval.csv", "omnipose_count", "omnipose_f1")}


def counts() -> pd.DataFrame:
    """Per image: ground-truth count and the fluorostats count."""
    rows = []
    for stem, img_path, mask_path in bbbc039_pairs():
        pred = denoised_otsu_cc(tifffile.imread(img_path).astype(np.float32), MIN_SIZE)
        rows.append({"image": stem, "gt_count": int(bbbc039_gt_instances(mask_path).max()),
                     DENOISED_OTSU_CC: int(pred.max())})
    return pd.DataFrame(rows)


def agreement_row(tool: str, pred: np.ndarray, gt: np.ndarray) -> dict:
    return {"tool": tool, "n_images": len(gt), "count_ccc": lins_ccc(pred, gt),
            "spearman_rho": float(sps.spearmanr(pred, gt).statistic),
            "count_mae": float(np.abs(pred - gt).mean()), "bias": bland_altman(pred, gt)["bias"]}


def main():
    df = counts()
    gt = df.gt_count.values.astype(float)
    rows = [agreement_row(DENOISED_OTSU_CC, df[DENOISED_OTSU_CC].values.astype(float), gt)]
    for tool, (table, count_col, f1_col) in DL.items():
        d = pd.read_csv(RESULTS / table)
        d = df[["image", "gt_count"]].merge(d[d[f1_col] >= 0][["image", count_col]], on="image")
        rows.append(agreement_row(tool, d[count_col].values.astype(float), d.gt_count.values.astype(float)))
    out = pd.DataFrame(rows)
    write_table(out, "b_nuclei_count_agreement", {"min_size": MIN_SIZE, "small_object_rule": SMALL_OBJECT_RULE,
                                                  "n_images": len(df)})
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
