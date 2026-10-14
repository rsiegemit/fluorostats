"""Produce results/b_dl_ci.csv — bootstrap 95% CIs on fluorostats vs DL instance F1 (BBBC039).

fluorostats per-image F1@0.5 is computed here on all 200 BBBC039 images
(denoise σ=1 -> Otsu binarize -> ``objects.label_3d``, min size 20; scored by
``validate.instance_f1`` and rounded to 4 d.p. like the DL tables). The StarDist
and Cellpose per-image F1 are read from the committed GPU-cluster outputs
``results/stardist_eval.csv`` / ``cellpose_eval.csv`` (see benchmarks/README.md).
Means and paired differences get percentile-bootstrap CIs (10k resamples, seed 0).
Data: ``$FLUOROSTATS_DATA/BBBC039``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from common import RESULTS, bbbc039_gt_instances, bbbc039_pairs
from fluorostats.objects import label_3d
from fluorostats.preprocess import denoise
from fluorostats.segment import binarize
from fluorostats.validate import instance_f1

SEED = 0
N_BOOT = 10000
SEG = dict(method="otsu", threshold_scale=1.0, sigma=1.0, min_size=20)


def fluorostats_f1() -> pd.DataFrame:
    rows = []
    for stem, img_path, mask_path in bbbc039_pairs():
        sm = denoise(tifffile.imread(img_path).astype(np.float32), sigma=SEG["sigma"])
        mask = binarize(sm, method=SEG["method"], threshold_scale=SEG["threshold_scale"],
                        min_size=SEG["min_size"])
        pred = label_3d(mask, min_size=SEG["min_size"])[0]
        f1 = instance_f1(pred, bbbc039_gt_instances(mask_path))["f1"]
        rows.append({"image": stem, "fs_f1": round(f1, 4)})
    return pd.DataFrame(rows)


def boot_mean_ci(x, rng):
    # exception REU-3: the library bootstraps fold changes (stats.bootstrap_fold_change_ci),
    # not a plain mean; this percentile bootstrap of a mean has no library equivalent.
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    sd = pd.read_csv(RESULTS / "stardist_eval.csv")[["image", "stardist_f1"]]
    cp = pd.read_csv(RESULTS / "cellpose_eval.csv")[["image", "cellpose_f1"]]
    df = fluorostats_f1().merge(sd, on="image").merge(cp, on="image")
    rng = np.random.default_rng(SEED)
    rows = []
    for name, col in [("fluorostats", "fs_f1"), ("StarDist", "stardist_f1"), ("Cellpose", "cellpose_f1")]:
        m, lo, hi = boot_mean_ci(df[col].values, rng)
        rows.append({"method": name, "mean_F1": round(m, 4), "CI_low": round(lo, 4), "CI_high": round(hi, 4)})
    for name, col in [("fluorostats - StarDist", "stardist_f1"), ("fluorostats - Cellpose", "cellpose_f1")]:
        m, lo, hi = boot_mean_ci((df["fs_f1"] - df[col]).values, rng)
        sig = "YES (excludes 0)" if (lo > 0 or hi < 0) else "no (CI spans 0)"
        rows.append({"method": name, "mean_F1": round(m, 4), "CI_low": round(lo, 4),
                     "CI_high": round(hi, 4), "diff_significant": sig})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "b_dl_ci.csv", index=False)
    print(f"n = {len(df)} matched images")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
