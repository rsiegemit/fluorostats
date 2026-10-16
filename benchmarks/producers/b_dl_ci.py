"""Produce results/b_dl_ci.csv — bootstrap 95% CIs on fluorostats vs DL instance F1 (BBBC039).

fluorostats per-image F1@0.5 is computed here on all 200 BBBC039 images
(``common.denoised_otsu_cc``: denoise σ=1 -> Otsu binarize -> label, min size 20; scored by
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

from common import (DENOISED_OTSU_CC, RESULTS, SMALL_OBJECT_RULE, bbbc039_gt_instances, bbbc039_pairs,
                    denoised_otsu_cc, write_table)
from fluorostats.stats import bootstrap_mean_ci
from fluorostats.validate import instance_f1

SEED = 0
N_BOOT = 10000
MIN_SIZE = 20


def fluorostats_f1() -> pd.DataFrame:
    rows = []
    for stem, img_path, mask_path in bbbc039_pairs():
        pred = denoised_otsu_cc(tifffile.imread(img_path).astype(np.float32), MIN_SIZE)
        f1 = instance_f1(pred, bbbc039_gt_instances(mask_path))["f1"]
        rows.append({"image": stem, "fs_f1": round(f1, 4)})
    return pd.DataFrame(rows)


def boot_mean_ci(x, rng):
    # one shared generator: the five intervals are successive draws from one seeded stream
    r = bootstrap_mean_ci(x, n_boot=N_BOOT, ci=0.95, rng=rng)
    return r["mean"], r["ci_low"], r["ci_high"]


def main():
    sd = pd.read_csv(RESULTS / "stardist_eval.csv")[["image", "stardist_f1"]]
    cp = pd.read_csv(RESULTS / "cellpose_eval.csv")[["image", "cellpose_f1"]]
    df = fluorostats_f1().merge(sd, on="image").merge(cp, on="image")
    rng = np.random.default_rng(SEED)
    rows = []
    for name, col in [(DENOISED_OTSU_CC, "fs_f1"), ("StarDist", "stardist_f1"), ("Cellpose", "cellpose_f1")]:
        m, lo, hi = boot_mean_ci(df[col].values, rng)
        rows.append({"method": name, "mean_F1": round(m, 4), "CI_low": round(lo, 4), "CI_high": round(hi, 4)})
    for name, col in [(f"{DENOISED_OTSU_CC} - StarDist", "stardist_f1"), (f"{DENOISED_OTSU_CC} - Cellpose", "cellpose_f1")]:
        m, lo, hi = boot_mean_ci((df["fs_f1"] - df[col]).values, rng)
        sig = "YES (excludes 0)" if (lo > 0 or hi < 0) else "no (CI spans 0)"
        rows.append({"method": name, "mean_F1": round(m, 4), "CI_low": round(lo, 4),
                     "CI_high": round(hi, 4), "diff_significant": sig})
    out = pd.DataFrame(rows)
    write_table(out, "b_dl_ci", {"seed": SEED, "n_boot": N_BOOT, "min_size": MIN_SIZE,
                                 "small_object_rule": SMALL_OBJECT_RULE, "n_images": len(df)})
    print(f"n = {len(df)} matched images")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
