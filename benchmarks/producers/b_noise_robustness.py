"""Produce results/b_noise_robustness.csv — foreground Dice vs added noise (BBBC024).

Gaussian noise (sd 0-160, seed 0) is added to the first 5 BBBC024 volumes; four
global thresholds (+ small-object removal) and fluorostats raw Otsu+CC are scored
by foreground Dice. Labelling does not change the foreground, so the fluorostats
column equals the Otsu column by construction.
Data: ``$FLUOROSTATS_DATA/BBBC024``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from common import RAW_OTSU_CC, SMALL_OBJECT_RULE, bbbc024_volumes, raw_otsu_cc, thresholds, write_table
from fluorostats.segment import remove_small
from fluorostats.validate import dice

SEED = 0
NOISE_SD = [0, 20, 40, 80, 160]
MIN_SIZE = 30
THRESHOLDS = thresholds(["otsu", "li", "isodata", "triangle"],
                        {"otsu": "Otsu", "li": "Li", "isodata": "Isodata", "triangle": "Triangle"})


def main():
    vols = [(tifffile.imread(fp).astype(np.float32), tifffile.imread(gp).astype(np.int32))
            for fp, gp in bbbc024_volumes(n=5)]
    rng = np.random.default_rng(SEED)
    rows = []
    for sd in NOISE_SD:
        acc = {m: [] for m in [*THRESHOLDS, RAW_OTSU_CC]}
        for vol, gt in vols:
            noisy = vol + rng.normal(0, sd, vol.shape)
            for name, fn in THRESHOLDS.items():
                acc[name].append(float(dice(
                    remove_small(noisy > fn(noisy), MIN_SIZE), gt)))
            acc[RAW_OTSU_CC].append(dice(raw_otsu_cc(noisy, MIN_SIZE), gt))
        rows.append({"noise_sd": sd, **{m: round(float(np.mean(v)), 3) for m, v in acc.items()}})
    df = pd.DataFrame(rows)
    write_table(df, "b_noise_robustness", {"seed": SEED, "noise_sd": NOISE_SD, "min_size": MIN_SIZE, "small_object_rule": SMALL_OBJECT_RULE,
                                      "n_volumes": 5})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
