"""Produce results/b_noise_robustness.csv — foreground Dice vs added noise (BBBC024).

Gaussian noise (sd 0-160, seed 0) is added to the first 5 BBBC024 volumes; four
global thresholds (+ small-object removal) are scored by foreground Dice. The
fluorostats column is its Otsu path, so it equals the Otsu column by design.
Data: ``$FLUOROSTATS_DATA/BBBC024``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
from skimage import filters
from skimage.morphology import remove_small_objects

from common import RESULTS, bbbc024_volumes, foreground_dice

SEED = 0
NOISE_SD = [0, 20, 40, 80, 160]
MIN_SIZE = 30
THRESHOLDS = {"Otsu": filters.threshold_otsu, "Li": filters.threshold_li,
              "Isodata": filters.threshold_isodata, "Triangle": filters.threshold_triangle}


def main():
    vols = [(tifffile.imread(fp).astype(np.float32), tifffile.imread(gp).astype(np.int32))
            for fp, gp in bbbc024_volumes(n=5)]
    rng = np.random.default_rng(SEED)
    rows = []
    for sd in NOISE_SD:
        acc = {m: [] for m in THRESHOLDS}
        for vol, gt in vols:
            noisy = vol + rng.normal(0, sd, vol.shape)
            for name, fn in THRESHOLDS.items():
                acc[name].append(float(foreground_dice(
                    remove_small_objects(noisy > fn(noisy), MIN_SIZE), gt)))
        acc["fluorostats(Otsu+CC)"] = acc["Otsu"]
        rows.append({"noise_sd": sd, **{m: round(float(np.mean(v)), 3) for m, v in acc.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b_noise_robustness.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
