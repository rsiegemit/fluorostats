"""Produce results/b_denoising.csv — denoiser effect on Otsu foreground Dice (BBBC024).

Gaussian noise (sd 40/80/160, seed 0) is added to the first 5 BBBC024 volumes;
each denoiser is followed by an Otsu threshold and scored by foreground Dice.
Data: ``$FLUOROSTATS_DATA/BBBC024``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import filters, restoration
from skimage.filters import median

from common import bbbc024_volumes, write_table
from fluorostats.preprocess import denoise
from fluorostats.validate import dice

SEED = 0
NOISE_SD = [40, 80, 160]
DENOISERS = {
    "none": lambda v: v,
    "fluorostats(gaussian σ1)": lambda v: denoise(v, sigma=1.0),
    "median_3": lambda v: median(v, footprint=np.ones((3, 3, 3))),
    "gaussian_σ2": lambda v: ndi.gaussian_filter(v, 2.0),
    "TV_chambolle": lambda v: restoration.denoise_tv_chambolle(v, weight=0.1),
}


def main():
    vols = bbbc024_volumes(n=5)
    rng = np.random.default_rng(SEED)
    rows = []
    for sd in NOISE_SD:
        acc = {m: [] for m in DENOISERS}
        for fp, gp in vols:
            gt = tifffile.imread(gp)
            clean = tifffile.imread(fp)
            noisy = (clean.astype(np.float32) + rng.normal(0, sd, clean.shape)).astype(np.float32)
            for name, fn in DENOISERS.items():
                d = fn(noisy)
                acc[name].append(dice(d > filters.threshold_otsu(d), gt))
        rows.append({"noise_sd": sd, **{m: round(float(np.mean(v)), 3) for m, v in acc.items()}})
    df = pd.DataFrame(rows)
    write_table(df, "b_denoising", {"seed": SEED, "noise_sd": NOISE_SD, "n_volumes": 5, "denoisers": list(DENOISERS)})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
