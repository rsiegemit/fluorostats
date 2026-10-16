"""Produce results/b_timing.csv — CPU runtime per BBBC039 image, fluorostats vs classical methods.

Mean wall-clock ms per image over the first 50 BBBC039 images (one warm-up
call). Wall-clock times depend on the machine, so a re-run reproduces the
ranking and order of magnitude, not the committed digits.

Deep-learning inference times are measured with fluorostats on one CPU in
``b_seg_timing.py``, not here.
Data: ``$FLUOROSTATS_DATA/BBBC039``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import tifffile
from skimage import filters

from common import (DENOISED_OTSU_CC, SMALL_OBJECT_RULE, YEAR_LABELS, bbbc039_pairs, denoised_otsu_cc,
                    threshold_cc, thresholds, write_table)
from fluorostats.objects import watershed_split
from fluorostats.segment import remove_small

N_IMAGES = 50
MIN_SIZE = 20

METHODS = {
    DENOISED_OTSU_CC: lambda im: denoised_otsu_cc(im, MIN_SIZE),
    **{name: (lambda im, fn=fn: threshold_cc(im, fn, MIN_SIZE))
       for name, fn in thresholds(["otsu", "li", "isodata", "triangle"], YEAR_LABELS).items()},
    "Watershed_1991": lambda im: watershed_split(
        remove_small(im > filters.threshold_otsu(im), MIN_SIZE), min_size=MIN_SIZE, min_distance=4)[0],
}


def ms_per_image(fn, images):
    fn(images[0])  # warm-up
    t = time.perf_counter()
    for im in images:
        fn(im)
    return (time.perf_counter() - t) / len(images) * 1000


def main():
    images = [tifffile.imread(p).astype(np.float32) for _, p, _ in bbbc039_pairs()[:N_IMAGES]]
    rows = []
    for name, fn in METHODS.items():
        ms = ms_per_image(fn, images)
        rows.append({"method": name, "device": "CPU", "ms_per_image": round(ms, 1),
                     "images_per_sec": round(1000 / ms, 1)})
    df = pd.DataFrame(rows).sort_values("ms_per_image")
    write_table(df, "b_timing", {"n_images": N_IMAGES, "min_size": MIN_SIZE, "small_object_rule": SMALL_OBJECT_RULE}, timing=True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
