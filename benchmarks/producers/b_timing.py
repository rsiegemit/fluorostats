"""Produce results/b_timing.csv — CPU runtime per BBBC039 image, fluorostats vs classical methods.

Mean wall-clock ms per image over the first 50 BBBC039 images (one warm-up
call). Wall-clock times depend on the machine, so a re-run reproduces the
ranking and order of magnitude, not the committed digits.

The two DL rows are not measured here: they are the per-image CPU inference
times measured on the cluster (cluster script, not included; 20 BBBC039 images,
StarDist ``2D_versatile_fluo`` and Cellpose ``nuclei``), carried as constants in
``common.DL_CPU_MS``.
Data: ``$FLUOROSTATS_DATA/BBBC039``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import tifffile
from skimage import filters
from skimage.morphology import remove_small_objects

from common import DL_CPU_MS, RESULTS, YEAR_LABELS, bbbc039_pairs, threshold_cc, thresholds
from fluorostats.objects import label_3d, watershed_split
from fluorostats.segment import segment_channel

N_IMAGES = 50
MIN_SIZE = 20

METHODS = {
    "fluorostats(Otsu+CC)": lambda im: label_3d(segment_channel(im, min_size=MIN_SIZE)[1], min_size=MIN_SIZE),
    **{name: (lambda im, fn=fn: threshold_cc(im, fn, max_size=MIN_SIZE))
       for name, fn in thresholds(["otsu", "li", "isodata", "triangle"], YEAR_LABELS).items()},
    "Watershed_1991": lambda im: watershed_split(
        remove_small_objects(im > filters.threshold_otsu(im), MIN_SIZE), min_size=MIN_SIZE, min_distance=4)[0],
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
    rows.append({"method": "StarDist_2018 (DL)", "device": "CPU(cluster)",
                 "ms_per_image": DL_CPU_MS["StarDist"],
                 "images_per_sec": round(1000 / DL_CPU_MS["StarDist"], 1)})
    rows.append({"method": "Cellpose_2021 (DL)", "device": "CPU(cluster)",
                 "ms_per_image": DL_CPU_MS["Cellpose"],
                 "images_per_sec": round(1000 / DL_CPU_MS["Cellpose"], 2)})
    df = pd.DataFrame(rows).sort_values("ms_per_image")
    df.to_csv(RESULTS / "b_timing.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
