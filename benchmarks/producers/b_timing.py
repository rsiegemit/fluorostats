"""Produce results/b_timing.csv — CPU runtime per BBBC039 image, fluorostats vs classical methods.

Mean wall-clock ms per image over the first 50 BBBC039 images (one warm-up
call). Wall-clock times depend on the machine, so a re-run reproduces the
ranking and order of magnitude, not the committed digits.

The two DL rows are not measured here: they are the per-image CPU inference
times measured on the cluster (``time_dl.py``, 20 BBBC039 images, StarDist
``2D_versatile_fluo`` and Cellpose ``nuclei``) and are carried as constants.
Data: ``$FLUOROSTATS_DATA/BBBC039``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import filters
from skimage.morphology import remove_small_objects

from common import RESULTS, bbbc039_pairs
from fluorostats.objects import label_3d, watershed_split
from fluorostats.preprocess import denoise
from fluorostats.segment import binarize

N_IMAGES = 50
MIN_SIZE = 20
DL_CPU_MS = {"StarDist_2018 (DL)": 215.1, "Cellpose_2021 (DL)": 5547.0}  # cluster-measured


def threshold_cc(thr):
    return lambda im: ndi.label(remove_small_objects(im > thr(im), MIN_SIZE))[0]


METHODS = {
    "fluorostats(Otsu+CC)": lambda im: label_3d(
        binarize(denoise(im, sigma=1.0), method="otsu", threshold_scale=1.0, min_size=MIN_SIZE),
        min_size=MIN_SIZE),
    "Otsu_1979": threshold_cc(filters.threshold_otsu),
    "Li_1993": threshold_cc(filters.threshold_li),
    "Isodata_1978": threshold_cc(filters.threshold_isodata),
    "Triangle_1977": threshold_cc(filters.threshold_triangle),
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
                 "ms_per_image": DL_CPU_MS["StarDist_2018 (DL)"],
                 "images_per_sec": round(1000 / DL_CPU_MS["StarDist_2018 (DL)"], 1)})
    rows.append({"method": "Cellpose_2021 (DL)", "device": "CPU(cluster)",
                 "ms_per_image": DL_CPU_MS["Cellpose_2021 (DL)"],
                 "images_per_sec": round(1000 / DL_CPU_MS["Cellpose_2021 (DL)"], 2)})
    df = pd.DataFrame(rows).sort_values("ms_per_image")
    df.to_csv(RESULTS / "b_timing.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
