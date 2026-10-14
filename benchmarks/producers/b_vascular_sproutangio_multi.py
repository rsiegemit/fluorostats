"""Produce results/b_vascular_sproutangio_multi.csv — sprout area fraction per method and VEGF group.

SproutAngio fibrin-bead assay CZI files (group number in the file name = VEGF-A
dose group); the first channel's plane is segmented by four global thresholds
(+ small-object removal) and by fluorostats (denoise σ=1 -> background subtract
r=15 -> Otsu x0.9, min size 100). Data: ``$FLUOROSTATS_DATA/sproutangio/*.czi``.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import czifile
import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects

from common import DATA, RESULTS, YEAR_LABELS, require, thresholds
from fluorostats.segment import segment_channel

MIN_SIZE = 100
THRESHOLDS = thresholds(["otsu", "li", "isodata", "triangle"], YEAR_LABELS)


def main():
    recs = []
    for fp in sorted(glob.glob(str(require(DATA / "sproutangio") / "*.czi"))):
        fa = np.squeeze(czifile.imread(fp)).astype(np.float32)[0]
        row = {"group": int(re.search(r"group(\d)", Path(fp).name).group(1))}
        row.update({name: float(remove_small_objects(fa > fn(fa), max_size=MIN_SIZE).mean())
                    for name, fn in THRESHOLDS.items()})
        _, mask = segment_channel(fa, method="otsu", threshold_scale=0.9, min_size=MIN_SIZE,
                                  sigma=1.0, bg_radius=15)
        row["fluorostats"] = float(mask.mean())
        recs.append(row)
    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "b_vascular_sproutangio_multi.csv", index=False)
    print(df.groupby("group").mean().round(4).to_string())


if __name__ == "__main__":
    main()
