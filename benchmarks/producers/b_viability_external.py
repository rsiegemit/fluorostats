"""Produce results/b_viability_external.csv — viability estimators vs exact GT (Kerkhoff set).

Each synthetic Live/Dead image (Zenodo 10395753, "N Live; M Dead" in the file
name gives the exact viability; ch 1 live, ch 0 dead) is scored by:
the Kerkhoff Fiji macro equivalent (Find-Maxima peak count), fluorostats
watershed object count (``objects.watershed_split``), area fraction
(``viability.live_dead_fractions``), native peak counting
(``viability.live_dead_by_count(method="maxima")``) and plain Otsu-CC counts.
Data: ``$FLUOROSTATS_DATA/kerkhoff_lda/synthetic``.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import filters

from common import DATA, SMALL_OBJECT_RULE, kerkhoff_peak_count, require, threshold_cc, write_table
from fluorostats.objects import watershed_split
from fluorostats.segment import remove_small
from fluorostats.viability import live_dead_by_count, live_dead_fractions, live_ratio

MIN_SIZE = 8


def score(path):
    im = tifffile.imread(path).astype(np.float32)
    live, dead = im[1], im[0]
    lm = remove_small(live > filters.threshold_otsu(live), MIN_SIZE)
    dm = remove_small(dead > filters.threshold_otsu(dead), MIN_SIZE)
    n_ws = [watershed_split(m, min_size=MIN_SIZE, min_distance=3)[0].max() for m in (lm, dm)]
    n_cc = [threshold_cc(im, filters.threshold_otsu, MIN_SIZE).max() for im in (live, dead)]
    return {
        "Kerkhoff_macro_peakcount": live_ratio(kerkhoff_peak_count(live), kerkhoff_peak_count(dead)),
        "fluorostats_objcount": live_ratio(*n_ws),
        "fluorostats_areafrac": live_dead_fractions(live, dead, method="otsu", min_size=MIN_SIZE)["viability"],
        "fluorostats_maxima(NEW)": live_dead_by_count(live, dead, method="maxima", min_distance=3,
                                                      threshold_rel=0.15)["viability"],
        "Otsu_CC_count": live_ratio(*n_cc),
    }


def main():
    root = require(DATA / "kerkhoff_lda" / "synthetic")
    rows = []
    for f in sorted(glob.glob(str(root / "**" / "*.tif"), recursive=True)):
        m = re.search(r"(\d+) Live; (\d+) Dead", Path(f).name)
        if not m:
            continue
        n_live, n_dead = int(m.group(1)), int(m.group(2))
        rows.append({"file": Path(f).name, "true_viability": n_live / (n_live + n_dead), **score(f)})
    df = pd.DataFrame(rows)
    write_table(df, "b_viability_external", {"min_size": MIN_SIZE, "small_object_rule": SMALL_OBJECT_RULE, "n_images": len(df)})
    print(f"n = {len(df)} images")
    print(df.drop(columns=["file"]).describe().to_string())


if __name__ == "__main__":
    main()
