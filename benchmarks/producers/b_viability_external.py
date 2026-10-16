"""Produce results/b_viability_external.csv — viability estimators vs exact GT (Kerkhoff set).

Each synthetic Live/Dead image (Zenodo 10395753, "N Live; M Dead" in the file
name gives the exact viability; ch 1 live, ch 0 dead) is scored by:
a direct skimage peak count with the library's parameters (``skimage_peak_count``;
the same algorithm as the maxima mode, so identical to it by construction), fluorostats
watershed object count (``objects.watershed_split``), area fraction
(``viability.live_dead_fractions``), native peak counting
(``viability.live_dead_by_count(method="maxima")``) and plain Otsu-CC counts.
``kerkhoff_macro`` is the published Fiji macro's own viability, read from the
``Summary.csv`` it wrote (``cluster/kerkhoff_macro_headless.py``; DATA_MANIFEST.md).
Data: ``$FLUOROSTATS_DATA/kerkhoff_lda/{synthetic,macro_output}``.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import filters

from common import (DATA, MAXIMA_MIN_DISTANCE, MAXIMA_THRESHOLD_REL, SMALL_OBJECT_RULE, skimage_peak_count, require,
                    threshold_cc, write_table)
from fluorostats.objects import watershed_split
from fluorostats.segment import remove_small
from fluorostats.viability import live_dead_by_count, live_dead_fractions, live_ratio

MIN_SIZE = 8
MACRO_CUT = 17   # the macro's "Cut Prefix": it names each image without its first 17 characters


def macro_viability() -> tuple[dict[str, float], dict]:
    """The macro's viability (fraction) per image file name, and its run record."""
    out = DATA / "kerkhoff_lda" / "macro_output"
    summary = pd.read_csv(require(out / "Summary.csv")).dropna(subset=["Image"])
    record = json.loads(require(out / "versions.json").read_text(encoding="utf-8"))
    return {name: v / 100 for name, v in zip(summary["Image"], summary["Cell Viability [%]"])}, record


def score(path):
    im = tifffile.imread(path).astype(np.float32)
    live, dead = im[1], im[0]
    lm = remove_small(live > filters.threshold_otsu(live), MIN_SIZE)
    dm = remove_small(dead > filters.threshold_otsu(dead), MIN_SIZE)
    n_ws = [watershed_split(m, min_size=MIN_SIZE, min_distance=3)[0].max() for m in (lm, dm)]
    n_cc = [threshold_cc(im, filters.threshold_otsu, MIN_SIZE).max() for im in (live, dead)]
    return {
        "skimage_peakcount": live_ratio(skimage_peak_count(live), skimage_peak_count(dead)),
        "fluorostats_objcount": live_ratio(*n_ws),
        "fluorostats_areafrac": live_dead_fractions(live, dead, method="otsu", min_size=MIN_SIZE)["viability"],
        "fluorostats_maxima": live_dead_by_count(live, dead, method="maxima", min_distance=MAXIMA_MIN_DISTANCE,
                                                 threshold_rel=MAXIMA_THRESHOLD_REL)["viability"],
        "Otsu_CC_count": live_ratio(*n_cc),
    }


def main():
    root = require(DATA / "kerkhoff_lda" / "synthetic")
    macro, macro_run = macro_viability()
    rows = []
    for f in sorted(glob.glob(str(root / "**" / "*.tif"), recursive=True)):
        m = re.search(r"(\d+) Live; (\d+) Dead", Path(f).name)
        if not m:
            continue
        n_live, n_dead = int(m.group(1)), int(m.group(2))
        name = Path(f).name
        rows.append({"file": name, "true_viability": n_live / (n_live + n_dead), **score(f),
                     "kerkhoff_macro": macro[name[MACRO_CUT:]]})
    df = pd.DataFrame(rows)
    write_table(df, "b_viability_external", {"min_size": MIN_SIZE, "small_object_rule": SMALL_OBJECT_RULE,
                                             "n_images": len(df), "kerkhoff_macro": macro_run})
    print(f"n = {len(df)} images")
    print(df.drop(columns=["file"]).describe().to_string())


if __name__ == "__main__":
    main()
