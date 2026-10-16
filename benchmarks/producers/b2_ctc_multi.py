"""Produce results/b2_ctc_multi.csv — CTC 3D foreground Dice, fluorostats vs global thresholds.

Every annotated GT slice / frame in sequences 01 and 02 of Fluo-C3DH-A549 and
Fluo-N3DH-CHO is paired with its raw plane; five global thresholds and the
fluorostats pipeline (denoise σ=1 -> background subtract r=25 -> Otsu x0.9,
min size 50) are scored by foreground Dice. ``ctc_top_SEG`` is the challenge's
published best instance SEG, carried as context only.
Data: ``$FLUOROSTATS_DATA/CTC/<dataset>/{01,02,01_GT,02_GT}``.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from common import DATA, SMALL_OBJECT_RULE, YEAR_LABELS, require, thresholds, write_table
from fluorostats.segment import segment_channel
from fluorostats.validate import dice

DATASETS = [("Fluo-C3DH-A549", 0.908), ("Fluo-N3DH-CHO", 0.925)]  # (name, CTC top SEG)
SEG = dict(method="otsu", threshold_scale=0.9, min_size=50, sigma=1.0, bg_radius=25)
THRESHOLDS = thresholds(["otsu", "li", "isodata", "triangle", "yen"], YEAR_LABELS)


def fluorostats_seg(v):
    return segment_channel(v.astype(np.float32), **SEG)[1]


def gt_pairs(ds_root):
    """(raw plane, GT mask) for each GT file: man_seg_T_Z = slice Z of frame T, man_segT = frame T."""
    out = []
    for sq in ("01", "02"):
        for gp in sorted(glob.glob(str(ds_root / f"{sq}_GT" / "SEG" / "*.tif"))):
            nm = Path(gp).name
            m3 = re.search(r"man_seg_(\d+)_(\d+)\.tif", nm)
            m2 = re.search(r"man_seg0*(\d+)\.tif", nm)
            t = int((m3 or m2).group(1)) if (m3 or m2) else None
            raw = ds_root / sq / f"t{t:03d}.tif" if t is not None else None
            if raw is None or not raw.exists():
                continue
            vol = tifffile.imread(raw)
            if m3:
                z = int(m3.group(2))
                if z < vol.shape[0]:
                    out.append((vol[z], tifffile.imread(gp)))
            else:
                out.append((vol, tifffile.imread(gp)))
    return out


def main():
    rows = []
    for ds, top in DATASETS:
        scores = {m: [] for m in list(THRESHOLDS) + ["fluorostats"]}
        for img, gt in gt_pairs(require(DATA / "CTC" / ds)):
            for name, fn in THRESHOLDS.items():
                scores[name].append(dice(img > fn(img.astype(np.float32)), gt))
            scores["fluorostats"].append(dice(fluorostats_seg(img), gt))
        rows += [{"dataset": ds, "method": m, "mean_dice": round(float(np.mean(v)), 3), "ctc_top_SEG": top}
                 for m, v in scores.items()]
    df = pd.DataFrame(rows)
    write_table(df, "b2_ctc_multi", {"datasets": [d for d, _ in DATASETS], "thresholds": list(THRESHOLDS), "fluorostats": SEG,
                                     "small_object_rule": SMALL_OBJECT_RULE})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
