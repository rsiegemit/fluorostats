"""Produce results/b4_reaver_ranking_perimage.csv — vessel area fraction per tool vs manual GT.

The REAVER dataset (Corliss et al. 2020) ships, for all 36 images, the expert
manual mask and the masks of REAVER, AngioTool, AngioQuant, ImageJ and RAVE as
RGB TIFFs whose red channel is the vessel mask. Each tool's area fraction is
read from those masks; fluorostats segments the raw image with its fixed 2D
pipeline (denoise σ=2 -> background subtract r=75 -> Otsu x0.85, min size 100).
Data: ``$FLUOROSTATS_DATA/REAVER/REAVER_Vascular_Networks_Image_Dataset``.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from common import DATA, RESULTS, require
from fluorostats.segment import segment_channel

TOOLS = ["REAVER_Auto", "AngioTool_Auto", "AngioQuant_Auto", "ImageJ_Auto", "RAVE_Auto"]
SEG = dict(method="otsu", threshold_scale=0.85, sigma=2.0, bg_radius=75, min_size=100)


def red_area_fraction(path: Path) -> float:
    a = tifffile.imread(path)
    red = a[..., 0] if a.ndim == 3 else a
    return float((red > 127).mean())


def fluorostats_area_fraction(path: Path) -> float:
    gray = tifffile.imread(path)
    if gray.ndim == 3:
        gray = gray[..., 0]
    _, mask = segment_channel(gray.astype(np.float32), **SEG)
    return float(mask.mean())


def main():
    root = require(DATA / "REAVER" / "REAVER_Vascular_Networks_Image_Dataset")
    rows = []
    for man in sorted(glob.glob(str(root / "Manual" / "*.tif"))):
        name = Path(man).stem
        rec = {"image": name, "manual_GT": red_area_fraction(Path(man))}
        for tool in TOOLS:
            tp = root / tool / f"{name}.tif"
            rec[tool.replace("_Auto", "")] = red_area_fraction(tp) if tp.exists() else np.nan
        rec["fluorostats"] = fluorostats_area_fraction(root / "_Original_Images" / f"{name}.tif")
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b4_reaver_ranking_perimage.csv", index=False)
    print(df.drop(columns=["image"]).describe().to_string())


if __name__ == "__main__":
    main()
