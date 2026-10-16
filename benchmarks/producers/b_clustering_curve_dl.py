"""Produce results/b_clustering_curve_dl.csv and b_clustering_curve_2d.csv — deep learning on BBBC024.

Scores the StarDist and Cellpose label predictions made by
``benchmarks/cluster/bbbc024_dl_infer.py`` (run on a GPU node; see DATA_MANIFEST.md)
with ``validate.instance_f1``, the scorer of ``b_clustering_curve.py``.

- ``b_clustering_curve_dl``: 3D, the same first 5 volumes per clustering level as
  ``b_clustering_curve``. Cellpose in its 3D mode; StarDist per plane, stitched in z
  (``zstitch``) twice: with default settings (model thresholds, volume
  normalisation, stitch IoU 0.5, no fragment filter) and with the setting
  ``b_stardist3d_tuning.py`` selected on volumes 5-9 (run that producer first).
- ``b_clustering_curve_2d``: the mid z plane of all 30 volumes per level, every
  method (four global thresholds + CC, fluorostats raw Otsu+CC and watershed,
  StarDist, Cellpose) on the identical planes; mean F1, its SD and n.

Predictions: ``$FLUOROSTATS_DATA/BBBC024_dl/{3d,2d}/{cellpose,stardist}/<set>/`` and,
for StarDist 3D, ``BBBC024_dl/sd_grid/p<prob>_<norm>/3d/<set>/`` (unstitched planes),
each with the ``versions.json`` the inference script wrote (copied into run.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import filters

from common import (DATA, RAW_OTSU_CC, RESULTS, SMALL_OBJECT_RULE, YEAR_LABELS, bbbc024_volumes, raw_otsu_cc,
                    require, threshold_cc, thresholds, write_table)
from fluorostats.objects import watershed_split
from fluorostats.segment import remove_small
from fluorostats.validate import instance_f1
from zstitch import drop_short, stitch_planes

LEVELS = {"c00": "BBBC024", "c25": "BBBC024_c25", "c50": "BBBC024_c50", "c75": "BBBC024_c75"}
DL_TOOLS = {"stardist": "StarDist", "cellpose": "Cellpose"}
N_VOL_3D = 5          # as b_clustering_curve
MIN_SIZE = 30         # as b_clustering_curve (voxels in 3D, pixels in 2D)
WATERSHED_MIN_DISTANCE = 6
THRESHOLDS = thresholds(["otsu", "li", "isodata", "triangle"], YEAR_LABELS)
PRED = DATA / "BBBC024_dl"
SD_DEFAULT = {"prob_thresh": "default", "normalise": "volume", "stitch_iou": 0.5, "min_planes": 0}


def sd_tuned() -> dict:
    tab = pd.read_csv(require(RESULTS / "b_stardist3d_tuning.csv"), dtype={"prob_thresh": str})
    row = tab.loc[tab["selected"]].iloc[0]
    return {"prob_thresh": row["prob_thresh"], "normalise": row["normalise"],
            "stitch_iou": float(row["stitch_iou"]), "min_planes": int(row["min_planes"])}


def stardist_3d(setting: dict, sub: str, image_path: str) -> np.ndarray:
    planes = tifffile.imread(require(PRED / "sd_grid" / f"p{setting['prob_thresh']}_{setting['normalise']}"
                                     / "3d" / sub / Path(image_path).name))
    stitched = stitch_planes(planes, setting["stitch_iou"])
    return drop_short(stitched, setting["min_planes"]) if setting["min_planes"] else stitched


def mid_plane(path: str) -> np.ndarray:
    vol = tifffile.imread(path)
    return vol[vol.shape[0] // 2]


def predicted(mode: str, tool: str, sub: str, image_path: str) -> np.ndarray:
    return tifffile.imread(require(PRED / mode / tool / sub / Path(image_path).name))


def versions(mode: str) -> dict:
    out = {}
    for tool in DL_TOOLS:
        folder = (PRED / "sd_grid" / "pdefault_volume" / "3d" if (mode, tool) == ("3d", "stardist")
                  else PRED / mode / tool)
        meta = json.loads(require(folder / LEVELS["c00"] / "versions.json").read_text())
        out[tool] = {k: v for k, v in meta.items() if k not in ("seconds", "host", "n_files")}
    return out


def curve_3d(tuned: dict) -> pd.DataFrame:
    rows = []
    for lvl, sub in LEVELS.items():
        for fp, gp in bbbc024_volumes(sub, N_VOL_3D):
            gt = tifffile.imread(gp).astype(np.int32)
            preds = {"Cellpose": predicted("3d", "cellpose", sub, fp),
                     "StarDist (default stitch)": stardist_3d(SD_DEFAULT, sub, fp),
                     "StarDist (tuned on volumes 5-9)": stardist_3d(tuned, sub, fp)}
            rows += [{"clustering": lvl, "method": m, "f1": instance_f1(p, gt)["f1"]}
                     for m, p in preds.items()]
        print(f"3d done {lvl}", flush=True)
    df = pd.DataFrame(rows).groupby(["method", "clustering"])["f1"].mean().round(3)
    return df.unstack("clustering")


def plane_methods(img: np.ndarray) -> dict[str, np.ndarray]:
    labels = {name: threshold_cc(img, fn, MIN_SIZE) for name, fn in THRESHOLDS.items()}
    labels[RAW_OTSU_CC] = raw_otsu_cc(img, MIN_SIZE)
    otsu = remove_small(img > filters.threshold_otsu(img), MIN_SIZE)
    labels["fluorostats_watershed"] = watershed_split(
        otsu, min_size=MIN_SIZE, min_distance=WATERSHED_MIN_DISTANCE)[0]
    return labels


def curve_2d() -> pd.DataFrame:
    rows = []
    for lvl, sub in LEVELS.items():
        for fp, gp in bbbc024_volumes(sub):
            img = mid_plane(fp).astype(np.float32)
            gt = mid_plane(gp).astype(np.int32)
            labels = plane_methods(img)
            labels.update({name: predicted("2d", tool, sub, fp) for tool, name in DL_TOOLS.items()})
            rows += [{"clustering": lvl, "method": m, "f1": instance_f1(lab, gt)["f1"]}
                     for m, lab in labels.items()]
        print(f"2d done {lvl}", flush=True)
    g = pd.DataFrame(rows).groupby(["method", "clustering"])["f1"]
    out = pd.DataFrame({"mean_F1": g.mean().round(3), "sd_F1": g.std(ddof=1).round(3),
                        "n": g.size()}).reset_index()
    return out.sort_values(["method", "clustering"])


def main():
    tuned = sd_tuned()
    piv = curve_3d(tuned)
    write_table(piv, "b_clustering_curve_dl",
                {"levels": LEVELS, "n_volumes": N_VOL_3D, "scorer": "validate.instance_f1",
                 "stardist_default": SD_DEFAULT, "stardist_tuned": tuned,
                 "models": versions("3d")}, index=True)
    print(piv.to_string())
    flat = curve_2d()
    write_table(flat, "b_clustering_curve_2d",
                {"levels": LEVELS, "plane": "mid z (shape[0] // 2)", "min_size": MIN_SIZE,
                 "small_object_rule": SMALL_OBJECT_RULE,
                 "watershed_min_distance": WATERSHED_MIN_DISTANCE,
                 "scorer": "validate.instance_f1", "models": versions("2d")})
    print(flat.pivot(index="method", columns="clustering", values="mean_F1").to_string())


if __name__ == "__main__":
    main()
