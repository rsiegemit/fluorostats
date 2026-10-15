"""Produce results/b_stardist3d_tuning.csv — StarDist 3D settings chosen on held-out volumes.

StarDist has no pretrained 3D fluorescence model, so in 3D it runs per plane and
the planes are stitched (``zstitch``). On BBBC024 the default settings segment the
defocus haze above and below each nucleus. This grid picks one setting on volumes
5-9 of every clustering level, which no figure or other table scores; the chosen
setting is then applied unchanged to the reported volumes 0-4 by
``b_clustering_curve_dl.py``.

Grid: detection probability {model default 0.479, 0.6, 0.7} x normalisation
{whole volume, each plane} (inference, ``cluster/bbbc024_dl_infer.py``) x stitch
IoU {0.05, 0.1, 0.2, 0.3, 0.5, 0.7} x minimum z extent {0, 5, 10, 20, 30, 40, 50, 60}
planes (here). The grid was widened twice because its optimum lay at the edge (first
IoU >= 0.3 and <= 20 planes, then IoU >= 0.1 and <= 40 planes); this is the last
widening, and 60 planes approaches the ~72-plane nucleus height. The reported
volumes play no part in the choice. Every
setting's mean F1 is written; ``selected`` marks the best (ties: first in grid order).
Predictions: ``$FLUOROSTATS_DATA/BBBC024_dl/sd_grid/p<prob>_<norm>/3d_tune/<set>/``.
"""
from __future__ import annotations

import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile

from common import DATA, bbbc024_volumes, require, write_table
from fluorostats.validate import instance_f1
from zstitch import drop_short, stitch_planes

LEVELS = {"c00": "BBBC024", "c25": "BBBC024_c25", "c50": "BBBC024_c50", "c75": "BBBC024_c75"}
TUNE = slice(5, 10)   # volumes used for tuning; the reported curves use 0-4
PROBS = ("default", "0.6", "0.7")
NORMS = ("volume", "plane")
STITCH_IOUS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
MIN_PLANES = (0, 5, 10, 20, 30, 40, 50, 60)
GRID = DATA / "BBBC024_dl" / "sd_grid"


def grid_dir(prob: str, norm: str, split: str, sub: str):
    return GRID / f"p{prob}_{norm}" / split / sub


def score(planes: np.ndarray, gt: np.ndarray) -> dict[tuple[float, int], float]:
    out = {}
    for iou in STITCH_IOUS:
        stitched = stitch_planes(planes, iou)
        for mp in MIN_PLANES:
            out[(iou, mp)] = instance_f1(drop_short(stitched, mp) if mp else stitched, gt)["f1"]
    return out


WORKERS = 7   # each holds a few 3D label volumes (~2.5 GB)


def score_task(task: tuple[str, str, str]) -> list[dict]:
    """Every stitch/filter combination for one inference setting and clustering level."""
    prob, norm, lvl = task
    sub = LEVELS[lvl]
    rows = []
    for fp, gp in bbbc024_volumes(sub)[TUNE]:
        name = fp.rsplit("/", 1)[-1]
        planes = tifffile.imread(require(grid_dir(prob, norm, "3d_tune", sub) / name))
        gt = tifffile.imread(gp).astype(np.int32)
        rows += [{"prob_thresh": prob, "normalise": norm, "stitch_iou": iou,
                  "min_planes": mp, "clustering": lvl, "f1": f1}
                 for (iou, mp), f1 in score(planes, gt).items()]
    print(f"done p{prob}_{norm} {lvl}", flush=True)
    return rows


def main():
    tasks = [(p, n, lvl) for p, n in itertools.product(PROBS, NORMS) for lvl in LEVELS]
    with Pool(WORKERS) as pool:   # results come back in task order
        rows = [r for part in pool.map(score_task, tasks, chunksize=1) for r in part]
    keys = ["prob_thresh", "normalise", "stitch_iou", "min_planes"]
    df = pd.DataFrame(rows)
    per_level = df.pivot_table(index=keys, columns="clustering", values="f1", sort=False)
    per_level["mean_F1"] = df.groupby(keys, sort=False)["f1"].mean()
    per_level = per_level.round(3).reset_index()
    per_level["selected"] = False
    per_level.loc[per_level["mean_F1"].idxmax(), "selected"] = True
    write_table(per_level, "b_stardist3d_tuning",
                {"tune_volumes": "5-9 of each set", "levels": LEVELS, "prob_thresh": PROBS,
                 "normalise": NORMS, "stitch_iou": STITCH_IOUS, "min_planes": MIN_PLANES,
                 "selection": "max mean F1 over all tuning volumes; ties first in grid order",
                 "scorer": "validate.instance_f1"})
    print(per_level.sort_values("mean_F1", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
