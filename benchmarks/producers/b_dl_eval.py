"""Produce results/{stardist,cellpose,omnipose}_eval.csv — DL baselines on all 200 BBBC039 images.

Scores the label predictions written by ``cluster/bbbc024_dl_infer.py <tool> 2d``
(StarDist ``2D_versatile_fluo``, Cellpose ``nuclei``, Omnipose ``cyto2_omni``;
the script stops rather than substitute a model) against the BBBC039 ground truth
reconstruction of ``common.bbbc039_gt_instances``, with ``validate.instance_f1``
at IoU 0.5. Per image: the predicted object count and F1 (4 d.p.). Each table's
``.run.json`` carries the cluster run's ``versions.json`` (library versions,
model, weights file and SHA-256, command).
Predictions: ``$FLUOROSTATS_DATA/BBBC039_dl/<tool>/<image>.tif`` (DATA_MANIFEST.md).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tifffile

from common import DATA, bbbc039_gt_instances, bbbc039_pairs, require, write_table
from fluorostats.validate import instance_f1

TOOLS = ("stardist", "cellpose", "omnipose")
PRED = DATA / "BBBC039_dl"


def evaluate(tool: str, pairs: list, gts: dict) -> pd.DataFrame:
    rows = []
    for stem, _, _ in pairs:
        pred = tifffile.imread(require(PRED / tool / f"{stem}.tif"))
        rows.append({"image": stem, f"{tool}_count": int(len(np.unique(pred[pred > 0]))),
                     f"{tool}_f1": round(instance_f1(pred, gts[stem])["f1"], 4)})
    return pd.DataFrame(rows)


def main():
    pairs = bbbc039_pairs()
    gts = {stem: bbbc039_gt_instances(mask) for stem, _, mask in pairs}
    for tool in TOOLS:
        run = json.loads(require(PRED / tool / "versions.json").read_text(encoding="utf-8"))
        run = {k: v for k, v in run.items() if k not in ("seconds", "host", "diameters_3d")}
        df = evaluate(tool, pairs, gts)
        write_table(df, f"{tool}_eval", {"images": "all BBBC039 with masks", "n_images": len(df),
                                         "iou_threshold": 0.5, "scorer": "validate.instance_f1",
                                         "inference": run})
        print(f"{tool}: n={len(df)} mean F1 {df[f'{tool}_f1'].mean():.4f}")


if __name__ == "__main__":
    main()
