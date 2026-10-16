"""Produce results/b_seg_timing.csv — 2D segmentation time per image, all methods on one CPU.

fluorostats (``cluster/fs_time.py``), StarDist and Cellpose
(``cluster/bbbc024_dl_infer.py ... 2d``) segmented the first 10 BBBC039 images in
5 passes each, one after the other in one cluster job on one node, on its CPU.
Each pass is a new process; the time is per image, model loading and file reading
excluded (Cellpose's diameter estimate included). The table gives the median and
quartiles over the 50 timings per method; the run record names the node's CPU,
the fluorostats commit installed there and the model versions.
Inputs: ``$FLUOROSTATS_DATA/BBBC039_timing/`` (see DATA_MANIFEST.md).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA, DENOISED_OTSU_CC, RAW_OTSU_CC, require, write_table

TIMING = DATA / "BBBC039_timing"
N_PASSES = 5
FS_METHODS = {"denoised Otsu+CC": DENOISED_OTSU_CC, "raw Otsu+CC": RAW_OTSU_CC}
DL_TOOLS = {"stardist": "StarDist", "cellpose": "Cellpose"}


def passes(tool: str) -> list[dict]:
    return [json.loads(require(TIMING / tool / f"pass_{k}" / "versions.json").read_text())
            for k in range(1, N_PASSES + 1)]


def row(method: str, seconds: list[float]) -> dict:
    ms = 1000 * np.asarray(seconds)
    return {"method": method, "median_ms": round(float(np.median(ms)), 2),
            "q25_ms": round(float(np.percentile(ms, 25)), 2),
            "q75_ms": round(float(np.percentile(ms, 75)), 2), "n_timings": int(ms.size)}


def main():
    node = json.loads(require(TIMING / "node.json").read_text())
    fs = passes("fluorostats")
    rows = [row(name, [s for p in fs for s in p["seconds"][key].values()])
            for key, name in FS_METHODS.items()]
    versions = {"fluorostats": {k: fs[0][k] for k in ("fluorostats", "numpy", "scikit-image",
                                                         "python", "min_size")}}
    for tool, name in DL_TOOLS.items():
        dl = passes(tool)
        rows.append(row(name, [s for p in dl for s in p["seconds"].values()]))
        versions[tool] = {k: v for k, v in dl[0].items() if k not in ("seconds", "host", "n_files")}
    df = pd.DataFrame(rows)
    write_table(df, "b_seg_timing",
                {"images": "first 10 BBBC039 (sorted)", "passes": N_PASSES,
                 "cpu": node["cpu"], "nproc": node["nproc"],
                 "fluorostats_commit": (TIMING / "fs_commit.txt").read_text().strip(),
                 "versions": versions})   # the timed CPU is in params, not this machine
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
