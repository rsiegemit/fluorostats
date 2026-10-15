"""Produce results/b_ve_metrics.csv — vessel volume fraction, fluorostats vs VesselExpress software.

For each light-sheet volume of the VesselExpress paper's Figure 2C-E set, the
VesselExpress software's own binary segmentation (``Binary/Binary_<name>*``) is
compared with fluorostats ``segment.binarize(method="auto", min_size=50)``
(auto resolves to ``common.VE_AUTO_RESOLVED`` on every volume; checked here) by ``metrics_3d.volume_fraction``.
Skeleton length is omitted: skeletonising the full dense volumes is intractable.

Data: the ``Figure2C_E/*/Raw`` and ``Figure2C_E/*/Binary`` folders of
``VesselExpress_Data.zip`` (Zenodo 6025935) unpacked under
``$FLUOROSTATS_DATA/vesselexpress`` — the full deposit, not just ``test.tiff``.
The committed table was produced on the cluster (9 volumes, ~250 MB each).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import tifffile

from common import DATA, SMALL_OBJECT_RULE, VE_AUTO_RESOLVED, require, write_table
from fluorostats.metrics_3d import volume_fraction
from fluorostats.segment import binarize, choose_threshold_method


def main():
    root = require(DATA / "vesselexpress")
    rows = []
    for rp in sorted(glob.glob(str(root / "**" / "Raw" / "*.tif"), recursive=True)):
        name = os.path.basename(rp)
        binary = glob.glob(os.path.dirname(rp).replace("/Raw", "/Binary") + "/Binary_" + name.replace(".tif", "*"))
        if not binary:
            print(f"skipped {name}: no VesselExpress binary")
            continue
        ve = tifffile.imread(binary[0]) > 0
        raw = tifffile.imread(rp).astype(np.float32)
        resolved = choose_threshold_method(raw.astype(np.float64))["method"]
        if resolved != VE_AUTO_RESOLVED:
            raise SystemExit(f"{name}: auto resolved to {resolved!r}, not {VE_AUTO_RESOLVED!r}; "
                             "update common.VE_AUTO_RESOLVED and the fig3 label")
        fs = binarize(raw, method="auto", min_size=50)
        rows.append({"volume": name, "VesselExpress_VF": round(volume_fraction(ve), 5),
                     "fluorostats_VF": round(volume_fraction(fs), 5)})
    df = pd.DataFrame(rows)
    write_table(df, "b_ve_metrics", {"method": "auto", "auto_resolved": VE_AUTO_RESOLVED, "min_size": 50, "small_object_rule": SMALL_OBJECT_RULE})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
