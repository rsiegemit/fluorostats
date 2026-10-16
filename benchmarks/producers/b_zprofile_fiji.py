"""Produce results/b_zprofile_fiji.csv — depth.intensity_depth_profile vs Fiji's Plot Z-axis Profile.

Fiji's "Image > Stacks > Plot Z-axis Profile" (no selection: each slice's mean over
the whole image) was run on two public 16-bit z-stacks by
``cluster/fiji_zprofile.ijm`` (ImageJ 1.54p; values exported at 9 decimals), and
compared here with ``depth.intensity_depth_profile(...).mean`` (double precision).
Fiji keeps profile values in single precision, so the table reports the largest
absolute and relative difference, and the largest difference once the fluorostats
value is rounded to single precision (what Fiji can represent; the remaining
difference is the 9-decimal export).
Data: ``$FLUOROSTATS_DATA/zprofile_fiji/<stack>.csv`` and the stacks below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from common import DATA, require, write_table
from fluorostats.depth import intensity_depth_profile

STACKS = {
    "bbbc024_c00_0000": DATA / "BBBC024" / "image-final_0000.tif",
    "vea_592-1-lepleft": DATA / "vesselexpress" / "Figure2C_E" / "VEA_0.25mm^3" / "Raw"
    / "210426_592-1-lepleft_500x_VEA.tif",
}


def compare(name, path) -> dict:
    fs = intensity_depth_profile(tifffile.imread(require(path)), 1.0).mean
    fiji = pd.read_csv(require(DATA / "zprofile_fiji" / f"{name}.csv"))["fiji_mean"].to_numpy()
    if fs.shape != fiji.shape:
        raise SystemExit(f"{name}: {fs.size} fluorostats slices vs {fiji.size} Fiji slices")
    diff = np.abs(fs - fiji)
    single = np.abs(fs.astype(np.float32).astype(np.float64) - fiji)
    return {"stack": name, "n_slices": int(fs.size), "max_abs_diff": float(diff.max()),
            "max_rel_diff": float((diff / np.abs(fiji)).max()),
            "max_abs_diff_at_single_precision": float(single.max())}


def main():
    df = pd.DataFrame([compare(n, p) for n, p in STACKS.items()])
    write_table(df, "b_zprofile_fiji", {"fiji": "ImageJ 1.54p (Fiji), Plot Z-axis Profile, no selection",
                                        "export_decimals": 9, "stacks": {k: str(v.relative_to(DATA))
                                                                         for k, v in STACKS.items()}})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
