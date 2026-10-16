"""Produce results/ve_crops.npz (+ ve_crops.csv) — the Fig 3(b) VesselExpress display crops.

Two VesselExpress Figure 2C-E volumes (``VOLUMES``, left and right of one
sample). For each: the maximum over the ``MIP_SLICES`` planes centred on the mid
plane (``shape[0] // 2``), cropped to the centred ``CROP`` x ``CROP`` window. Saved
per crop ``i``: ``raw<i>`` (raw intensities, float32), ``ve<i>`` (the VesselExpress
software's Binary segmentation) and ``fs<i>`` (fluorostats
``segment.binarize(method="auto", min_size=50)`` on the whole volume, as
``b_ve_metrics``), each projected the same way; masks 0/1 uint8. ``ve_crops.csv``
records each window and the two vessel fractions inside it.
Data: ``$FLUOROSTATS_DATA/vesselexpress/Figure2C_E/VEA_0.25mm^3/{Raw,Binary}``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from common import DATA, RESULTS, require, write_table
from fluorostats.segment import binarize

ROOT = DATA / "vesselexpress" / "Figure2C_E" / "VEA_0.25mm^3"
VOLUMES = ("210426_592-1-lepleft_500x_VEA", "210426_592-1-lepright_500x_VEA")
MIP_SLICES = 3
CROP = 400
MIN_SIZE = 50   # as b_ve_metrics


def mid_mip_crop(vol: np.ndarray) -> np.ndarray:
    """Max over the MIP_SLICES planes centred on the mid plane, centred CROP window."""
    z = vol.shape[0] // 2
    mip = vol[z - MIP_SLICES // 2: z + MIP_SLICES // 2 + 1].max(axis=0)
    y0, x0 = (mip.shape[0] - CROP) // 2, (mip.shape[1] - CROP) // 2
    return mip[y0:y0 + CROP, x0:x0 + CROP]


def main():
    arrays, rows = {}, []
    for i, name in enumerate(VOLUMES):
        raw = tifffile.imread(require(ROOT / "Raw" / f"{name}.tif")).astype(np.float32)
        ve = tifffile.imread(require(ROOT / "Binary" / f"Binary_{name}.tiff")) > 0
        fs = binarize(raw, method="auto", min_size=MIN_SIZE)
        arrays[f"raw{i}"] = mid_mip_crop(raw)
        arrays[f"ve{i}"] = mid_mip_crop(ve).astype(np.uint8)
        arrays[f"fs{i}"] = mid_mip_crop(fs).astype(np.uint8)
        rows.append({"crop": i, "volume": name, "z_mid": raw.shape[0] // 2,
                     "y0": (raw.shape[1] - CROP) // 2, "x0": (raw.shape[2] - CROP) // 2,
                     "size": CROP,
                     "VesselExpress_fraction": round(float(arrays[f"ve{i}"].mean()), 5),
                     "fluorostats_fraction": round(float(arrays[f"fs{i}"].mean()), 5)})
    np.savez_compressed(RESULTS / "ve_crops.npz", **arrays)
    df = pd.DataFrame(rows)
    write_table(df, "ve_crops", {"volumes": list(VOLUMES), "mip_slices": MIP_SLICES,
                                 "plane": "centred on shape[0] // 2", "crop": f"{CROP} centred",
                                 "fluorostats": f'binarize(method="auto", min_size={MIN_SIZE}) on the volume',
                                 "arrays": "results/ve_crops.npz"})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
