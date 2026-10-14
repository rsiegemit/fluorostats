"""Produce results/b_density_normalization.csv and b_density_normalization_cv.csv.

One physical sample (the first BBBC024 volume's exact GT object count) is
"re-imaged" at several digital zooms. Five count normalisations are compared by
their coefficient of variation across zoom: only fluorostats' per-mm³ density
(``objects.object_density_per_mm3``) is invariant in both regimes:

  A) fixed physical FOV, varying voxel size (pure digital zoom / rebinning)
  B) fixed voxel size, varying crop extent (objects scale with FOV fraction)

Data: ``$FLUOROSTATS_DATA/BBBC024`` (only the first label volume is read).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from common import RESULTS, bbbc024_volumes
from fluorostats.metrics_3d import fov_volume_mm3
from fluorostats.objects import object_density_per_mm3
from fluorostats.stats import coefficient_of_variation

NATIVE_VOXEL_UM = (1.0, 0.5, 0.5)  # (z, y, x) µm per voxel assigned at zoom 1.0
ZOOM_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
FOV_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SCHEMES = ["fluorostats_per_mm3", "raw_count", "per_megavoxel", "per_mm2_area", "per_z_slice"]


def schemes(count, shape_zyx, voxel_um):
    n_vox = float(np.prod(shape_zyx))
    area_mm2 = shape_zyx[1] * shape_zyx[2] * voxel_um[1] * voxel_um[2] * 1e-6
    return {
        "fluorostats_per_mm3": object_density_per_mm3(count, shape_zyx, voxel_um),
        "raw_count": float(count),
        "per_megavoxel": count / (n_vox / 1e6),
        "per_mm2_area": count / area_mm2 if area_mm2 > 0 else np.nan,
        "per_z_slice": count / shape_zyx[0],
    }


def regime_a(native_shape, count):
    fov_um = tuple(s * v for s, v in zip(native_shape, NATIVE_VOXEL_UM))
    rows = []
    for z in ZOOM_FACTORS:
        shape = tuple(max(1, int(round(s * z))) for s in native_shape)
        voxel = tuple(f / n for f, n in zip(fov_um, shape))
        rows.append({**schemes(count, shape, voxel), "regime": "A_fixed_FOV", "zoom": z,
                     "shape": str(shape), "voxel_um": str(tuple(round(v, 4) for v in voxel)),
                     "fov_vol_mm3": round(fov_volume_mm3(shape, voxel), 6)})
    return rows


def regime_b(native_shape, count):
    rows = []
    for frac in FOV_FRACTIONS:
        shape = tuple(max(1, int(round(s * (frac ** (1 / 3))))) for s in native_shape)
        c = max(1, int(round(count * frac)))
        rows.append({**schemes(c, shape, NATIVE_VOXEL_UM), "regime": "B_fixed_voxel", "zoom": frac,
                     "shape": str(shape), "voxel_um": str(NATIVE_VOXEL_UM),
                     "fov_vol_mm3": round(fov_volume_mm3(shape, NATIVE_VOXEL_UM), 6)})
    return rows


def main():
    gt = tifffile.imread(bbbc024_volumes(n=1)[0][1])
    native_shape = tuple(int(s) for s in gt.shape)
    count = int(len(np.unique(gt)) - 1)
    df = pd.DataFrame(regime_a(native_shape, count) + regime_b(native_shape, count))
    df.to_csv(RESULTS / "b_density_normalization.csv", index=False)
    summ = pd.DataFrame([{"regime": regime, "scheme": col,
                          "CV_percent_across_zoom": round(coefficient_of_variation(sub[col].values, ddof=0) * 100, 3),
                          "mean_value": round(float(sub[col].mean()), 4)}
                         for regime, sub in df.groupby("regime") for col in SCHEMES])
    summ.to_csv(RESULTS / "b_density_normalization_cv.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
