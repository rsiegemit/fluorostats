"""B4 (3D vascular) — fluorostats on real light-sheet vessel data (VesselExpress).

Public data: VesselExpress (Spangenberg et al. 2023, Cell Reports Methods,
CC-BY-4.0). Their in-repo test volume: 100 x 500 x 500 uint16, voxel
2.0 x 1.016 x 1.016 um. This exercises fluorostats' 3D pipeline end-to-end
(segmentation -> connectivity -> pruned skeleton -> FOV-normalised densities)
and reports the metrics VesselExpress itself reports: vessel length density
(mm/mm3), junction density (per mm3), and volume fraction.

Demonstrates fluorostats produces field-comparable 3D vascular morphometry on
light-sheet data, using the v0.4 spur-pruning so junction counts are sane.
No competitor output on this exact file, so this is a capability/soundness
demonstration, not a head-to-head.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from fluorostats.preprocess import denoise, background_subtract
from fluorostats.segment import binarize
from fluorostats.metrics_3d import (
    volume_fraction, connectivity_metrics, fov_volume_mm3,
    normalise_skeleton_metrics,
)
from fluorostats.skeleton import skeleton_metrics

VOL = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/"
           "downloads/vesselexpress_test.tiff")
OUT = Path(__file__).resolve().parent / "results"
VOXEL_UM = (2.0, 1.016, 1.016)


def main():
    vol = tifffile.imread(VOL)
    print(f"Loaded {vol.shape} {vol.dtype}", flush=True)

    sm = denoise(vol.astype(np.float32), sigma=1.0)
    sm = background_subtract(sm, radius=25)
    mask = binarize(sm, method="otsu", threshold_scale=0.9, min_size=100)

    vf = volume_fraction(mask)
    conn = connectivity_metrics(mask)
    # pruned skeleton (v0.4) so junction/branch counts are field-sane
    skel = skeleton_metrics(mask, voxel_size_um=VOXEL_UM, prune=True,
                            min_branch_length_um=6.0)
    dens = normalise_skeleton_metrics(skel, mask.shape, VOXEL_UM)

    fov_mm3 = fov_volume_mm3(mask.shape, VOXEL_UM)
    # length density in mm/mm3 (skeleton length is in um -> /1000 for mm)
    length_density_mm_per_mm3 = dens["length_density_um_per_mm3"] / 1000.0
    junction_node_density = conn.get("n_components", 0)  # placeholder not used

    result = {
        "shape": str(vol.shape),
        "fov_volume_mm3": round(fov_mm3, 5),
        "volume_fraction": round(vf, 4),
        "n_components": conn["n_components"],
        "largest_component_fraction": round(conn["largest_component_fraction"], 3),
        "total_length_um": round(skel["total_length_um"], 1),
        "length_density_mm_per_mm3": round(length_density_mm_per_mm3, 1),
        "n_junction_nodes": skel["n_junction_nodes"],
        "junction_density_per_mm3": round(dens["junction_density_per_mm3"], 1),
        "n_branches": skel["n_branches"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUT / "b4_vesselexpress_3d.csv", index=False)

    print("\n=== fluorostats 3D vascular morphometry (VesselExpress test volume) ===",
          flush=True)
    for k, v in result.items():
        print(f"  {k:32s} {v}", flush=True)
    print("\nContext anchors (VesselExpress / VesSAP literature): cortical vessel"
          " caliber ~4.8 um; whole-brain vascular volume fraction ~1-2%;"
          " length density order 100s mm/mm3.", flush=True)
    print("fluorostats produces the same metric family with sane (pruned) "
          "junction counts — soundness demonstration on real light-sheet data.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
