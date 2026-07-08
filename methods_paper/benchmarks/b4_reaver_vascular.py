"""B4 (vascular) — fluorostats vs manual ground truth on the REAVER dataset.

Public, citable, manually-annotated benchmark (Corliss et al. 2020,
Microcirculation; Zenodo 10.5281/zenodo.3340165, CC-BY-4.0). 36 confocal
vascular images across 9 murine tissues, with expert manual ground truth.

Ground truth (per image):
    vessel area fraction = fraction of the manual RED channel that is vessel
    branchpoint count    = number of manually-marked branchpoints
                           (BranchpointsByName.mat, rows per image)

fluorostats runs its standard 2D pipeline on the raw greyscale image and we
compare against the manual GT with the agreement harness (Bland-Altman + CCC).
REAVER's paper reports it cut AngioTool's error by -75.8% (area fraction) and
-94.6% (branchpoints) vs manual; the bar is REAVER-class low MAE.

Honest scope: fluorostats is a general quantifier, not a vessel-specialised
tool — the claim is parity-class accuracy on a competitor's own benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.io import loadmat
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from skan import Skeleton, summarize

from fluorostats.preprocess import denoise, background_subtract
from fluorostats.segment import binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import agreement_report, plot_agreement  # noqa: E402

ROOT = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/"
            "downloads/REAVER/REAVER_Vascular_Networks_Image_Dataset")
OUT = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent / "figures"

# fluorostats 2D segmentation params (fixed, not per-image tuned)
SEG = dict(method="otsu", threshold_scale=0.85, sigma=2.0, bg_radius=75, min_size=100)
MIN_BRANCH_PX = 15   # spur length below which a junction-to-tip branch is pruned


def _branchpoint_nodes(skel: np.ndarray) -> int:
    k = np.ones((3, 3), int); k[1, 1] = 0
    nbr = ndi.convolve(skel.astype(int), k, mode="constant")
    bp = skel & (nbr >= 3)
    return int(ndi.label(bp)[1])


def count_branchpoints(mask: np.ndarray, prune: bool = True) -> int:
    """Branchpoint NODE count (REAVER/AngioTool definition).

    With prune=True, smooths the mask and iteratively removes short spur
    branches (junction-to-endpoint segments below MIN_BRANCH_PX) before
    counting — the standard step every vascular tool performs. Without it,
    a raw skeleton of a rough mask hugely over-counts junctions.
    """
    m = ndi.binary_closing(mask, iterations=2)
    m = ndi.binary_opening(m, iterations=1)
    skel = skeletonize(m)
    if not prune or skel.sum() < 10:
        return _branchpoint_nodes(skel)
    for _ in range(4):
        try:
            s = Skeleton(skel)
            df = summarize(s, separator="_")
        except Exception:
            break
        spurs = df.index[(df["branch_type"] == 1)
                         & (df["branch_distance"] < MIN_BRANCH_PX)]
        if len(spurs) == 0:
            break
        for i in spurs:
            coords = s.path_coordinates(i).astype(int)
            skel[coords[:, 0], coords[:, 1]] = False
        skel = skeletonize(skel)
    return _branchpoint_nodes(skel)


def fluorostats_measure(gray: np.ndarray) -> dict:
    sm = denoise(gray.astype(np.float32), sigma=SEG["sigma"])
    sm = background_subtract(sm, radius=SEG["bg_radius"])
    mask = binarize(sm, method=SEG["method"],
                    threshold_scale=SEG["threshold_scale"], min_size=SEG["min_size"])
    return {
        "area_fraction": float(mask.mean()),
        "branchpoints": count_branchpoints(mask, prune=True),
        "branchpoints_raw": count_branchpoints(mask, prune=False),
    }


def main():
    bp_mat = loadmat(ROOT / "Manual" / "BranchpointsByName.mat")
    names = [k for k in bp_mat if not k.startswith("__")]
    gt_bp = {n: int(bp_mat[n].shape[0]) for n in names}

    rows = []
    for name in names:
        orig = tifffile.imread(ROOT / "_Original_Images" / f"{name}.tif")
        if orig.ndim == 3:
            orig = orig[..., 0]
        man = tifffile.imread(ROOT / "Manual" / f"{name}.tif")
        gt_af = float((man[..., 0] > 127).mean())   # red channel = vessel mask
        fs = fluorostats_measure(orig)
        rows.append({
            "image": name,
            "gt_area_fraction": gt_af,
            "fs_area_fraction": fs["area_fraction"],
            "gt_branchpoints": gt_bp[name],
            "fs_branchpoints": fs["branchpoints"],
        })
        print(f"  {name:20s} AF gt={gt_af:.3f} fs={fs['area_fraction']:.3f}  "
              f"BP gt={gt_bp[name]:3d} fs={fs['branchpoints']:3d}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "b4_reaver_vascular.csv", index=False)

    # Agreement + MAE
    af = agreement_report(df["fs_area_fraction"], df["gt_area_fraction"],
                          "fluorostats", "manual GT")
    bp = agreement_report(df["fs_branchpoints"], df["gt_branchpoints"],
                          "fluorostats", "manual GT")
    af_mae = float(np.abs(df["fs_area_fraction"] - df["gt_area_fraction"]).mean())
    bp_mae = float(np.abs(df["fs_branchpoints"] - df["gt_branchpoints"]).mean())

    summary = pd.DataFrame([
        {"metric": "vessel_area_fraction", "n": af["n"], "CCC": round(af["ccc"], 3),
         "ICC": round(af["icc"], 3), "spearman": round(af["spearman"], 3),
         "bias": round(af["bias"], 4), "MAE": round(af_mae, 4)},
        {"metric": "branchpoint_count", "n": bp["n"], "CCC": round(bp["ccc"], 3),
         "ICC": round(bp["icc"], 3), "spearman": round(bp["spearman"], 3),
         "bias": round(bp["bias"], 2), "MAE": round(bp_mae, 2)},
    ])
    summary.to_csv(OUT / "b4_reaver_summary.csv", index=False)
    print("\n=== fluorostats vs manual GT (REAVER dataset, n=36) ===", flush=True)
    print(summary.to_string(index=False), flush=True)

    FIG.mkdir(exist_ok=True)
    plot_agreement(df["fs_area_fraction"], df["gt_area_fraction"],
                   FIG / "b4_reaver_area_fraction.png",
                   "fluorostats", "manual GT",
                   title="B4 vascular — vessel area fraction vs manual GT (REAVER dataset)")
    plot_agreement(df["fs_branchpoints"], df["gt_branchpoints"],
                   FIG / "b4_reaver_branchpoints.png",
                   "fluorostats", "manual GT",
                   title="B4 vascular — branchpoint count vs manual GT (REAVER dataset)")
    print(f"\nSaved results + figures. Area-fraction CCC={af['ccc']:.3f}, "
          f"branchpoint CCC={bp['ccc']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
