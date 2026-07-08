"""B2 (3D segmentation) — fluorostats foreground accuracy vs Cell Tracking
Challenge gold-standard 3D annotations.

Public, citable benchmark: Cell Tracking Challenge (Maška 2014 Bioinformatics;
Ulman 2017 Nat Methods; Maška 2023). We use Fluo-C3DH-A549 (real confocal A549,
single cell per field) — its gold ground truth gives a clean 3D foreground mask.

fluorostats is a semantic quantifier, NOT an instance segmenter, so we report
the metrics it can legitimately win on — voxel-wise foreground Dice/Jaccard and
volume-fraction agreement vs the gold mask — NOT the CTC instance SEG/DET/TRA
scores. We cite the CTC top SEG (A549 = 0.908) only as context.
"""

from __future__ import annotations

import sys
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from fluorostats.preprocess import denoise, background_subtract
from fluorostats.segment import binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import agreement_report, plot_agreement  # noqa: E402

CTC = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/"
           "downloads/CTC")
OUT = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent / "figures"

SEG = dict(method="otsu", threshold_scale=0.9, sigma=1.0, bg_radius=25, min_size=50)


def dice_jaccard(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = pred > 0; g = gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0
    jac = inter / union if union else 1.0
    return float(dice), float(jac)


def segment(vol: np.ndarray) -> np.ndarray:
    sm = denoise(vol.astype(np.float32), sigma=SEG["sigma"])
    sm = background_subtract(sm, radius=SEG["bg_radius"])
    return binarize(sm, method=SEG["method"],
                    threshold_scale=SEG["threshold_scale"], min_size=SEG["min_size"])


def run_dataset(name: str) -> pd.DataFrame:
    rows = []
    for seq in ("01", "02"):
        gt_dir = CTC / name / f"{seq}_GT" / "SEG"
        for gt_path in sorted(glob.glob(str(gt_dir / "*.tif"))):
            m = re.search(r"man_seg0*(\d+)\.tif$", Path(gt_path).name)
            if not m:
                continue   # skip per-slice (CHO-style) GT for this foreground test
            t = int(m.group(1))
            img_path = CTC / name / seq / f"t{t:03d}.tif"
            if not img_path.exists():
                continue
            vol = tifffile.imread(img_path)
            gt = tifffile.imread(gt_path)
            mask = segment(vol)
            dice, jac = dice_jaccard(mask, gt)
            rows.append({
                "dataset": name, "seq": seq, "frame": t,
                "dice": dice, "jaccard": jac,
                "fs_vol_fraction": float((mask > 0).mean()),
                "gt_vol_fraction": float((gt > 0).mean()),
            })
    return pd.DataFrame(rows)


def main():
    df = run_dataset("Fluo-C3DH-A549")
    if df.empty:
        print("No matching A549 GT frames found.")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "b2_ctc_3d.csv", index=False)

    vf = agreement_report(df["fs_vol_fraction"], df["gt_vol_fraction"],
                          "fluorostats", "gold GT")
    summary = pd.DataFrame([{
        "dataset": "Fluo-C3DH-A549", "n_frames": len(df),
        "mean_dice": round(df["dice"].mean(), 3),
        "mean_jaccard": round(df["jaccard"].mean(), 3),
        "vol_fraction_CCC": round(vf["ccc"], 3),
        "vol_fraction_spearman": round(vf["spearman"], 3),
        "ctc_top_SEG_context": 0.908,
    }])
    summary.to_csv(OUT / "b2_ctc_3d_summary.csv", index=False)
    print("=== fluorostats vs CTC gold GT (Fluo-C3DH-A549) ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nNote: mean_dice/jaccard are SEMANTIC foreground overlap vs gold "
          "manual masks; not the CTC instance SEG score.", flush=True)

    FIG.mkdir(exist_ok=True)
    plot_agreement(df["fs_vol_fraction"], df["gt_vol_fraction"],
                   FIG / "b2_ctc_volfraction.png", "fluorostats", "gold GT",
                   title="B2 3D — volume fraction vs CTC gold GT (Fluo-C3DH-A549)")
    print(f"\nSaved. mean Dice={df['dice'].mean():.3f}, "
          f"mean Jaccard={df['jaccard'].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
