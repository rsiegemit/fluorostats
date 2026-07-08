"""B2 (nuclei) — fluorostats connected-component counting vs GT on BBBC039.

Public, citable benchmark: BBBC039 "Nuclei of U2OS cells" (Broad Bioimage
Benchmark Collection; Caicedo et al. 2019, Nat Methods; CC0). 200 single-channel
Hoechst fluorescence fields with expert instance ground truth (3-class masks:
background / interior / boundary — the boundary class separates touching nuclei,
so GT count = connected components of the interior class).

fluorostats counts nuclei by thresholding + 3D-style connected-component
labeling (objects.label_3d on the 2D field). Honest expectation: parity with GT
when nuclei are well separated, growing under-count as local density rises
(CC labeling merges touching nuclei — the exact regime StarDist/Cellpose exist
to fix). We report agreement (Bland-Altman, CCC, ICC) and %error stratified by
density tertile.
"""

from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from fluorostats.preprocess import denoise
from fluorostats.segment import binarize
from fluorostats.objects import label_3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import agreement_report, plot_agreement  # noqa: E402

DL = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/"
          "downloads/BBBC039")
OUT = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent / "figures"

SEG = dict(method="otsu", threshold_scale=1.0, sigma=1.0, min_size=20)


def gt_count(mask_path: Path) -> int:
    m = np.array(Image.open(mask_path))
    r = m[..., 0] if m.ndim == 3 else m       # interior class == 1
    return int(ndi.label(r == 1)[1])


def fs_count(img: np.ndarray) -> int:
    sm = denoise(img.astype(np.float32), sigma=SEG["sigma"])
    mask = binarize(sm, method=SEG["method"],
                    threshold_scale=SEG["threshold_scale"], min_size=SEG["min_size"])
    _, n = label_3d(mask, min_size=SEG["min_size"])
    return int(n)


def main():
    imgs = {Path(p).stem: p for p in
            glob.glob(str(DL / "images" / "images" / "*.tif"))}
    masks = {Path(p).stem: p for p in
             glob.glob(str(DL / "masks" / "masks" / "*.png"))}
    common = sorted(set(imgs) & set(masks))
    print(f"Matched {len(common)} image/mask pairs", flush=True)

    rows = []
    for i, stem in enumerate(common):
        img = tifffile.imread(imgs[stem])
        g = gt_count(Path(masks[stem]))
        f = fs_count(img)
        rows.append({"image": stem, "gt_count": g, "fs_count": f,
                     "err": f - g, "abs_pct_err": abs(f - g) / max(1, g) * 100})
        if i % 40 == 0:
            print(f"  [{i}/{len(common)}] {stem[:22]} gt={g} fs={f}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "b2_nuclei_bbbc039.csv", index=False)

    rep = agreement_report(df["fs_count"], df["gt_count"], "fluorostats", "GT")
    mae = float(df["err"].abs().mean())

    # density stratification by GT tertile
    df["density_bin"] = pd.qcut(df["gt_count"], 3, labels=["sparse", "medium", "crowded"])
    strat = df.groupby("density_bin", observed=True).agg(
        n=("image", "count"),
        gt_median=("gt_count", "median"),
        mean_signed_err=("err", "mean"),
        mean_abs_pct_err=("abs_pct_err", "mean"),
    ).reset_index()

    summary = pd.DataFrame([{
        "n": rep["n"], "CCC": round(rep["ccc"], 3), "ICC": round(rep["icc"], 3),
        "spearman": round(rep["spearman"], 3), "bias": round(rep["bias"], 2),
        "MAE": round(mae, 2),
        "mean_abs_pct_err": round(float(df["abs_pct_err"].mean()), 1),
    }])
    summary.to_csv(OUT / "b2_nuclei_summary.csv", index=False)
    strat.to_csv(OUT / "b2_nuclei_by_density.csv", index=False)

    print("\n=== fluorostats vs GT nucleus count (BBBC039, n=200) ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nBy density (honest under-count expected as density rises):", flush=True)
    print(strat.to_string(index=False), flush=True)

    FIG.mkdir(exist_ok=True)
    plot_agreement(df["fs_count"], df["gt_count"], FIG / "b2_nuclei_bbbc039.png",
                   "fluorostats", "GT count",
                   title="B2 nuclei — fluorostats CC count vs GT (BBBC039, n=200)")
    print(f"\nSaved. CCC={rep['ccc']:.3f}, Spearman={rep['spearman']:.3f}, MAE={mae:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
