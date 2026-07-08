"""B2 nuclei — tri-tool comparison: fluorostats vs Cellpose vs StarDist vs GT.

Merges per-image nucleus counts from:
  - fluorostats (connected-component labeling)  -> b2_nuclei_bbbc039.csv
  - Cellpose (Cellpose-SAM, GPU)                 -> cellpose_counts.csv
  - StarDist (2D_versatile_fluo, GPU)            -> stardist_counts.csv (if present)
  - ground truth (BBBC039 interior-class CC)     -> b2_nuclei_bbbc039.csv

Cellpose/StarDist were run on the AMD ROCm cluster (login1.hpcfund). Reports,
for each tool vs GT: CCC, Spearman, MAE, and density-stratified error — placing
fluorostats' training-free CC counting directly against the DL instance
segmenters on the same 200 images with instance ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import agreement_report  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent / "figures"


def main():
    fs = pd.read_csv(RES / "b2_nuclei_bbbc039.csv")   # image, gt_count, fs_count
    df = fs[["image", "gt_count", "fs_count"]].copy()

    # Prefer the validated eval files (count + F1); fall back to plain counts.
    for tool in ("cellpose", "stardist"):
        ev = RES / f"{tool}_eval.csv"
        ct = RES / f"{tool}_counts.csv"
        if ev.exists():
            e = pd.read_csv(ev)[["image", f"{tool}_count", f"{tool}_f1"]]
            df = df.merge(e, on="image", how="left")
        elif ct.exists():
            df = df.merge(pd.read_csv(ct), on="image", how="left")

    df.to_csv(RES / "b2_nuclei_tritool.csv", index=False)

    tools = {"fluorostats": "fs_count", "Cellpose": "cellpose_count",
             "StarDist": "stardist_count"}
    rank = []
    for name, col in tools.items():
        if col not in df.columns:
            continue
        sub = df[["gt_count", col]].dropna()
        sub = sub[sub[col] >= 0]   # drop error sentinels (-1)
        if len(sub) < 5:
            continue
        rep = agreement_report(sub[col].values, sub["gt_count"].values, name, "GT")
        mae = float((sub[col] - sub["gt_count"]).abs().mean())
        bias = float((sub[col] - sub["gt_count"]).mean())
        rank.append({"tool": name, "n": len(sub), "CCC": round(rep["ccc"], 3),
                     "spearman": round(rep["spearman"], 3), "MAE": round(mae, 2),
                     "bias": round(bias, 2)})
    rank_df = pd.DataFrame(rank).sort_values("MAE").reset_index(drop=True)
    rank_df.to_csv(RES / "b2_nuclei_tritool_ranking.csv", index=False)

    # Baseline validation: DL instance F1 vs published values
    published = {"StarDist": 0.864, "Cellpose": 0.80}
    print("=== Baseline validation — DL instance F1@0.5 vs published ===", flush=True)
    for name, col in [("StarDist", "stardist_f1"), ("Cellpose", "cellpose_f1")]:
        if col in df.columns:
            v = df[col][df[col] >= 0]
            if len(v):
                print(f"  {name}: observed mean F1 = {v.mean():.3f}  "
                      f"(published ~{published[name]:.3f})  -> "
                      f"{'FAITHFUL' if abs(v.mean() - published[name]) < 0.08 else 'CHECK'}",
                      flush=True)
    print("\n=== Nucleus count accuracy vs GT (BBBC039, n=200) — tri-tool ===", flush=True)
    print(rank_df.to_string(index=False), flush=True)

    # density-stratified MAE per tool
    if len(df):
        df["density_bin"] = pd.qcut(df["gt_count"], 3,
                                    labels=["sparse", "medium", "crowded"])
        strat_rows = []
        for name, col in tools.items():
            if col not in df.columns:
                continue
            for b in ["sparse", "medium", "crowded"]:
                sub = df[(df.density_bin == b)][["gt_count", col]].dropna()
                sub = sub[sub[col] >= 0]
                if len(sub) < 3:
                    continue
                strat_rows.append({"tool": name, "density": b, "n": len(sub),
                                   "MAE": round(float((sub[col] - sub["gt_count"]).abs().mean()), 2)})
        strat = pd.DataFrame(strat_rows)
        strat.to_csv(RES / "b2_nuclei_tritool_by_density.csv", index=False)
        print("\nMAE by nuclear density (fluorostats CC-merge should worsen with density):",
              flush=True)
        print(strat.pivot(index="tool", columns="density", values="MAE").to_string(),
              flush=True)

    # grouped bar figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from fluorostats.style import apply_style, PALETTE, material_color
        apply_style()
        colors = {"fluorostats": PALETTE["accent"], "Cellpose": PALETTE["primary"],
                  "StarDist": PALETTE["highlight"]}
    except Exception:
        colors = {"fluorostats": "#E25C5C", "Cellpose": "#3F6AB3", "StarDist": "#E8A946"}
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(rank_df["tool"], rank_df["MAE"],
           color=[colors.get(t, "#999") for t in rank_df["tool"]],
           edgecolor="black", linewidth=0.6)
    ax.set_ylabel("MAE of nucleus count vs GT")
    ax.set_title("B2 nuclei — fluorostats vs DL segmenters (BBBC039, n=200)")
    fig.tight_layout()
    fig.savefig(FIG / "b2_nuclei_tritool.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved tri-tool ranking + figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
