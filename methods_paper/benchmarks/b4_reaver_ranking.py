"""B4 competitor ranking — fluorostats placed among all tools on REAVER data.

The REAVER dataset ships the segmentation output of every tool it compared
(REAVER, AngioTool, AngioQuant, ImageJ, RAVE) alongside expert manual ground
truth, as RGB TIFFs whose RED channel is the binary vessel mask. We compute
each tool's vessel area fraction, compare to manual GT across all 36 images,
and rank every tool by mean absolute error — inserting fluorostats into that
exact ranking on the competitors' own home-turf data.

fluorostats area fractions are reused from b4_reaver_vascular.csv (same images,
same fixed pipeline) to avoid re-segmenting.
"""

from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import agreement_report  # noqa: E402

ROOT = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/"
            "downloads/REAVER/REAVER_Vascular_Networks_Image_Dataset")
OUT = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent / "figures"

TOOLS = ["REAVER_Auto", "AngioTool_Auto", "AngioQuant_Auto", "ImageJ_Auto", "RAVE_Auto"]


def red_area_fraction(path: Path) -> float:
    a = tifffile.imread(path)
    red = a[..., 0] if a.ndim == 3 else a
    return float((red > 127).mean())


def main():
    # image names from manual GT tifs
    names = [Path(p).stem for p in sorted(glob.glob(str(ROOT / "Manual" / "*.tif")))]

    # fluorostats area fractions from the B4 run
    fs_csv = OUT / "b4_reaver_vascular.csv"
    fs_af = {}
    if fs_csv.exists():
        d = pd.read_csv(fs_csv)
        fs_af = dict(zip(d["image"], d["fs_area_fraction"]))

    rows = []
    for name in names:
        man = ROOT / "Manual" / f"{name}.tif"
        if not man.exists():
            continue
        rec = {"image": name, "manual_GT": red_area_fraction(man)}
        for tool in TOOLS:
            tp = ROOT / tool / f"{name}.tif"
            rec[tool.replace("_Auto", "")] = red_area_fraction(tp) if tp.exists() else np.nan
        if name in fs_af:
            rec["fluorostats"] = fs_af[name]
        rows.append(rec)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "b4_reaver_ranking_perimage.csv", index=False)

    # Rank every tool by MAE vs manual GT
    tools = [c for c in df.columns if c not in ("image", "manual_GT")]
    rank = []
    for t in tools:
        sub = df[["manual_GT", t]].dropna()
        if len(sub) < 3:
            continue
        mae = float((sub[t] - sub["manual_GT"]).abs().mean())
        rep = agreement_report(sub[t].values, sub["manual_GT"].values, t, "manual")
        rank.append({"tool": t, "n": len(sub), "MAE": round(mae, 4),
                     "CCC": round(rep["ccc"], 3), "spearman": round(rep["spearman"], 3),
                     "bias": round(rep["bias"], 4)})
    rank_df = pd.DataFrame(rank).sort_values("MAE").reset_index(drop=True)
    rank_df.insert(0, "rank", rank_df.index + 1)
    rank_df.to_csv(OUT / "b4_reaver_ranking.csv", index=False)

    print("=== Vessel area fraction accuracy vs manual GT (REAVER dataset, ranked) ===",
          flush=True)
    print(rank_df.to_string(index=False), flush=True)
    fs_row = rank_df[rank_df["tool"] == "fluorostats"]
    if not fs_row.empty:
        r = int(fs_row["rank"].iloc[0])
        print(f"\nfluorostats ranks #{r} of {len(rank_df)} tools by MAE "
              f"(vs the vessel-specialised tools on their own data).", flush=True)

    # Bar chart of MAE by tool
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from fluorostats.style import apply_style, PALETTE
        apply_style()
    except Exception:
        PALETTE = {"accent": "#E25C5C", "primary": "#3F6AB3", "muted": "#9AA4B0"}
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [PALETTE["accent"] if t == "fluorostats" else PALETTE["muted"]
              for t in rank_df["tool"]]
    ax.bar(rank_df["tool"], rank_df["MAE"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("MAE of vessel area fraction vs manual GT")
    ax.set_title("B4 — tool accuracy on the REAVER dataset (lower is better)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "b4_reaver_ranking.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved ranking + figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
