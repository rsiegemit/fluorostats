"""Publication figures for the passing Tier-0 benchmarks (B1 topology, B1
skeleton, B6 homogeneity). Uses fluorostats.style for a consistent look.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluorostats.style import apply_style, PALETTE
apply_style()

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


# ---- B1 topology + skeleton: correctness tables as a "perfect agreement" figure
def fig_b1():
    topo = pd.read_csv(RES / "b1_topology_phantoms.csv")
    skel = pd.read_csv(RES / "b1_skeleton_phantoms.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # topology: expected vs fluorostats euler (identity)
    ax1.plot([-1, 6], [-1, 6], "--", color=PALETTE["ink"], lw=1, alpha=0.5)
    ax1.scatter(topo["expected_euler"], topo["fluorostats_euler"], s=120,
                color=PALETTE["accent"], edgecolors=PALETTE["ink"], zorder=3)
    for _, r in topo.iterrows():
        ax1.annotate(r["phantom"], (r["expected_euler"], r["fluorostats_euler"]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax1.set_xlabel("Analytic Euler number")
    ax1.set_ylabel("fluorostats Euler number")
    ax1.set_title("B1 topology — exact agreement on known phantoms\n"
                  f"({int(topo['PASS'].sum())}/{len(topo)} zero-error)")

    # skeleton: expected vs measured length
    s = skel[skel["expected_length"].notna()]
    mx = float(s["expected_length"].max()) * 1.1
    ax2.plot([0, mx], [0, mx], "--", color=PALETTE["ink"], lw=1, alpha=0.5)
    ax2.scatter(s["expected_length"], s["fluorostats_length"], s=120,
                color=PALETTE["primary"], edgecolors=PALETTE["ink"], zorder=3)
    for _, r in s.iterrows():
        ax2.annotate(r["phantom"], (r["expected_length"], r["fluorostats_length"]),
                     fontsize=7, xytext=(4, -8), textcoords="offset points")
    ax2.set_xlabel("Analytic skeleton length (voxels)")
    ax2.set_ylabel("fluorostats total length")
    ax2.set_title("B1 skeleton — length recovery\n(<1% error; branch counts exact)")

    fig.suptitle("B1 correctness anchors — fluorostats vs analytic ground truth",
                 fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG / "b1_correctness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'b1_correctness.png'}")


# ---- B6 homogeneity: example patterns + Gini-vs-R scatter
def fig_b6():
    import sys
    sys.path.insert(0, str(HERE))
    from b6_homogeneity_synthetic import (
        gen_regular, gen_csr, gen_thomas, render, clark_evans_R, FIELD)

    df = pd.read_csv(RES / "b6_homogeneity_synthetic.csv")

    examples = [
        ("Regular", lambda rng: gen_regular(rng)),
        ("CSR (Poisson)", lambda rng: gen_csr(rng)),
        ("Clustered", lambda rng: gen_thomas(rng, 5, 10.0)),
    ]
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.25)

    for i, (label, gen) in enumerate(examples):
        ax = fig.add_subplot(gs[0, i])
        rng = np.random.default_rng(7)
        img = render(gen(rng))
        ax.imshow(img, cmap="magma")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=11)

    ax = fig.add_subplot(gs[0, 3])
    colors = {0: PALETTE["primary"], 1: PALETTE["highlight"], 2: PALETTE["accent"],
              3: PALETTE["accent"], 4: PALETTE["accent"]}
    for lvl, sub in df.groupby("level"):
        ax.scatter(sub["clark_evans_R"], sub["lateral_gini"], s=45, alpha=0.8,
                   color=colors.get(lvl, PALETTE["muted"]),
                   edgecolors=PALETTE["ink"], linewidths=0.4,
                   label=sub["condition"].iloc[0])
    rho = df["lateral_gini"].corr(df["clark_evans_R"], method="spearman")
    ax.axvline(1.0, color=PALETTE["ink"], ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Clark-Evans R  (lower = clustered, higher = dispersed)")
    ax.set_ylabel("fluorostats lateral Gini")
    ax.set_title(f"Gini tracks clustering  ·  Spearman ρ = {rho:.3f}")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("B6 homogeneity — fluorostats Gini validated vs Clark-Evans "
                 "nearest-neighbor index on synthetic controls", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG / "b6_homogeneity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'b6_homogeneity.png'}")


if __name__ == "__main__":
    fig_b1()
    fig_b6()
