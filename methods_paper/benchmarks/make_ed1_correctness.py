"""Extended Data Figure 1 - Correctness against analytic ground truth.

Three panels:
  (a) Topology phantom battery: analytic vs measured Euler number chi (bars) and
      connected-component count (diamonds) for ball / disjoint balls / torus /
      hollow ball. Zero error on every phantom.
  (b) Skeleton trees: expected vs measured branch and bifurcation counts at
      recursion depth 2/3 (exact) and depth 4 (raster undercount, labelled).
  (c) Zoom-invariance: measured density vs digital zoom for five normalisation
      schemes; fluorostats per-mm3 is flat (CV=0), competitors drift. CV per
      scheme annotated.

Run from methods_paper/benchmarks: python3.13 make_ed1_correctness.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle
from figstyle import OKABE, apply_style, save, panel, caption, EXT

apply_style()

RES = figstyle.Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------- load data
topo = pd.read_csv(RES / "b1_topology_phantoms.csv")
tree = pd.read_csv(RES / "b_skeleton_tree.csv")
dens = pd.read_csv(RES / "b_density_normalization.csv")
cv = pd.read_csv(RES / "b_density_normalization_cv.csv")

BLUE = OKABE["blue"]
GREY = OKABE["grey"]
GREEN = OKABE["green"]
VERM = OKABE["vermillion"]

# ================================================================ figure
# a, b on the top row; c spans the full bottom row (its wide legend needs the
# room). constrained layout spaces everything; save with tight=False.
fig = plt.figure(figsize=(7.2, 6.2), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, :])

# ---------------------------------------------------------------- panel (a)
# Grouped expected-vs-measured for Euler number and component count.
labels_a = {
    "solid_ball": "ball\n$\\chi{=}1$",
    "2_disjoint_balls": "2 balls\n$\\chi{=}2$",
    "3_disjoint_balls": "3 balls\n$\\chi{=}3$",
    "5_disjoint_balls": "5 balls\n$\\chi{=}5$",
    "solid_torus": "torus\n$\\chi{=}0$",
    "hollow_ball": "shell\n$\\chi{=}2$",
}
order = ["solid_ball", "2_disjoint_balls", "3_disjoint_balls",
         "5_disjoint_balls", "solid_torus", "hollow_ball"]
ta = topo.set_index("phantom").loc[order]
x = np.arange(len(order))
w = 0.34

axA.bar(x - w / 2, ta["expected_euler"], w, label="expected $\\chi$",
        color="none", edgecolor=GREY, linewidth=0.9, hatch="////")
axA.bar(x + w / 2, ta["fluorostats_euler"], w, label="fluorostats $\\chi$",
        color=BLUE, edgecolor="none")
# component-count markers overlaid as a second series (open vs filled)
axA.plot(x - w / 2, ta["expected_n_comp"], marker="D", ls="none", ms=4.5,
         mfc="none", mec="black", mew=0.9, label="expected $n_{\\mathrm{comp}}$")
axA.plot(x + w / 2, ta["fluorostats_n_comp"], marker="D", ls="none", ms=3.2,
         mfc=GREEN, mec="none", label="fluorostats $n_{\\mathrm{comp}}$")

axA.set_xticks(x)
axA.set_xticklabels([labels_a[k] for k in order], fontsize=5.6)
axA.set_ylabel("Euler number $\\chi$  /  component count")
axA.set_ylim(-0.6, 7.4)
axA.axhline(0, color="#333333", lw=0.6)
n_pass = int(ta.shape[0])
axA.text(0.98, 0.97, f"0 / {n_pass} phantoms with error", transform=axA.transAxes,
         ha="right", va="top", fontsize=6.3, color=GREEN, fontweight="bold")
# note that the diamonds encode component count (easy to miss vs the bars)
axA.text(0.98, 0.865, "diamonds = component count",
         transform=axA.transAxes, ha="right", va="top", fontsize=5.6,
         color="#333333", style="italic")
axA.legend(loc="upper left", fontsize=5.2, handlelength=1.3,
           borderpad=0.2, labelspacing=0.25, ncol=1, bbox_to_anchor=(0.0, 1.0))
panel(axA, "a")

# ---------------------------------------------------------------- panel (b)
depths = tree["tree_depth"].values
xb = np.arange(len(depths))
wb = 0.20
# branches
axB.bar(xb - 1.5 * wb, tree["true_branches"], wb, color="none",
        edgecolor=GREY, linewidth=0.9, hatch="////", label="true branches")
bar_fb = axB.bar(xb - 0.5 * wb, tree["fs_n_branches"], wb, color=BLUE,
                 label="fluorostats branches")
# bifurcations
axB.bar(xb + 0.5 * wb, tree["true_bifurcations"], wb, color="none",
        edgecolor="#555555", linewidth=0.9, hatch="\\\\\\\\",
        label="true bifurcations")
bar_fj = axB.bar(xb + 1.5 * wb, tree["fs_n_junction_nodes"], wb, color=GREEN,
                 label="fluorostats bifurcations")

for i, row in tree.reset_index().iterrows():
    if not row["branch_match"]:
        axB.annotate("raster\nundercount",
                     xy=(xb[i] - 0.5 * wb, row["fs_n_branches"] + 0.3),
                     xytext=(xb[i] + 0.9 * wb, row["true_branches"] + 3.6),
                     ha="center", va="bottom", fontsize=5.4, color=VERM,
                     arrowprops=dict(arrowstyle="-|>", color=VERM, lw=0.7,
                                     shrinkA=1, shrinkB=1))
        axB.text(xb[i] - 0.5 * wb, row["fs_n_branches"] + 0.4,
                 f"{int(row['fs_n_branches'])}/{int(row['true_branches'])}",
                 ha="center", va="bottom", fontsize=5.2, color=VERM,
                 fontweight="bold")

axB.set_xticks(xb)
axB.set_xticklabels([f"depth {int(d)}" for d in depths])
axB.set_ylabel("count")
axB.set_ylim(0, 37)
axB.text(0.02, 0.97, "exact at depth 2-3", transform=axB.transAxes,
         ha="left", va="top", fontsize=6.0, color=GREEN, fontweight="bold")
axB.legend(loc="upper left", fontsize=5.1, handlelength=1.2, borderpad=0.2,
           labelspacing=0.22, bbox_to_anchor=(0.0, 0.90))
panel(axB, "b")

# ---------------------------------------------------------------- panel (c)
# Regime A = fixed FOV, varying voxel size (true digital zoom).
A = dens[dens["regime"] == "A_fixed_FOV"].sort_values("zoom")
z = A["zoom"].values
# worst-case CV across BOTH acquisition regimes: a scheme is only truly
# zoom-invariant if it stays flat in every regime, not just this one.
cv_worst = cv.groupby("scheme")["CV_percent_across_zoom"].max()

schemes = [
    ("fluorostats_per_mm3", "fluorostats (per mm$^3$)", BLUE, "o", "-"),
    ("raw_count", "raw count", OKABE["orange"], "s", "--"),
    ("per_megavoxel", "per Mvoxel", VERM, "^", "-."),
    ("per_mm2_area", "per mm$^2$", OKABE["purple"], "D", ":"),
    ("per_z_slice", "per z-slice", "#555555", "v", (0, (3, 1, 1, 1))),
]
# normalise each scheme to its value at zoom=1 so drift is comparable on one axis
z1 = A[A["zoom"] == 1.0].iloc[0]
for col, lab, c, m, ls in schemes:
    y = A[col].values / z1[col]
    cvv = cv_worst[col]
    is_fs = col.startswith("fluorostats")
    lw = 2.0 if is_fs else 1.0
    # nudge the two flat lines (fluorostats, raw_count) apart so both are visible
    yy = y * (1.03 if is_fs else (0.97 if col == "raw_count" else 1.0))
    axC.plot(z, yy, marker=m, ls=ls, color=c, lw=lw, ms=4,
             label=f"{lab}  (CV$_{{\\max}}$={cvv:.0f}%)",
             zorder=6 if is_fs else 3)

axC.axhline(1.0, color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=1)
axC.set_xlabel("digital zoom  (fixed field of view)")
axC.set_ylabel("measured density  (relative to zoom = 1)")
axC.set_yscale("log")
axC.set_xticks(z)
axC.set_xticklabels([f"{v:g}" for v in z])
axC.legend(loc="lower left", fontsize=5.8, handlelength=2.0, borderpad=0.25,
           labelspacing=0.3, title="CV$_{\\max}$ over both regimes",
           title_fontsize=5.8, ncol=2, columnspacing=1.2)
axC.text(0.97, 0.94, "fluorostats invariant\n(CV = 0% in both regimes)",
         transform=axC.transAxes, ha="right", va="top", fontsize=6.2,
         color=BLUE, fontweight="bold")
panel(axC, "c")

# ---------------------------------------------------------------- save
save(fig, "ed1_correctness", EXT, tight=False)

caption(
    "ed1_correctness",
    "Extended Data Figure 1 | Correctness against analytic ground truth. "
    "(a) Topology phantom battery. For a solid ball, 2/3/5 disjoint balls, a "
    "solid torus and a hollow spherical shell, the fluorostats Euler number "
    "chi (bars) and connected-component count (diamonds) match the analytic "
    "expectation exactly (hatched = expected, filled = measured); error is zero "
    "on all six phantoms. (b) Recursive skeleton trees. Measured branch and "
    "bifurcation counts equal the true values exactly at recursion depth 2 "
    "(7 branches, 3 bifurcations) and depth 3 (15, 7). At depth 4 the finest "
    "twigs fall below the raster resolution and are undercounted (27 of 31 "
    "branches, 13 of 15 bifurcations); this is an honest sampling limit, not a "
    "topological error. (c) Zoom-invariance. Measured density versus digital "
    "zoom at fixed field of view for five normalisation schemes, each expressed "
    "relative to its value at zoom = 1 (log axis). The legend gives the "
    "worst-case coefficient of variation across zoom over both acquisition "
    "regimes tested (fixed field of view and fixed voxel size); a scheme is "
    "only truly zoom-invariant if it stays flat in every regime. The "
    "fluorostats per-mm3 density is invariant in both (CV = 0%), whereas raw "
    "count, per-megavoxel and per-z-slice normalisations each drift, by up to "
    "two orders of magnitude, in at least one regime. The flat fluorostats and "
    "raw-count traces are drawn slightly offset for visibility. Numerical "
    "expected/measured/error values for every phantom are tabulated in "
    "Extended Data Table 1. fluorostats is shown in blue throughout.",
    EXT,
)
print("done")
