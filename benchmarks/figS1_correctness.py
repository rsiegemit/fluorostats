"""Figure S1 - Correctness against analytic ground truth.

Three panels:
  (a) Topology phantom battery: analytic vs measured Euler number chi (bars) and
      connected-component count (diamonds) for ball / disjoint balls / torus /
      hollow ball. Zero error on every phantom.
  (b) Skeleton trees: expected vs measured branch and bifurcation counts at
      recursion depth 2/3 (exact) and depth 4 (raster undercount, labelled).
  (c) Zoom-invariance: measured density vs digital zoom for five normalisation
      schemes; fluorostats per-mm3 is flat (CV<=0.2%), competitors drift. CV per
      scheme annotated.

Run: python3.13 benchmarks/figS1_correctness.py (from any directory)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402
from figstyle import OKABE, apply_style, save, panel, caption, SUPP  # noqa: E402

apply_style()

RES = Path(__file__).resolve().parent / "results"

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
# Authored at the text width (F.TW = 5.147 in) so
# \includegraphics[width=\textwidth] is 1:1. At this narrow width a and b cannot
# share a row without crushing their grouped bars + multi-entry legends, so the
# three panels stack vertically (a, b, c), each spanning the full width.
# constrained layout spaces everything; save with tight=False.
# Height trimmed to fit a 7.65 in text height with room for the short
# caption: 6.5 in overall. constrained-layout padding pulled in tight so the three
# stacked panels use the reduced height without empty gaps between rows.
fig = plt.figure(figsize=(figstyle.TW, 6.5), layout="constrained")
fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.03, wspace=0.02)
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[2, 0])

# ---------------------------------------------------------------- panel (a)
# Grouped expected-vs-measured for Euler number and component count.
labels_a = {
    "solid_ball": "ball\n$\\chi=1$",
    "2_disjoint_balls": "2 balls\n$\\chi=2$",
    "3_disjoint_balls": "3 balls\n$\\chi=3$",
    "5_disjoint_balls": "5 balls\n$\\chi=5$",
    "solid_torus": "torus\n$\\chi=0$",
    "hollow_ball": "shell\n$\\chi=2$",
}
order = ["solid_ball", "2_disjoint_balls", "3_disjoint_balls",
         "5_disjoint_balls", "solid_torus", "hollow_ball"]
ta = topo.set_index("phantom").loc[order]
x = np.arange(len(order))
w = 0.34

axA.bar(x - w / 2, ta["expected_euler"], w, label="expected $\\chi$",
        color="none", edgecolor="#333333", linewidth=0.9, hatch="////")
axA.bar(x + w / 2, ta["fluorostats_euler"], w, label="fluorostats $\\chi$",
        color=BLUE, edgecolor="#333333", linewidth=0.4)
# component-count markers overlaid as a second series (open vs filled)
axA.plot(x - w / 2, ta["expected_n_comp"], marker="D", ls="none", ms=4.5,
         mfc="none", mec="black", mew=0.9, label="expected n comp")
axA.plot(x + w / 2, ta["fluorostats_n_comp"], marker="D", ls="none", ms=3.2,
         mfc=GREEN, mec="none", label="fluorostats n comp")

axA.set_xticks(x)
axA.set_xticklabels([labels_a[k] for k in order], fontsize=6.0)
axA.set_ylabel("Euler number $\\chi$  /  component count")
axA.set_ylim(-0.6, 7.8)
axA.axhline(0, color="#333333", lw=0.6)
n_err = int((~ta["PASS"]).sum())
axA.text(0.98, 0.97, f"{n_err} / {len(ta)} phantoms with error", transform=axA.transAxes,
         ha="right", va="top", fontsize=6.0, color=GREEN, fontweight="bold")
# the torus has chi = 0, so BOTH bars have zero height and the column reads as
# missing data unless it is labelled explicitly (y=0.16 clears the two diamonds at y=1).
i_torus = order.index("solid_torus")
axA.text(i_torus, 0.16, "$\\chi=0$ (exact)", ha="center", va="bottom",
         fontsize=5.5, color=GREEN, fontweight="bold")
# note that the diamonds encode component count (easy to miss vs the bars)
axA.text(0.98, 0.85, "diamonds = component count",
         transform=axA.transAxes, ha="right", va="top", fontsize=6.0,
         color="#333333", style="italic")
axA.legend(loc="upper left", fontsize=6.0, handlelength=1.4,
           borderpad=0.2, labelspacing=0.28, ncol=2, columnspacing=1.0,
           bbox_to_anchor=(0.0, 1.0))
panel(axA, "a", xanchor=0.0315)   # row-leader

# ---------------------------------------------------------------- panel (b)
depths = tree["tree_depth"].values
xb = np.arange(len(depths))
wb = 0.20
# branches
axB.bar(xb - 1.5 * wb, tree["true_branches"], wb, color="none",
        edgecolor="#333333", linewidth=0.9, hatch="////", label="true branches")
bar_fb = axB.bar(xb - 0.5 * wb, tree["fs_n_branches"], wb, color=BLUE,
                 edgecolor="#333333", linewidth=0.4, label="fluorostats branches")
# bifurcations
axB.bar(xb + 0.5 * wb, tree["true_bifurcations"], wb, color="none",
        edgecolor="#333333", linewidth=0.9, hatch="\\\\\\\\",
        label="true bifurcations")
bar_fj = axB.bar(xb + 1.5 * wb, tree["fs_n_junction_nodes"], wb, color=GREEN,
                 edgecolor="#333333", linewidth=0.4, label="fluorostats bifurcations")

for i, row in tree.reset_index().iterrows():
    if not row["branch_match"]:
        # arrow lands on the RIGHT edge of the blue bar top so its shaft stays
        # clear of the "27/31" label, which sits at the bar-top centre-left
        axB.annotate("raster\nundercount",
                     xy=(xb[i], row["fs_n_branches"] + 0.3),
                     xytext=(xb[i] + 1.0 * wb, row["true_branches"] + 3.6),
                     ha="center", va="bottom", fontsize=5.0, color=VERM,
                     arrowprops=dict(arrowstyle="-|>", color=VERM, lw=0.7,
                                     shrinkA=1, shrinkB=1))
        axB.text(xb[i] - 0.5 * wb, row["fs_n_branches"] + 0.4,
                 f"{int(row['fs_n_branches'])}/{int(row['true_branches'])}",
                 ha="center", va="bottom", fontsize=5.0, color=VERM,
                 fontweight="bold")

axB.set_xticks(xb)
axB.set_xticklabels([f"depth {int(d)}" for d in depths])
axB.set_ylabel("count")
axB.set_ylim(0, 37)
exact = tree[(tree.true_branches == tree.fs_n_branches)
             & (tree.true_bifurcations == tree.fs_n_junction_nodes)].tree_depth
assert list(exact) == list(range(exact.min(), exact.max() + 1)), "exact depths must be contiguous"
axB.text(0.40, 0.62, f"exact at depth {exact.min()}–{exact.max()}", transform=axB.transAxes,
         ha="center", va="top", fontsize=6.0, color=GREEN, fontweight="bold")
axB.legend(loc="upper left", fontsize=6.0, handlelength=1.4, borderpad=0.2,
           labelspacing=0.28, ncol=2, columnspacing=1.0,
           bbox_to_anchor=(0.0, 1.0))
panel(axB, "b", xanchor=0.0315)   # row-leader

# ---------------------------------------------------------------- panel (c)
# Regime A = fixed FOV, varying voxel size (true digital zoom).
A = dens[dens["regime"] == "A_fixed_FOV"].sort_values("zoom")
z = A["zoom"].values
# worst-case CV across BOTH acquisition regimes: a scheme is only truly
# zoom-invariant if it stays flat in every regime, not just this one.
cv_worst = cv.groupby("scheme")["CV_percent_across_zoom"].max()
# bound stated for the fluorostats scheme: its worst-case CV rounded up to 0.1%
fs_cv_bound = np.ceil(cv_worst["fluorostats_per_mm3"] * 10) / 10

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
    y = A[col].values / z1[col]                       # TRUE value, never displaced
    cvv = cv_worst[col]
    is_fs = col.startswith("fluorostats")
    lw = 2.0 if is_fs else 1.0
    # sub-percent CVs (fluorostats = 0.16%) must not round to "0%" — that contradicts
    # the panel annotation ("CV <= 0.2%") and the caption (0.16%)
    cvs = f"{cvv:.0f}" if cvv >= 1 else f"{cvv:.2f}"
    # no CV_max subscript on the entries (the legend title carries "worst-case CV");
    # the 0.7x-scaled subscript was 4.06 pt, below the 5 pt floor
    base = dict(ls=ls, lw=lw, color=c, label=f"{lab}  ({cvs}%)")
    # fluorostats per-mm3, raw_count and per-mm2 coincide EXACTLY at 1.000 in this
    # fixed-FOV regime (CV = 0.0%); keep all three readable by nesting the markers
    # (large hollow diamond / medium hollow square / small filled circle) rather than
    # offsetting the data on a correctness axis.
    if is_fs:
        axC.plot(z, y, marker="o", ms=3, zorder=6, **base)
    elif col == "raw_count":
        axC.plot(z, y, marker="s", ms=4.5, mfc="none", mew=1.0, zorder=4, **base)
    elif col == "per_mm2_area":
        axC.plot(z, y, marker="D", ms=6, mfc="none", mew=1.0, zorder=5, **base)
    else:
        axC.plot(z, y, marker=m, ms=4, zorder=3, **base)

axC.axhline(1.0, color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=1)
axC.set_xlabel("digital zoom  (fixed field of view)")
axC.set_ylabel("measured density  (relative to zoom = 1)")
axC.set_yscale("log")
axC.set_xticks(z)
axC.set_xticklabels([f"{v:g}" for v in z])
axC.legend(loc="lower left", fontsize=6.0, handlelength=2.0, borderpad=0.25,
           labelspacing=0.3, title="worst-case CV over both regimes",
           title_fontsize=6.0, ncol=2, columnspacing=1.2)
axC.text(0.97, 0.94, f"fluorostats invariant\n(CV ≤ {fs_cv_bound:.1f}% in both regimes)",
         transform=axC.transAxes, ha="right", va="top", fontsize=6.0,
         color=BLUE, fontweight="bold")
axC.text(0.97, 0.72, "three traces coincide exactly at 1.000",
         transform=axC.transAxes, fontsize=5.0, color="#555555",
         ha="right", va="bottom")
panel(axC, "c", xanchor=0.0315)   # row-leader

# ---------------------------------------------------------------- save
save(fig, "figS1_correctness", SUPP, tight=False)

# Caption numbers are read from the tables plotted above, never typed.
_trees = tree.set_index("tree_depth")
_fs_cv = cv[cv.scheme == "fluorostats_per_mm3"].set_index("regime")["CV_percent_across_zoom"]


def _topo_result():
    """'error is zero on all six phantoms', or the failing count, from the PASS column."""
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    n, n_fail = len(topo), int((~topo["PASS"]).sum())
    if n_fail == 0:
        return f"error is zero on all {words[n] if n < len(words) else n} phantoms"
    return f"{n_fail} of {n} phantoms fail"


def _tree(depth):
    r = _trees.loc[depth]
    return r.true_branches, r.true_bifurcations, r.fs_n_branches, r.fs_n_junction_nodes


caption(
    "figS1_correctness",
    "Figure S1 | Correctness against analytic ground truth. "
    "(a) Topology phantom battery. For a solid ball, 2/3/5 disjoint balls, a "
    "solid torus and a hollow spherical shell, the fluorostats Euler number "
    "chi (bars) and connected-component count (diamonds) match the analytic "
    "expectation exactly (hatched = expected, filled = measured); "
    f"{_topo_result()}. (b) Recursive skeleton trees. Measured branch and "
    "bifurcation counts equal the true values exactly at recursion depth 2 "
    f"({_tree(2)[0]} branches, {_tree(2)[1]} bifurcations) and depth 3 "
    f"({_tree(3)[0]}, {_tree(3)[1]}). At depth 4 the finest "
    f"twigs fall below the raster resolution and are undercounted ({_tree(4)[2]} of "
    f"{_tree(4)[0]} branches, {_tree(4)[3]} of {_tree(4)[1]} bifurcations); this is an "
    "honest sampling limit, not a "
    "topological error. (c) Zoom-invariance. Measured density versus digital "
    "zoom at fixed field of view for five normalisation schemes, each expressed "
    "relative to its value at zoom = 1 (log axis). The legend gives the "
    "worst-case coefficient of variation across zoom over both acquisition "
    "regimes tested (fixed field of view and fixed voxel size); a scheme is "
    "only truly zoom-invariant if it stays flat in every regime. The "
    f"fluorostats per-mm3 density is invariant in both (CV <= {fs_cv_bound:.1f}%; "
    f"{_fs_cv['A_fixed_FOV']:.1f}% fixed field of view, {_fs_cv['B_fixed_voxel']:.2f}% "
    "fixed voxel), whereas raw count, per-megavoxel and "
    "per-z-slice normalisations each drift, by up to two orders of magnitude, in "
    "at least one regime. In the fixed-field-of-view regime shown, the fluorostats "
    "per-mm3, raw-count and per-mm2 traces coincide exactly at 1.000 and are drawn "
    "as nested markers (no data offset). Numerical "
    "expected/measured/error values for every phantom are tabulated in "
    "Table S1. fluorostats is shown in blue throughout.",
    SUPP,
)
print("done")
