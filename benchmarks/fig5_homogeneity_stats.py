"""Figure 5 — Spatial homogeneity and the integrated statistics layer (a-c)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F
from figstyle import OKABE, panel, save, caption
import matplotlib.pyplot as plt
from matplotlib import gridspec
from fluorostats.objects import centroid_homogeneity
from fluorostats.stats import mann_whitney, cliffs_delta, bh_fdr, bootstrap_fold_change_ci
from fluorostats.power import power_curve
from scipy import stats as sps
F.apply_style()
R = Path(__file__).resolve().parent / "results"
BLUE = OKABE["blue"]; GREY = OKABE["grey"]; VERM = OKABE["vermillion"]; GREEN = OKABE["green"]

fig = plt.figure(figsize=(F.TW, 6.2), layout="constrained")
fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.02)
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.78, 1.10, 1.05],
                       width_ratios=[0.55, 1.45], hspace=0.05, wspace=0.30)

# (a) three canonical point patterns with tile Gini
rng = np.random.default_rng(1); N, S = 225, 500
def pattern(kind):
    if kind == "regular":
        g = int(np.sqrt(N)); xs = np.linspace(30, S-30, g)
        return np.array([(y, x) for y in xs for x in xs])
    if kind == "random":
        return rng.uniform(20, S-20, (N, 2))
    k = 7; cen = rng.uniform(60, S-60, (k, 2))
    return np.clip(np.array([cen[i % k] + rng.normal(0, 22, 2) for i in range(N)]), 5, S-5)
titles = {"regular": "regular", "random": "random (CSR)", "clustered": "clustered"}
gs_a = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.14)
for i, kind in enumerate(["regular", "random", "clustered"]):
    p = pattern(kind)
    p3 = np.column_stack([np.zeros(len(p)), p[:, 0], p[:, 1]])
    gini = centroid_homogeneity(p3, (1, S, S))["centroid_gini"]
    ax = fig.add_subplot(gs_a[i]); ax.scatter(p[:, 1], p[:, 0], s=8, color=BLUE, edgecolor="none")
    ax.set_xlim(0, S); ax.set_ylim(0, S); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#999"); sp.set_linewidth(0.5)
    ax.set_title(titles[kind], fontsize=7.0)
    ax.text(0.5, -0.09, f"tile Gini = {gini:.3f}", transform=ax.transAxes, ha="center", fontsize=6.0,
            fontweight="bold", color=BLUE)
    if i == 0: panel(ax, "a", xanchor=0.0315)   # row-leader -> common page anchor

# (b) tracking scatter across the full regular→clustered sweep (50 synthetic
# fields), NOT a zoomed correlation dot-plot: x = mean sign-aligned z-score of the
# five reference statistics (a "clustering consensus"), y = fluorostats tile Gini.
# A tight monotonic band shows the metric tracks the references honestly, without an
# axis floor that visually exaggerates the near-perfect agreement.
axb = fig.add_subplot(gs[1, 0])
raw = pd.read_csv(R/"b_homogeneity_multi.csv")
REFSIGN = {"clark_evans": -1, "ripley_L_dev": 1, "morisita": 1, "lacunarity": 1, "quadrat_var": 1}
Z = np.zeros(len(raw))
for r, s in REFSIGN.items():
    v = raw[r].to_numpy(float) * s
    Z += (v - v.mean()) / (v.std() + 1e-9)
consensus = Z / len(REFSIGN)
gini = raw["gini"].to_numpy(float); lvl = raw["level"].to_numpy(int)
ramp = ["#BFBFBF", "#8C8C8C", "#6FA8CF", "#2E7BB5", "#0A4C8A"]   # regular(grey)→clustered(blue)
for L in range(5):
    m = lvl == L
    axb.scatter(consensus[m], gini[m], s=15, color=ramp[L], edgecolor="white", lw=0.3, zorder=3)
rho = sps.spearmanr(consensus, gini).statistic
axb.text(0.05, 0.97, f"ρ = {rho:.3f}\nper-stat 0.96–0.997", transform=axb.transAxes,
         va="top", ha="left", fontsize=5.5, color=BLUE, fontweight="bold")
axb.text(0.03, 0.04, "regular", transform=axb.transAxes, fontsize=5.0, color="#888", ha="left")
axb.text(0.97, 0.04, "clustered", transform=axb.transAxes, fontsize=5.0, color=ramp[-1], ha="right")
axb.set_xlabel("clustering consensus (z)", fontsize=6.0)
axb.set_ylabel("fluorostats tile Gini", fontsize=6.0)
panel(axb, "b", "tracks 5 spatial statistics", xanchor=0.0315)   # row-leader

# (c) end-to-end statistics worked example — SproutAngio VEGF dose (real .czi)
sa = pd.read_csv(R/"b_vascular_sproutangio_multi.csv")
groups = {g: sa[sa.group == g]["fluorostats"].values * 100 for g in (1, 3, 5)}  # % VF
axc = fig.add_subplot(gs[1, 1])
# non-blue sequential grey ramp for dose (blue reserved for fluorostats)
gcol = {1: "#CFCFCF", 3: "#8A8A8A", 5: "#3A3A3A"}
for j, (g, v) in enumerate(groups.items()):
    x = np.full(len(v), j) + rng.normal(0, 0.05, len(v))
    axc.scatter(x, v, s=26, color=gcol[g], edgecolor="black", lw=0.4, zorder=3)
    axc.plot([j-0.22, j+0.22], [v.mean(), v.mean()], color="black", lw=1.3, zorder=2)
axc.set_xticks([0,1,2]); axc.set_xticklabels(["V1","V3","V5"], fontsize=6.0)
axc.set_xlabel("VEGF dose", fontsize=6.0)
axc.set_ylabel("vessel volume fraction (%)")
# headroom so the stats readout sits ABOVE every data point (it was opaque-white over
# two of the four V3 markers, making the group mean read as higher than all its points)
axc.set_ylim(0.20, 1.25)
# real fluorostats.stats output on groups 1 vs 3
a, b = groups[1], groups[3]
mw = mann_whitney(a, b); cd = cliffs_delta(a, b)
fc = bootstrap_fold_change_ci(a, b, n_boot=5000)
qs = bh_fdr([mann_whitney(groups[i], groups[j])["p"] for i,j in [(1,3),(1,5),(3,5)]])
fc_ratio = fc["fold_change_median"]; fc_lo = fc["ci_low"]; fc_hi = fc["ci_high"]

_cds = f"{cd:+.2f}".replace("-", "−")  # real minus, matching tick labels
axc.text(0.02, 0.985,
         f"VEGF 1 vs 3:\nMWU p = {mw['p']:.3f}\nCliff’s δ = {_cds}\nFC {fc_ratio:.1f}× [{fc_lo:.1f}, {fc_hi:.1f}]\nBH q = {qs.min():.3f}\n(illustrative, n = 4/group)",
         transform=axc.transAxes, va="top", ha="left", fontsize=5.0, color="#333",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
panel(axc, "c", "statistics layer output")

# power curve (real fluorostats.power) — span the full width of its row
axp = fig.add_subplot(gs[2, :])
ns = [3, 4, 6, 8, 12, 16]
pc = power_curve(a, b, ns, n_sims=400)
ycol = [c for c in pc.columns if "power" in c.lower()][0]; ncol = [c for c in pc.columns if c.lower() in ("n","sample_size","n_per_group")][0]
axp.plot(pc[ncol], pc[ycol], marker="o", color=BLUE, lw=1.8, ms=4)
axp.axhline(0.8, ls="--", color=VERM, lw=0.9); axp.text(ns[-1], 0.82, "80%", fontsize=5.0, color=VERM, ha="right")
axp.set_xlabel("n per group"); axp.set_ylabel("power"); axp.set_ylim(0, 1.02)
axp.text(0.97, 0.18, "bootstrap power from an\nn = 4 pilot — optimistic\n(see text)", transform=axp.transAxes,
         ha="right", va="bottom", fontsize=5.0, color="#555", style="italic")
panel(axp, "d", "power (same pipeline)", xanchor=0.0315)   # row-leader

cap = ("Figure 5 | Spatial homogeneity and the integrated statistics layer. "
 "(a) Three canonical spatial patterns (regular, random/CSR, clustered) with fluorostats' "
 "tile-based Gini index beneath each — the segmentation-free homogeneity metric increases from "
 "regular to clustered. (b) Across 50 synthetic fields spanning a regular→clustered sweep, the tile "
 "Gini tracks the consensus of five independently-computed spatial statistics (Clark–Evans "
 "nearest-neighbour, Ripley's K/L, Morisita, quadrat variance, gliding-box lacunarity): each point "
 "is one field (coloured regular→clustered), x = mean sign-aligned z-score of the five references, "
 "y = tile Gini; per-statistic |Spearman ρ| = 0.96–0.997, uniform-vs-clustered AUC = 1.0. "
 "(c) End-to-end statistics on a small VEGF dose experiment (SproutAngio, .czi): vessel "
 "volume fraction by dose (group means as bars), with the fluorostats.stats output "
 "computed live for VEGF 1 vs 3 — Mann–Whitney U, Cliff's δ, bootstrap fold-change CI and "
 "BH-FDR across contrasts — showing image→statistic with no manual export. This contrast is an "
 "illustrative toy example (n=4/group; Cliff's δ=−1.00 reflects complete separation), not a headline "
 "result. (d) Bootstrap power curve from the same pipeline; the 80% reference line is dashed. Because "
 "it is estimated from an n=4 pilot the power is optimistic (see text).")
save(fig, "fig5_homogeneity_stats", tight=False); caption("fig5_homogeneity_stats", cap)
print("Figure 5 done")
