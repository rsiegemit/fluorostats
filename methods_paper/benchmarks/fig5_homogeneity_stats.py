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
F.apply_style()
R = Path(__file__).resolve().parent / "results"
BLUE = OKABE["blue"]; GREY = OKABE["grey"]; VERM = OKABE["vermillion"]; GREEN = OKABE["green"]

fig = plt.figure(figsize=(7.2, 5.4))
gs = gridspec.GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.05], hspace=0.5, wspace=1.0)

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
gs_a = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.18)
for i, kind in enumerate(["regular", "random", "clustered"]):
    p = pattern(kind)
    p3 = np.column_stack([np.zeros(len(p)), p[:, 0], p[:, 1]])
    gini = centroid_homogeneity(p3, (1, S, S))["centroid_gini"]
    ax = fig.add_subplot(gs_a[i]); ax.scatter(p[:, 1], p[:, 0], s=8, color=BLUE, edgecolor="none")
    ax.set_xlim(0, S); ax.set_ylim(0, S); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#999"); sp.set_linewidth(0.5)
    ax.set_title(titles[kind], fontsize=6.8)
    ax.text(0.5, -0.09, f"tile Gini = {gini:.3f}", transform=ax.transAxes, ha="center", fontsize=6.4,
            fontweight="bold", color=BLUE)
    if i == 0: panel(ax, "a")

# (b) five-statistic correlation
axb = fig.add_subplot(gs[1, :2])
corr = pd.read_csv(R/"b_homogeneity_multi_corr.csv").copy()
corr["abs"] = corr["spearman_gini_vs_ref"].abs()
corr = corr.sort_values("abs")
names = {"clark_evans":"Clark–Evans NN","ripley_L_dev":"Ripley's K/L","morisita":"Morisita",
         "lacunarity":"lacunarity","quadrat_var":"quadrat variance"}
axb.barh([names[r] for r in corr.reference], corr["abs"], color=BLUE, edgecolor="black", lw=0.4, height=0.66)
for i, v in enumerate(corr["abs"]): axb.text(v-0.02, i, f"{v:.3f}", va="center", ha="right", color="white", fontsize=5.6)
axb.set_xlim(0, 1.02); axb.set_xlabel("|Spearman ρ| vs fluorostats tile Gini")
panel(axb, "b", "tracks 5 spatial statistics")

# (c) end-to-end statistics worked example — SproutAngio VEGF dose (real .czi)
sa = pd.read_csv(R/"b_vascular_sproutangio_multi.csv")
groups = {g: sa[sa.group == g]["fluorostats"].values * 100 for g in (1, 3, 5)}  # % VF
axc = fig.add_subplot(gs[1, 2:4])
gcol = {1: GREY, 3: GREEN, 5: BLUE}
for j, (g, v) in enumerate(groups.items()):
    x = np.full(len(v), j) + rng.normal(0, 0.05, len(v))
    axc.scatter(x, v, s=26, color=gcol[g], edgecolor="black", lw=0.4, zorder=3)
    axc.plot([j-0.22, j+0.22], [v.mean(), v.mean()], color="black", lw=1.3, zorder=2)
axc.set_xticks([0,1,2]); axc.set_xticklabels(["VEGF 1","VEGF 3","VEGF 5"], fontsize=6.5)
axc.set_ylabel("vessel volume fraction (%)")
# real fluorostats.stats output on groups 1 vs 3
a, b = groups[1], groups[3]
mw = mann_whitney(a, b); cd = cliffs_delta(a, b)
fc = bootstrap_fold_change_ci(a, b, n_boot=5000)
qs = bh_fdr([mann_whitney(groups[i], groups[j])["p"] for i,j in [(1,3),(1,5),(3,5)]])
fc_ratio = fc["fold_change_median"]; fc_lo = fc["ci_low"]; fc_hi = fc["ci_high"]

axc.text(0.03, 0.97, f"MWU p = {mw['p']:.3f}\nCliff's δ = {cd:+.2f}\nFC {fc_ratio:.1f}× [{fc_lo:.1f}, {fc_hi:.1f}]\nBH q = {qs.min():.3f}",
         transform=axc.transAxes, va="top", ha="left", fontsize=5.4, color="#333",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", lw=0.4))
panel(axc, "c", "statistics layer output")

# power curve (real fluorostats.power)
axp = fig.add_subplot(gs[1, 4:])
ns = [3, 4, 6, 8, 12, 16]
pc = power_curve(a, b, ns, n_sims=400)
ycol = [c for c in pc.columns if "power" in c.lower()][0]; ncol = [c for c in pc.columns if c.lower() in ("n","sample_size","n_per_group")][0]
axp.plot(pc[ncol], pc[ycol], marker="o", color=BLUE, lw=1.8, ms=4)
axp.axhline(0.8, ls="--", color=VERM, lw=0.9); axp.text(ns[-1], 0.82, "80%", fontsize=5.4, color=VERM, ha="right")
axp.set_xlabel("n per group"); axp.set_ylabel("power"); axp.set_ylim(0, 1.02)
panel(axp, "", "power (same pipeline)")

cap = ("Figure 5 | Spatial homogeneity and the integrated statistics layer. "
 "(a) Three canonical spatial patterns (regular, random/CSR, clustered) with fluorostats' "
 "tile-based Gini index beneath each — the segmentation-free homogeneity metric increases from "
 "regular to clustered. (b) Across a regular→clustered sweep, the tile Gini tracks five established "
 "spatial statistics (Clark–Evans nearest-neighbour, Ripley's K/L, Morisita, quadrat variance, "
 "gliding-box lacunarity) at |Spearman ρ| 0.96–0.997 (uniform-vs-clustered AUC = 1.0). "
 "(c) End-to-end statistics on a real VEGF dose experiment (SproutAngio, .czi; n=4/group): vessel "
 "volume fraction by dose (SuperPlot; bars = group means), with the fluorostats.stats / power output "
 "computed live — Mann–Whitney U, Cliff's δ, bootstrap fold-change CI, BH-FDR across contrasts, and "
 "a power curve — showing image→statistic with no manual export.")
save(fig, "fig5_homogeneity_stats"); caption("fig5_homogeneity_stats", cap)
print("Figure 5 done")
