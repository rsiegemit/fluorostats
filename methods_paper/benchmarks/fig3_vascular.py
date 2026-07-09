"""Figure 3 — Vascular networks (a-d)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F
from figstyle import OKABE, panel, save, caption
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats as sps, ndimage as ndi
from skimage.segmentation import find_boundaries
F.apply_style()
R = Path(__file__).resolve().parent / "results"
BLUE = OKABE["blue"]; GREY = OKABE["grey"]

fig = plt.figure(figsize=(7.2, 5.6))
gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.95],
                       hspace=0.5, wspace=0.75)

# (a) REAVER 6-tool: accuracy (MAE) + precision (residual std) + unbiased test
pi = pd.read_csv(R/"b4_reaver_ranking_perimage.csv")
tools = ["REAVER","ImageJ","AngioTool","fluorostats","RAVE","AngioQuant"]
gt = pi["manual_GT"].values
rec = []
for t in tools:
    res = pi[t].values - gt
    mae = np.abs(res).mean(); sd = res.std(ddof=1)
    p = sps.ttest_1samp(res, 0).pvalue                      # bias test
    rec.append((t, mae, sd, p))
rec.sort(key=lambda r: r[1])
axa = fig.add_subplot(gs[0, :2])
ybias = 0.05 / len(tools)                                   # Bonferroni
for i, (t, mae, sd, p) in enumerate(rec):
    c = BLUE if t == "fluorostats" else GREY
    axa.barh(i, mae, xerr=sd, color=c, edgecolor="black", lw=0.4, height=0.66,
             error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#444"), zorder=2)
    tag = t + (" #" if p > ybias else "")                   # # = unbiased (mean err = 0)
    axa.text(-0.001, i, tag, ha="right", va="center", fontsize=6.3,
             fontweight="bold" if t == "fluorostats" else "normal")
axa.set_yticks([]); axa.set_xlabel("area-fraction error vs manual GT  (mean ± residual s.d.)")

axa.text(0.98, 0.04, "# unbiased (mean error = 0,\nBonferroni)", transform=axa.transAxes,
         ha="right", va="bottom", fontsize=5.3, color="#333")
panel(axa, "a", "vessel-tool accuracy   (REAVER, n = 36)")

# (d) Bland-Altman VF fluorostats vs VesselExpress + rank inset
axd = fig.add_subplot(gs[0, 2:])
m = pd.read_csv(R/"b_ve_metrics.csv")
ve, fs = m.VesselExpress_VF.values, m.fluorostats_VF.values
bias, loa = F.bland_altman(axd, fs, ve, label=" · ~1.7×")
axd.set_ylim(-0.005, bias+loa+0.006)
axd.set_xlabel("mean vessel VF"); axd.set_ylabel("fluorostats − VesselExpress")

axins = axd.inset_axes([0.66, 0.14, 0.30, 0.34])
axins.scatter(sps.rankdata(ve), sps.rankdata(fs), s=11, color=BLUE, edgecolor="none")
axins.set_title(f"rank ρ = {sps.spearmanr(ve,fs).statistic:.2f}", fontsize=5.2, pad=1.5)
axins.set_xticks([]); axins.set_yticks([])
for sp in axins.spines.values(): sp.set_linewidth(0.5); sp.set_color("#999")
panel(axd, "d", "metric agreement")

# (b) VE overlay: raw / VesselExpress seg / fluorostats(auto→li), 2 crops
crops = np.load(R/"ve_crops.npz")
gs_b = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, :3], hspace=0.05, wspace=0.05)
labs = ["raw", "VesselExpress", "fluorostats (auto→li)"]
for row in range(2):
    raw = crops[f"raw{row}"]; ve_m = crops[f"ve{row}"]; fs_m = crops[f"fs{row}"]
    p = F.imnorm(raw, hi=99.7)
    for col, (lab, ov) in enumerate(zip(labs, [None, ve_m, fs_m])):
        ax = fig.add_subplot(gs_b[row, col]); ax.imshow(p, cmap="gray"); F.image_axes(ax)
        if ov is not None:
            F.outline(ax, ov, (0.80,0.47,0.65) if col==1 else (0.0,0.45,0.70), width=1)
        if row == 0: ax.set_title(lab, fontsize=6.8)
        if row == 0 and col == 0: F.scalebar(ax, 100, "100 µm"); panel(ax, "b")

# (c) 3D phantom accuracy vs exact GT
axc = fig.add_subplot(gs[1, 3])
ph = pd.read_csv(R/"b_vascular_phantom_3d.csv")
x = np.arange(len(ph)); w = 0.36
axc.bar(x-w/2, ph.len_err_pct, w, color=BLUE, edgecolor="black", lw=0.4, label="length")
axc.bar(x+w/2, ph.vf_err_pct, w, color=OKABE["sky"], edgecolor="black", lw=0.4, label="vol. frac.")
axc.set_xticks(x); axc.set_xticklabels([f"{p}\nbranches {b}/{t}✓" for p,b,t in
    zip(ph.phantom, ph.fs_branches, ph.true_segments)], fontsize=5.2)
axc.set_ylabel("error vs exact GT (%)")
axc.set_ylim(0, max(3.2, ph.len_err_pct.max()*1.5)); axc.legend(fontsize=5.2, loc="upper right")
panel(axc, "c", "3D phantom")

cap = ("Figure 3 | Vascular networks. (a) fluorostats inserted into the six-tool REAVER benchmark "
 "(n=36): area-fraction error vs manual ground truth (mean ± residual s.d. = accuracy and precision); "
 "'#' marks tools statistically unbiased (one-sample t-test mean error = 0, Bonferroni). fluorostats "
 "(blue) is level with AngioTool and unbiased. (b) Real light-sheet vessels (VesselExpress, Zenodo "
 "6025935), 3-slice MIP crops: raw, VesselExpress software segmentation (magenta), fluorostats "
 "auto→li (blue); scale bar 100 µm. (c) 3D synthetic phantom vs exact ground truth: centreline-length "
 "and volume-fraction error (%), branch counts exact. (d) Bland–Altman of vessel volume fraction, "
 "fluorostats vs VesselExpress (n=9): a ~1.7× systematic offset (bias line ± 95% limits) with "
 "consistent ranking (inset, Spearman 0.75) — a software-agreement comparison (VesselExpress GT is "
 "pipeline-generated, not manual).")
save(fig, "fig3_vascular"); caption("fig3_vascular", cap)
print("Figure 3 done")
