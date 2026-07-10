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

fig = plt.figure(figsize=(7.2, 5.8))
gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.02],
                       hspace=0.52, wspace=0.75)

# (a) REAVER 6-tool: accuracy (MAE) + precision (residual std) + unbiased test  [top-left]
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
    unbiased = p > ybias
    axa.barh(i, mae, xerr=sd, color=c, edgecolor="black", lw=0.4, height=0.66,
             error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#444"), zorder=2)
    axa.text(-0.001, i, t, ha="right", va="center", fontsize=6.3,
             fontweight="bold" if t == "fluorostats" else "normal")
    if unbiased:                                            # only REAVER qualifies
        axa.text(mae + sd + max(mae+sd for _,mae,sd,_ in rec)*0.02, i, "unbiased",
                 ha="left", va="center", fontsize=5.3, style="italic", color="#333")
axa.set_yticks([]); axa.set_xlabel("area-fraction error vs manual GT  (mean ± residual s.d.)")
axa.set_xlim(0, max(mae+sd for _,mae,sd,_ in rec)*1.28)
axa.text(0.98, 0.05, "'unbiased' = mean error not\nsignificantly ≠ 0 (t-test, Bonferroni)",
         transform=axa.transAxes, ha="right", va="bottom", fontsize=5.1, color="#333")
panel(axa, "a", "vessel-tool accuracy   (REAVER, n = 36)")

# (b) VE overlay: raw / VesselExpress seg / fluorostats(auto->li), 2 crops  [top-right]
crops = np.load(R/"ve_crops.npz")
gs_b = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 2:], hspace=0.06, wspace=0.06)
def fill_mask(ax, mask, rgb, alpha=0.85):
    ov = np.zeros((*mask.shape, 4)); ov[mask > 0] = (*rgb, alpha)
    ax.imshow(ov, interpolation="nearest")
labs = ["raw", "VesselExpress", r"fluorostats (auto$\rightarrow$li)"]
for row in range(2):
    raw = crops[f"raw{row}"]; ve_m = crops[f"ve{row}"]; fs_m = crops[f"fs{row}"]
    p = F.imnorm(raw, hi=99.7)
    for col, (lab, ov) in enumerate(zip(labs, [None, ve_m, fs_m])):
        ax = fig.add_subplot(gs_b[row, col]); ax.imshow(p, cmap="gray"); F.image_axes(ax)
        if ov is not None:
            fill_mask(ax, ov, (0.85,0.30,0.60) if col==1 else (0.13,0.55,0.85))
        if row == 0: ax.set_title(lab, fontsize=6.0)
        if row == 0 and col == 0: F.scalebar(ax, 100, "100 µm"); panel(ax, "b")

# (c) 3D phantom accuracy vs exact GT  [bottom-left]
axc = fig.add_subplot(gs[1, :2])
ph = pd.read_csv(R/"b_vascular_phantom_3d.csv")
x = np.arange(len(ph)); w = 0.36
axc.bar(x-w/2, ph.len_err_pct, w, color=BLUE, edgecolor="black", lw=0.4, label="centreline length")
# volume-fraction error is exactly 0 for every phantom -> avoid an invisible/"missing" bar
axc.bar(x+w/2, np.zeros(len(ph)), w, color=OKABE["sky"], edgecolor="black", lw=0.4)
for xi in x:
    axc.text(xi+w/2, max(3.2, ph.len_err_pct.max()*1.5)*0.02, "0",
             ha="center", va="bottom", fontsize=5.0, color="#555", rotation=0)
axc.set_xticks(x)
axc.set_xticklabels([f"{p}\nbranches {b}/{t} exact" for p,b,t in
    zip(ph.phantom, ph.fs_branches, ph.true_segments)], fontsize=5.4)
axc.set_ylabel("error vs exact GT (%)")
axc.set_ylim(0, max(3.2, ph.len_err_pct.max()*1.5))
# legend explains both series; annotate the exact-zero vol-frac in clear space
from matplotlib.patches import Patch
axc.legend(handles=[Patch(facecolor=BLUE, edgecolor="black", lw=0.4, label="centreline length"),
                    Patch(facecolor=OKABE["sky"], edgecolor="black", lw=0.4,
                          label="volume fraction = 0.0 (exact)")],
           fontsize=5.2, loc="upper right", handlelength=1.1, handleheight=0.9)
panel(axc, "c", "3D synthetic phantom")

# (d) Bland-Altman VF fluorostats vs VesselExpress + rank sub-panel  [bottom-right]
gs_d = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 2:], wspace=0.62)
axd = fig.add_subplot(gs_d[0, :2])
m = pd.read_csv(R/"b_ve_metrics.csv")
ve, fs = m.VesselExpress_VF.values, m.fluorostats_VF.values
bias, loa = F.bland_altman(axd, fs, ve, label=r" · ~1.7$\times$", annotate=False)
axd.set_ylim(-0.005, bias+loa+0.010)
axd.set_xlabel("mean vessel VF"); axd.set_ylabel("fluorostats − VesselExpress")
# bias/LoA annotation placed in clear lower-right white space, clear of the zero line
axd.text(0.97, 0.16, f"bias {bias:+.3f}\n±{loa:.3f} (95% LoA)\n~1.7$\\times$",
         transform=axd.transAxes, ha="right", va="bottom", fontsize=5.4, color="#333")
panel(axd, "d", "metric agreement")

# rank-agreement gets its own small labelled panel (no longer a cramped inset)
axr = fig.add_subplot(gs_d[0, 2])
rho = sps.spearmanr(ve, fs).statistic
axr.scatter(sps.rankdata(ve), sps.rankdata(fs), s=13, color=BLUE, edgecolor="none")
axr.set_title(f"rank agreement\n(Spearman ρ = {rho:.2f})", fontsize=5.4, pad=2)
axr.set_xlabel("VesselExpress rank", fontsize=5.4)
axr.set_ylabel("fluorostats rank", fontsize=5.4)
axr.set_xticks([1, len(ve)]); axr.set_yticks([1, len(fs)])
axr.tick_params(labelsize=5.0)
for sp in axr.spines.values(): sp.set_linewidth(0.5); sp.set_color("#999")

cap = ("Figure 3 | Vascular networks. (a) fluorostats inserted into the six-tool REAVER benchmark "
 "(n=36): area-fraction error vs manual ground truth (mean ± residual s.d. = accuracy and precision); "
 "'unbiased' marks tools whose mean error is not significantly different from 0 (one-sample t-test, "
 "Bonferroni across the six tools); only REAVER qualifies. fluorostats (blue) ranks alongside AngioTool "
 "in accuracy but, like every tool except REAVER, carries a small systematic underestimate. (b) Real "
 "light-sheet vessels (VesselExpress, Zenodo 6025935), 3-slice MIP crops: raw, VesselExpress software "
 "segmentation (magenta), fluorostats auto->li (blue); scale bar 100 µm. (c) 3D synthetic phantom vs "
 "exact ground truth: centreline-length error (%) with branch counts recovered exactly; volume-fraction "
 "error is 0.0 (exact) for every phantom. (d) Bland–Altman of vessel volume fraction, fluorostats vs "
 "VesselExpress (n=9): a ~1.7× systematic offset (bias line ± 95% limits) with consistent ranking "
 "(rank-agreement panel, Spearman 0.75) — a software-agreement comparison (VesselExpress GT is "
 "pipeline-generated, not manual).")
save(fig, "fig3_vascular"); caption("fig3_vascular", cap)
print("Figure 3 done")
