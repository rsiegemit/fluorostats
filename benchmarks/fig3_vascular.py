"""Figure 3 — Vascular networks (a-d)."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F
from figstyle import OKABE, panel, save, caption
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import gridspec
from scipy import stats as sps
F.apply_style()
R = Path(__file__).resolve().parent / "results"
BLUE = OKABE["blue"]
GREY = OKABE["grey"]

# Authored at the manuscript text width (F.TW = 5.147 in) so the LaTeX
# \includegraphics[width=\textwidth] is 1:1 and the 6-7 pt type stays 6-7 pt.
# Narrower canvas -> taller stacked layout, three rows, reading order a->b->c->d:
#   row 0  (a) six-tool REAVER accuracy ranking (full width)
#   row 1  (b) light-sheet crops raw/VesselExpress/fluorostats (full-width image band)
#   row 2  (c) 3D phantom  |  (d) Bland-Altman + rank-agreement sub-panel
fig = plt.figure(figsize=(F.TW, 5.827))
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.98, 0.86, 1.02],
                       hspace=0.342, wspace=0.34,
                       left=0.1268, right=0.9518, top=0.9675, bottom=0.0535)

# (a) REAVER 6-tool: accuracy (MAE) + precision (residual std) + unbiased test  [row 0, full width]
# derived tables are read with round_trip so every float is the one the producer wrote
rk = pd.read_csv(R/"b4_reaver_ranking.csv", float_precision="round_trip")   # sorted by MAE
rec = list(zip(rk.tool, rk.MAE, rk.residual_SD, rk.unbiased))
axa = fig.add_subplot(gs[0, :])
for i, (t, mae, sd, unbiased) in enumerate(rec):
    c = BLUE if t == "fluorostats" else GREY
    axa.barh(i, mae, xerr=sd, color=c, edgecolor="#333333", lw=0.4, height=0.66,
             error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#444"), zorder=2)
    axa.text(-0.001, i, t, ha="right", va="center", fontsize=6.0,
             fontweight="bold" if t == "fluorostats" else "normal")
    if unbiased:
        axa.text(mae + sd + max(mae+sd for _,mae,sd,_ in rec)*0.02, i, "unbiased",
                 ha="left", va="center", fontsize=5.0, style="italic", color="#333")
axa.set_yticks([])
axa.set_xlabel("area-fraction error vs manual GT  (mean ± residual s.d.)")
axa.set_xlim(0, max(mae+sd for _,mae,sd,_ in rec)*1.28)
axa.text(0.98, 0.05, "‘unbiased’ = mean error not\nsignificantly ≠ 0 (t-test, Bonferroni)",
         transform=axa.transAxes, ha="right", va="bottom", fontsize=5.0, color="#333")
panel(axa, "a", f"vessel-tool accuracy   (REAVER, n = {rk.n_images.iloc[0]})", xanchor=0.0315)   # row-leader

# (b) VE overlay: raw / VesselExpress seg / fluorostats(auto->li), 2 crops  [row 1, full width]
crops = np.load(R/"ve_crops.npz")
gs_b = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, :], hspace=0.06, wspace=0.05)
# Landscape crop: keep full 400 px width, take a 215 px vertical band (rows 90:305)
# so the vessel band is shorter on the page while raw/VE/fluorostats columns, the
# scale bar and auto->li label are preserved. Same window for every panel so the
# raw / VesselExpress / fluorostats overlays stay pixel-aligned.
YB0, YB1 = 90, 305
labs = ["raw", "VesselExpress", r"fluorostats (auto$\rightarrow$li)"]
for row in range(2):
    raw = crops[f"raw{row}"][YB0:YB1]
    ve_m = crops[f"ve{row}"][YB0:YB1]
    fs_m = crops[f"fs{row}"][YB0:YB1]
    p = F.imnorm(raw, hi=99.7)
    for col, (lab, ov) in enumerate(zip(labs, [None, ve_m, fs_m])):
        ax = fig.add_subplot(gs_b[row, col])
        ax.imshow(p, cmap="gray")
        F.image_axes(ax)
        if ov is not None:
            F.fill_mask(ax, ov, (0.85,0.30,0.60) if col==1 else (0.13,0.55,0.85))
        if row == 0:
            ax.set_title(lab, fontsize=6.0)
        if row == 0 and col == 0:
            F.scalebar(ax, 100, "100 µm")
            panel(ax, "b", xanchor=0.0315)  # row-leader

# (c) 3D phantom accuracy vs exact GT  [row 2, left]
axc = fig.add_subplot(gs[2, 0])
ph = pd.read_csv(R/"b_vascular_phantom_3d.csv")
x = np.arange(len(ph))
w = 0.36
axc.bar(x-w/2, ph.len_err_pct, w, color=BLUE, edgecolor="#333333", lw=0.4, label="centreline length")
# volume-fraction error is exactly 0 for every phantom -> zero-height bars. Annotate it
# directly (green, the S1 torus "exact" convention) rather than legending a sky-blue
# swatch that never appears in the panel.
_ytop = max(4.0, ph.len_err_pct.max()*1.6)   # headroom so the above-bar labels fit
# annotate ON each bar (centred at x-w/2, just above its top), NOT offset to the right —
# offsetting right pushed the depth-2 label out of panel (c) and onto panel (d)'s y label.
for xi, h, vf in zip(x, ph.len_err_pct.values, ph.vf_err_pct.values):
    axc.text(xi-w/2, h + _ytop*0.035, "vol. frac.\n" + ("0% (exact)" if vf == 0 else f"{vf:.1f}%"),
             ha="center", va="bottom",
             fontsize=4.8, color=OKABE["green"], fontweight="bold", linespacing=1.05)
axc.set_xticks(x)
# prose "depth N" labels (tree_d1 -> depth 1), matching Fig S1b's naming
axc.set_xticklabels([f"{p.replace('tree_d', 'depth ')}\nbranches {b}/{t}{' exact' if b == t else ''}" for p,b,t in
    zip(ph.phantom, ph.fs_branches, ph.true_segments)], fontsize=5.0)
axc.set_ylabel("error vs exact GT (%)")
axc.set_ylim(0, _ytop)
# legend now describes only the one visible series (centreline-length error bars)
axc.legend(handles=[Patch(facecolor=BLUE, edgecolor="#333333", lw=0.4, label="centreline length")],
           fontsize=5.0, loc="upper right", handlelength=1.1, handleheight=0.9)
panel(axc, "c", "3D synthetic phantom", xanchor=0.0315)   # row-leader

# (d) Bland-Altman VF fluorostats vs VesselExpress + rank sub-panel  [row 2, right]
gs_d = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, 1], wspace=0.70)
axd = fig.add_subplot(gs_d[0, :2])
m = pd.read_csv(R/"b_ve_metrics.csv")
ve, fs = m.VesselExpress_VF.values, m.fluorostats_VF.values
offset = fs.mean() / ve.mean()                              # systematic fluorostats/VE ratio
bias, loa = F.bland_altman(axd, fs, ve, label=rf" · ~{offset:.1f}$\times$", annotate=False)
axd.set_ylim(-0.005, bias+loa+0.010)
axd.set_xlabel("mean vessel VF")
axd.set_ylabel("fluorostats − VesselExpress")
# bias/LoA annotation placed in clear lower-right white space, clear of the zero line
axd.text(0.97, 0.16, f"bias {bias:+.3f}\n±{loa:.3f} (95% LoA)\n~{offset:.1f}$\\times$",
         transform=axd.transAxes, ha="right", va="bottom", fontsize=5.0, color="#333")
panel(axd, "d", "metric agreement")

# rank-agreement gets its own small labelled panel (no longer a cramped inset)
axr = fig.add_subplot(gs_d[0, 2])
rho = pd.read_csv(R/"b_ve_agreement.csv", float_precision="round_trip").spearman_rho.iloc[0]
axr.scatter(sps.rankdata(ve), sps.rankdata(fs), s=13, color=BLUE, edgecolor="none")
# left-anchored title so the descriptor never overflows the narrow sub-panel edge
axr.set_title(f"rank agreement\nSpearman ρ = {rho:.2f}", fontsize=5.0, pad=2, loc="right")
axr.set_xlabel("VesselExpress rank", fontsize=5.0)
axr.set_ylabel("fluorostats rank", fontsize=5.0)
axr.set_xticks([1, len(ve)])
axr.set_yticks([1, len(fs)])
axr.set_xlim(0, len(ve) + 1)
axr.set_ylim(0, len(fs) + 1)
axr.tick_params(labelsize=5.0)
for sp in axr.spines.values():
    sp.set_linewidth(0.5)
    sp.set_color("#999")

# the caption states exact branch counts and volume fraction; hold it to the table
assert (ph.fs_branches == ph.true_segments).all() and (ph.vf_err_pct == 0).all(), \
    "b_vascular_phantom_3d.csv no longer exact: update the panel (c) caption"
qualifying = list(rk.tool[rk.unbiased])   # tools whose bias test is not significant
cap = ("Figure 3 | Vascular networks. (a) fluorostats inserted into the six-tool REAVER benchmark "
 f"(n={rk.n_images.iloc[0]}): area-fraction error vs manual ground truth (mean ± residual s.d. = accuracy and precision); "
 "'unbiased' marks tools whose mean error is not significantly different from 0 (one-sample t-test, "
 f"Bonferroni across the six tools); only {' and '.join(qualifying)} qualifies. fluorostats (blue) ranks alongside AngioTool "
 "in accuracy but, like every tool except REAVER, carries a small systematic underestimate. (b) Real "
 "light-sheet vessels (VesselExpress, Zenodo 6025935), 3-slice MIP crops: raw, VesselExpress software "
 "segmentation (magenta), fluorostats auto->li (blue); scale bar 100 µm. (c) 3D synthetic phantom vs "
 "exact ground truth: centreline-length error (%) with branch counts recovered exactly; volume-fraction "
 "error is 0.0 (exact) for every phantom. (d) Bland–Altman of vessel volume fraction, fluorostats vs "
 f"VesselExpress (n={len(fs)}): a ~{offset:.1f}× systematic offset (bias line ± 95% limits) with consistent ranking "
 f"(rank-agreement panel, Spearman {rho:.2f}) — a software-agreement comparison (VesselExpress GT is "
 "pipeline-generated, not manual).")
save(fig, "fig3_vascular", tight=False)
caption("fig3_vascular", cap)
print("Figure 3 done")
