"""Figure 4 — Depth-resolved viability (a-c)."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F
from figstyle import OKABE, panel, save, caption
import matplotlib.pyplot as plt
from matplotlib import gridspec
from fluorostats.segment import binarize
F.apply_style()
R = Path(__file__).resolve().parent / "results"
BLUE = OKABE["blue"]
GREEN = OKABE["green"]
MAG = OKABE["purple"]
VERM = OKABE["vermillion"]
GREY = OKABE["grey"]

# Authored at the text width (F.TW = 5.147 in) so
# \includegraphics[width=\textwidth] is 1:1 and the 6-7 pt type stays 6-7 pt on the
# page. Narrower canvas -> taller figure; the 2x2 plot grid (a-d) sits above a
# full-width 3-across image row (e). Extra top/left margin + hspace keeps panel
# letters, legends and titles clear of neighbours at this width.
fig = plt.figure(figsize=(F.TW, 5.570))
gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.05, 1.05, 0.80],
                       hspace=0.42, wspace=0.85,
                       left=0.150, right=0.975, top=0.964, bottom=0.005)

# (a) 2D/heuristic bias vs true 3D + depth profile
axa = fig.add_subplot(gs[0, :2])
bm = pd.read_csv(R/"b3_viability_multi.csv")
bm = bm[bm.method != "full_3D_voxelwise(REF)"].copy()
lab = {"midplane_slice":"mid-plane","MIP":"MIP","mean_of_per_slice":"mean-of-slices",
       "attenuation_corrected_3D":"attn-corrected","brightest_focus_slice":"brightest focus"}
bm["name"] = bm.method.map(lab)
bm = bm.sort_values("rel_bias_pct").reset_index(drop=True)
rbias = dict(zip(bm.method, bm.rel_bias_pct))   # per-method relative bias (for the caption)
# attenuation-corrected 3D is the fluorostats result -> blue (the signal); the 2D
# reductions are context -> neutral grey (never Cellpose vermillion, which is locked)
y = np.arange(len(bm))
cols = [BLUE if "attenuation" in m else GREY for m in bm.method]
axa.hlines(y, 0, bm.rel_bias_pct, color=cols, lw=1.3, zorder=1)
axa.scatter(bm.rel_bias_pct, y, color=cols, s=44, edgecolor="black", lw=0.4, zorder=3)
for yi, rb in zip(y, bm.rel_bias_pct):
    axa.text(rb + 0.8, yi, f"{rb:+.1f}%", va="center", fontsize=F.FS["annot"])
axa.axvline(0, color="black", lw=0.6)
axa.set_xlim(-1, 32)
axa.set_ylim(-0.6, len(bm)-0.4)
axa.set_yticks(y)
axa.set_yticklabels(bm.name)
axa.set_xlabel("overestimate of live fraction vs true 3D (%)")
panel(axa, "a", "2D inflates viability", xanchor=0.0315)   # row-leader

# (b) depth gradient
axb = fig.add_subplot(gs[0, 2:])
dp = pd.read_csv(R/"b3_viability_depth_profile.csv")
axb.plot(dp.depth_um, dp.live_fraction_raw, color=GREY, lw=1.8, label="raw")
axb.plot(dp.depth_um, dp.live_fraction_attn_corrected, color=BLUE, lw=1.8, ls="--", label="attn-corrected")
axb.set_xlabel("depth (µm)")
axb.set_ylabel("live fraction per z-slice")
axb.legend(fontsize=6, loc="lower left")
panel(axb, "b", "depth gradient")

# (c) maxima mode vs exact ground truth (Kerkhoff synthetic set)
ext = pd.read_csv(R/"b_viability_external.csv")
tv = ext.true_viability.values
skp = ext["skimage_peakcount"].values
fx = ext["fluorostats_maxima"].values
# fluorostats-vs-ground-truth agreement (Lin's CCC + MAE), from the producer's table
agr = pd.read_csv(R/"b_viability_external_agreement.csv", float_precision="round_trip").set_index("estimator")
ccc_v = agr.loc["fluorostats_maxima", "lins_ccc"]
mae_v = agr.loc["fluorostats_maxima", "MAE"]
axc = fig.add_subplot(gs[1, :2])
axc.scatter(tv, ext["fluorostats_objcount"].values, s=10, color=GREY, alpha=0.55,
            zorder=1, label="fs alt. modes (count, area)")
axc.scatter(tv, ext["fluorostats_areafrac"].values, s=10, color=OKABE["lgrey"], alpha=0.55, zorder=1)
axc.scatter(tv, skp, s=20, marker="s", facecolor="none", edgecolor=OKABE["orange"], lw=0.7, label="skimage peak count", zorder=3)
axc.scatter(tv, fx, s=22, color=BLUE, edgecolor="black", lw=0.3, label="fluorostats maxima", zorder=4)
axc.plot([0.3,1],[0.3,1], ls="--", color="black", lw=0.7)
axc.set_xlabel("true viability")
axc.set_ylabel("measured viability")
axc.set_xlim(0.3,1.0)
axc.set_ylim(0.3,1.0)
axc.legend(fontsize=6.0, loc="upper left")
panel(axc, "c", "maxima vs ground truth", xanchor=0.0315)   # row-leader

# (d) Bland-Altman: fluorostats maxima vs the direct skimage call (same algorithm -> identical)
axd = fig.add_subplot(gs[1, 2:])
_bd, _ld = F.bland_altman(axd, fx, skp, annotate=False)
axd.set_ylim(-0.02, 0.02)
axd.text(0.02, 0.96, f"bias {_bd:+.3f}\n±{_ld:.3f} (95% LoA)", transform=axd.transAxes,
         va="top", fontsize=6.0, color="#333")
axd.set_xlabel("mean viability (fs, skimage)")
axd.set_ylabel("fluorostats $-$ skimage")
axd.text(0.5, 0.30, "maxima = skimage\n(same peak counting)", transform=axd.transAxes, ha="center", va="center", fontsize=6.0, color="#555")
axd.text(0.5, 0.08, f"vs ground truth: CCC {ccc_v:.3f}, MAE {mae_v:.3f}", transform=axd.transAxes,
         ha="center", va="bottom", fontsize=5.0, color="#777")
panel(axd, "d", "fluorostats $-$ skimage = 0")

# (e) Live/Dead qualitative overlay
# exception DAT-3: an overlay shows the segmentation on the image itself, so the
# display masks are computed here from the raw stack with the tables' settings.
live = F.load_sbiad2130_channel(F.SBIAD2130_LIVE_CH)
dead = F.load_sbiad2130_channel(F.SBIAD2130_DEAD_CH)
z = live.shape[0]//2
SCALEBAR_UM = 500
gs_c = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :], wspace=0.10)
lm_mid = binarize(live[z], **F.SBIAD2130_SEG)
dm_mid = binarize(dead[z], **F.SBIAD2130_SEG)
# short, wrapped titles so each ~1.6-in crop keeps its label inside its own cell
panels_c = [
    (F.composite2ch(live[z], dead[z]), "raw mid-plane\n(green live / magenta dead)"),
    (np.dstack([dm_mid*0.85, lm_mid*0.85, dm_mid*0.85]), "fluorostats\nclassification\n(grey = both, black = neither)"),
    (F.composite2ch(live.max(0), dead.max(0)), f"MIP — 2D collapses\ndepth ({rbias['MIP']:+.0f}%)"),
]
for i,(img,ttl) in enumerate(panels_c):
    ax = fig.add_subplot(gs_c[i])
    ax.imshow(np.clip(img,0,1))
    F.image_axes(ax)
    ax.set_title(ttl, fontsize=6.0, loc="center")
    if i == 0:
        # 500 µm at the file's calibration (the stack is XY-downsampled on load)
        F.scalebar(ax, SCALEBAR_UM / F.sbiad2130_voxel_um()[2], f"{SCALEBAR_UM} µm",
                   center_label=True)
        panel(ax, "e", xanchor=0.0315)  # row-leader

cap = ("Figure 4 | Depth-resolved viability. (a) On a Day-14 Live/Dead z-stack (BioImage Archive "
 "S-BIAD2130), 2D/heuristic reductions overestimate the live fraction relative to the true voxelwise "
 f"3D value: mid-plane {rbias['midplane_slice']:+.1f}%, brightest-focus {rbias['brightest_focus_slice']:+.1f}%, "
 f"MIP {rbias['MIP']:+.1f}%, naive mean-of-slices {rbias['mean_of_per_slice']:+.1f}%; "
 f"attenuation-corrected 3D stays within {rbias['attenuation_corrected_3D']:+.1f}% (blue). (b) Per-z live fraction shows the depth "
 "gradient (raw) that a single 2D readout misses, and how attenuation correction flattens it. "
 "(c) On the synthetic Live/Dead set published with the Kerkhoff Fiji macro (Zenodo 10395753, exact "
 f"ground-truth viability), fluorostats live_dead_by_count(maxima; blue) against ground truth: CCC {ccc_v:.3f}, "
 f"MAE {mae_v:.3f}. Orange squares: a direct skimage peak_local_max count with the same parameters, "
 "which coincides with the maxima mode by construction; the macro itself was not run. fluorostats "
 "count/area modes shown faint (grey). (d) Bland–Altman of fluorostats maxima minus the direct skimage "
 "count: 0 because both apply the same peak-counting algorithm, a consistency check rather than an "
 "external comparison. (e) Two-channel Live/Dead crop (green live, magenta dead): raw mid-plane, "
 "fluorostats live/dead classification, and the maximum-intensity projection that collapses depth "
 f"and inflates viability. Scale bar {SCALEBAR_UM} µm.")
save(fig, "fig4_viability", tight=False)
caption("fig4_viability", cap)
print("Figure 4 done")
