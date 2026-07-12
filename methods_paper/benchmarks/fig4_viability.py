"""Figure 4 — Depth-resolved viability (a-c)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F
from figstyle import OKABE, panel, save, caption
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats as sps
from fluorostats.segment import binarize
F.apply_style()
R = Path(__file__).resolve().parent / "results"
STACK = F.DATA / "viability" / "zccs1035_Day14_LiveDead.tif"
BLUE = OKABE["blue"]; GREEN = OKABE["green"]; MAG = OKABE["purple"]; VERM = OKABE["vermillion"]; GREY = OKABE["grey"]

# Authored at the true submission text width (F.TW = 5.147 in) so
# \includegraphics[width=\textwidth] is 1:1 and the 6-7 pt type stays 6-7 pt on the
# page. Narrower canvas -> taller figure; the 2x2 plot grid (a-d) sits above a
# full-width 3-across image row (e). Extra top/left margin + hspace keeps panel
# letters, legends and titles clear of neighbours at this width.
fig = plt.figure(figsize=(F.TW, 6.6))
gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.05, 1.05, 1.95],
                       hspace=0.62, wspace=0.85,
                       left=0.135, right=0.975, top=0.955, bottom=0.045)

# (a) 2D/heuristic bias vs true 3D + depth profile
axa = fig.add_subplot(gs[0, :2])
bm = pd.read_csv(R/"b3_viability_multi.csv")
bm = bm[bm.method != "full_3D_voxelwise(REF)"].copy()
lab = {"midplane_slice":"mid-plane","MIP":"MIP","mean_of_per_slice":"mean-of-slices",
       "attenuation_corrected_3D":"attn-corrected","brightest_focus_slice":"brightest focus"}
bm["name"] = bm.method.map(lab); bm = bm.sort_values("rel_bias_pct").reset_index(drop=True)
rbias = dict(zip(bm.method, bm.rel_bias_pct))   # per-method relative bias (for the caption)
# attenuation-corrected 3D is the fluorostats result -> blue (the signal); the 2D
# reductions are context -> neutral grey (never Cellpose vermillion, which is locked)
y = np.arange(len(bm)); cols = [BLUE if "attenuation" in m else GREY for m in bm.method]
axa.hlines(y, 0, bm.rel_bias_pct, color=cols, lw=1.3, zorder=1)
axa.scatter(bm.rel_bias_pct, y, color=cols, s=44, edgecolor="black", lw=0.4, zorder=3)
for yi, rb in zip(y, bm.rel_bias_pct):
    axa.text(rb + 0.8, yi, f"+{rb:.1f}%", va="center", fontsize=F.FS["annot"])
axa.axvline(0, color="black", lw=0.6); axa.set_xlim(-1, 32); axa.set_ylim(-0.6, len(bm)-0.4)
axa.set_yticks(y); axa.set_yticklabels(bm.name)
axa.set_xlabel("overestimate of live fraction vs true 3D (%)")
panel(axa, "a", "2D inflates viability")

# (b) depth gradient
axb = fig.add_subplot(gs[0, 2:])
dp = pd.read_csv(R/"b3_viability_depth_profile.csv")
axb.plot(dp.depth_um, dp.live_fraction_raw, color=GREY, lw=1.8, label="raw")
axb.plot(dp.depth_um, dp.live_fraction_attn_corrected, color=BLUE, lw=1.8, ls="--", label="attn-corrected")
axb.set_xlabel("depth (µm)"); axb.set_ylabel("live fraction per z-slice")
axb.legend(fontsize=6, loc="lower left"); panel(axb, "b", "depth gradient")

# (c) tie to published Fiji macro (Kerkhoff synthetic GT)
ext = pd.read_csv(R/"b_viability_external.csv")
tv = ext.true_viability.values; mac = ext["Kerkhoff_macro_peakcount"].values; fx = ext["fluorostats_maxima(NEW)"].values
# fluorostats-vs-ground-truth agreement (computed, not hand-typed): Lin's CCC + MAE
def _ccc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return 2 * ((a - a.mean()) * (b - b.mean())).mean() / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)
ccc_v = _ccc(fx, tv); mae_v = float(np.abs(fx - tv).mean())
axc = fig.add_subplot(gs[1, :2])
axc.scatter(tv, ext["fluorostats_objcount"].values, s=10, color=GREY, alpha=0.55,
            zorder=1, label="fs alt. modes (count, area)")
axc.scatter(tv, ext["fluorostats_areafrac"].values, s=10, color=OKABE["lgrey"], alpha=0.55, zorder=1)
axc.scatter(tv, mac, s=20, marker="s", facecolor="none", edgecolor=OKABE["orange"], lw=0.7, label="Kerkhoff Fiji macro", zorder=3)
axc.scatter(tv, fx, s=22, color=BLUE, edgecolor="black", lw=0.3, label="fluorostats maxima", zorder=4)
axc.plot([0.3,1],[0.3,1], ls="--", color="black", lw=0.7)
axc.set_xlabel("true viability"); axc.set_ylabel("measured viability"); axc.set_xlim(0.3,1.0); axc.set_ylim(0.3,1.0)
axc.legend(fontsize=5.6, loc="upper left")
panel(axc, "c", "ties the published Fiji macro")

# (d) Bland-Altman: fluorostats maxima vs macro (same algorithm -> identical)
axd = fig.add_subplot(gs[1, 2:])
F.bland_altman(axd, fx, mac); axd.set_ylim(-0.02, 0.02)
axd.set_xlabel("mean viability (fs, macro)"); axd.set_ylabel("fluorostats $-$ macro")
axd.text(0.5, 0.30, "maxima = macro\n(same peak counting)", transform=axd.transAxes, ha="center", va="center", fontsize=5.6, color="#555")
axd.text(0.5, 0.08, f"vs ground truth: CCC {ccc_v:.3f}, MAE {mae_v:.3f}", transform=axd.transAxes,
         ha="center", va="bottom", fontsize=5.4, color="#777")
panel(axd, "d", "fluorostats $-$ macro = 0")

# (e) Live/Dead qualitative overlay
live = F.load_channels(STACK, 2, down=8); dead = F.load_channels(STACK, 1, down=8); z = live.shape[0]//2
gs_c = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :], wspace=0.10)
seg = dict(method="otsu", min_size=8)
lm_mid = binarize(live[z], **seg); dm_mid = binarize(dead[z], **seg)
# short, wrapped titles so each ~1.6-in crop keeps its label inside its own cell
panels_c = [
    (F.composite2ch(live[z], dead[z]), "raw mid-plane\n(green live / magenta dead)"),
    (np.dstack([dm_mid*0.85, lm_mid*0.85, dm_mid*0.85]), "fluorostats\nclassification"),
    (F.composite2ch(live.max(0), dead.max(0)), "MIP — 2D collapses\ndepth (+5%)"),
]
for i,(img,ttl) in enumerate(panels_c):
    ax = fig.add_subplot(gs_c[i]); ax.imshow(np.clip(img,0,1)); F.image_axes(ax)
    ax.set_title(ttl, fontsize=6.0, loc="center")
    if i == 0: F.scalebar(ax, 12, "200 µm"); panel(ax, "e")

cap = ("Figure 4 | Depth-resolved viability. (a) On a Day-14 Live/Dead z-stack (BioImage Archive "
 "S-BIAD2130), 2D/heuristic reductions overestimate the live fraction relative to the true voxelwise "
 f"3D value: mid-plane +{rbias['midplane_slice']:.1f}%, brightest-focus +{rbias['brightest_focus_slice']:.1f}%, "
 f"MIP +{rbias['MIP']:.1f}%, naive mean-of-slices +{rbias['mean_of_per_slice']:.1f}%; "
 f"attenuation-corrected 3D stays within +{rbias['attenuation_corrected_3D']:.1f}% (blue). (b) Per-z live fraction shows the depth "
 "gradient (raw) that a single 2D readout misses, and how attenuation correction flattens it. "
 "(c) On the published Kerkhoff Fiji Live/Dead macro's synthetic dataset (Zenodo 10395753, exact "
 "ground-truth viability), fluorostats live_dead_by_count(maxima; blue) reproduces the macro exactly "
 "(identity scatter); fluorostats count/area modes shown faint (grey). (d) Bland–Altman of "
 "fluorostats maxima minus macro: the difference is 0 because both apply the same peak-counting "
 f"algorithm; the agreement with ground truth (CCC {ccc_v:.3f}, MAE {mae_v:.3f}) is a separate, each-method-vs-"
 "truth comparison. (e) Two-channel Live/Dead crop (green live, magenta dead): raw mid-plane, "
 "fluorostats live/dead classification, and the maximum-intensity projection that collapses depth "
 "and inflates viability. Scale bar 200 µm (approx., downsampled).")
save(fig, "fig4_viability", tight=False); caption("fig4_viability", cap)
print("Figure 4 done")
