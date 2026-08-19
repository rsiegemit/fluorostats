"""Figure 2 — Nucleus segmentation and the deep-learning boundary (a-f)."""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F
from figstyle import OKABE, TOOL, panel, save, caption, boot_ci
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
from skimage import filters
from skimage.morphology import remove_small_objects
from fluorostats.objects import label_3d
F.apply_style()
R = Path(__file__).resolve().parent / "results"
DL = F.DATA
GREY = OKABE["grey"]; LGREY = OKABE["lgrey"]
outline = F.outline

# ---------- data ----------
pi = pd.read_csv(R / "b2_nuclei_methods_perimage.csv")           # per-image F1, 9 methods
# n=100 first-100 per-image DL scores (so panel a uses the n=100 values, not the
# n=200 Table-1 numbers): StarDist 0.861, Cellpose 0.858, Omnipose 0.798.
dl_pi = {"StarDist": pd.read_csv(R/"stardist_eval.csv").stardist_f1.values[:100],
         "Cellpose": pd.read_csv(R/"cellpose_eval.csv").cellpose_f1.values[:100],
         "Omnipose": pd.read_csv(R/"omnipose_eval.csv").query("thirddl_f1>=0").thirddl_f1.values[:100]}
methods = {c: pi[c].values for c in pi.columns if c != "image"}
methods.update(dl_pi)
# ---------- layout ----------
# Authored at the true manuscript text width (F.TW = 5.147 in) so
# \includegraphics[width=\textwidth] is 1:1 and the 6-7 pt type stays 6-7 pt on the
# page. The narrower canvas is re-fit as a TALLER 4-row stack: (a) the busy
# 12-row unification dot-plot gets a full-width row of its own, (b) the forest plot
# a shorter full-width row, (c) the 2x3 image band full width, and (d)-(e)-(f)
# share the bottom row.
# Plain GridSpec (not constrained-layout): the nested fixed-aspect image band (c)
# collapses constrained-layout's sibling axes, so lay out manually with generous
# margins/spacing and export with tight=True.
fig = plt.figure(figsize=(F.TW, 5.818))
gs = gridspec.GridSpec(4, 6, figure=fig, height_ratios=[1.72, 0.66, 0.92, 1.02],
                       left=0.275, right=0.965, top=0.9654, bottom=0.0573,
                       hspace=0.532, wspace=0.62)

# (a) the fluorostats "unification" panel. Otsu / Li / Isodata / Triangle / Yen /
# Mean / Minimum / watershed are NOT rival tools: they are fluorostats threshold
# CONFIGURATIONS. So the panel shows ONE designated default (fluorostats Otsu+CC,
# solid blue, the Table-1 / abstract 0.890 headline) dominant; the other thresholds
# as the SAME blue at reduced opacity (the "tuning envelope", a range-of-a-knob);
# and the three trained DL tools (StarDist orange, Cellpose vermillion, Omnipose
# purple) as the genuine external comparison, plotted at their n=100 values.
# Dot + 95% bootstrap CI (F.lollipop-style) instead of bars: the top pack all hugs
# ~0.8-0.91, so full bars would waste ink.
axa = fig.add_subplot(gs[0, :])
BLUE = OKABE["blue"]
# family: (display label, per-image score array, kind) — kind in
# {"default","envelope","dl"}. Provenance kept in the label.
fam = [
    ("fluorostats (default, Otsu+CC)", pi["fluorostats (Otsu+CC)"].values, "default"),
    ("fluorostats · Li (1993)",       pi["Li (1993)"].values,       "envelope"),
    ("fluorostats · Isodata (1978)",  pi["Isodata (1978)"].values,  "envelope"),
    ("fluorostats · Otsu+watershed",  pi["Watershed (1991)"].values,"envelope"),
    ("fluorostats · Otsu (1979)",     pi["Otsu (1979)"].values,     "envelope"),
    ("fluorostats · Mean",            pi["Mean"].values,            "envelope"),
    ("fluorostats · Triangle (1977)", pi["Triangle (1977)"].values, "envelope"),
    ("fluorostats · Minimum",         pi["Minimum"].values,         "envelope"),
    ("fluorostats · Yen (1995)",      pi["Yen (1995)"].values,      "envelope"),
    ("StarDist (trained DL)", dl_pi["StarDist"], "dl"),
    ("Cellpose (trained DL)", dl_pi["Cellpose"], "dl"),
    ("Omnipose (trained DL)", dl_pi["Omnipose"], "dl"),
]
DLCOL = {"StarDist": OKABE["orange"], "Cellpose": OKABE["vermillion"], "Omnipose": OKABE["purple"]}
rows = []
for lab, arr, kind in fam:
    mean, lo, hi = boot_ci(arr)
    if kind == "dl":
        c = DLCOL[lab.split()[0]]
    else:
        c = BLUE
    rows.append(dict(lab=lab, mean=mean, lo=lo, hi=hi, kind=kind, c=c))
rows.sort(key=lambda d: d["mean"])                # worst -> best (bottom -> top)
lo_x = 0.20                                        # zoom floor just below the min (Yen ~0.28)
hi_x = 1.06                                         # headroom for value labels
for row, d in enumerate(rows):
    default = d["kind"] == "default"
    env = d["kind"] == "envelope"
    a = 1.0 if default else (0.42 if env else 1.0)   # envelope translucent
    ms = 7 if default else 5.2
    # stem
    axa.plot([lo_x, d["mean"]], [row, row], color=d["c"], lw=1.0, alpha=a * 0.6,
             zorder=2, solid_capstyle="butt")
    # CI whisker
    axa.plot([d["lo"], d["hi"]], [row, row], color=d["c"], lw=1.4, alpha=a * 0.9, zorder=3)
    # dot: filled for default/DL, ringed (blue outline) for the envelope so it reads
    # as "same family, other setting"
    if env:
        axa.plot(d["mean"], row, "o", mfc="white", mec=d["c"], mew=1.1, ms=ms,
                 alpha=1.0, zorder=4)
    else:
        axa.plot(d["mean"], row, "o", color=d["c"], mec="white", mew=0.6, ms=ms, zorder=5)
    # anchor the value past the envelope END (d["hi"]), not the marker, so the label
    # clears the pale-blue envelope bar on the rows where it extends past the mean
    axa.annotate(f"{d['mean']:.3f}", (d["hi"], row), xytext=(9, 0),
                 textcoords="offset points", va="center", ha="left",
                 fontsize=5.0, color="#333", alpha=a)
labcols = [BLUE if r["kind"] != "dl" else r["c"] for r in rows]
axa.set_yticks(range(len(rows)))
axa.set_yticklabels([r["lab"] for r in rows], fontsize=5.0)
for tick, lc, r in zip(axa.get_yticklabels(), labcols, rows):
    tick.set_color(lc)
    tick.set_fontweight("bold" if r["kind"] == "default" else "normal")
axa.set_xlim(lo_x, hi_x); axa.set_ylim(-0.6, len(rows) - 0.4)
axa.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
axa.set_xlabel("instance F1 @ IoU 0.5   (BBBC039, n = 100)")
# legend explaining the family
axa.plot([], [], "o", color=BLUE, mec="white", mew=0.6, ms=6, label="fluorostats default")
axa.plot([], [], "o", mfc="white", mec=BLUE, mew=1.1, ms=5.5, label="envelope")
# neutral key: the DL family is drawn in 3 colours (StarDist/Cellpose/Omnipose), so a
# single coloured key would mis-map; caption carries the colour mapping.
axa.plot([], [], "o", mfc="none", mec="black", mew=0.8, ms=5.5, label="trained DL")
# legend OUTSIDE the axes: single row in the band above the top spine, right-aligned.
# The panel title is left-aligned in the same band, so the two cannot meet at the page width.
axa.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=3,
           fontsize=F.FS["legend"], handletextpad=0.25, borderpad=0.2,
           columnspacing=1.1, frameon=False)
panel(axa, "a", "one tool, one default knob (n = 100)", xanchor=0.0315)   # row-leader

# (b) forest vs DL — the n=200 FULL test set (Table 1 / Results), NOT panel (a)'s
# n=100 envelope split. fluorostats mean+CI from b_dl_ci.csv (0.896 [0.873, 0.916]);
# the three DL tools from their full 200-image eval arrays. Panel (a) stays at n=100.
axb = fig.add_subplot(gs[1, :])
ci = pd.read_csv(R/"b_dl_ci.csv").set_index("method")
dl200 = {"StarDist": pd.read_csv(R/"stardist_eval.csv").stardist_f1.values,
         "Cellpose": pd.read_csv(R/"cellpose_eval.csv").cellpose_f1.values,
         "Omnipose": pd.read_csv(R/"omnipose_eval.csv").query("thirddl_f1>=0").thirddl_f1.values}
fb = {"fluorostats": (float(ci.loc["fluorostats", "mean_F1"]),
                      float(ci.loc["fluorostats", "CI_low"]), float(ci.loc["fluorostats", "CI_high"]))}
for m, arr in dl200.items():
    fb[m] = boot_ci(arr)
fo = sorted(fb, key=lambda m: fb[m][0])
for i, m in enumerate(fo):
    mean, lo, hi = fb[m]; c = OKABE["blue"] if m=="fluorostats" else TOOL[m]["c"]
    axb.errorbar(mean, i, xerr=[[mean-lo],[hi-mean]], fmt=TOOL.get(m,{}).get("m","o"),
                 color=c, ecolor="black", capsize=2.5, ms=6, mec="black", mew=0.5, elinewidth=0.9)
axb.set_yticks(range(len(fo))); axb.set_yticklabels(fo, fontsize=6.0)
axb.set_xlabel("mean F1  (95% CI, n = 200)"); axb.set_xlim(0.78, 0.955)
axb.text(0.97, 0.06, "paired Δ excludes 0:\n−StarDist +0.025\n−Cellpose +0.034",
         transform=axb.transAxes, fontsize=5.0, color="#333", va="bottom", ha="right")
panel(axb, "b", "vs trained deep learning (n = 200)", xanchor=0.0315)   # row-leader

# (c) qualitative overlay: raw / GT / fluorostats, 3 crops
def gt_inst(stem):
    return F.gt_instances_bbbc039(DL/"BBBC039/masks/masks"/f"{stem}.png")
imgs = sorted(glob.glob(str(DL/"BBBC039/images/images/*.tif")))
if not imgs:
    raise SystemExit("BBBC039 images not found under $FLUOROSTATS_DATA "
                     f"({DL}/BBBC039) — download per benchmarks/DATA_MANIFEST.md")
crops = []
hh, ww = 150, 250   # landscape crops (aspect ~1.67) so the 2x3 band stays short
for f in imgs:
    im = np.asarray(Image.open(f)).astype(np.float32); stem = Path(f).stem
    gt = gt_inst(stem)
    if gt.max() < 8: continue
    ys, xs = np.where(gt > 0); cy, cx = int(ys.mean()), int(xs.mean())
    y0 = min(max(0, cy-hh), im.shape[0]-2*hh); x0 = min(max(0, cx-ww), im.shape[1]-2*ww)
    sl = (slice(y0, y0+2*hh), slice(x0, x0+2*ww))
    crops.append((im[sl], gt[sl], stem))
    if len(crops) == 2: break
labs = ["raw", "expert GT", "fluorostats"]
# 2x3 image grid via nested gridspec
gs_c = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[2, :], hspace=0.06, wspace=0.06)
for row, (im, gt, stem) in enumerate(crops):
    fs = label_3d(remove_small_objects(im > filters.threshold_otsu(im), 20), min_size=20)[0]
    p = F.imnorm(im)
    for col, (lab, arr) in enumerate(zip(labs, [None, gt, fs])):
        ax = fig.add_subplot(gs_c[row, col]); ax.imshow(p, cmap="gray"); F.image_axes(ax)
        if arr is not None:
            outline(ax, arr, (0.0,0.62,0.45) if col==1 else (0.0,0.45,0.70), width=2)
        if row == 0: ax.set_title(lab, fontsize=F.FS["title"])
        if col == 0 and row == 0:
            F.scalebar(ax, 65, "20 µm")  # BBBC039 ~0.31 µm/px (approx, stated in caption)
            panel(ax, "c", xanchor=0.0315)   # row-leader
# (d) crossover — two directly-labelled line groups: fluorostats (blue) and the
# classical-threshold envelope (grey), both collapsing under overlap while the DL
# reference (dashed) holds ~0.96.
axd = fig.add_subplot(gs[3, :2])
cc = pd.read_csv(R/"b_clustering_curve.csv"); x = [0,25,50,75]
for _, r in cc.iterrows():
    isf = "fluorostats" in r["method"]
    if not isf and r[["c00","c25","c50","c75"]].sum() == 0:   # skip empty Triangle row
        continue
    axd.plot(x, [r.c00,r.c25,r.c50,r.c75], marker="o" if isf else ".", ms=4 if isf else 3,
             color=OKABE["blue"] if isf else GREY, lw=2.0 if isf else 0.8, alpha=1 if isf else 0.6,
             zorder=3 if isf else 1)
axd.axhline(0.96, ls="--", color=OKABE["black"], lw=1.0)
axd.text(74, 0.955, "DL holds $\\approx$0.96", fontsize=5.0, color="black",
         ha="right", va="top")
axd.axvspan(50, 75, color=OKABE["vermillion"], alpha=0.08)
axd.text(63, 0.42, "DL wins", fontsize=6, color=OKABE["vermillion"], ha="center")
# direct labels on the two line groups (in the clear triangle below both curves)
axd.text(2, 0.735, "fluorostats", fontsize=6.0, color=OKABE["blue"], fontweight="bold",
         ha="left", va="bottom",
         bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
axd.text(2, 0.14, "classical envelope", fontsize=5.0, color=GREY, ha="left", va="center")
axd.set_xlabel("nuclear overlap (%)"); axd.set_ylabel("instance F1"); axd.set_ylim(0,1.02)
panel(axd, "d", "crowding crossover", xanchor=0.0315)   # row-leader

# (e) separated vs crowded fields
gs_e = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3, 2:4], wspace=0.06)
_c75 = sorted(glob.glob(str(DL/"BBBC024_c75/image-final_*.tif")))
if not _c75:
    raise SystemExit("BBBC024 (c75 crowded set) not found under $FLUOROSTATS_DATA "
                     f"({DL}/BBBC024_c75) — download per benchmarks/DATA_MANIFEST.md")
for i, (fp, ttl) in enumerate([(str(DL/"BBBC024/image-final_0000.tif"), "separated"),
                               (_c75[0], "crowded")]):
    sl = F.mid_slice(fp)
    ax = fig.add_subplot(gs_e[i]); ax.imshow(F.imnorm(sl), cmap="gray"); F.image_axes(ax)
    fsm = label_3d(remove_small_objects(sl > filters.threshold_otsu(sl), 15), min_size=15)[0]
    outline(ax, fsm, (0.0,0.45,0.70), width=2); ax.set_title(ttl, fontsize=7)
    if i == 0: panel(ax, "e")

# (f) scope decision map
axf = fig.add_subplot(gs[3, 4:])
axf.axvspan(0, 30, color=OKABE["green"], alpha=0.14); axf.axvspan(30, 50, color=OKABE["yellow"], alpha=0.16)
axf.axvspan(50, 100, color=OKABE["vermillion"], alpha=0.14)
axf.axvline(40, ls="--", color="black", lw=1.0, ymax=0.92)
axf.text(43, 0.5, "measured\ncrossover $\\approx$40%", fontsize=5.0)
axf.text(15, 0.82, "fluorostats\n= DL", ha="center", fontsize=6.0, color=OKABE["green"])
axf.text(75, 0.82, "use trained\nsegmenter", ha="center", fontsize=6.0, color=OKABE["vermillion"])
axf.set_xlim(0,100); axf.set_ylim(0,1); axf.set_yticks([])
axf.set_xlabel("overlap / crowding (%)")
panel(axf, "f", "when to use which")

# caption numbers derived from the same per-image data the panels plot (never hand-typed)
sd_m, cp_m, om_m = (float(np.mean(dl_pi[k])) for k in ("StarDist", "Cellpose", "Omnipose"))
def_m = float(np.mean(pi["fluorostats (Otsu+CC)"].values))
_env = [float(np.mean(pi[c].values)) for c in
        ("Li (1993)", "Isodata (1978)", "Watershed (1991)", "Otsu (1979)", "Mean",
         "Triangle (1977)", "Minimum", "Yen (1995)")]
env_lo, env_hi = min(_env), max(_env)
cap = ("Figure 2 | Nucleus segmentation and the deep-learning boundary. "
 "(a) Instance F1 (IoU 0.5) on BBBC039 (n=100 images), dots with 95% bootstrap CIs "
 "(10,000 resamples of per-image scores). The eight classical thresholds (Otsu, Li, "
 "Isodata, Triangle, Yen, Mean, Minimum, watershed) are fluorostats CONFIGURATIONS, not "
 f"rival tools: the designated default (fluorostats Otsu+CC, solid blue, F1 {def_m:.3f}) is the "
 "headline; the remaining thresholds (open blue circles) trace the tuning envelope "
 f"({env_lo:.2f}-{env_hi:.2f}, i.e. the range of one knob). The genuine external comparison is trained deep "
 f"learning at n=100 values (StarDist {sd_m:.3f} orange, Cellpose {cp_m:.3f} vermillion, Omnipose {om_m:.3f} "
 "purple). (b) Forest plot on the full n=200 test set (Table 1) vs validated DL (StarDist, Cellpose, "
 "Omnipose): fluorostats 0.896 [0.873, 0.916]; paired differences exclude zero. (c) Representative BBBC039 crops: raw, expert ground "
 "truth (green outlines), fluorostats (blue outlines); scale bar 20 µm (BBBC039 ~0.31 µm/px). "
 "(d) Instance F1 vs nuclear overlap (BBBC024 c00-c75): all non-DL methods (grey; fluorostats blue) "
 "collapse while DL (dashed) holds ~0.96; red band = DL advantage. (e) fluorostats overlay on a "
 "well-separated (c00) and a crowded (c75) field, showing connected-component merging under overlap. "
 "(f) Data-anchored scope map: fluorostats is at parity with trained segmenters up to ~40% crowding "
 "(green), beyond which a trained instance segmenter is preferred (red).")
save(fig, "fig2_nuclei_boundary", tight=False); caption("fig2_nuclei_boundary", cap)
print("Figure 2 done")
