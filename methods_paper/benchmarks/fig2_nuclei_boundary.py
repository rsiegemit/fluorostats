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
from skimage.segmentation import watershed, find_boundaries
from scipy import ndimage as ndi
from fluorostats.objects import label_3d
F.apply_style()
R = Path(__file__).resolve().parent / "results"
DL = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads")
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
fig = plt.figure(figsize=(7.2, 7.6), layout="constrained")
fig.get_layout_engine().set(hspace=0.06, wspace=0.06)
gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1.05, 1.75, 0.82])

# (a) the fluorostats "unification" panel. Otsu / Li / Isodata / Triangle / Yen /
# Mean / Minimum / watershed are NOT rival tools: they are fluorostats threshold
# CONFIGURATIONS. So the panel shows ONE designated default (fluorostats Otsu+CC,
# solid blue, the Table-1 / abstract 0.890 headline) dominant; the other thresholds
# as the SAME blue at reduced opacity (the "tuning envelope", a range-of-a-knob);
# and the three trained DL tools (StarDist orange, Cellpose vermillion, Omnipose
# purple) as the genuine external comparison, plotted at their n=100 values.
# Dot + 95% bootstrap CI (F.lollipop-style) instead of bars: the top pack all hugs
# ~0.8-0.91, so full bars would waste ink.
axa = fig.add_subplot(gs[0, :4])
BLUE = OKABE["blue"]
# family: (display label, per-image score array, kind) — kind in
# {"default","envelope","dl"}. Provenance kept in the label.
fam = [
    ("fluorostats (default, Otsu+CC)", pi["fluorostats (Otsu+CC)"].values, "default"),
    ("fluorostats $\\cdot$ Li (1993)",       pi["Li (1993)"].values,       "envelope"),
    ("fluorostats $\\cdot$ Isodata (1978)",  pi["Isodata (1978)"].values,  "envelope"),
    ("fluorostats $\\cdot$ Otsu+watershed",  pi["Watershed (1991)"].values,"envelope"),
    ("fluorostats $\\cdot$ Otsu (1979)",     pi["Otsu (1979)"].values,     "envelope"),
    ("fluorostats $\\cdot$ Mean",            pi["Mean"].values,            "envelope"),
    ("fluorostats $\\cdot$ Triangle (1977)", pi["Triangle (1977)"].values, "envelope"),
    ("fluorostats $\\cdot$ Minimum",         pi["Minimum"].values,         "envelope"),
    ("fluorostats $\\cdot$ Yen (1995)",      pi["Yen (1995)"].values,      "envelope"),
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
    axa.text(d["mean"] + 0.012, row, f"{d['mean']:.3f}", va="center", ha="left",
             fontsize=5.4, color="#333", alpha=a)
labcols = [BLUE if r["kind"] != "dl" else r["c"] for r in rows]
axa.set_yticks(range(len(rows)))
axa.set_yticklabels([r["lab"] for r in rows], fontsize=5.6)
for tick, lc, r in zip(axa.get_yticklabels(), labcols, rows):
    tick.set_color(lc)
    tick.set_fontweight("bold" if r["kind"] == "default" else "normal")
axa.set_xlim(lo_x, hi_x); axa.set_ylim(-0.6, len(rows) - 0.4)
axa.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
axa.set_xlabel("instance F1 @ IoU 0.5   (BBBC039, n = 100)")
# legend explaining the family (in axes white space, lower right)
axa.plot([], [], "o", color=BLUE, mec="white", mew=0.6, ms=7, label="fluorostats default")
axa.plot([], [], "o", mfc="white", mec=BLUE, mew=1.1, ms=6, label="threshold envelope")
axa.plot([], [], "o", color=OKABE["orange"], mec="white", mew=0.6, ms=6, label="trained deep learning")
axa.legend(loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=5.0,
           handletextpad=0.3, borderpad=0.3, labelspacing=0.3, frameon=False)
panel(axa, "a", "one tool, one default knob (n=100)")

# (b) forest vs DL
axb = fig.add_subplot(gs[0, 4:])
ci = pd.read_csv(R/"b_dl_ci.csv")
fb = {"fluorostats": boot_ci(methods["fluorostats (Otsu+CC)"]),
      "StarDist": boot_ci(dl_pi["StarDist"]), "Cellpose": boot_ci(dl_pi["Cellpose"]),
      "Omnipose": boot_ci(dl_pi["Omnipose"])}
fo = sorted(fb, key=lambda m: fb[m][0])
for i, m in enumerate(fo):
    mean, lo, hi = fb[m]; c = OKABE["blue"] if m=="fluorostats" else TOOL[m]["c"]
    axb.errorbar(mean, i, xerr=[[mean-lo],[hi-mean]], fmt=TOOL.get(m,{}).get("m","o"),
                 color=c, ecolor="black", capsize=2.5, ms=6, mec="black", mew=0.5, elinewidth=0.9)
axb.set_yticks(range(len(fo))); axb.set_yticklabels(fo, fontsize=6.5)
axb.set_xlabel("mean F1  (95% CI)"); axb.set_xlim(0.78, 0.955)
axb.text(0.97, 0.06, "paired Δ excludes 0:\n−StarDist +0.025\n−Cellpose +0.034",
         transform=axb.transAxes, fontsize=5.4, color="#333", va="bottom", ha="right")
panel(axb, "b", "vs trained deep learning")

# (c) qualitative overlay: raw / GT / fluorostats, 3 crops
def gt_inst(stem):
    m = np.array(Image.open(DL/"BBBC039/masks/masks"/f"{stem}.png")); r = m[...,0] if m.ndim==3 else m
    seeds,_ = ndi.label(r==1); return watershed((r>=2).astype(np.uint8), seeds, mask=(r>0))
imgs = sorted(glob.glob(str(DL/"BBBC039/images/images/*.tif")))
crops = []
hh, ww = 178, 235   # crops sized to fill the image band (aspect ~1.3)
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
gs_c = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, :], hspace=0.06, wspace=0.06)
for row, (im, gt, stem) in enumerate(crops):
    fs = label_3d(remove_small_objects(im > filters.threshold_otsu(im), 20), min_size=20)[0]
    p = F.imnorm(im)
    for col, (lab, arr) in enumerate(zip(labs, [None, gt, fs])):
        ax = fig.add_subplot(gs_c[row, col]); ax.imshow(p, cmap="gray"); F.image_axes(ax)
        if arr is not None:
            outline(ax, arr, (0.0,0.62,0.45) if col==1 else (0.0,0.45,0.70), width=2)
        if row == 0: ax.set_title(lab, fontsize=7.5)
        if col == 0 and row == 0:
            F.scalebar(ax, 65, "20 µm")  # BBBC039 ~0.31 µm/px (approx, stated in caption)
            panel(ax, "c")
# (d) crossover — two directly-labelled line groups: fluorostats (blue) and the
# classical-threshold envelope (grey), both collapsing under overlap while the DL
# reference (dashed) holds ~0.96.
axd = fig.add_subplot(gs[2, :2])
cc = pd.read_csv(R/"b_clustering_curve.csv"); x = [0,25,50,75]
for _, r in cc.iterrows():
    isf = "fluorostats" in r["method"]
    if not isf and r[["c00","c25","c50","c75"]].sum() == 0:   # skip empty Triangle row
        continue
    axd.plot(x, [r.c00,r.c25,r.c50,r.c75], marker="o" if isf else ".", ms=4 if isf else 3,
             color=OKABE["blue"] if isf else GREY, lw=2.0 if isf else 0.8, alpha=1 if isf else 0.6,
             zorder=3 if isf else 1)
axd.axhline(0.96, ls="--", color=OKABE["black"], lw=1.0)
axd.text(2, 0.905, "deep learning holds $\\approx$0.96", fontsize=5.4, color="black")
axd.axvspan(50, 75, color=OKABE["vermillion"], alpha=0.08)
axd.text(62, 0.55, "DL wins", fontsize=6, color=OKABE["vermillion"], ha="center")
# direct labels on the two line groups (in white space, coloured to match)
axd.text(30, 0.60, "fluorostats", fontsize=6.2, color=OKABE["blue"], fontweight="bold",
         ha="left", va="bottom")
axd.text(30, 0.35, "classical envelope", fontsize=5.8, color=GREY, ha="left", va="top")
axd.set_xlabel("nuclear overlap (%)"); axd.set_ylabel("instance F1"); axd.set_ylim(0,1.02)
panel(axd, "d", "crowding crossover")

# (e) separated vs crowded fields
def mid(fp):
    v = np.asarray(Image.open(fp)) if fp.endswith(".png") else None
    import tifffile; vol = tifffile.imread(fp); return vol[vol.shape[0]//2].astype(np.float32)
gs_e = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 2:4], wspace=0.06)
for i, (fp, ttl) in enumerate([(str(DL/"BBBC024/image-final_0000.tif"), "separated"),
                               (sorted(glob.glob(str(DL/"BBBC024_c75/image-final_*.tif")))[0], "crowded")]):
    import tifffile; vol = tifffile.imread(fp).astype(np.float32); sl = vol[vol.shape[0]//2]
    ax = fig.add_subplot(gs_e[i]); ax.imshow(F.imnorm(sl), cmap="gray"); F.image_axes(ax)
    fsm = label_3d(remove_small_objects(sl > filters.threshold_otsu(sl), 15), min_size=15)[0]
    outline(ax, fsm, (0.0,0.45,0.70), width=2); ax.set_title(ttl, fontsize=7)
    if i == 0: panel(ax, "e")

# (f) scope decision map
axf = fig.add_subplot(gs[2, 4:])
axf.axvspan(0, 30, color=OKABE["green"], alpha=0.14); axf.axvspan(30, 50, color=OKABE["yellow"], alpha=0.16)
axf.axvspan(50, 100, color=OKABE["vermillion"], alpha=0.14)
axf.axvline(40, ls="--", color="black", lw=1.0); axf.text(41, 0.5, "measured\ncrossover ~c40", fontsize=5.4)
axf.text(15, 0.82, "fluorostats\n= DL", ha="center", fontsize=6.2, color=OKABE["green"])
axf.text(75, 0.82, "use trained\nsegmenter", ha="center", fontsize=6.2, color=OKABE["vermillion"])
axf.set_xlim(0,100); axf.set_ylim(0,1); axf.set_yticks([])
axf.set_xlabel("instance overlap / crowding (%)")
panel(axf, "f", "when to use which")

cap = ("Figure 2 | Nucleus segmentation and the deep-learning boundary. "
 "(a) Instance F1 (IoU 0.5) on BBBC039 (n=100 images), dots with 95% bootstrap CIs "
 "(10,000 resamples of per-image scores). The eight classical thresholds (Otsu, Li, "
 "Isodata, Triangle, Yen, Mean, Minimum, watershed) are fluorostats CONFIGURATIONS, not "
 "rival tools: the designated default (fluorostats Otsu+CC, solid blue, F1 0.890) is the "
 "headline; the remaining thresholds (open blue circles) trace the tuning envelope "
 "(0.28-0.91, i.e. the range of one knob). The genuine external comparison is trained deep "
 "learning at n=100 values (StarDist 0.861 orange, Cellpose 0.858 vermillion, Omnipose 0.798 "
 "purple). (b) Forest plot vs validated DL (StarDist, Cellpose, "
 "Omnipose); paired differences exclude zero. (c) Representative BBBC039 crops: raw, expert ground "
 "truth (green outlines), fluorostats (blue outlines); scale bar 20 µm (BBBC039 ~0.31 µm/px). "
 "(d) Instance F1 vs nuclear overlap (BBBC024 c00-c75): all non-DL methods (grey; fluorostats blue) "
 "collapse while DL (dashed) holds ~0.96; red band = DL advantage. (e) fluorostats overlay on a "
 "well-separated (c00) and a crowded (c75) field, showing connected-component merging under overlap. "
 "(f) Data-anchored scope map: fluorostats is at parity with trained segmenters up to ~40% crowding "
 "(green), beyond which a trained instance segmenter is preferred (red).")
save(fig, "fig2_nuclei_boundary", tight=False); caption("fig2_nuclei_boundary", cap)
print("Figure 2 done")
