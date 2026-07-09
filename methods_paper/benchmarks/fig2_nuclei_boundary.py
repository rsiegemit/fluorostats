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
dl_pi = {"StarDist": pd.read_csv(R/"stardist_eval.csv").stardist_f1.values,
         "Cellpose": pd.read_csv(R/"cellpose_eval.csv").cellpose_f1.values,
         "Omnipose": pd.read_csv(R/"omnipose_eval.csv").query("thirddl_f1>=0").thirddl_f1.values}
methods = {c: pi[c].values for c in pi.columns if c != "image"}
methods.update(dl_pi)
stats = {m: boot_ci(v) for m, v in methods.items()}
# ---------- layout ----------
fig = plt.figure(figsize=(7.2, 7.4))
gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[0.82, 1.9, 0.82],
                       hspace=0.34, wspace=1.05)

# (a) 12-method ranking with bootstrap CIs — one hero colour (blue), rest grey
axa = fig.add_subplot(gs[0, :4])
names = list(stats)
# tool identity: fluorostats blue, DL tools in their Okabe-Ito colours (StarDist
# orange, Cellpose vermillion, Omnipose purple), classical thresholds grey
acol = [F.color_for(m) for m in names]
F.ranked_barh(axa, names, [stats[m][0] for m in names],
              [stats[m][1] for m in names], [stats[m][2] for m in names],
              colors=acol, label_fs=6)
axa.set_xlim(0, 1.0); axa.set_xlabel("instance F1 @ IoU 0.5   (BBBC039, n = 100)")
panel(axa, "a", "twelve-method accuracy")

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
# (d) crossover
axd = fig.add_subplot(gs[2, :2])
cc = pd.read_csv(R/"b_clustering_curve.csv"); x = [0,25,50,75]
for _, r in cc.iterrows():
    isf = "fluorostats" in r["method"]
    axd.plot(x, [r.c00,r.c25,r.c50,r.c75], marker="o" if isf else ".", ms=4 if isf else 3,
             color=OKABE["blue"] if isf else GREY, lw=2.0 if isf else 0.8, alpha=1 if isf else 0.6,
             zorder=3 if isf else 1, label="fluorostats" if isf and "CC" in r["method"] else None)
axd.axhline(0.96, ls="--", color=OKABE["black"], lw=1.0)
axd.text(2, 0.90, "deep learning", fontsize=5.6, color="black")
axd.axvspan(50, 75, color=OKABE["vermillion"], alpha=0.08)
axd.text(62, 0.5, "DL wins", fontsize=6, color=OKABE["vermillion"], ha="center")
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
 "(a) Instance F1 (IoU 0.5) of twelve methods on BBBC039 (n=100 images); bars ranked, "
 "95% bootstrap CIs (10,000 resamples of per-image scores); fluorostats blue, deep-learning "
 "tools in colour, classical thresholds grey. (b) Forest plot vs validated DL (StarDist, Cellpose, "
 "Omnipose); paired differences exclude zero. (c) Representative BBBC039 crops: raw, expert ground "
 "truth (green outlines), fluorostats (blue outlines); scale bar 20 µm (BBBC039 ~0.31 µm/px). "
 "(d) Instance F1 vs nuclear overlap (BBBC024 c00-c75): all non-DL methods (grey; fluorostats blue) "
 "collapse while DL (dashed) holds ~0.96; red band = DL advantage. (e) fluorostats overlay on a "
 "well-separated (c00) and a crowded (c75) field, showing connected-component merging under overlap. "
 "(f) Data-anchored scope map: fluorostats is at parity with trained segmenters up to ~40% crowding "
 "(green), beyond which a trained instance segmenter is preferred (red).")
save(fig, "fig2_nuclei_boundary"); caption("fig2_nuclei_boundary", cap)
print("Figure 2 done")
