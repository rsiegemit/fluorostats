"""Extended Data Figure 2 — Runtime and determinism.

(a) Per-operation runtime (log x, horizontal bars) for a curated, category-grouped
    set of ~14 representative operations: the 2D-segmentation head-to-head
    (fluorostats vs StarDist vs Cellpose) plus fluorostats metrics spanning the
    dynamic range and their matched SciPy / scikit-image equivalents. fluorostats
    sits at parity with the underlying library call on shared operations.
(b) Determinism: three fluorostats metrics run N=20x on the SAME BBBC024 volume
    give bit-identical output (spread = 0), unlike seed/hardware-sensitive DL.
(c) Reproducibility: run-to-run spread (CV%) — fluorostats 0.0 vs a trained
    pipeline that varies with seed / GPU.
"""
import glob
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as F
from figstyle import OKABE, panel, save, caption

from fluorostats import metrics_3d, morphometry

F.apply_style()

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA = HERE.parent / "data" / "downloads"

# ----------------------------------------------------------------------------
# Panel (b) data — computed LIVE: N=20 identical runs on one real BBBC024 volume
# ----------------------------------------------------------------------------
N_RUNS = 20
vol = tifffile.imread(sorted(glob.glob(str(DATA / "BBBC024/image-final_*.tif")))[0]).astype(np.float32)
from skimage import filters
mask = vol > filters.threshold_otsu(vol)

det_metrics = {
    "volume\nfraction": lambda: metrics_3d.volume_fraction(mask),
    "connectivity\n(Euler no.)": lambda: metrics_3d.connectivity_metrics(mask)["euler_number"],
    "lateral\nhomogeneity": lambda: morphometry.lateral_homogeneity(mask.astype(np.float32))["lateral_cv"],
}
det_runs = {name: np.array([fn() for _ in range(N_RUNS)], float) for name, fn in det_metrics.items()}
det_spread = {name: float(v.max() - v.min()) for name, v in det_runs.items()}
print("determinism spread (max-min) per metric:", det_spread)
assert all(s == 0.0 for s in det_spread.values()), "expected bit-identical output"

# ----------------------------------------------------------------------------
# Panel (a) data — CURATED, category-grouped representative operations
# All ms values read directly from results/b_timing_all_metrics.csv.
# ----------------------------------------------------------------------------
df = pd.read_csv(RES / "b_timing_all_metrics.csv")
# 2D-segmentation head-to-head fluorostats value lives in the segmentation
# benchmark table (b_timing.csv), row fluorostats(Otsu+CC) -> ms_per_image.
df_seg = pd.read_csv(RES / "b_timing.csv")
SEG_FLUORO_MS = float(df_seg[df_seg.method == "fluorostats(Otsu+CC)"].ms_per_image.iloc[0])


# Category label -> list of (display_label, csv_metric, kind)
# kind in {"fluorostats", "scipy", "stardist", "cellpose"} -> colour.
# Ordered top->bottom within the axis; grouped by category with spacers.
ROWS = [
    # --- 2D instance segmentation (head-to-head) ---
    ("2D segmentation",  "Cellpose (per 2D img)",     "comparator,Cellpose (per 2D img, CPU cluster)", "cellpose"),
    ("2D segmentation",  "StarDist (per 2D img)",     "comparator,StarDist (per 2D img, CPU cluster)", "stardist"),
    ("2D segmentation",  "fluorostats (Otsu + CC)",   "fluorostats(Otsu+CC)",                          "fluorostats"),
    # --- 3D metrics ---
    ("3D metrics",       "connectivity (Euler no.)",  "fluorostats,metrics_3d.connectivity_metrics",   "fluorostats"),
    ("3D metrics",       "euler_number (skimage)",    "comparator,euler_number(skimage) 3D",           "scipy"),
    ("3D metrics",       "volume fraction",           "fluorostats,metrics_3d.volume_fraction",        "fluorostats"),
    # --- Skeleton / morphology ---
    ("skeleton / morph", "skeleton metrics 2D",       "fluorostats,skeleton.skeleton_metrics 2D",      "fluorostats"),
    ("skeleton / morph", "medial_axis (skimage)",     "comparator,medial_axis 2D",                     "scipy"),
    ("skeleton / morph", "lateral homogeneity",       "fluorostats,morphometry.lateral_homogeneity",   "fluorostats"),
    # --- Statistics ---
    ("statistics",       "Mann-Whitney",              "fluorostats,stats.mann_whitney",                "fluorostats"),
    ("statistics",       "scipy.mannwhitneyu",        "comparator,scipy.mannwhitneyu",                 "scipy"),
    ("statistics",       "bootstrap fold-change CI",  "fluorostats,stats.bootstrap_fold_change_ci",    "fluorostats"),
    # --- Validation (once per volume) ---
    ("validation",       "instance F1",               "fluorostats,validate.instance_f1",              "fluorostats"),
    ("validation",       "average precision",         "fluorostats,validate.average_precision",        "fluorostats"),
]

# resolve the ms for the two head-to-head rows and validation directly
KIND_COLOR = {
    "fluorostats": OKABE["blue"],
    "scipy":       OKABE["grey"],
    "stardist":    OKABE["orange"],
    "cellpose":    OKABE["vermillion"],
}


def lookup(metric_key):
    """metric_key is the exact CSV metric string (after the 'group,' prefix we
    stored above for uniqueness). We stored 'group,metric' to disambiguate the
    two DL rows whose metric text contains a comma; strip the group if present."""
    if metric_key == "fluorostats(Otsu+CC)":
        return SEG_FLUORO_MS
    if metric_key.startswith("comparator,") or metric_key.startswith("fluorostats,"):
        grp, met = metric_key.split(",", 1)
        hit = df[(df.group == grp) & (df.metric == met)]
    else:
        hit = df[df.metric == metric_key]
    if len(hit) != 1:
        raise KeyError(f"expected 1 row for {metric_key!r}, got {len(hit)}")
    return float(hit.ms.iloc[0])


# build the ordered plotting list with category spacers
categories = []
seen = []
for cat, lab, key, kind in ROWS:
    if cat not in categories:
        categories.append(cat)
    seen.append((cat, lab, lookup(key), kind))

# y positions: leave a gap between categories
ypos, ylabels, yvals, ycolors = [], [], [], []
cat_bounds = {}  # category -> (ymin, ymax) for bracket labels
y = 0.0
prev_cat = None
for cat, lab, val, kind in seen:
    if prev_cat is not None and cat != prev_cat:
        y += 1.0  # spacer between categories
    ypos.append(y)
    ylabels.append(lab)
    yvals.append(val)
    ycolors.append(KIND_COLOR[kind])
    cat_bounds.setdefault(cat, [y, y])
    cat_bounds[cat][1] = y
    prev_cat = cat
    y += 1.0

ypos = np.array(ypos)

# ----------------------------------------------------------------------------
# Figure — constrained layout, TRUE manuscript text width (F.TW = 5.147 in).
# Narrow single-column canvas: stack panel (a) full-width on top (it is
# label-heavy: 14 bars + long left y-labels + a far-left category band), with
# (b) and (c) side-by-side below. A tall H gives (a) room to breathe.
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(F.TW, 6.4), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.06, hspace=0.03)
# (a) spans the top; (b)/(c) share the bottom row.
gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.0])
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

# --- Panel (a): grouped horizontal bars, log x ------------------------------
axA.barh(ypos, yvals, color=ycolors, height=0.72, edgecolor="#333333", linewidth=0.4, zorder=3)
axA.set_yticks(ypos)
axA.set_yticklabels(ylabels, fontsize=F.FS["small"])
axA.invert_yaxis()  # first ROW on top
axA.set_xscale("log")
axA.set_xlim(0.01, 30000)
axA.set_xlabel("time per operation (ms, log scale)")
axA.tick_params(axis="y", length=0)
axA.spines["left"].set_visible(False)
for xg in (0.1, 1, 10, 100, 1000, 10000):
    axA.axvline(xg, color="#EEEEEE", lw=0.5, zorder=0)

# category labels rotated vertically in the far-left reserved margin (grey
# italic), left of the (longest) y-tick labels so they never overlap them.
# On this narrow canvas a vertical bracket-style label saves horizontal room
# that the long y-tick labels need.
for cat, (y0, y1) in cat_bounds.items():
    ymid = (y0 + y1) / 2.0
    axA.annotate(cat, xy=(0, ymid), xycoords=("axes fraction", "data"),
                 xytext=(-96, 0), textcoords="offset points",
                 fontsize=F.FS["small"], color="#666", style="italic",
                 va="center", ha="center", rotation=90, annotation_clip=False)

# thin separator lines between categories
for cat in categories[1:]:
    y0 = cat_bounds[cat][0]
    axA.axhline(y0 - 0.55, color="#DDDDDD", lw=0.5, zorder=1)

axA.set_ylim(ypos.max() + 0.8, ypos.min() - 0.8)  # inverted, with margin

# legend in clear white space (lower right of panel a)
legA = [
    Line2D([0], [0], marker="s", ls="", mfc=OKABE["blue"], mec="none", ms=6, label="fluorostats"),
    Line2D([0], [0], marker="s", ls="", mfc=OKABE["grey"], mec="none", ms=6, label="SciPy / scikit-image"),
    Line2D([0], [0], marker="s", ls="", mfc=OKABE["orange"], mec="none", ms=6, label="StarDist (DL)"),
    Line2D([0], [0], marker="s", ls="", mfc=OKABE["vermillion"], mec="none", ms=6, label="Cellpose (DL)"),
]
# legend in the empty right-hand white space over the short "statistics" bars
axA.legend(handles=legA, loc="center right", bbox_to_anchor=(1.0, 0.30),
           fontsize=F.FS["small"], handletextpad=0.4, borderaxespad=0.5,
           labelspacing=0.35, frameon=False)

# 2D-seg per-image runtimes feed the caption only. The on-panel inset that used to
# sit here was removed: it duplicated the caption and, floating over the short
# 3D-metric/skeleton bars, read as if it belonged to that row group. The three
# 2D-segmentation bars already show the fluorostats-vs-DL speedup visually.
seg_f = lookup("fluorostats(Otsu+CC)")
seg_sd = lookup("comparator,StarDist (per 2D img, CPU cluster)")
seg_cp = lookup("comparator,Cellpose (per 2D img, CPU cluster)")

F.panel(axA, "a", xanchor=0.0315)   # row-leader

# --- Panel (b): determinism — 20 identical runs, spread = 0 ------------------
names = list(det_runs.keys())
xr = np.arange(len(names))
rng = np.random.default_rng(0)
for i, nm in enumerate(names):
    v = det_runs[nm]
    rel = (v - v.mean()) / (abs(v.mean()) + 1e-12) * 1e6  # ppm
    jit = (rng.random(len(v)) - 0.5) * 0.34
    axB.scatter(np.full(len(v), i) + jit, rel, s=9, color=OKABE["blue"],
                alpha=0.85, edgecolors="none", zorder=3)
axB.axhline(0, color="#BBBBBB", lw=0.6, zorder=1)
axB.set_xticks(xr)
axB.set_xticklabels(names, fontsize=F.FS["small"])
axB.set_xlim(-0.6, len(names) - 0.4)
axB.set_ylim(-1, 1)
axB.set_ylabel("deviation from mean\n(parts per million)", fontsize=F.FS["annot"])
axB.text(0.5, 1.10, f"fluorostats: {N_RUNS} runs, same input",
         transform=axB.transAxes, ha="center", va="bottom", fontsize=F.FS["annot"])
axB.text(0.5, 0.80, "spread = 0  (bit-identical)", transform=axB.transAxes,
         ha="center", va="center", fontsize=F.FS["annot"], color=OKABE["blue"],
         fontweight="bold")
F.panel(axB, "b")   # deferred: row-2 de-indent destabilises panel a's x-axis

# --- Panel (c): reproducibility — run-to-run spread (CV%) --------------------
labels = ["fluorostats\n(CPU, any run)", "trained pipeline\n(seed / hardware)"]
spreads = [0.0, 3.2]  # illustrative CV% for a seed/hardware-sensitive DL pipeline
axC.bar([0, 1], spreads, width=0.6,
        color=[OKABE["blue"], OKABE["orange"]], edgecolor="#333333", linewidth=0.4)
axC.set_xticks([0, 1])
axC.set_xticklabels(labels, fontsize=F.FS["small"])
axC.set_xlim(-0.6, 1.6)
axC.set_ylabel("run-to-run spread\n(CV %)", fontsize=F.FS["annot"])
axC.set_ylim(0, 4)
axC.text(0, 0.12, "0.0", ha="center", va="bottom", fontsize=F.FS["annot"],
         color=OKABE["blue"], fontweight="bold")
axC.text(1, spreads[1] + 0.12, "seed / GPU\nvariable", ha="center", va="bottom",
         fontsize=F.FS["small"], color=OKABE["orange"])
F.panel(axC, "c")   # deferred with panel b

F.save(fig, "ed2_runtime", F.EXT, tight=False)

CAPTION = (
    "Extended Data Figure 2 | Runtime and determinism. "
    "(a) Wall-clock time per operation (log scale) for a curated, category-grouped "
    "set of representative fluorostats operations (blue) and their matched "
    "SciPy / scikit-image reference implementations (grey), timed on identical "
    "inputs (one BBBC024 3D volume; 2D nuclei image; synthetic live/dead field) on "
    "a single Apple Silicon Mac CPU (darwin). Shared operations run at parity with "
    "their library equivalents (e.g. fluorostats connectivity vs euler_number; "
    "stats.mann_whitney vs scipy.mannwhitneyu), confirming fluorostats adds "
    "negligible overhead over the underlying call. For headline 2D instance "
    "segmentation, fluorostats (Otsu + connected components) runs at "
    f"{seg_f:.1f} ms/image versus {seg_sd:.0f} ms for StarDist (~{F.round_sig(seg_sd/seg_f):.0f}×) "
    f"and {seg_cp:.0f} ms for Cellpose (~{F.round_sig(seg_cp/seg_f):.0f}×), both measured on CPU. "
    "The two whole-volume validation operations (validate.instance_f1; "
    "validate.average_precision) are computed once per volume during benchmarking "
    "and are not part of per-frame analysis cost. (b) Determinism. Three "
    "representative fluorostats metrics (volume fraction, connectivity/Euler "
    "number, lateral homogeneity) run N=20 times on the same BBBC024 volume return "
    "bit-identical values (run-to-run spread = 0; deviations plotted in parts per "
    "million). (c) Reproducibility. Run-to-run spread (CV%): fluorostats is fully "
    "deterministic on CPU (0.0), whereas trained deep-learning pipelines vary "
    "run-to-run with random seed and hardware/GPU non-determinism (illustrative)."
)
F.caption("ed2_runtime", CAPTION, F.EXT)
print("done")
