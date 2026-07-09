"""Extended Data Figure 2 — Runtime and determinism.

(a) Per-operation runtime (log x, horizontal bars): fluorostats metrics vs the
    matched SciPy/scikit-image/DL comparators, timed on the same data per op-class.
    Shared-operation parity is visible where a fluorostats bar sits beside its
    library equivalent; the two whole-volume validation ops are set apart as
    "once per volume (validation)" so they don't read as per-frame cost.
(b) Determinism: three fluorostats metrics run N=20x on the SAME BBBC024 volume
    give bit-identical output (spread = 0), unlike seed/hardware-sensitive DL.
"""
import glob
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle
from figstyle import OKABE
from fluorostats import metrics_3d, morphometry

figstyle.apply_style()

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA = HERE.parent / "data" / "downloads"

# ----------------------------------------------------------------------------
# Panel (b) data — computed LIVE: N=20 identical runs on one real BBBC024 volume
# ----------------------------------------------------------------------------
N_RUNS = 20
vol = tifffile.imread(sorted(glob.glob(str(DATA / "BBBC024/image-final_*.tif")))[0]).astype(np.float32)
# fixed binary mask (deterministic threshold) so every run sees identical input
from skimage import filters
mask = vol > filters.threshold_otsu(vol)

det_metrics = {
    "volume_fraction": lambda: metrics_3d.volume_fraction(mask),
    "connectivity\n(Euler no.)": lambda: metrics_3d.connectivity_metrics(mask)["euler_number"],
    "lateral\nhomogeneity": lambda: morphometry.lateral_homogeneity(mask.astype(np.float32))["lateral_cv"],
}
det_runs = {name: np.array([fn() for _ in range(N_RUNS)], float) for name, fn in det_metrics.items()}
det_spread = {name: float(v.max() - v.min()) for name, v in det_runs.items()}
print("determinism spread (max-min) per metric:", det_spread)
assert all(s == 0.0 for s in det_spread.values()), "expected bit-identical output"

# ----------------------------------------------------------------------------
# Panel (a) data — runtime table
# ----------------------------------------------------------------------------
df = pd.read_csv(RES / "b_timing_all_metrics.csv")

# the two whole-volume validation ops are amortised once per volume, not per frame
VALIDATION = {"validate.instance_f1", "validate.average_precision"}
val = df[df.metric.isin(VALIDATION)].copy()
main = df[~df.metric.isin(VALIDATION)].copy().sort_values("ms").reset_index(drop=True)


def bar_color(row):
    if row.group == "fluorostats":
        return OKABE["blue"]
    m = row.metric.lower()
    if "stardist" in m:
        return OKABE["orange"]
    if "cellpose" in m:
        return OKABE["vermillion"]
    return OKABE["lgrey"]


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.55)
axA = fig.add_subplot(gs[0, 0])
gsB = gs[0, 1].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.5)
axB = fig.add_subplot(gsB[0, 0])
axH = fig.add_subplot(gsB[1, 0])

# --- Panel (a): horizontal bars, log x -------------------------------------
y = np.arange(len(main))
colors = [bar_color(r) for _, r in main.iterrows()]
axA.barh(y, main.ms.values, color=colors, height=0.72, edgecolor="none")
axA.set_yticks(y)
axA.set_yticklabels(main.metric.values, fontsize=4.6)
axA.set_xscale("log")
axA.set_xlim(0.005, 30000)
axA.set_xlabel("time per operation (ms, log scale)")
axA.set_ylim(-0.8, len(main) - 0.2)
axA.tick_params(axis="y", length=0)
axA.spines["left"].set_visible(False)
for x in (0.01, 0.1, 1, 10, 100, 1000, 10000):
    axA.axvline(x, color="#EEEEEE", lw=0.5, zorder=0)

# validation ops drawn as an offset callout band at the top (once per volume)
val_sorted = val.sort_values("ms")
yv = np.arange(len(main) + 1.0, len(main) + 1.0 + len(val_sorted))
axA.barh(yv, val_sorted.ms.values, color=OKABE["purple"], height=0.72,
         edgecolor="none", alpha=0.9)
for yy, (_, r) in zip(yv, val_sorted.iterrows()):
    axA.text(r.ms * 1.3, yy, f"{r.metric}", va="center", ha="left", fontsize=4.6)
axA.axhline(len(main) + 0.1, color="#CCCCCC", lw=0.5, ls=(0, (2, 2)))
axA.text(0.006, len(main) + len(val_sorted) + 0.35,
         "once per volume (validation) — amortised, not per frame",
         fontsize=5.2, color=OKABE["purple"], fontweight="bold", va="center")
axA.set_ylim(-0.8, len(main) + len(val_sorted) + 0.9)

# headline annotation: 2D-seg speedups (upper-left, over the empty short-bar region)
axA.text(1.4, len(main) * 0.80,
         "2D segmentation, per image:\n"
         "fluorostats  14.5 ms\n"
         "StarDist  215 ms  (15x)\n"
         "Cellpose  5547 ms  (380x)",
         fontsize=5.4, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", lw=0.5))

legA = [
    Line2D([0], [0], marker="s", ls="", mfc=OKABE["blue"], mec="none", ms=6, label="fluorostats"),
    Line2D([0], [0], marker="s", ls="", mfc=OKABE["lgrey"], mec="none", ms=6, label="SciPy / scikit-image"),
    Line2D([0], [0], marker="s", ls="", mfc=OKABE["orange"], mec="none", ms=6, label="StarDist (DL)"),
    Line2D([0], [0], marker="s", ls="", mfc=OKABE["vermillion"], mec="none", ms=6, label="Cellpose (DL)"),
    Line2D([0], [0], marker="s", ls="", mfc=OKABE["purple"], mec="none", ms=6, label="validation op"),
]
axA.legend(handles=legA, loc="lower right", fontsize=5.2, handletextpad=0.4,
           borderaxespad=0.3, labelspacing=0.35)
figstyle.panel(axA, "a", dx=0.0, dy=1.005)

# --- Panel (b): determinism — 20 identical runs, spread = 0 -----------------
names = list(det_runs.keys())
xr = np.arange(len(names))
rng = np.random.default_rng(0)
for i, nm in enumerate(names):
    v = det_runs[nm]
    # normalise to relative deviation from the mean so all three share one y-axis
    rel = (v - v.mean()) / (abs(v.mean()) + 1e-12) * 1e6  # ppm
    jit = (rng.random(len(v)) - 0.5) * 0.34
    axB.scatter(np.full(len(v), i) + jit, rel, s=9, color=OKABE["blue"],
                alpha=0.85, edgecolors="none", zorder=3)
axB.axhline(0, color="#BBBBBB", lw=0.6, zorder=1)
axB.set_xticks(xr)
axB.set_xticklabels(names, fontsize=5.6)
axB.set_ylim(-1, 1)
axB.set_ylabel("deviation from mean\n(parts per million)", fontsize=6)
axB.text(0.5, 1.02, f"fluorostats: {N_RUNS} runs, same input", transform=axB.transAxes, ha="center", va="bottom", fontsize=6)
axB.text(0.5, 0.82, "spread = 0  (bit-identical)", transform=axB.transAxes,
         ha="center", va="center", fontsize=6.5, color=OKABE["blue"],
         fontweight="bold")
figstyle.panel(axB, "b", dx=-0.28, dy=1.02)

# --- Panel (b, lower): contrast bar — reproducibility spread ----------------
labels = ["fluorostats\n(CPU, any run)", "trained pipeline\n(seed / hardware)"]
spreads = [0.0, 3.2]  # illustrative CV% for a seed/hardware-sensitive DL pipeline
bars = axH.bar([0, 1], spreads, width=0.6,
               color=[OKABE["blue"], OKABE["orange"]], edgecolor="none")
axH.set_xticks([0, 1])
axH.set_xticklabels(labels, fontsize=5.6)
axH.set_ylabel("run-to-run spread\n(CV %)", fontsize=6)
axH.set_ylim(0, 4)
axH.text(0, 0.12, "0.0", ha="center", va="bottom", fontsize=6,
         color=OKABE["blue"], fontweight="bold")
axH.text(1, spreads[1] + 0.1, "seed / GPU\nvariable", ha="center", va="bottom",
         fontsize=5.2, color=OKABE["orange"])
axH.text(0.5, 1.02, "reproducibility", transform=axH.transAxes, ha="center", va="bottom", fontsize=6)

figstyle.save(fig, "ed2_runtime", figstyle.EXT, tight=False)

CAPTION = (
    "Extended Data Figure 2 | Runtime and determinism. "
    "(a) Wall-clock time per operation (log scale) for fluorostats metrics (blue) "
    "and their matched reference implementations, timed on identical inputs "
    "(one BBBC024 3D volume; 2D nuclei image; synthetic live/dead field) on a "
    "single Apple Silicon Mac CPU (darwin). Shared operations run at parity with "
    "their SciPy / scikit-image equivalents (e.g. stats.mann_whitney vs "
    "scipy.mannwhitneyu; objects.count_local_maxima vs skimage.peak_local_max), "
    "confirming fluorostats adds negligible overhead over the underlying library "
    "call. For headline 2D instance segmentation, fluorostats (Otsu + connected "
    "components) runs at 14.5 ms/image versus 215 ms for StarDist (~15x) and "
    "5547 ms for Cellpose (~380x), both measured on CPU. The two whole-volume "
    "validation operations (validate.instance_f1 ~2.2 s; validate.average_precision "
    "~19.5 s, purple) are computed once per volume during benchmarking and are "
    "not part of per-frame analysis cost. (b) Determinism. Three representative "
    "fluorostats metrics (volume_fraction, connectivity/Euler number, lateral "
    "homogeneity) run N=20 times on the same BBBC024 volume return bit-identical "
    "values (run-to-run spread = 0; deviations plotted in parts per million). "
    "fluorostats is fully deterministic on CPU, whereas trained deep-learning "
    "pipelines vary run-to-run with random seed and hardware/GPU non-determinism "
    "(lower bar, illustrative)."
)
figstyle.caption("ed2_runtime", CAPTION, figstyle.EXT)
print("done")
