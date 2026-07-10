"""Extended Data Figure 3 — Robustness of fluorostats to noise and denoising.

(a) Foreground Dice vs added Gaussian-noise sd for classic thresholders + fluorostats.
(b) Dice recovery under denoising strategies across noise levels.
(c) Per-nucleus median-diameter recovery error (%) vs BBBC024 ground truth.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle
from figstyle import OKABE, apply_style, save, panel, caption, EXT

apply_style()

noise = pd.read_csv("results/b_noise_robustness.csv")
denoise = pd.read_csv("results/b_denoising.csv")
size = pd.read_csv("results/b_nuclei_size.csv")

FLU = OKABE["blue"]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax_a, ax_b, ax_c = axes

# ---- (a) Noise robustness ----------------------------------------------------
flu_col = "fluorostats(Otsu+CC)"
# other methods: grey/coloured with distinct markers; fluorostats blue bold on top
others = {
    "Otsu":     dict(c=OKABE["grey"],       m="s", ls="--"),
    "Li":       dict(c=OKABE["orange"],     m="^", ls="-."),
    "Isodata":  dict(c=OKABE["green"],      m="D", ls=":"),
    "Triangle": dict(c=OKABE["vermillion"], m="v", ls="--"),
}
for name, st in others.items():
    if name == "Otsu":
        continue  # identical to fluorostats; shown as white dashes over the blue line
    ax_a.plot(noise["noise_sd"], noise[name], color=st["c"], marker=st["m"],
              ls=st["ls"], lw=1.0, ms=3.2, mew=0.4, mec="white", label=name)
# fluorostats (Otsu + CC): bold blue; white dashes on top mark the identical Otsu curve
ax_a.plot(noise["noise_sd"], noise[flu_col], color=FLU, marker="o", ls="-",
          lw=2.2, ms=4.2, mew=0.5, mec="white", label="fluorostats", zorder=6)
ax_a.plot(noise["noise_sd"], noise["Otsu"], color="white", ls=(0, (2, 2.2)),
          lw=1.3, zorder=7)
ax_a.set_xlabel("Added Gaussian noise (s.d., grey levels)")
ax_a.set_ylabel("Foreground Dice")
ax_a.set_xticks(noise["noise_sd"])
ax_a.set_ylim(0.5, 0.96)
# order legend so fluorostats is first; note it coincides with Otsu
h, l = ax_a.get_legend_handles_labels()
order = [l.index("fluorostats")] + [i for i, x in enumerate(l) if x != "fluorostats"]
ax_a.legend([h[i] for i in order], [l[i] for i in order],
            loc="lower left", handlelength=1.8, borderpad=0.2, labelspacing=0.25,
            title="(white dashes = Otsu)", title_fontsize=5.5)
panel(ax_a, "a")

# ---- (b) Denoising recovery --------------------------------------------------
# grouped bars per noise level; fluorostats denoiser highlighted
strat = ["none", "median_3", "gaussian_σ2", "TV_chambolle", "fluorostats(gaussian σ1)"]
labels = ["none", "median", r"gaussian $\sigma$2", "TV", "fluorostats"]
bar_c = [OKABE["lgrey"], OKABE["grey"], OKABE["green"], OKABE["orange"], FLU]
levels = denoise["noise_sd"].to_list()
x = np.arange(len(levels))
w = 0.16
for i, (s, c) in enumerate(zip(strat, bar_c)):
    off = (i - (len(strat) - 1) / 2) * w
    ax_b.bar(x + off, denoise[s], width=w, color=c,
             edgecolor="white", linewidth=0.3,
             label=labels[i], zorder=3 if s.startswith("fluoro") else 2)
ax_b.set_xticks(x)
ax_b.set_xticklabels([f"{v}" for v in levels])
ax_b.set_xlabel("Added Gaussian noise (s.d.)")
ax_b.set_ylabel("Foreground Dice")
ax_b.set_ylim(0.35, 0.96)
# annotate the recovery at the highest noise level
i_hi = len(levels) - 1
ax_b.annotate("", xy=(i_hi, 0.905), xytext=(i_hi, 0.42),
              arrowprops=dict(arrowstyle="->", color=OKABE["black"], lw=0.7))
# horizontal label near the arrow head, nudged left so it clears the bars
ax_b.text(i_hi - 0.14, 0.905, r"0.41 $\rightarrow$ 0.91", rotation=0, va="center",
          ha="right", fontsize=6, color=OKABE["black"])
ax_b.legend(loc="lower left", ncol=2, columnspacing=0.9, handlelength=1.0,
            borderpad=0.2, labelspacing=0.25)
panel(ax_b, "b")

# ---- (c) Per-nucleus size recovery ------------------------------------------
sz = size.sort_values("pct_error").reset_index(drop=True)
# collapse identical Otsu/Isodata/fluorostats tie into readable short labels
name_map = {
    "Otsu_1979": "Otsu", "Isodata_1978": "Isodata",
    "fluorostats(Otsu+CC)": "fluorostats", "Li_1993": "Li",
    "Triangle_1977": "Triangle",
}
sz["short"] = sz["method"].map(name_map)
cols = [FLU if s == "fluorostats" else OKABE["lgrey"] for s in sz["short"]]
ypos = np.arange(len(sz))[::-1]  # best at top
bars = ax_c.barh(ypos, sz["pct_error"], color=cols, edgecolor="white",
                 linewidth=0.3, height=0.68, zorder=3)
ax_c.set_yticks(ypos)
ax_c.set_yticklabels(sz["short"])
ax_c.set_xlabel("Median-diameter error (%)")
ax_c.set_xlim(0, 45)
for y, v in zip(ypos, sz["pct_error"]):
    ax_c.text(v + 0.8, y, f"{v:.1f}", va="center", ha="left", fontsize=6)
panel(ax_c, "c")

save(fig, "ed3_robustness", EXT)

cap = (
    "Extended Data Figure 3 | Robustness of fluorostats segmentation to noise. "
    "(a) Foreground Dice versus added zero-mean Gaussian noise (standard deviation "
    "in grey levels; 8-bit images) for classic global thresholders and fluorostats "
    "(Otsu + connected-component post-processing, blue). fluorostats matches the best "
    "classic method (Otsu/Isodata) at every noise level and degrades gracefully, "
    "retaining Dice 0.74 at s.d.=160 where naive Triangle and Li fail. "
    "(b) Foreground Dice under denoising strategies (none, 3x3 median, Gaussian sigma=2, "
    "total-variation Chambolle, and the fluorostats built-in Gaussian sigma=1 pre-filter) "
    "across increasing noise. Without denoising, Dice collapses to 0.41 at s.d.=160; the "
    "fluorostats pre-filter recovers it to 0.91, matching dedicated denoisers. "
    "(c) Per-nucleus median-diameter recovery error relative to BBBC024 ground truth "
    "(74.5 um), sorted best-to-worst. fluorostats is best-tied at 3.5% error, versus "
    "15.2% (Li) and 39.8% (Triangle). Blue denotes fluorostats throughout."
)
caption("ed3_robustness", cap, EXT)
print("done")
