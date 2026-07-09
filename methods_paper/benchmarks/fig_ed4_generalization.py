"""Extended Data Figure 4 — Generalization across datasets.

Two panels showing that fluorostats, being training-free, holds up across
independent public datasets and imaging modalities it was never tuned on:

(a) DSB2018 fluorescence-nuclei subset (StarDist's own training dataset):
    instance F1 @ IoU 0.5 for fluorostats vs classical threshold baselines,
    with StarDist's published trained-DL AP@0.5 marked as a reference line.
    fluorostats ranks first among heuristics and reaches ~91% of the trained
    model's score on the trained model's home dataset.

(b) Cell Tracking Challenge 3D foreground Dice on two unseen modalities
    (Fluo-C3DH-A549, Fluo-N3DH-CHO), fluorostats vs threshold baselines.

Run:  python3.13 fig_ed4_generalization.py
Outputs PDF+PNG to figures/extended/ plus a .txt caption.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

STARDIST_AP50 = 0.864  # published StarDist trained-DL AP@0.5 on DSB2018 test

# pretty method labels for display
LABELS = {
    "fluorostats": "fluorostats",
    "fluorostats-li": "fluorostats (Li)",
    "otsu": "Otsu",
    "Otsu_1979": "Otsu",
    "isodata": "IsoData",
    "Isodata_1978": "IsoData",
    "li": "Li",
    "Li_1993": "Li",
    "yen": "Yen",
    "Yen_1995": "Yen",
    "triangle": "Triangle",
    "Triangle_1977": "Triangle",
}


def is_fluorostats(name: str) -> bool:
    return "fluorostats" in str(name).lower()


def panel_a(ax):
    """DSB2018 F1@0.5 ranking as horizontal bars, best at top."""
    df = pd.read_csv(os.path.join(RESULTS, "b_dsb2018.csv"))
    df = df.sort_values("mean_f1@0.5", ascending=True)  # ascending -> best on top

    f1 = df["mean_f1@0.5"].to_numpy()
    names = df["method"].tolist()
    labels = [LABELS.get(n, n) for n in names]
    colors = [figstyle.OKABE["blue"] if is_fluorostats(n) else figstyle.OKABE["lgrey"]
              for n in names]
    y = np.arange(len(names))

    ax.barh(y, f1, color=colors, edgecolor="#333333", linewidth=0.5, height=0.68,
            zorder=3)

    # value labels at bar ends
    for yi, v in zip(y, f1):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", ha="left", fontsize=6,
                color="#222222")

    # StarDist trained-DL reference line
    ax.axvline(STARDIST_AP50, color=figstyle.OKABE["orange"], ls="--", lw=1.1,
               zorder=2)
    ax.text(STARDIST_AP50 + 0.015, 1.6, "StarDist (trained DL), published",
            rotation=90, va="center", ha="left", fontsize=6,
            color=figstyle.OKABE["orange"], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    # emphasise the fluorostats tick label
    for tick, n in zip(ax.get_yticklabels(), names):
        if is_fluorostats(n) and "li" not in n.lower():
            tick.set_fontweight("bold")
            tick.set_color(figstyle.OKABE["blue"])

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Instance F1 @ IoU 0.5")


    # annotate the "% of trained model" headline, placed in the clear
    # bottom-right whitespace beside the two shortest bars (Yen, Triangle)
    best = float(df["mean_f1@0.5"].max())
    pct = 100.0 * best / STARDIST_AP50
    ax.text(0.30, 0.04,
            f"training-free fluorostats =\n{pct:.0f}% of the trained model",
            transform=ax.transAxes, fontsize=6.2, va="bottom", ha="left",
            color=figstyle.OKABE["blue"])
    ax.margins(y=0.02)


def panel_b(ax):
    """CTC 3D foreground Dice, A549 vs CHO grouped bars, fluorostats highlighted."""
    df = pd.read_csv(os.path.join(RESULTS, "b2_ctc_multi.csv"))
    datasets = ["Fluo-C3DH-A549", "Fluo-N3DH-CHO"]
    ds_short = {"Fluo-C3DH-A549": "A549", "Fluo-N3DH-CHO": "CHO"}

    # consistent method order across both datasets, fluorostats last so it reads
    methods = ["Otsu_1979", "Isodata_1978", "Li_1993", "Triangle_1977",
               "Yen_1995", "fluorostats"]
    labels = [LABELS.get(m, m) for m in methods]

    n_m = len(methods)
    n_d = len(datasets)
    group_w = 0.8
    bar_w = group_w / n_m
    x = np.arange(n_d)

    for j, m in enumerate(methods):
        vals = []
        for d in datasets:
            row = df[(df["dataset"] == d) & (df["method"] == m)]
            vals.append(float(row["mean_dice"].iloc[0]))
        offset = (j - (n_m - 1) / 2) * bar_w
        if is_fluorostats(m):
            color = figstyle.OKABE["blue"]
            edge = "#333333"
            lw = 0.7
        else:
            # distinct grey shades so baselines are separable in greyscale
            shade = 0.55 + 0.30 * (j / max(1, n_m - 2))
            color = plt.matplotlib.colors.to_hex((shade, shade, shade))
            edge = "#333333"
            lw = 0.4
        ax.bar(x + offset, vals, bar_w * 0.92, color=color, edgecolor=edge,
               linewidth=lw, zorder=3, label=labels[j])

    ax.set_xticks(x)
    ax.set_xticklabels([ds_short[d] for d in datasets])
    ax.set_xlabel("Cell Tracking Challenge 3D dataset")
    ax.set_ylabel("Foreground Dice")
    ax.set_ylim(0, 1.0)


    # legend: one entry per method
    handles, leg_labels = ax.get_legend_handles_labels()
    # de-duplicate while preserving order
    seen = {}
    for h, l in zip(handles, leg_labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), handlelength=1.0, columnspacing=1.0,
              handletextpad=0.4, fontsize=6)


def main():
    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    panel_a(axes[0])
    panel_b(axes[1])
    figstyle.panel(axes[0], "a", "DSB2018 — StarDist\u2019s own dataset")
    figstyle.panel(axes[1], "b", "unseen 3D modalities (training-free)")
    fig.subplots_adjust(wspace=0.55, bottom=0.28, left=0.16, right=0.97, top=0.86)

    figstyle.save(fig, "ed4_generalization", figstyle.EXT, tight=False)

    caption = (
        "Extended Data Figure 4 | Generalization across independent datasets and "
        "imaging modalities. Because fluorostats is training-free, it applies the "
        "same estimator to every dataset with no per-dataset fitting, so it does "
        "not collapse on modalities outside a training distribution the way a "
        "trained model can. (a) Instance F1 at IoU 0.5 on the StarDist DSB2018 "
        "fluorescence-nuclei test set (50 images) — the exact curated dataset that "
        "the StarDist deep-learning model was trained and evaluated on. fluorostats "
        "(blue) ranks first among all classical threshold heuristics and reaches "
        f"{100.0 * 0.7886 / STARDIST_AP50:.0f}% of StarDist's published trained-DL "
        f"AP@0.5 ({STARDIST_AP50}, orange dashed reference) on the trained model's "
        "home dataset, without ever seeing the training split. Note AP@0.5 and F1@0.5 "
        "are related but not identical instance metrics; the reference line is a "
        "published deep-learning benchmark shown for context, not a re-scored row. "
        "(b) Foreground Dice on two unseen Cell Tracking Challenge 3D datasets "
        "(Fluo-C3DH-A549, Fluo-N3DH-CHO), for fluorostats (blue) versus classical "
        "global-threshold baselines. fluorostats stays within the baseline band on "
        "both volumes despite the shift from 2D nuclei to 3D fluorescence cells, with "
        "no tuning. For context, the best published CTC leaderboard instance-"
        "segmentation SEG scores are 0.908 (A549) and 0.925 (CHO); these are instance "
        "SEG scores from trained pipelines and are not directly comparable to the "
        "training-free foreground-Dice values plotted here, and so are reported in "
        "text rather than overlaid. Grey bars use distinct shades to remain separable "
        "in greyscale; the Okabe-Ito palette is colourblind-safe."
    )
    figstyle.caption("ed4_generalization", caption, figstyle.EXT)
    print("caption written")


if __name__ == "__main__":
    main()
