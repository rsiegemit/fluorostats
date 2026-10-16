"""Figure S4 — Generalization across datasets.

Three panels showing that fluorostats, being training-free, holds up across
independent public datasets and imaging modalities it was never tuned on:

(a) DSB2018 fluorescence-nuclei subset (StarDist's own training dataset):
    instance F1 @ IoU 0.5 for fluorostats vs classical threshold baselines,
    with StarDist's published trained-DL AP@0.5 marked as a reference line.
    fluorostats ranks first among heuristics and reaches ~91% of the trained
    model's score on the trained model's home dataset.

(b) Cell Tracking Challenge 3D foreground Dice on two unseen modalities
    (Fluo-C3DH-A549, Fluo-N3DH-CHO), fluorostats vs threshold baselines.

(c) BBBC024 crowding series on 2D mid planes, where the 2D deep-learning models
    run as published: instance F1 of fluorostats, classical thresholds, StarDist
    and Cellpose on the identical planes (b_clustering_curve_2d).

Run:  python3.13 benchmarks/figS4_generalization.py (from any directory)
Outputs PDF+PNG to figures/supplementary/ plus a .txt caption.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# published StarDist trained-DL AP@0.5 on DSB2018 test (Schmidt et al. 2018; VALIDATION.md):
# an external literature value, so it has no results table to be read from
STARDIST_AP50 = 0.864

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
    """DSB2018 F1@0.5 ranking as horizontal bars, best at top.

    §1 unification: the classical thresholds ARE fluorostats configurations
    (Otsu/IsoData/Li/Yen/Triangle). The former separate `fluorostats` vs `Otsu`
    and `fluorostats (Li)` vs `Li` duplicate pairs are collapsed: Otsu+CC is the
    designated default (solid-blue headline) and the remaining thresholds are the
    fluorostats "tuning envelope" (same blue at reduced opacity). StarDist stays
    the orange trained-DL reference line.
    """
    df = pd.read_csv(os.path.join(RESULTS, "b_dsb2018.csv"))
    df = df.set_index("method")

    BLUE = figstyle.OKABE["blue"]
    # (label, F1, is_default) — headline uses the fluorostats(Otsu+CC) score with
    # connected-component post-processing; the envelope rows are the other
    # thresholder configs. Sorted best->worst; drawn best-on-top.
    rows = [
        ("fluorostats (Otsu+CC)", float(df.loc["fluorostats", "mean_f1@0.5"]), True),
        ("fluorostats · IsoData",     float(df.loc["isodata", "mean_f1@0.5"]),     False),
        ("fluorostats · Li",          float(df.loc["fluorostats-li", "mean_f1@0.5"]), False),
        ("fluorostats · Yen",         float(df.loc["yen", "mean_f1@0.5"]),         False),
        ("fluorostats · Triangle",    float(df.loc["triangle", "mean_f1@0.5"]),    False),
    ]
    rows.sort(key=lambda r: r[1])  # ascending -> best on top
    labels = [r[0] for r in rows]
    f1 = np.array([r[1] for r in rows])
    is_default = [r[2] for r in rows]
    # default = solid blue headline; envelope = same blue at reduced opacity
    alphas = [1.0 if d else 0.42 for d in is_default]
    y = np.arange(len(rows))

    for yi, v, a in zip(y, f1, alphas):
        ax.barh(yi, v, color=BLUE, alpha=a, edgecolor="#333333", linewidth=0.4,
                height=0.68, zorder=3)

    # value labels at bar ends
    for yi, v in zip(y, f1):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", ha="left", fontsize=5,
                color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    # emphasise the default (headline) fluorostats tick label
    for tick, d in zip(ax.get_yticklabels(), is_default):
        if d:
            tick.set_fontweight("bold")
            tick.set_color(BLUE)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("instance F1 @ IoU 0.5")
    stardist_reference(ax, len(rows), float(df.loc["fluorostats", "mean_f1@0.5"]))


def stardist_reference(ax, n_rows, best):
    """StarDist's published AP@0.5 as a reference line, and the default row's share of it."""
    # reference line raised so its vertical label sits above the annotation cluster
    # in the lower-left
    ax.axvline(STARDIST_AP50, color=figstyle.OKABE["orange"], ls="--", lw=1.1,
               zorder=2)
    ax.text(STARDIST_AP50 + 0.02, n_rows - 1, "StarDist (trained DL), published",
            rotation=90, va="top", ha="left", fontsize=6,
            color=figstyle.OKABE["orange"], fontweight="bold")
    # "% of trained model" headline on the bottom (shortest-bar) row, in the clear
    # space right of that bar and left of the reference line
    pct = 100.0 * best / STARDIST_AP50
    ax.text(0.335, 0.0,
            f"training-free fluorostats =\n{pct:.0f}% of the trained model",
            fontsize=6.0, va="center", ha="left", color=figstyle.OKABE["blue"], zorder=5)


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
            lw = 0.4
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
    ax.set_ylabel("foreground Dice")
    ax.set_ylim(0, 1.0)


    # legend: one entry per method
    handles, leg_labels = ax.get_legend_handles_labels()
    # de-duplicate while preserving order
    seen = {}
    for h, lab in zip(handles, leg_labels):
        if lab not in seen:
            seen[lab] = h
    # single row so legend order == bar left-to-right order (matplotlib fills a
    # multi-col legend column-major, which otherwise scrambles the order)
    ax.legend(seen.values(), seen.keys(), ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.19), handlelength=1.0, columnspacing=1.2,
              handletextpad=0.4, fontsize=6)


LEVELS = ["c00", "c25", "c50", "c75"]
PLANE_LINES = {   # method in b_clustering_curve_2d -> (label, colour, linestyle, marker)
    "fluorostats(raw Otsu+CC)": ("fluorostats default", figstyle.OKABE["blue"], "-", "o"),
    "fluorostats_watershed": ("fluorostats watershed", figstyle.OKABE["sky"], "--", "s"),
    "Cellpose": ("Cellpose", figstyle.TOOL["Cellpose"]["c"], figstyle.TOOL["Cellpose"]["ls"],
                 figstyle.TOOL["Cellpose"]["m"]),
    "StarDist": ("StarDist", figstyle.TOOL["StarDist"]["c"], figstyle.TOOL["StarDist"]["ls"],
                 figstyle.TOOL["StarDist"]["m"]),
}


def plane_table() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, "b_clustering_curve_2d.csv"))


def panel_c(ax):
    """Mean F1 +/- SD over the mid planes at each crowding level, every method."""
    tab = plane_table()
    x = [int(lv[1:]) for lv in LEVELS]
    for m, g in tab.groupby("method"):
        g = g.set_index("clustering").loc[LEVELS]
        if m in PLANE_LINES or g["mean_F1"].sum() == 0:
            continue
        ax.plot(x, g["mean_F1"], marker=".", ms=3, color="0.6", lw=0.8, alpha=0.7, zorder=1)
    for i, (m, (lab, col, ls, mk)) in enumerate(PLANE_LINES.items()):
        g = tab[tab["method"] == m].set_index("clustering").loc[LEVELS]
        ax.errorbar([v + (i - 1.5) * 0.8 for v in x], g["mean_F1"], yerr=g["sd_F1"], marker=mk,
                    ms=4, color=col, ls=ls, lw=1.3, capsize=1.5, elinewidth=0.7, label=lab,
                    zorder=3 + i)
    ax.plot([], [], marker=".", ms=3, color="0.6", lw=0.8, label="classical")
    ax.legend(fontsize=5, loc="lower left", frameon=False, ncol=1, handlelength=1.6)
    ax.set_xticks(x)
    ax.set_xlabel("nuclear overlap (%)")
    ax.set_ylabel("instance F1 (mid plane)")
    ax.set_ylim(0, 1.06)


def main():
    figstyle.apply_style()
    # Author at the text width (TW = 5.147 in) for a 1:1 include.
    # At this narrower canvas the former 1x2 row is too cramped, so stack the two
    # panels vertically: (a) DSB2018 ranking on top, (b) CTC grouped bars below.
    fig = plt.figure(figsize=(figstyle.TW, 7.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.02, 0.9],
                          hspace=0.46, left=0.235, right=0.965,
                          top=0.974, bottom=0.062)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    figstyle.panel(axes[0], "a", "DSB2018 — StarDist\u2019s own dataset", xanchor=0.0315)   # row-leader
    figstyle.panel(axes[1], "b", "unseen 3D modalities (training-free)", xanchor=0.0315)   # row-leader
    figstyle.panel(axes[2], "c", "BBBC024 crowding, 2D mid planes", xanchor=0.0315)   # row-leader
    figstyle.save(fig, "figS4_generalization", figstyle.SUPP, tight=False)

    dsb = pd.read_csv(os.path.join(RESULTS, "b_dsb2018.csv")).set_index("method")
    with open(os.path.join(RESULTS, "b_dsb2018.run.json")) as fh:
        n_dsb = json.load(fh)["params"]["n_images"]
    best_f1 = float(dsb.loc["fluorostats", "mean_f1@0.5"])
    ctc_seg = (pd.read_csv(os.path.join(RESULTS, "b2_ctc_multi.csv"))
               .groupby("dataset")["ctc_top_SEG"].first())
    caption = (
        "Figure S4 | Generalization across independent datasets and "
        "imaging modalities. Because fluorostats is training-free, it applies the "
        "same estimator to every dataset with no per-dataset fitting, so it does "
        "not collapse on modalities outside a training distribution the way a "
        "trained model can. (a) Instance F1 at IoU 0.5 on the StarDist DSB2018 "
        f"fluorescence-nuclei test set ({n_dsb} images) — the exact curated dataset that "
        "the StarDist deep-learning model was trained and evaluated on. fluorostats "
        "(blue) ranks first among all classical threshold heuristics and reaches "
        f"{100.0 * best_f1 / STARDIST_AP50:.0f}% of StarDist's published trained-DL "
        f"AP@0.5 ({STARDIST_AP50}, orange dashed reference) on the trained model's "
        "home dataset, without ever seeing the training split. Note AP@0.5 and F1@0.5 "
        "are related but not identical instance metrics; the reference line is a "
        "published deep-learning benchmark shown for context, not a re-scored row. "
        "(b) Foreground Dice on two unseen Cell Tracking Challenge 3D datasets "
        "(Fluo-C3DH-A549, Fluo-N3DH-CHO), for fluorostats (blue) versus classical "
        "global-threshold baselines. fluorostats stays within the baseline band on "
        "both volumes despite the shift from 2D nuclei to 3D fluorescence cells, with "
        "no tuning. For context, the best published CTC leaderboard instance-"
        f"segmentation SEG scores are {ctc_seg['Fluo-C3DH-A549']:.3f} (A549) and "
        f"{ctc_seg['Fluo-N3DH-CHO']:.3f} (CHO); these are instance "
        "SEG scores from trained pipelines and are not directly comparable to the "
        "training-free foreground-Dice values plotted here, and so are reported in "
        "text rather than overlaid. "
        + panel_c_caption() +
        " Grey bars use distinct shades to remain separable "
        "in greyscale; the Okabe-Ito palette is colourblind-safe."
    )
    figstyle.caption("figS4_generalization", caption, figstyle.SUPP)
    print("caption written")


def panel_c_caption() -> str:
    tab = plane_table().set_index(["method", "clustering"])
    n = int(tab["n"].min())

    def span(m):
        v = tab.loc[m, "mean_F1"].loc[LEVELS]
        return f"{v.iloc[0]:.2f} to {v.iloc[-1]:.2f}"
    return (
        f"(c) Instance F1 (IoU 0.5, mean \u00b1 SD over the mid z plane of {n} volumes per "
        "level) on the BBBC024 crowding series, every method on the identical planes, "
        "where the 2D deep-learning models run as published (StarDist 2D_versatile_fluo, "
        "Cellpose 3 nuclei, pretrained, diameter estimated). From no overlap to 75% "
        f"overlap: Cellpose {span('Cellpose')}, StarDist {span('StarDist')}, the "
        f"fluorostats watershed split {span('fluorostats_watershed')} and the fluorostats "
        f"default (Otsu+CC) {span('fluorostats(raw Otsu+CC)')}; classical thresholds grey. "
        "Fig. 2d shows the same series on whole 3D volumes.")


if __name__ == "__main__":
    main()
