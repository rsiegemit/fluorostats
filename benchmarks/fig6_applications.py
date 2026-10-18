"""Figure 6 — two laboratory GelMA assays quantified by fluorostats.

Row 1, probe penetration (FITC-dextran 2000 kDa in GelMA, 4/12/24/72 h):
(a) normalized retention vs depth per time point; (b) per-stack AUC (0–200 µm)
as fold change on 4 h, surface-normalized against raw absolute intensity.
Row 2, Live/Dead viability (HUVEC on GelMA, day 1 vs day 5): (c) representative
maximum-intensity projections; (d) live and dead coverage; (e) area- vs
count-based viability and the three counting modes.

Panels a, b, d, e read only results/fig6_*.csv (``producers/fig6_tables.py``).
Panel c renders two laboratory .oib stacks that are not deposited; set
``$FIG6_LIVEDEAD_DIR`` to the folder holding them. Needs ``oiffile``.
Run:  python3.13 benchmarks/fig6_applications.py (from any directory)
Outputs PDF+PNG to figures/main/ plus a .txt caption.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
from figstyle import FS, OKABE, image_axes, panel, scalebar  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
NAME = "fig6_applications"
REP = {"Day 1": "GelMA 1 day (1).oib", "Day 5": "GelMA 5 days (2).oib"}
FOV_UM = 848.0                      # the same physical field for the 800² and 1024² stacks
GROUPS = ["4h", "12h", "24h", "72h"]
TIME_H = {"4h": 4, "12h": 12, "24h": 24, "72h": 72}
DEPTH_ACQ = 475.0                   # acquired depth (100 slices × 4.8 µm)
BLUE, GREY = OKABE["blue"], OKABE["grey"]
# light → dark = short → long incubation; capped below pure yellow for contrast
TGRAD = {g: plt.cm.viridis(v) for g, v in zip(GROUPS, np.linspace(0.80, 0.08, 4))}
# row 2 has no tool comparison, so blue is not reserved; day is encoded by x position
LIVE, DEAD, AREA, COUNT = OKABE["green"], OKABE["purple"], OKABE["orange"], OKABE["sky"]
DAYS = ["Day 1", "Day 5"]


def table(stem):
    return pd.read_csv(RESULTS / f"fig6_{stem}.csv")


def primary_auc():
    a = table("depth_auc_per_stack")
    return a[a.role.fillna("primary") == "primary"]


def viability():
    v = table("viability_per_stack")
    return v.assign(live_cov_pct=100 * v.live_cov_frac, dead_cov_pct=100 * v.dead_cov_frac)


def col(v, day, k):
    return v.loc[v.group == day, k].dropna().to_numpy(float)


def sem(a):
    return a.std(ddof=1) / len(a) ** 0.5 if len(a) > 1 else 0.0


def draw_panel_a(ax):
    gs = table("depth_group_summary")
    for g in reversed(GROUPS):            # 4 h drawn on top
        p, c = gs[gs.group == g], TGRAD[g]
        d, m, s = p.depth_um.to_numpy(), p.normalized_mean.to_numpy(), p.normalized_sem.to_numpy()
        ax.fill_between(d, m - s, m + s, color=c, alpha=0.18, lw=0)
        ax.plot(d, m, color=c, lw=1.3)
        r240 = float(np.interp(240, d, m)) * 100
        ax.text(DEPTH_ACQ + 10, float(np.clip(r240 / 100, 0.05, 1.0)), f"{TIME_H[g]} h · {r240:.0f}%",
                color=c, fontsize=FS["small"], va="center", ha="left")
    # the dotted line stops below its label so it cannot run into the glyphs
    ax.axvline(240, color="#555", ls=":", lw=0.7, zorder=1, ymax=0.95)
    ax.text(240, 1.05, "240 µm", fontsize=FS["small"], color="#555", ha="center", va="bottom")
    ax.plot([0, DEPTH_ACQ], [0, 0], color=GREY, lw=0.7, alpha=0.6, zorder=0)
    ax.set_xlim(0, 575)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([0, 240, 475])
    ax.set_xlabel("depth (µm)")
    ax.set_ylabel("normalized retention")
    panel(ax, "a", "deepens with time")


def fold_series(a, column):
    """Per-group fold change on the 4 h mean: (per-stack values, means, SEMs)."""
    base = a.loc[a.group == "4h", column].mean()
    vals = [a.loc[a.group == g, column].to_numpy() / base for g in GROUPS]
    return vals, [v.mean() for v in vals], [sem(v) for v in vals]


def draw_panel_b(ax):
    """Surface normalization against raw intensity, which acquisition gain confounds."""
    a, rng = primary_auc(), np.random.default_rng(0)
    ax.axhline(1.0, ls=":", lw=0.7, color="#888", zorder=1)
    _, af_m, af_s = fold_series(a, "auc_absolute_0_200um")
    ax.errorbar(range(4), af_m, yerr=af_s, color=GREY, lw=1.1, ls="--",
                marker="o", mfc="white", mec=GREY, mew=1.2, ms=5, capsize=2, zorder=4)
    nf, nf_m, nf_s = fold_series(a, "auc_normalized_0_200um")
    for i, v in enumerate(nf):
        ax.scatter(i + (rng.random(len(v)) - 0.5) * 0.30, v, s=11, color=BLUE, edgecolor="none",
                   alpha=0.40, zorder=3)
    ax.plot(range(4), nf_m, color=BLUE, lw=1.3, zorder=5)
    ax.errorbar(range(4), nf_m, yerr=nf_s, color=BLUE, lw=1.1, marker="o", ms=4.5,
                mec="white", mew=0.5, capsize=2, zorder=6)
    a = a[a.group.isin(GROUPS)]
    rho, p = stats.spearmanr(a.group.map(TIME_H), a.auc_normalized_0_200um)
    ax.set_xticks(range(4))
    ax.set_xticklabels([TIME_H[g] for g in GROUPS])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 1.48)
    ax.set_xlabel("incubation time (h)")
    ax.set_ylabel("AUC (0–200 µm), fold-change vs 4 h")
    ax.annotate("raw intensity:\ngain-confounded\n(12 h imaged dim)", xy=(1, af_m[1]), xytext=(1.25, 0.60),
                fontsize=FS["annot"], color="#666", va="center", ha="left", linespacing=1.05,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
    ax.text(0.97, 0.045, f"Spearman ρ = {rho:.3f}\np = {p:.1e}  (n = {len(a)})",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=FS["annot"], color=BLUE)
    ax.plot([], [], color=BLUE, lw=1.3, marker="o", ms=4, mec="white", label="surface-normalized (fluorostats)")
    ax.plot([], [], color=GREY, lw=1.1, ls="--", marker="o", mfc="white", mec=GREY,
            label="absolute intensity (raw)")
    ax.legend(loc="upper left", fontsize=FS["annot"], handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    panel(ax, "b", "normalization recovers the trend")


def mips(path):
    """Live and dead maximum-intensity projections of a 2-channel .oib stack."""
    import oiffile
    a = np.asarray(oiffile.imread(str(path))).astype(np.float32)   # (2, Z, Y, X)
    return a[0].max(0), a[1].max(0)


def draw_panel_c(ax1, ax2):
    if "FIG6_LIVEDEAD_DIR" not in os.environ:
        raise SystemExit("Figure 6(c) renders laboratory .oib stacks that are not deposited: "
                         "set $FIG6_LIVEDEAD_DIR to the folder holding " + " and ".join(REP.values()))
    folder = Path(os.environ["FIG6_LIVEDEAD_DIR"])
    (live1, dead1), (live5, dead5) = (mips(folder / REP[d]) for d in DAYS)

    def rng(x, y):                      # one display range per channel, pooled over both days
        p = np.concatenate([x.ravel(), y.ravel()])
        return np.percentile(p, 1), np.percentile(p, 99.5)
    (lv0, lv1), (dv0, dv1) = rng(live1, live5), rng(dead1, dead5)

    def rgb(live, dead):
        L = np.clip((live - lv0) / (lv1 - lv0 + 1e-9), 0, 1)
        D = np.clip((dead - dv0) / (dv1 - dv0 + 1e-9), 0, 1)
        return np.clip(np.dstack([D, L, D]), 0, 1)          # magenta dead, green live
    stroke = [pe.withStroke(linewidth=1.6, foreground="black")]
    for ax, (live, dead), day in [(ax1, (live1, dead1), "Day 1"), (ax2, (live5, dead5), "Day 5")]:
        ax.imshow(rgb(live, dead), extent=[0, FOV_UM, FOV_UM, 0], interpolation="nearest")
        image_axes(ax)
        ax.text(0.035, 0.965, day, transform=ax.transAxes, ha="left", va="top", color="white",
                fontsize=FS["label"], fontweight="bold", path_effects=stroke)
    scalebar(ax2, 100, "100 µm")         # both share the physical extent
    panel(ax1, "c", "same scale", pad=(-2, 5))


def live_mwu(v):
    u = stats.mannwhitneyu(col(v, "Day 1", "live_cov_pct"), col(v, "Day 5", "live_cov_pct"),
                           alternative="two-sided")
    return u.statistic, u.pvalue


def draw_panel_d(ax, v):
    rng = np.random.default_rng(1)
    means = {"live": [], "dead": []}
    for di, g in enumerate(DAYS):
        for kind, color, x in (("live", LIVE, di - 0.16), ("dead", DEAD, di + 0.16)):
            c = col(v, g, f"{kind}_cov_pct")
            ax.scatter(x + (rng.random(len(c)) - 0.5) * 0.14, c, s=9, color=color, alpha=0.45,
                       edgecolor="none", zorder=3)
            means[kind].append(c.mean())
        for kind, color, x in (("live", LIVE, di - 0.16), ("dead", DEAD, di + 0.16)):
            c = col(v, g, f"{kind}_cov_pct")
            ax.errorbar(x, c.mean(), yerr=sem(c), color=color, marker="o", ms=4.5,
                        mec="white", mew=0.5, capsize=2, lw=1.0, zorder=5)
    ax.plot([-0.16, 0.84], means["live"], color=LIVE, lw=1.0, zorder=4)
    ax.plot([0.16, 1.16], means["dead"], color=DEAD, lw=1.0, zorder=4)
    ax.text(1.24, means["live"][1], "live", color=LIVE, fontsize=FS["small"], va="center", ha="left")
    ax.text(1.24, means["dead"][1], "dead", color=DEAD, fontsize=FS["small"], va="center", ha="left")
    yb = 39.5                            # significance bracket on the live series only
    ax.plot([-0.16, -0.16, 0.84, 0.84], [means["live"][0] + 2.5, yb, yb, means["live"][1] + 2.5],
            color="#333", lw=0.7)
    u, p = live_mwu(v)
    ax.text(0.34, yb + 0.6, f"U = {u:.0f}, p = {p:.2g}", ha="center", va="bottom", fontsize=FS["small"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 44)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(DAYS)
    ax.set_ylabel("coverage (%)")
    panel(ax, "d", "live rises, dead changes little")


def draw_panel_e(ax, v):
    area = [col(v, g, "viability_area").mean() for g in DAYS]
    count = [col(v, g, "viability_count_consensus").mean() for g in DAYS]
    ax.fill_between([0, 1], count, area, color="#555", alpha=0.12, lw=0, zorder=2)
    for di, g in enumerate(DAYS):        # the three counting modes, spread so each shows
        for k, m in enumerate(("cc", "watershed", "maxima")):
            ax.plot(di - 0.26 + k * 0.10, col(v, g, f"viability_{m}").mean(), "o",
                    mfc="none", mec=GREY, mew=0.8, ms=3, zorder=3)
    ax.plot([0, 1], area, color=AREA, lw=1.3, marker="o", ms=5, mec="white", mew=0.5, zorder=5)
    ax.plot([0, 1], count, color=COUNT, lw=1.3, marker="s", ms=5, mec="white", mew=0.5, zorder=5)
    for di in range(2):
        ax.text(di, 0.24, f"gap {area[di] - count[di]:.3f}", fontsize=FS["small"],
                va="bottom", ha="center", color="#333")
    ax.text(1.09, area[1], "area-based", color=AREA, fontsize=FS["small"], va="bottom", ha="left")
    ax.text(1.09, count[1], "count-based", color=COUNT, fontsize=FS["small"], va="top", ha="left")
    ax.text(-0.42, 0.02, "cc / ws /\nmaxima", color=GREY, fontsize=FS["small"],
            va="bottom", ha="left", linespacing=1.0)
    ax.set_xlim(-0.5, 1.95)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(DAYS)
    ax.set_ylabel("viability estimate")
    panel(ax, "e", "area vs count diverge")


def build():
    """The laid-out figure and its two row subfigures."""
    F.apply_style()
    v = viability()
    fig = plt.figure(figsize=(F.TW, 4.75), layout="constrained")
    sfs = fig.subfigures(2, 1, height_ratios=[1, 1.18])
    tagkw = dict(x=0.012, ha="left", fontsize=FS["small"], style="italic", color="#444")
    sfs[0].suptitle("Probe penetration — FITC-dextran in GelMA", **tagkw)
    axa, axb = sfs[0].subplots(1, 2)
    draw_panel_a(axa)
    draw_panel_b(axb)
    sfs[1].suptitle("Live/Dead viability — HUVEC on GelMA", **tagkw)
    gs = sfs[1].add_gridspec(2, 3, width_ratios=[30, 35, 35], wspace=0.14, hspace=0.00)
    draw_panel_c(sfs[1].add_subplot(gs[0, 0]), sfs[1].add_subplot(gs[1, 0]))
    draw_panel_d(sfs[1].add_subplot(gs[:, 1]), v)
    draw_panel_e(sfs[1].add_subplot(gs[:, 2]), v)
    fig.canvas.draw()                    # hairline rule at the top of row 2
    yb = sfs[1].bbox.y1 / fig.bbox.height
    fig.add_artist(Line2D([0.03, 0.97], [yb, yb], transform=fig.transFigure, color="#bbb", lw=0.6))
    return fig, sfs


def caption_text(v):
    """The caption, every number computed from the fig6_* tables."""
    def m(g, k):
        return col(v, g, k).mean()

    def both(k):
        return m("Day 1", k), m("Day 5", k)
    n1, n5 = len(col(v, "Day 1", "live_cov_pct")), len(col(v, "Day 5", "live_cov_pct"))
    lc, dc, dn = both("live_cov_pct"), both("dead_cov_pct"), both("live_density_permm2")
    ar, ct, mip = both("viability_area"), both("viability_count_consensus"), both("overestimate_mip")
    modes = ["viability_cc", "viability_watershed", "viability_maxima"]
    spread = [float((v.loc[v.group == g, modes].max(axis=1) - v.loc[v.group == g, modes].min(axis=1)).mean())
              for g in DAYS]
    u, p = live_mwu(v)
    a = primary_auc()
    a = a[a.group.isin(GROUPS)]
    rho, prho = stats.spearmanr(a.group.map(TIME_H), a.auc_normalized_0_200um)
    gs = table("depth_group_summary")
    r240 = [float(np.interp(240, gs[gs.group == g].depth_um, gs[gs.group == g].normalized_mean)) * 100
            for g in GROUPS]
    return (
        f"Figure 6 | fluorostats applied to two laboratory GelMA assays. (a,b) FITC-dextran (2000 kDa) "
        f"penetration time course (n = {', '.join(str((a.group == g).sum()) for g in GROUPS)} stacks at "
        f"4/12/24/72 h). (a) Normalized retention vs depth (mean +/- SEM); retention at 240 um "
        f"{'/'.join(f'{r:.0f}' for r in r240)}%. (b) AUC (0-200 um) as fold change on 4 h: surface-normalized "
        f"(blue; per-stack points, mean +/- SEM) rises monotonically (Spearman rho = {rho:.3f}, p = {prho:.1e}, "
        f"n = {len(a)}); raw absolute intensity (grey) does not, because acquisition gain differed. "
        f"(c-e) Live/Dead of HUVECs on GelMA, day 1 (n = {n1}) vs day 5 (n = {n5}); live AF488, dead PI. "
        f"(c) Representative maximum-intensity projections, matched display range, field {FOV_UM:.0f} um. "
        f"(d) Live coverage {lc[0]:.2f} -> {lc[1]:.2f}% ({lc[1] / lc[0]:.2f}x; Mann-Whitney U = {u:.0f}, "
        f"p = {p:.2g}); dead coverage changes little ({dc[0]:.2f} -> {dc[1]:.2f}%); points are stacks, bars "
        f"mean +/- SEM. Live density {dn[0]:.1f} -> {dn[1]:.1f} cells/mm^2 ({dn[1] / dn[0]:.2f}x). "
        f"(e) Area-based viability ({ar[0]:.3f} -> {ar[1]:.3f}) and count-based consensus "
        f"({ct[0]:.3f} -> {ct[1]:.3f}) separate, gap {ar[0] - ct[0]:.3f} -> {ar[1] - ct[1]:.3f}; the spread "
        f"across counting modes widens {spread[0]:.2f} -> {spread[1]:.2f} (day-5 modes cc "
        f"{m('Day 5', modes[0]):.2f}, watershed {m('Day 5', modes[1]):.2f}, maxima {m('Day 5', modes[2]):.2f}); "
        f"the projection overstates coverage {mip[0]:.2f}x (day 1) and {mip[1]:.2f}x (day 5).")


def main():
    fig, _ = build()
    F.save(fig, NAME, folder=F.MAIN, tight=False)
    F.caption(NAME, caption_text(viability()), folder=F.MAIN)


if __name__ == "__main__":
    main()
