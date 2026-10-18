"""Figure S5 — the depth module on synthetic Beer–Lambert stacks with known λ.

(a) Surface-normalized profile of noisy stacks against the analytic exp(-z/λ).
(b) Recovered λ against the known λ (identity line), with the λ = 40 µm fit.
(c) Retained fraction over 0–100 µm for two conditions (λ = 30 and 70 µm) against
    the truth, with fluorostats' Mann–Whitney test between them.

Reads only results/b_depth_penetration_{profile,correctness,conditions,parity}.csv.
Run:  python3.13 benchmarks/figS5_depth_validation.py (from any directory)
Outputs PDF+PNG to figures/supplementary/ plus a .txt caption.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
from figstyle import OKABE, panel  # noqa: E402

from fluorostats.stats import mann_whitney  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NAME = "figS5_depth_validation"
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")   # full-size superscripts, not mathtext


def table(stem):
    return pd.read_csv(os.path.join(RESULTS, f"b_depth_penetration_{stem}.csv"))


def draw_profile(ax, prof):
    z, m, sd = prof.depth_um, prof["mean"], prof.sd
    ax.plot(z, prof.analytic, color="black", lw=1.3, ls="--", label="analytic exp(−z/λ)", zorder=3)
    ax.fill_between(z, m - sd, m + sd, color=OKABE["blue"], alpha=0.25, lw=0, zorder=1)
    ax.plot(z, m, color=OKABE["blue"], lw=1.8, label="fluorostats (mean±s.d.)", zorder=2)
    ax.set_xlabel("depth (µm)")
    ax.set_ylabel("normalized intensity")
    ax.set_xlim(0, z.max())
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5.0, loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.2)
    panel(ax, "a", "ground-truth recovery")


def draw_lambda(ax, lam):
    known, rec = lam.gt_lambda.to_numpy(), lam.fs_lambda.to_numpy()
    hi = known.max() * 1.12
    ax.plot([0, hi], [0, hi], color=OKABE["grey"], lw=1.0, ls="--", zorder=1, label="identity")
    ax.scatter(known, rec, color=OKABE["blue"], edgecolor="white", s=34, lw=0.7, zorder=3, label="recovered")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("known λ (µm)")
    ax.set_ylabel("recovered λ (µm)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=5.0, loc="upper left")
    ex = lam[lam.gt_lambda == 40.0].iloc[0]
    ax.text(0.97, 0.06, "example λ = %g µm\nfit %.1f µm, R² = %.3f" % (ex.gt_lambda, ex.fs_lambda, ex.r_squared),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color="#333")
    panel(ax, "b", "λ recovery")


def draw_conditions(ax, cond, mw):
    # both bars are fluorostats output for two known conditions, not a tool comparison,
    # so the tool blue stays reserved and two other hues are used
    colors = {"short": OKABE["orange"], "long": OKABE["sky"]}
    labels = []
    for i, (name, c) in enumerate(cond.groupby("condition", sort=False)):
        vals, col = c.retained_fraction.to_numpy(), colors[name]
        ax.bar(i, vals.mean(), width=0.6, color=col, edgecolor=col, lw=0.8, zorder=1)
        ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), color=col, capsize=3, lw=1.2, zorder=2)
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.1, 0.1, len(vals)), vals,
                   color=col, edgecolor="white", s=22, lw=0.5, zorder=3)
        ax.plot([i - 0.34, i + 0.34], [c.truth.iloc[0]] * 2, color="black", lw=1.2, ls="--", zorder=4)
        labels.append("%s λ = %g" % (name, c.lambda_um.iloc[0]))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("retained fraction, 0–100 µm depth")
    ax.set_ylim(0, 1.0)
    exp = int(np.floor(np.log10(mw["p"])))
    delta = f"{mw['cliffs_delta']:.2f}".replace("-", "−")
    ax.text(0.98, 0.96, f"MW p = {mw['p'] / 10 ** exp:.1f} × 10{str(exp).translate(SUP)}\nCliff's δ = {delta}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.0, color="#333")
    panel(ax, "c", "condition discrimination")


def caption_text(corr, cond, parity, mw):
    lam = corr[corr.test == "Alam_exact_noiseless"]
    noisy = corr[corr.test == "Alam_noisy_recovery"].iloc[0]
    a1 = corr[corr.test == "A1_exact_noiseless"]
    p = parity.set_index("check").value
    n = cond.groupby("condition", sort=False).size()
    return (
        "Depth module on synthetic Beer-Lambert z-stacks with known lambda. "
        "(a) Surface-normalized intensity vs depth: the analytic decay exp(-z/lambda) (black dashed) "
        "and the fluorostats profile of noisy stacks (blue, mean +/- s.d. over 8 replicates, surface "
        "SNR ~10). (b) lambda recovered by the single-exponential fit (fluorostats.depth."
        "fit_penetration_depth) against the known lambda (%s um) on noiseless stacks: largest error "
        "%.1g%%; on the noisy stacks lambda = %.2f um for a true 40 um (%.2f%%). (c) Two conditions "
        "(lambda = 30 vs 70 um): retained fraction over 0-100 um per replicate (dots; bars mean +/- s.d.) "
        "against the noiseless truth (dashed); Mann-Whitney p = %.1e, Cliff's delta = %.2f (n = %d, %d). "
        "Normalized AUC on noiseless stacks is recovered within %.3f%%; the trapezoidal AUC is within "
        "%.3f%% of the analytic integral, against %.2f%% for a rectangular sum (%.0fx smaller error)."
        % ("/".join(f"{v:g}" for v in lam.gt_lambda), lam.lambda_err_pct.max(), noisy.fs_lambda,
           noisy.lambda_err_pct, mw["p"], mw["cliffs_delta"], n.iloc[0], n.iloc[1], a1.err_norm_pct.max(),
           p["trapezoid_err_pct_vs_analytic"], p["rectangular_err_pct_vs_analytic"],
           p["rectangular_err_pct_vs_analytic"] / p["trapezoid_err_pct_vs_analytic"]))


def main():
    corr, cond, parity = table("correctness"), table("conditions"), table("parity")
    by = [c.retained_fraction.to_numpy() for _, c in cond.groupby("condition", sort=False)]
    mw = mann_whitney(by[0], by[1])
    F.apply_style()
    fig = plt.figure(figsize=(F.TW, 2.237))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 0.95], wspace=0.46,
                          left=0.075, right=0.985, top=0.915, bottom=0.148)
    draw_profile(fig.add_subplot(gs[0, 0]), table("profile"))
    draw_lambda(fig.add_subplot(gs[0, 1]), corr[corr.test == "Alam_exact_noiseless"])
    draw_conditions(fig.add_subplot(gs[0, 2]), cond, mw)
    F.save(fig, NAME, folder=F.SUPP, tight=False)
    F.caption(NAME, caption_text(corr, cond, parity, mw), folder=F.SUPP)


if __name__ == "__main__":
    main()
