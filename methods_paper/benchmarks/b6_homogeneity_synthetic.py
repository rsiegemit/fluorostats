"""B6 (homogeneity) — validate fluorostats lateral Gini/CV on synthetic controls.

Generates point patterns spanning a known uniformity gradient:
    regular (jittered lattice)  ->  CSR (Poisson)  ->  clustered (Thomas process)
renders each as a fluorescence image, and checks that fluorostats' 8x8 tile
Gini/CV:
    (1) increases monotonically from regular -> CSR -> clustered, and
    (2) correlates with the classic Clark-Evans nearest-neighbor index R
        (R>1 dispersed, R=1 CSR, R<1 clustered) — Spearman rho.

Ground truth is the generative clustering level; the Clark-Evans R is an
independent, established reference statistic (scipy only, no extra deps).
Needs no external data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy import stats as sps

from fluorostats.morphometry import lateral_homogeneity

OUT = Path(__file__).resolve().parent / "results"
FIELD = 512          # image size (px)
N_PTS = 400          # points per pattern
REPS = 12            # replicates per condition
BLOB_SIGMA = 3.0     # gaussian blob radius per point (px)


def clark_evans_R(pts, area, n):
    """Clark-Evans nearest-neighbor index R (edge-uncorrected)."""
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    mean_nn = d[:, 1].mean()
    expected = 0.5 / np.sqrt(n / area)
    return float(mean_nn / expected)


def gen_regular(rng, jitter=0.15):
    g = int(np.ceil(np.sqrt(N_PTS)))
    step = FIELD / g
    xs, ys = np.meshgrid(np.arange(g), np.arange(g))
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)[:N_PTS]
    pts = (pts + 0.5) * step + rng.uniform(-jitter, jitter, pts.shape) * step
    return np.clip(pts, 0, FIELD - 1)


def gen_csr(rng):
    return rng.uniform(0, FIELD, size=(N_PTS, 2))


def gen_thomas(rng, n_parents, sigma):
    """Thomas cluster process: children gaussian around random parents."""
    parents = rng.uniform(0, FIELD, size=(n_parents, 2))
    per = N_PTS // n_parents
    pts = []
    for p in parents:
        pts.append(p + rng.normal(0, sigma, size=(per, 2)))
    pts = np.vstack(pts)
    return np.clip(pts, 0, FIELD - 1)


def render(pts):
    img = np.zeros((FIELD, FIELD), np.float32)
    idx = np.clip(pts.astype(int), 0, FIELD - 1)
    img[idx[:, 1], idx[:, 0]] = 1.0
    return ndi.gaussian_filter(img, BLOB_SIGMA)


CONDITIONS = [
    ("regular",      lambda rng: gen_regular(rng),               0),
    ("csr",          lambda rng: gen_csr(rng),                   1),
    ("cluster_wide", lambda rng: gen_thomas(rng, 20, 30.0),      2),
    ("cluster_mid",  lambda rng: gen_thomas(rng, 10, 18.0),      3),
    ("cluster_tight",lambda rng: gen_thomas(rng, 5, 10.0),       4),
]


def main():
    rows = []
    area = FIELD * FIELD
    for name, gen, level in CONDITIONS:
        for r in range(REPS):
            rng = np.random.default_rng(1000 * level + r)
            pts = gen(rng)
            R = clark_evans_R(pts, area, len(pts))
            img = render(pts)[None, :, :]     # (1, H, W) volume
            hom = lateral_homogeneity(img, tiles=8)
            rows.append({
                "condition": name, "level": level, "rep": r,
                "clark_evans_R": R,
                "lateral_gini": hom["lateral_gini"],
                "lateral_cv": hom["lateral_cv"],
            })
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "b6_homogeneity_synthetic.csv", index=False)

    # (1) monotonic trend across ordered conditions
    med = df.groupby("level").agg(
        condition=("condition", "first"),
        R=("clark_evans_R", "median"),
        gini=("lateral_gini", "median"),
        cv=("lateral_cv", "median"),
    ).reset_index()
    print("Median by condition (ordered regular -> clustered):", flush=True)
    print(med.to_string(index=False), flush=True)
    gini_monotonic = bool(np.all(np.diff(med["gini"].values) > 0))
    R_monotonic = bool(np.all(np.diff(med["R"].values) < 0))  # R falls as clustering rises

    # (2) correlation gini vs R (expect strong negative)
    rho_gini = sps.spearmanr(df["lateral_gini"], df["clark_evans_R"]).statistic
    rho_cv = sps.spearmanr(df["lateral_cv"], df["clark_evans_R"]).statistic

    # (3) separation: uniform (regular+csr) vs clustered (levels>=2), AUC via Mann-Whitney
    uni = df[df.level <= 1]["lateral_gini"].values
    clu = df[df.level >= 2]["lateral_gini"].values
    U = sps.mannwhitneyu(clu, uni, alternative="greater")
    auc = U.statistic / (len(clu) * len(uni))

    summary = pd.DataFrame([{
        "gini_monotonic_up": gini_monotonic,
        "R_monotonic_down": R_monotonic,
        "spearman_gini_vs_R": round(float(rho_gini), 3),
        "spearman_cv_vs_R": round(float(rho_cv), 3),
        "AUC_clustered_vs_uniform_gini": round(float(auc), 3),
        "mannwhitney_p": float(U.pvalue),
    }])
    summary.to_csv(OUT / "b6_homogeneity_summary.csv", index=False)
    print("\nValidation summary:", flush=True)
    print(summary.to_string(index=False), flush=True)

    passed = (gini_monotonic and rho_gini < -0.6 and auc > 0.9)
    print(f"\nOVERALL: {'PASS — Gini tracks known clustering' if passed else 'REVIEW'}")
    print(f"Saved {OUT / 'b6_homogeneity_synthetic.csv'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
