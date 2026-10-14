"""Produce results/b_homogeneity_multi.csv and b_homogeneity_multi_corr.csv.

Point patterns spanning a known uniformity gradient (jittered lattice -> CSR ->
three Thomas cluster processes, 10 seeded replicates each) are rendered as
fluorescence images. fluorostats' 8x8-tile lateral Gini is compared against five
established point-pattern statistics (Clark-Evans R, Ripley L deviation,
Morisita index, lacunarity, quadrat variance-to-mean). No data needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats as sps

from common import RESULTS, clark_evans, lacunarity, morisita, quadrat_var, ripley_L_dev
from fluorostats.morphometry import lateral_homogeneity
from fluorostats.stats import mann_whitney

FIELD = 512          # image size (px)
N_PTS = 400          # points per pattern
BLOB_SIGMA = 3.0     # gaussian blob radius per point (px)
REPS = 10


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
    """Thomas cluster process: gaussian children around uniform parents."""
    parents = rng.uniform(0, FIELD, size=(n_parents, 2))
    per = N_PTS // n_parents
    pts = np.vstack([p + rng.normal(0, sigma, size=(per, 2)) for p in parents])
    return np.clip(pts, 0, FIELD - 1)


def render(pts):
    img = np.zeros((FIELD, FIELD), np.float32)
    idx = np.clip(pts.astype(int), 0, FIELD - 1)
    img[idx[:, 1], idx[:, 0]] = 1.0
    return ndi.gaussian_filter(img, BLOB_SIGMA)


CONDITIONS = [(gen_regular, 0), (gen_csr, 1),
              (lambda r: gen_thomas(r, 20, 30), 2),
              (lambda r: gen_thomas(r, 10, 18), 3),
              (lambda r: gen_thomas(r, 5, 10), 4)]
REFS = ["clark_evans", "ripley_L_dev", "morisita", "lacunarity", "quadrat_var"]


def main():
    recs = []
    area = FIELD * FIELD
    for gen, lvl in CONDITIONS:
        for rep in range(REPS):
            pts = gen(np.random.default_rng(1000 * lvl + rep))
            gini = lateral_homogeneity(render(pts)[None], tiles=8)["lateral_gini"]
            recs.append({"level": lvl, "gini": gini, "clark_evans": clark_evans(pts, area, len(pts)),
                         "ripley_L_dev": ripley_L_dev(pts, area), "morisita": morisita(pts, FIELD),
                         "lacunarity": lacunarity(pts, FIELD), "quadrat_var": quadrat_var(pts, FIELD)})
    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "b_homogeneity_multi.csv", index=False)

    uni = df[df.level <= 1]["gini"]
    clu = df[df.level >= 2]["gini"]
    auc = mann_whitney(clu, uni, alternative="greater")["u"] / (len(clu) * len(uni))
    rows = [{"reference": ref,
             "spearman_gini_vs_ref": round(float(sps.spearmanr(df["gini"], df[ref]).statistic), 3),
             "clustered_vs_uniform_AUC(gini)": round(float(auc), 3)} for ref in REFS]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "b_homogeneity_multi_corr.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
