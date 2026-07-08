"""Validate fluorostats.stats against independent references (scipy + hand-coded).

Proves the statistics layer reproduces trusted implementations exactly — a
requirement for a methods paper that ships statistics inside the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from fluorostats.stats import (
    cliffs_delta, bh_fdr, mann_whitney, bootstrap_fold_change_ci,
    stouffer_combine, scheirer_ray_hare,
)

RES = Path(__file__).resolve().parent / "results"
rng = np.random.default_rng(0)
rows = []


def check(name, fs_val, ref_val, tol=1e-9):
    diff = abs(fs_val - ref_val)
    rows.append({"test": name, "fluorostats": fs_val, "reference": ref_val,
                 "abs_diff": diff, "PASS": bool(diff <= tol)})


# 1. Mann-Whitney U + p vs scipy
x = rng.normal(0, 1, 30); y = rng.normal(0.7, 1, 25)
fs = mann_whitney(x, y)
u_ref, p_ref = sps.mannwhitneyu(x, y, alternative="two-sided")
check("mann_whitney_U", fs["u"], float(u_ref))
check("mann_whitney_p", fs["p"], float(p_ref))

# 2. Cliff's delta vs brute force
def cliffs_ref(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean(np.sign(a[:, None] - b[None, :])))
check("cliffs_delta", cliffs_delta(x, y), cliffs_ref(x, y))

# 3. BH-FDR vs hand-coded reference
p = rng.uniform(0, 1, 20) ** 2
def bh_ref(pv):
    pv = np.asarray(pv); n = len(pv); order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked, 0, 1); return out
q_fs = bh_fdr(p); q_ref = bh_ref(p)
check("bh_fdr_max_abs_diff", float(np.abs(q_fs - q_ref).max()), 0.0)

# 4. Stouffer vs scipy combine_pvalues (like-for-like: scipy uses one-sided p->Z).
# NOTE: fluorostats defaults to a two-sided convention (isf(p/2)); one_sided=True
# reproduces scipy exactly. Both are correct; the default suits pooling two-sided
# test p-values, the common bioimaging case.
pv = np.array([0.01, 0.04, 0.2, 0.5])
fs_s = stouffer_combine(pv, one_sided=True)
res = sps.combine_pvalues(pv, method="stouffer")
check("stouffer_Z(one_sided vs scipy)", fs_s["z"], float(res.statistic), tol=1e-6)
check("stouffer_p(one_sided vs scipy)", fs_s["p"], float(res.pvalue), tol=1e-6)

# 5. Bootstrap fold-change CI — coverage of known ratio (lognormal, true ratio e^1)
a = rng.lognormal(0, 0.25, 40); b = rng.lognormal(1.0, 0.25, 40)
ci = bootstrap_fold_change_ci(a, b, n_boot=5000, seed=1)
covered = ci["ci_low"] < np.e < ci["ci_high"]
rows.append({"test": "bootstrap_CI_covers_true_ratio(e)", "fluorostats": ci["fold_change_median"],
             "reference": float(np.e), "abs_diff": abs(ci["fold_change_median"] - np.e),
             "PASS": bool(covered)})

# 6. Scheirer-Ray-Hare interaction detection on data with a known interaction
recs = []
for mat in ["A", "B"]:
    for day in [1, 7, 14]:
        shift = 6 if (mat == "B" and day == 14) else 0
        for _ in range(12):
            recs.append({"mat": mat, "day": day, "v": rng.normal(shift, 1)})
srh = scheirer_ray_hare(pd.DataFrame(recs), "v", "mat", "day")
p_int = float(srh[srh.source == "mat:day"]["p"].iloc[0])
rows.append({"test": "SRH_interaction_p<0.05", "fluorostats": p_int, "reference": 0.05,
             "abs_diff": None, "PASS": bool(p_int < 0.05)})

df = pd.DataFrame(rows)
RES.mkdir(parents=True, exist_ok=True)
df.to_csv(RES / "validate_stats.csv", index=False)
print(df.to_string(index=False))
print(f"\n{int(df.PASS.sum())}/{len(df)} checks PASS")
