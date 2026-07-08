"""Validate fluorostats.agreement against independent references.

Cross-checks Lin's CCC, ICC(A,1), and Bland-Altman limits of agreement against
hand-coded reference formulas (numpy/scipy only) on shared synthetic paired
data — a requirement for a methods paper that ships agreement statistics inside
the pipeline. Also verifies the qualitative invariants: identical vectors give
CCC = ICC = 1, and a pure constant offset leaves Pearson r at 1 while CCC drops.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from fluorostats.agreement import bland_altman, lins_ccc, icc

RES = Path(__file__).resolve().parent / "results"
rng = np.random.default_rng(0)
rows = []


def check(name, fs_val, ref_val, tol=1e-9):
    diff = abs(fs_val - ref_val)
    rows.append({"test": name, "fluorostats": fs_val, "reference": ref_val,
                 "abs_diff": diff, "PASS": bool(diff <= tol)})


# Shared synthetic paired data: method B is method A plus a small bias and noise.
a = rng.normal(10.0, 2.0, 60)
b = a + 0.5 + rng.normal(0.0, 1.0, 60)


# 1. Lin's CCC vs the closed-form Lin (1989) formula computed directly.
#    rho_c = 2*cov / (var_x + var_y + (mean_x - mean_y)^2), population moments.
ma, mb = a.mean(), b.mean()
va = ((a - ma) ** 2).mean()
vb = ((b - mb) ** 2).mean()
cov = ((a - ma) * (b - mb)).mean()
ccc_lin = 2 * cov / (va + vb + (ma - mb) ** 2)
check("lins_ccc vs Lin(1989) closed form", lins_ccc(a, b), float(ccc_lin))

# 2. Lin's CCC vs precision x accuracy decomposition: rho_c = r * C_b.
#    r = Pearson correlation; C_b = bias-correction factor = 2 / (v + 1/v + u^2),
#    with v = sd_x/sd_y (scale shift) and u = (mean_x-mean_y)/sqrt(sd_x*sd_y).
sda, sdb = np.sqrt(va), np.sqrt(vb)
r = cov / (sda * sdb)
v = sda / sdb
u = (ma - mb) / np.sqrt(sda * sdb)
c_b = 2.0 / (v + 1.0 / v + u ** 2)
check("lins_ccc vs r * C_b decomposition", lins_ccc(a, b), float(r * c_b), tol=1e-12)

# 3. ICC(A,1) vs hand-coded two-way ANOVA from explicit sums of squares.
#    Shrout & Fleiss (1979) two-way random, absolute agreement, single measure.
#    n targets x k=2 raters. Built from SS -> MS independently of the module.
Y = np.column_stack([a, b])
n, k = Y.shape
grand = Y.mean()
row_means = Y.mean(axis=1)
col_means = Y.mean(axis=0)
ss_rows = k * ((row_means - grand) ** 2).sum()
ss_cols = n * ((col_means - grand) ** 2).sum()
ss_total = ((Y - grand) ** 2).sum()
ss_err = ss_total - ss_rows - ss_cols
msr = ss_rows / (n - 1)
msc = ss_cols / (k - 1)
mse = ss_err / ((n - 1) * (k - 1))
icc_ref = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
check("icc vs two-way ANOVA ICC(A,1)", icc(a, b), float(icc_ref))

# 4. Bland-Altman bias / SD / LoA vs direct numpy on the differences.
ba = bland_altman(a, b)
d = a - b
bias_ref = d.mean()
sd_ref = d.std(ddof=1)
check("bland_altman bias vs mean(a-b)", ba["bias"], float(bias_ref))
check("bland_altman sd_diff vs std(a-b, ddof=1)", ba["sd_diff"], float(sd_ref))
check("bland_altman loa_lower vs bias-1.96sd", ba["loa_lower"], float(bias_ref - 1.96 * sd_ref))
check("bland_altman loa_upper vs bias+1.96sd", ba["loa_upper"], float(bias_ref + 1.96 * sd_ref))

# 5a. Identical vectors: CCC = ICC = 1 exactly.
check("identical vectors: CCC == 1", lins_ccc(a, a), 1.0)
check("identical vectors: ICC == 1", icc(a, a), 1.0)

# 5b. Constant offset: Pearson r stays 1, but CCC drops below 1 (accuracy loss).
offset = a + 3.0
r_off = sps.pearsonr(a, offset).statistic
ccc_off = lins_ccc(a, offset)
check("constant offset: Pearson r == 1", float(r_off), 1.0, tol=1e-9)
rows.append({"test": "constant offset: CCC < r (accuracy penalty)",
             "fluorostats": ccc_off, "reference": float(r_off),
             "abs_diff": float(r_off - ccc_off),
             "PASS": bool(ccc_off < r_off - 1e-6)})

df = pd.DataFrame(rows)
RES.mkdir(parents=True, exist_ok=True)
df.to_csv(RES / "b_agreement_validation.csv", index=False)

n_pass = int(df.PASS.sum())


def md_table(frame):
    cols = list(frame.columns)
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([head, sep, *body])


lines = ["# Agreement statistics validation", "",
         f"Cross-checked `fluorostats.agreement` against independent numpy/scipy "
         f"references on shared synthetic paired data (n={n}).", "",
         f"**{n_pass}/{len(df)} checks PASS**", "",
         md_table(df)]
(RES / "b_agreement_validation.md").write_text("\n".join(lines) + "\n")

print(df.to_string(index=False))
print(f"\n{n_pass}/{len(df)} checks PASS")
