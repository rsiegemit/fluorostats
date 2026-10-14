"""Produce results/b4_reaver_ranking.csv — per-tool area-fraction error vs manual GT (REAVER).

Summarises the committed per-image table ``results/b4_reaver_ranking_perimage.csv``
(no raw data needed): for each tool, the residual (tool - manual area fraction)
over the 36 images gives the mean absolute error, the residual s.d. (ddof 1) and a
two-sided one-sample t-test of zero mean bias; ``unbiased`` marks p above the
Bonferroni level 0.05 / number of tools. Rows are sorted by MAE (best first).
"""
from __future__ import annotations

import pandas as pd
from scipy import stats as sps

from common import RESULTS, write_table

ALPHA = 0.05


def main():
    pi = pd.read_csv(RESULTS / "b4_reaver_ranking_perimage.csv")
    tools = [c for c in pi.columns if c not in ("image", "manual_GT")]
    gt = pi["manual_GT"].values
    alpha = ALPHA / len(tools)                               # Bonferroni
    rows = []
    for t in tools:
        res = pi[t].values - gt
        p = sps.ttest_1samp(res, 0).pvalue
        rows.append({"tool": t, "n_images": len(res), "MAE": abs(res).mean(), "residual_SD": res.std(ddof=1),
                     "bias_ttest_p": p, "bonferroni_alpha": alpha, "unbiased": bool(p > alpha)})
    df = pd.DataFrame(rows).sort_values("MAE", kind="stable")
    write_table(df, "b4_reaver_ranking", {"alpha": ALPHA, "correction": "Bonferroni"})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
