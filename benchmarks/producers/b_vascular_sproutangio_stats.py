"""Produce results/b_vascular_sproutangio_stats.csv — pairwise VEGF-dose contrasts of fluorostats sprout area.

Summarises the committed table ``results/b_vascular_sproutangio_multi.csv`` (no
raw data needed). The fluorostats volume fraction, in percent, is compared
between every pair of VEGF groups: two-sided Mann-Whitney U, Cliff's delta,
BH-FDR q across the contrasts, and the bootstrap CI (5000 resamples, seed 0) of
the median fold change group_b / group_a.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from common import RESULTS, write_table
from fluorostats.stats import bh_fdr, bootstrap_fold_change_ci, cliffs_delta, mann_whitney

N_BOOT = 5000


def main():
    sa = pd.read_csv(RESULTS / "b_vascular_sproutangio_multi.csv")
    groups = {g: sa[sa.group == g]["fluorostats"].values * 100 for g in sorted(sa.group.unique())}  # % VF
    rows = []
    for ga, gb in combinations(groups, 2):
        a, b = groups[ga], groups[gb]
        fc = bootstrap_fold_change_ci(a, b, n_boot=N_BOOT)
        rows.append({"group_a": ga, "group_b": gb, "n_a": len(a), "n_b": len(b),
                     "mwu_p": mann_whitney(a, b)["p"], "cliffs_delta": cliffs_delta(a, b),
                     "fold_change_median": fc["fold_change_median"],
                     "fc_ci_low": fc["ci_low"], "fc_ci_high": fc["ci_high"]})
    df = pd.DataFrame(rows)
    df.insert(5, "bh_q", bh_fdr(df["mwu_p"].values))
    write_table(df, "b_vascular_sproutangio_stats", {"n_boot": N_BOOT, "fold_change_seed": 0})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
