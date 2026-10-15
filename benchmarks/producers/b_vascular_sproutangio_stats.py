"""Produce results/b_vascular_sproutangio_stats.csv — pairwise VEGF-dose contrasts of fluorostats sprout area.

Summarises the committed table ``results/b_vascular_sproutangio_multi.csv`` (no
raw data needed). The fluorostats volume fraction, in percent, is compared
between every pair of VEGF groups by the library's pairwise grid
(``stats.compute_pvalues``: two-sided Mann-Whitney U, Cliff's delta, BH-FDR q
across the contrasts), plus the bootstrap CI (5000 resamples, seed 0) of the
median fold change group_b / group_a (``stats.bootstrap_fold_change_ci``).
"""
from __future__ import annotations

import pandas as pd

from common import RESULTS, write_table
from fluorostats.stats import compute_pvalues
from fluorostats.stats import bootstrap_fold_change_ci

N_BOOT = 5000
SEED = 0


def main():
    sa = pd.read_csv(RESULTS / "b_vascular_sproutangio_multi.csv")
    pct = pd.DataFrame({"condition": sa.group, "vf_pct": sa.fluorostats * 100})   # % VF
    rows = []
    for _, r in compute_pvalues(pct, ["vf_pct"]).iterrows():
        a = pct.vf_pct[pct.condition == r.condition_a].values
        b = pct.vf_pct[pct.condition == r.condition_b].values
        fc = bootstrap_fold_change_ci(a, b, n_boot=N_BOOT, seed=SEED)
        rows.append({"group_a": r.condition_a, "group_b": r.condition_b, "n_a": r.n_a, "n_b": r.n_b,
                     "mwu_p": r.p_value, "bh_q": r.q_value, "cliffs_delta": r.cliffs_delta,
                     "fold_change_median": fc["fold_change_median"],
                     "fc_ci_low": fc["ci_low"], "fc_ci_high": fc["ci_high"]})
    df = pd.DataFrame(rows)
    write_table(df, "b_vascular_sproutangio_stats", {"n_boot": N_BOOT, "fold_change_seed": SEED})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
