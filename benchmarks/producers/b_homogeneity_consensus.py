"""Produce results/b_homogeneity_consensus.csv and b_homogeneity_consensus_corr.csv.

Summarises the committed table ``results/b_homogeneity_multi.csv`` (no data
needed). Per synthetic field, the clustering consensus is the mean of the five
reference statistics' z-scores, each signed so that larger means more clustered
(Clark-Evans R is negated); the second table gives its Spearman correlation with
the fluorostats tile Gini over all fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from common import RESULTS, write_table

REFSIGN = {"clark_evans": -1, "ripley_L_dev": 1, "morisita": 1, "lacunarity": 1, "quadrat_var": 1}


def main():
    raw = pd.read_csv(RESULTS / "b_homogeneity_multi.csv")
    Z = np.zeros(len(raw))
    for r, s in REFSIGN.items():
        v = raw[r].to_numpy(float) * s
        Z += (v - v.mean()) / (v.std() + 1e-9)
    df = pd.DataFrame({"level": raw["level"], "gini": raw["gini"], "consensus_z": Z / len(REFSIGN)})
    write_table(df, "b_homogeneity_consensus", {"refsign": REFSIGN})
    rho = sps.spearmanr(df["consensus_z"], df["gini"].to_numpy(float)).statistic
    corr = pd.DataFrame([{"n_fields": len(df), "spearman_consensus_vs_gini": rho}])
    write_table(corr, "b_homogeneity_consensus_corr", {"refsign": REFSIGN})
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
