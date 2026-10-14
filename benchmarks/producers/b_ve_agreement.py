"""Produce results/b_ve_agreement.csv — rank agreement of fluorostats and VesselExpress volume fraction.

Summarises the committed per-volume table ``results/b_ve_metrics.csv`` (no raw
data needed): Spearman rank correlation of the two volume fractions over the
light-sheet volumes.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats as sps

from common import RESULTS


def main():
    m = pd.read_csv(RESULTS / "b_ve_metrics.csv")
    rho = sps.spearmanr(m.VesselExpress_VF.values, m.fluorostats_VF.values).statistic
    df = pd.DataFrame([{"n_volumes": len(m), "spearman_rho": rho}])
    df.to_csv(RESULTS / "b_ve_agreement.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
