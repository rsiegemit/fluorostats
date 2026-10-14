"""Produce results/b_viability_external_agreement.csv — each viability estimator vs exact GT (Kerkhoff set).

Summarises the committed per-image table ``results/b_viability_external.csv``
(no raw data needed): for every estimator column, Lin's concordance correlation
(``agreement.lins_ccc``) and the mean absolute error against ``true_viability``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS
from fluorostats.agreement import lins_ccc


def main():
    ext = pd.read_csv(RESULTS / "b_viability_external.csv")
    tv = ext.true_viability.values
    rows = [{"estimator": c, "n_images": len(ext), "lins_ccc": lins_ccc(ext[c].values, tv),
             "MAE": float(np.abs(ext[c].values - tv).mean())}
            for c in ext.columns if c not in ("file", "true_viability")]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b_viability_external_agreement.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
