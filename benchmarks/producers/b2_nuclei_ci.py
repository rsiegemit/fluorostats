"""Produce results/b2_nuclei_ci.csv — mean instance F1 with a 95% bootstrap CI per method (BBBC039).

Summarises committed per-image tables (no raw data needed). Subset ``first100``:
every method column of ``b2_nuclei_methods_perimage.csv`` (first 100 sorted
images) and the StarDist / Cellpose / Omnipose per-image F1 of the same images,
matched on ``image``. Subset ``all200``: the three DL tools on all 200 images.
Images a DL table scores negative (no result) are left out. Each row is its own
percentile bootstrap of the mean (10,000 resamples, a fresh seed-0 generator).
"""
from __future__ import annotations

import pandas as pd

from common import RESULTS
from fluorostats.stats import bootstrap_mean_ci

SEED = 0
N_BOOT = 10000
DL = {"StarDist": ("stardist_eval.csv", "stardist_f1"), "Cellpose": ("cellpose_eval.csv", "cellpose_f1"),
      "Omnipose": ("omnipose_eval.csv", "thirddl_f1")}


def dl_scores() -> dict[str, pd.DataFrame]:
    """``{tool: DataFrame[image, <tool>]}`` of the scored images of each DL table."""
    out = {}
    for tool, (table, col) in DL.items():
        d = pd.read_csv(RESULTS / table)
        out[tool] = d[d[col] >= 0][["image", col]].rename(columns={col: tool})
    return out


def ci_row(subset: str, method: str, x) -> dict:
    r = bootstrap_mean_ci(x, n_boot=N_BOOT, ci=0.95, seed=SEED)
    return {"subset": subset, "method": method, "n_images": len(x),
            "mean_F1": r["mean"], "CI_low": r["ci_low"], "CI_high": r["ci_high"]}


def main():
    pi = pd.read_csv(RESULTS / "b2_nuclei_methods_perimage.csv")
    dl = dl_scores()
    rows = [ci_row("first100", c, pi[c].values) for c in pi.columns if c != "image"]
    rows += [ci_row("first100", t, pi[["image"]].merge(d, on="image")[t].values) for t, d in dl.items()]
    rows += [ci_row("all200", t, d[t].values) for t, d in dl.items()]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b2_nuclei_ci.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
