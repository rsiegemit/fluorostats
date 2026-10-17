"""Produce results/fig6_*.csv — the Figure 6 tables, from the two batch drivers' outputs.

Figure 6 rests on laboratory confocal stacks that are not deposited. The tables
come from ``fluorostats depth`` (FITC-dextran penetration time course) and
``fluorostats viability`` (HUVEC/GelMA Live/Dead) run on them; this producer
copies the drivers' own CSVs under ``fig6_*`` names and derives
``fig6_depth_lambda_timecourse`` from the per-stack λ fits:

- time points with fitted stacks: the mean, SEM and mean R² of λ over the
  primary stacks whose fit passed the range guard (``lambda_fit_ok``);
- time points with none (λ beyond the acquired depth): the unguarded λ of the
  first stack, i.e. the same exponential fit with the range guard not applied,
  reported to show what an unguarded fit would claim.

Each table's .run.json carries the driver's run record: tool, fluorostats commit
and package versions, and the analysis settings (data paths omitted).
Inputs: the driver output folders in ``$FIG6_DEPTH_OUT`` and ``$FIG6_VIABILITY_OUT``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import require, write_table
from fluorostats import depth

TIME_H = {"4h": 4, "12h": 12, "24h": 24, "72h": 72}
DEPTH_TABLES = {"depth_profiles_long": "fig6_depth_profiles_long",
                "auc_per_stack": "fig6_depth_auc_per_stack",
                "group_depth_summary": "fig6_depth_group_summary"}
VIABILITY_TABLES = {"viability_per_stack": "fig6_viability_per_stack",
                    "viability_group_summary": "fig6_viability_group_summary"}


def driver_record(folder: Path) -> dict:
    """The driver's run_config.json without data paths."""
    cfg = json.loads(require(folder / "run_config.json").read_text(encoding="utf-8"))
    resolved = {k: v for k, v in cfg.get("resolved", {}).items() if k not in ("groups", "output_dir")}
    groups = cfg.get("resolved", {}).get("groups", [])
    return {"tool": cfg["tool"], "status": cfg.get("status"), "settings": resolved,
            "groups": {g["name"]: len(g["stacks"]) for g in groups},
            "environment": cfg.get("_environment", {})}


def unguarded_lambda(profile: pd.DataFrame) -> tuple[float, float]:
    """λ and R² of the exponential fit with the range guard not applied."""
    z, y = profile.depth_um.to_numpy(float), profile.bg_subtracted.to_numpy(float)
    i0, lam = depth._initial_guess(z, y, len(z))   # the fit fit_penetration_depth runs
    model, _, lam_fit, _ = depth._fit_exponential(z, y, False, i0, lam)
    return float(lam_fit), float(depth._fit_quality(y, model)[0])


def lambda_timecourse(folder: Path) -> pd.DataFrame:
    auc = pd.read_csv(folder / "auc_per_stack.csv")
    prof = pd.read_csv(folder / "depth_profiles_long.csv")
    rows = []
    for g, t in TIME_H.items():
        a = auc[(auc.group == g) & (auc.role.fillna("primary") == "primary")]
        ok = a[a.lambda_fit_ok.astype(bool)]
        row = {"group": g, "time_h": t, "n_ok": len(ok)}
        if len(ok):
            row.update(guarded_lambda_mean=ok.lambda_um.mean(),
                       guarded_lambda_sem=ok.lambda_um.std(ddof=1) / np.sqrt(len(ok)),
                       guarded_r2_mean=ok.lambda_r_squared.mean())
        else:
            first = f"{g} #1"
            lam, r2 = unguarded_lambda(prof[(prof.group == g) & (prof["stack"] == first)])
            row.update(naive_repr_lambda=round(lam, 1), naive_repr_r2=round(r2, 4), naive_repr_stack=first)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    dep = require(Path(os.environ["FIG6_DEPTH_OUT"]))
    via = require(Path(os.environ["FIG6_VIABILITY_OUT"]))
    for folder, tables in ((dep, DEPTH_TABLES), (via, VIABILITY_TABLES)):
        rec = driver_record(folder)
        for src, stem in tables.items():
            write_table(pd.read_csv(require(folder / f"{src}.csv")), stem, {"driver": rec, "source": f"{src}.csv"})
    write_table(lambda_timecourse(dep), "fig6_depth_lambda_timecourse",
                {"driver": driver_record(dep), "derived_from": ["auc_per_stack.csv", "depth_profiles_long.csv"],
                 "rule": "guarded: mean/SEM of lambda over fit_ok primary stacks; else unguarded fit of stack #1"})
    print("wrote the six fig6_* tables")


if __name__ == "__main__":
    main()
