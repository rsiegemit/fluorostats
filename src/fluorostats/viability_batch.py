"""Batch Live/Dead viability analysis for two-channel confocal stacks — manifest driven.

Turns a set of Live/Dead z-stacks (grouped by condition, e.g. culture day) into
the quantitative viability readouts the library exposes, from one JSON config:

  - **area / coverage** (``viability.live_dead_fractions`` on the surface MIP):
    live and dead area fraction and area-based viability;
  - **counts** (``viability.live_dead_by_count``, ``method="all"``): connected-
    component / watershed / maxima cell counts, their consensus viability and
    spread (disagreement flags a crowded regime the counting choice matters in);
  - **regime** (``viability.choose_count_method``): SNR + crowding + the method
    the heuristic would auto-pick, so the choice is transparent;
  - **2D-vs-3D bias** (``viability.viability_2d_vs_3d``): how much a MIP overstates
    coverage relative to the full volume.

Everything is scale-aware: cells are counted on the surface maximum-intensity
projection (surface culture is a quasi-monolayer), and a per-FOV cell density is
reported in cells/mm^2 using the stack's own pixel calibration, so 800^2 and
1024^2 acquisitions of the same physical field are directly comparable.

Usage::

    fluorostats viability manifest.json [--output-dir DIR]
    # or
    from fluorostats import viability_batch
    viability_batch.run(Path("manifest.json"))

Manifest schema (all fields except groups/stacks optional)::

    {
      "title": "...",
      "live_channel": 0,          # channel index of the live (e.g. Calcein/AF488) signal
      "dead_channel": 1,          # channel index of the dead (e.g. PI) signal
      "seg": {"method": "otsu", "min_size": 32},   # segmentation kwargs (segment.binarize)
      "count_method": "all",      # live_dead_by_count method: all | cc | watershed | maxima | auto
      "output_dir": "viability_out",
      "groups": [
        {"name": "Day 1", "stacks": [{"label": "Day1 #1", "path": "/abs/f.oib"}, ...]},
        {"name": "Day 5", "stacks": [...]}
      ]
    }
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .io import load_volume
from . import viability as viab

DEFAULT_SEG = {"method": "otsu", "min_size": 32}


def load_manifest(path: Path) -> dict:
    m = json.loads(Path(path).read_text())
    if "groups" not in m or not m["groups"]:
        raise ValueError("manifest needs a non-empty 'groups' list")
    m.setdefault("live_channel", 0)
    m.setdefault("dead_channel", 1)
    m.setdefault("seg", dict(DEFAULT_SEG))
    m.setdefault("count_method", "all")
    m.setdefault("output_dir", "viability_out")
    return m


def _counts(by_method: dict, m: str) -> tuple[float, float, float]:
    """(n_live, n_dead, viability) for one method from a live_dead_by_count 'all' result."""
    d = by_method.get(m, {})
    return d.get("n_live", float("nan")), d.get("n_dead", float("nan")), d.get("viability", float("nan"))


def analyse_stack(path: Path, cfg: dict) -> dict:
    """All viability readouts for one two-channel stack."""
    vol, meta = load_volume(Path(path))                 # (C, Z, Y, X)
    if vol.ndim != 4:
        raise ValueError(f"{path.name}: expected 4D (C,Z,Y,X), got shape {vol.shape}")
    live = vol[cfg["live_channel"]].astype(np.float32)   # (Z, Y, X)
    dead = vol[cfg["dead_channel"]].astype(np.float32)
    seg = cfg["seg"]

    # surface culture -> analyse the maximum-intensity projection (quasi-monolayer)
    live_mip, dead_mip = live.max(0), dead.max(0)
    nz, ny, nx = live.shape
    px = float(meta.get("voxel_size_um", (1.0, 1.0, 1.0))[-1])   # xy micrometres/pixel
    fov_mm2 = (px * nx) * (px * ny) / 1e6

    # live_dead_by_count reserves `method` for the COUNT mode, so the segmentation
    # method (e.g. otsu, its default in binarize) is passed via the other seg keys only.
    seg_count = {k: v for k, v in seg.items() if k != "method"}
    area = viab.live_dead_fractions(live_mip, dead_mip, **seg)     # area / coverage
    counts = viab.live_dead_by_count(live_mip, dead_mip, method=cfg["count_method"], **seg_count)
    regime = viab.choose_count_method(live_mip, **seg)             # transparent regime
    bias = viab.viability_2d_vs_3d(live, **seg)                    # MIP vs full 3D

    row = {
        "px_um": px, "n_z": nz, "n_xy": nx, "fov_mm2": fov_mm2,
        # area / coverage on the surface MIP
        "live_cov_pct": 100.0 * area["live_fraction"],
        "dead_cov_pct": 100.0 * area["dead_fraction"],
        "viability_area": area["viability"],
        # 2D (MIP) vs full-3D coverage bias
        "live_frac_mip": bias["live_fraction_mip"],
        "live_frac_3d": bias["live_fraction_3d"],
        "overestimate_mip": bias["overestimate_mip"],
        # regime detection (live channel)
        "regime_snr": regime["snr"],
        "regime_crowding": regime["crowding"],
        "regime_method": regime["method"],
    }
    if cfg["count_method"] == "all":
        bm = counts["by_method"]
        for m in ("cc", "watershed", "maxima"):
            nl, nd, v = _counts(bm, m)
            row[f"n_live_{m}"] = nl
            row[f"n_dead_{m}"] = nd
            row[f"viability_{m}"] = v
        row["viability_count_consensus"] = counts["consensus"]
        row["viability_count_spread"] = counts["spread"]
        n_live_ref = bm.get("cc", {}).get("n_live", float("nan"))   # CC = robust count
    else:
        row["n_live"] = counts["n_live"]
        row["n_dead"] = counts["n_dead"]
        row["viability_count_consensus"] = counts["viability"]
        row["viability_count_spread"] = 0.0
        n_live_ref = counts["n_live"]
    row["live_density_permm2"] = n_live_ref / fov_mm2 if fov_mm2 > 0 else float("nan")
    return row


def _mean_sem(vals: list[float]) -> tuple[float, float, int]:
    a = np.array([v for v in vals if v == v], float)      # drop NaN
    if a.size == 0:
        return float("nan"), float("nan"), 0
    sem = a.std(ddof=1) / a.size ** 0.5 if a.size > 1 else 0.0
    return float(a.mean()), float(sem), int(a.size)


def run(manifest_path: Path, output_override: str | None = None) -> Path:
    m = load_manifest(Path(manifest_path))
    out = Path(output_override or m["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cfg = {k: m[k] for k in ("live_channel", "dead_channel", "seg", "count_method")}

    rows: list[dict] = []
    for grp in m["groups"]:
        gname = grp["name"]
        for st in grp["stacks"]:
            print(f"  [{gname}] {st['label']}")
            r = analyse_stack(Path(st["path"]), cfg)
            rows.append({"group": gname, "stack": st["label"], **r})

    # per-stack CSV
    if not rows:
        raise ValueError("no stacks analysed — every group's 'stacks' list is empty")
    fields = list(rows[0].keys())
    with open(out / "viability_per_stack.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # group summary CSV (mean +/- SEM for the numeric metrics)
    metrics = [k for k in fields if k not in ("group", "stack", "regime_method")]
    groups = [g["name"] for g in m["groups"]]
    with open(out / "viability_group_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "metric", "mean", "sem"])
        for g in groups:
            gr = [r for r in rows if r["group"] == g]
            for mk in metrics:
                mean, sem, n = _mean_sem([r[mk] for r in gr])
                w.writerow([g, n, mk, f"{mean:.5g}", f"{sem:.5g}"])

    _print_summary(rows, groups)
    print(f"\nOutputs written to {out}/")
    return out


def _print_summary(rows: list[dict], groups: list[str]) -> None:
    from scipy import stats
    print("\n=== Group means (mean +/- SEM) ===")
    key_metrics = ["live_cov_pct", "dead_cov_pct", "viability_area",
                   "viability_count_consensus", "live_density_permm2", "overestimate_mip"]
    for mk in key_metrics:
        parts = []
        for g in groups:
            mean, sem, n = _mean_sem([r[mk] for r in rows if r["group"] == g])
            parts.append(f"{g}: {mean:.3g}±{sem:.2g} (n={n})")
        print(f"  {mk:<26} " + "   ".join(parts))
    # two-group contrast (Mann-Whitney U — small n, non-normal safe) on coverage & density
    if len(groups) == 2:
        print("\n=== Day contrast (Mann-Whitney U, two-sided) ===")
        for mk in ("live_cov_pct", "live_density_permm2", "viability_area"):
            a = [r[mk] for r in rows if r["group"] == groups[0] and r[mk] == r[mk]]
            b = [r[mk] for r in rows if r["group"] == groups[1] and r[mk] == r[mk]]
            if len(a) >= 1 and len(b) >= 1:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                fold = (np.mean(b) / np.mean(a)) if np.mean(a) else float("nan")
                print(f"  {mk:<26} {groups[0]} vs {groups[1]}: U={u:.0f}, p={p:.4g}, "
                      f"fold={fold:.2f}x")
