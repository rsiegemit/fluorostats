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

    fluorostats viability manifest.json [--output DIR]
    # or
    from fluorostats import viability_batch
    viability_batch.run(Path("manifest.json"))

Manifest schema (all fields except groups/stacks optional)::

    {
      "title": "...",
      "live_channel": 0,          # channel index of the live (e.g. Calcein/AF488) signal
      "dead_channel": 1,          # channel index of the dead (e.g. PI) signal
      "seg": {"method": "otsu", "min_size": 32},   # segment.binarize kwargs: method,
                                  # min_size, threshold_scale. Cell counts always
                                  # threshold with otsu (recorded in run_config.json).
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
from . import stats
from .manifest import check_groups
from . import report
from . import segment
from . import viability as viab

DEFAULT_SEG = {"method": viab.DEFAULT_SEG_METHOD, "min_size": viab.AREA_MIN_SIZE}
# Viability runs analyse every stack; there is no auxiliary stack role.
_ROLES = ("primary",)


def load_manifest(path: Path) -> dict:
    """Read a viability manifest (see the module docstring), filling defaults."""
    m = json.loads(Path(path).read_text())
    if "groups" not in m or not m["groups"]:
        raise ValueError("manifest needs a non-empty 'groups' list")
    m.setdefault("live_channel", 0)
    m.setdefault("dead_channel", 1)
    m.setdefault("seg", dict(DEFAULT_SEG))
    m.setdefault("count_method", "all")
    m.setdefault("output_dir", "viability_out")
    _validate(m)
    return m


def _validate(m: dict) -> None:
    """Check a viability manifest before any image is loaded; raise listing every problem."""
    problems = []
    for key in ("live_channel", "dead_channel"):
        if not isinstance(m[key], int) or isinstance(m[key], bool) or m[key] < 0:
            problems.append(f"{key} must be a non-negative integer, got {m[key]!r}")
    if m["live_channel"] == m["dead_channel"]:
        problems.append(f"live_channel and dead_channel are both {m['live_channel']}")
    methods = ("all", "auto", *viab.COUNT_METHODS)
    if m["count_method"] not in methods:
        problems.append(f"count_method must be one of {methods}, got {m['count_method']!r}")
    problems += _check_seg(m["seg"])
    problems += check_groups(m["groups"], roles=_ROLES)
    if problems:
        raise ValueError("invalid viability manifest:\n  - " + "\n  - ".join(problems))


def _check_seg(seg) -> list[str]:
    """Problems with the manifest's ``seg`` block (keyword arguments of `segment.binarize`)."""
    if not isinstance(seg, dict):
        return [f"seg must be an object of segmentation settings, got {seg!r}"]
    problems = []
    unknown = sorted(set(seg) - {"method", "min_size", "threshold_scale"})
    if unknown:
        problems.append(f"seg has unknown keys {unknown} (use method, min_size, threshold_scale)")
    if "method" in seg and seg["method"] not in segment.METHODS:
        problems.append(f"seg.method must be one of {segment.METHODS}, got {seg['method']!r}")
    ms = seg.get("min_size", 0)
    if not isinstance(ms, int) or isinstance(ms, bool) or ms < 0:
        problems.append(f"seg.min_size must be a non-negative integer, got {ms!r}")
    ts = seg.get("threshold_scale", 1.0)
    if not isinstance(ts, (int, float)) or isinstance(ts, bool) or ts <= 0:
        problems.append(f"seg.threshold_scale must be a positive number, got {ts!r}")
    return problems


def _counts(by_method: dict, m: str) -> tuple[float, float, float]:
    """(n_live, n_dead, viability) for one method from a live_dead_by_count 'all' result."""
    d = by_method.get(m, {})
    return d.get("n_live", float("nan")), d.get("n_dead", float("nan")), d.get("viability", float("nan"))


def analyse_stack(path: Path, cfg: dict) -> dict:
    """All viability readouts for one two-channel stack."""
    vol, meta = load_volume(Path(path))                 # (C, Z, Y, X)
    if vol.ndim != 4:  # pragma: no cover — load_volume always canonicalises to (C,Z,Y,X)
        raise ValueError(f"{path.name}: expected 4D (C,Z,Y,X), got shape {vol.shape}")
    live = vol[cfg["live_channel"]].astype(np.float32)   # (Z, Y, X)
    dead = vol[cfg["dead_channel"]].astype(np.float32)
    seg = cfg["seg"]

    # surface culture -> analyse the maximum-intensity projection (quasi-monolayer)
    live_mip, dead_mip = live.max(0), dead.max(0)
    nz, ny, nx = live.shape
    py, px = (float(v) for v in meta.get("voxel_size_um", (1.0, 1.0, 1.0))[-2:])  # µm/pixel
    fov_mm2 = (px * nx) * (py * ny) / 1e6

    # live_dead_by_count's `method` is the COUNT mode; the threshold goes as `threshold`
    seg_count = {k: v for k, v in seg.items() if k != "method"}
    area = viab.live_dead_fractions(live_mip, dead_mip, **seg)     # area / coverage
    counts = viab.live_dead_by_count(live_mip, dead_mip, method=cfg["count_method"],
                                     threshold=seg.get("method", viab.DEFAULT_SEG_METHOD),
                                     **seg_count)
    regime = viab.choose_count_method(live_mip, **seg)             # transparent regime
    bias = viab.viability_2d_vs_3d(live, **seg)                    # MIP vs full 3D

    row = {
        "px_um": px, "voxel_size_source": meta.get("voxel_size_source", ""), "n_z": nz, "n_xy": nx, "fov_mm2": fov_mm2,
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
        for m in viab.COUNT_METHODS:
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
    """Mean, SEM and n of the non-NaN values (`stats.describe`: SEM NaN for n < 2)."""
    d = stats.describe(vals)
    return d["mean"], d["sem"], d["n"]


def run(manifest_path: Path, output_override: str | None = None) -> Path:
    """Analyse every stack in a manifest; write per-stack and group-summary CSVs."""
    m = load_manifest(Path(manifest_path))
    out = Path(output_override or m["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    count_seg = m["seg"].get("method", viab.DEFAULT_SEG_METHOD)   # counts use seg.method too
    report.write_run_config(out, {"manifest": Path(manifest_path), "count_seg_method": count_seg},
                            nested={"resolved": m})
    cfg = {k: m[k] for k in ("live_channel", "dead_channel", "seg", "count_method")}

    rows: list[dict] = []
    for grp in m["groups"]:
        gname = grp["name"]
        for st in grp["stacks"]:
            print(f"  [{gname}] {st['label']}")
            r = analyse_stack(Path(st["path"]), cfg)
            rows.append({"group": gname, "stack": st["label"], **r})

    # per-stack CSV (load_manifest guarantees every group has stacks)
    fields = list(rows[0].keys())
    with open(out / "viability_per_stack.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # group summary CSV (mean +/- SEM for the numeric metrics)
    metrics = [k for k in fields
               if k not in ("group", "stack", "voxel_size_source", "regime_method")]
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
    print("\n=== Group means (mean +/- SEM) ===")
    key_metrics = ["live_cov_pct", "dead_cov_pct", "viability_area",
                   "viability_count_consensus", "live_density_permm2", "overestimate_mip"]
    for mk in key_metrics:
        parts = []
        for g in groups:
            mean, sem, n = _mean_sem([r[mk] for r in rows if r["group"] == g])
            parts.append(f"{g}: {mean:.3g}±{sem:.2g} (n={n})")
        print(f"  {mk:<26} " + "   ".join(parts))
    _print_contrast(rows, groups)


def _print_contrast(rows: list[dict], groups: list[str]) -> None:
    """Two-group Mann-Whitney U with Cliff's delta, BH-FDR across the printed tests."""
    if len(groups) != 2:
        print(f"\n(two-group contrast skipped: {len(groups)} group(s); it compares exactly two)")
        return
    g0, g1 = groups
    tests = []
    for mk in ("live_cov_pct", "live_density_permm2", "viability_area"):
        a = [r[mk] for r in rows if r["group"] == g0 and r[mk] == r[mk]]
        b = [r[mk] for r in rows if r["group"] == g1 and r[mk] == r[mk]]
        if len(a) >= 1 and len(b) >= 1:
            fold = (np.mean(b) / np.mean(a)) if np.mean(a) else float("nan")
            tests.append((mk, stats.mann_whitney(a, b), fold))
    q = stats.bh_fdr(np.array([mw["p"] for _, mw, _ in tests]))
    print("\n=== Two-group contrast (Mann-Whitney U, two-sided; Cliff's delta; "
          "BH-FDR q across these tests) ===")
    for (mk, mw, fold), qv in zip(tests, q):
        print(f"  {mk:<26} {g0} vs {g1}: U={mw['u']:.0f}, p={mw['p']:.4g}, q={qv:.4g}, "
              f"delta={mw['cliffs_delta']:+.2f}, fold={fold:.2f}x (n={mw['n_x']},{mw['n_y']})")
