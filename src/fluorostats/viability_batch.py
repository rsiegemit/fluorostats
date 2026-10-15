"""Batch Live/Dead viability analysis for two-channel confocal stacks — manifest driven.

Turns a set of Live/Dead z-stacks (grouped by condition, e.g. culture day) into
the quantitative viability readouts the library exposes, from one JSON config:

  - **area / coverage** (``viability.live_dead_fractions`` on the intensity MIP:
    each channel is max-projected over z, then segmented): live and dead area
    fraction and area-based viability;
  - **counts** (``viability.live_dead_by_count``, ``method="all"`` by default):
    connected-component / watershed / maxima cell counts, their consensus viability
    and spread (disagreement flags a crowded regime the counting choice matters in).
    Every ``count_method`` writes the same columns (NaN / empty for what it did not
    compute; the spread is NaN when only one method ran);
  - **regime** (``viability.choose_count_method``): SNR + crowding + the method
    the heuristic would auto-pick, so the choice is transparent;
  - **2D-vs-3D bias** (``viability.viability_2d_vs_3d``): how much a MIP overstates
    coverage relative to the full volume. Its MIP is the projection of the 3D
    MASK (the stack is segmented in 3D, then max-projected), so ``live_frac_mip``
    is not ``live_cov_pct / 100``, which segments the intensity MIP. Both are kept;
    `MIP_COLUMNS` (also written to run_config.json) states which column is which.

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
      "dead_channel": 1,          # channel index of the dead (e.g. PI) signal; when
                                  # omitted, each is found from the file's channel
                                  # names (viability.live_dead_channels), else 0 / 1
                                  # with a warning. The names used are written to
                                  # live_channel_name / dead_channel_name.
      "seg": {"method": "otsu", "min_size": 32},   # segment.binarize kwargs: method,
                                  # min_size, threshold_scale (see "Segmentation").
      "count_method": "all",      # live_dead_by_count method: all | cc | watershed | maxima | auto
      "output_dir": "viability_out",
      "groups": [
        {"name": "Day 1", "stacks": [{"label": "Day1 #1", "path": "/abs/f.oib"}, ...]},
        {"name": "Day 5", "stacks": [...]}
      ]
    }

Stack labels must be unique within a group; a group ``background`` is rejected
(viability runs subtract no background).

Segmentation: ``seg.method`` thresholds everything, the cell counts included
(``count_seg_method`` in run_config.json). ``min_size`` differs by readout:

  - area / coverage and the 2D-vs-3D bias use ``seg.min_size``, or
    `viability.AREA_MIN_SIZE` when the ``seg`` block gives none;
  - the counts use ``seg.min_size`` when given (so the default ``seg`` block's 32
    applies to them too), else `viability.COUNT_MIN_SIZE`; the ``cc`` count then
    also drops labelled objects below `viability.COUNT_MIN_SIZE`;
  - the regime estimate (``regime_*`` columns, and ``auto``'s choice) always uses
    `viability.COUNT_MIN_SIZE`, whatever ``seg.min_size`` is.
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
# The two maximum-intensity-projection definitions in the per-stack table.
MIP_COLUMNS = {
    "live_cov_pct": "percent of the intensity MIP segmented live: each channel is "
                    "max-projected over z, then thresholded (likewise dead_cov_pct, "
                    "viability_area and the cell counts)",
    "live_frac_mip": "fraction of the MIP of the 3D live mask: the stack is "
                     "thresholded in 3D, then the mask is max-projected "
                     "(viability.viability_2d_vs_3d; overestimate_mip = live_frac_mip "
                     "/ live_frac_3d)",
}
# Viability runs analyse every stack; there is no auxiliary stack role.
_ROLES = ("primary",)


def load_manifest(path: Path) -> dict:
    """Read a viability manifest (see the module docstring), filling defaults."""
    m = json.loads(Path(path).read_text())
    if "groups" not in m or not m["groups"]:
        raise ValueError("manifest needs a non-empty 'groups' list")
    m.setdefault("live_channel", None)         # None: resolved from channel names
    m.setdefault("dead_channel", None)
    m.setdefault("seg", dict(DEFAULT_SEG))
    m.setdefault("count_method", "all")
    m.setdefault("output_dir", "viability_out")
    _validate(m)
    return m


def _validate(m: dict) -> None:
    """Check a viability manifest before any image is loaded; raise listing every problem."""
    problems = []
    for key in ("live_channel", "dead_channel"):
        v = m[key]
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 0):
            problems.append(f"{key} must be a non-negative integer, got {v!r}")
    if m["live_channel"] is not None and m["live_channel"] == m["dead_channel"]:
        problems.append(f"live_channel and dead_channel are both {m['live_channel']}")
    methods = ("all", "auto", *viab.COUNT_METHODS)
    if m["count_method"] not in methods:
        problems.append(f"count_method must be one of {methods}, got {m['count_method']!r}")
    problems += _check_seg(m["seg"])
    problems += check_groups(m["groups"], roles=_ROLES)
    for grp in m["groups"]:
        name = grp.get("name")
        if "background" in grp:
            problems.append(f"group {name!r}: 'background' is not used by viability runs")
        labels = [st.get("label") for st in grp.get("stacks", [])]
        for lab in sorted({lb for lb in labels if lb is not None and labels.count(lb) > 1}):
            problems.append(f"group {name!r}: label {lab!r} is used by more than one stack")
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


# `live_dead_by_count` keys written under another column name; every other key is
# written as is, except ``viability`` (equals ``consensus``, not repeated) and the
# deprecated nested ``by_method`` (its values are the per-method keys).
_COUNT_COLUMNS = {"consensus": "viability_count_consensus",
                  "spread": "viability_count_spread",
                  "method_live": "count_method_live", "method_dead": "count_method_dead",
                  "reason_live": "count_reason_live", "reason_dead": "count_reason_dead"}


def analyse_stack(path: Path, cfg: dict) -> dict:
    """All viability readouts for one two-channel stack."""
    vol, meta = load_volume(Path(path))                 # (C, Z, Y, X)
    if vol.ndim != 4:  # pragma: no cover — load_volume always canonicalises to (C,Z,Y,X)
        raise ValueError(f"{path.name}: expected 4D (C,Z,Y,X), got shape {vol.shape}")
    names = meta["channel_names"]
    li, di = viab.live_dead_channels(names, cfg["live_channel"], cfg["dead_channel"])
    live = vol[li].astype(np.float32)   # (Z, Y, X)
    dead = vol[di].astype(np.float32)
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
        "live_channel_name": names[li], "dead_channel_name": names[di],
        "px_um": px, "voxel_size_source": meta.get("voxel_size_source", ""), "n_z": nz, "n_xy": nx, "fov_mm2": fov_mm2,
        # area / coverage on the intensity MIP (segment the projection; MIP_COLUMNS)
        "live_cov_pct": 100.0 * area["live_fraction"],
        "dead_cov_pct": 100.0 * area["dead_fraction"],
        "viability_area": area["viability"],
        # 2D vs full-3D coverage bias; this MIP projects the 3D mask (MIP_COLUMNS)
        "live_frac_mip": bias["live_fraction_mip"],
        "live_frac_3d": bias["live_fraction_3d"],
        "overestimate_mip": bias["overestimate_mip"],
        # regime detection (live channel)
        "regime_snr": regime["snr"],
        "regime_crowding": regime["crowding"],
        "regime_method": regime["method"],
    }
    row.update({_COUNT_COLUMNS.get(k, k): v for k, v in counts.items()
                if k not in ("viability", "by_method")})
    # density from the single count, or (method "all": no single count) the CC count
    n_live_ref = counts["n_live"] if counts["n_live"] == counts["n_live"] else counts["n_live_cc"]
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
                            nested={"resolved": m, "mip_columns": MIP_COLUMNS})
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
    metrics = [k for k in fields if k not in ("group", "stack")
               and not isinstance(rows[0][k], str)]      # numeric columns only
    groups = [g["name"] for g in m["groups"]]
    with open(out / "viability_group_summary.csv", "w", newline="") as f:
        sw = csv.writer(f)
        sw.writerow(["group", "n", "metric", "mean", "sem"])
        for g in groups:
            gr = [r for r in rows if r["group"] == g]
            for mk in metrics:
                mean, sem, n = _mean_sem([r[mk] for r in gr])
                sw.writerow([g, n, mk, f"{mean:.5g}", f"{sem:.5g}"])

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
