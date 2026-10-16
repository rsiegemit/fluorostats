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
    is not ``live_cov_frac``, which segments the intensity MIP. Both are kept;
    `MIP_COLUMNS` (also written to run_config.json) states which column is which.

Coverage is a fraction in [0, 1] (``live_cov_frac``, ``dead_cov_frac``).

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
                                  # min_size, threshold_scale, opening (see
                                  # "Segmentation").
      "count_method": "all",      # live_dead_by_count method: all | cc | watershed | maxima | auto
      "output_dir": "viability_out",
      "groups": [
        {"name": "Day 1", "stacks": [{"label": "Day1 #1", "path": "/abs/f.oib"}, ...]},
        {"name": "Day 5", "stacks": [...]}
      ]
    }

Validation is strict, as for depth manifests: any field not shown above (``role``
apart, whose only value is "primary") is refused, a group ``background`` included
(viability runs subtract none), and stack labels must be unique within a group.

Outputs: ``viability_per_stack.csv``, ``viability_group_summary.csv`` (mean and SEM
per group), ``contrasts.csv`` (the printed two-group statistics, in the columns of
`depth_batch`'s contrasts.csv) and, written last, ``run_config.json`` (the manifest
with its paths resolved, plus ``files_read``: the absolute paths of every stack read
and of the output folder). The output folder is created only once every stack has
been analysed, with an "incomplete" ``run_config.json`` marker that the full record
replaces. Text files are UTF-8. A run
first removes those files left by an earlier viability run in the output folder; it
refuses a folder whose run_config.json names another tool, or that holds those
names with no run_config.json, unless ``overwrite`` is set.

Relative paths (stack ``path``, ``output_dir`` and its default) resolve against the folder that
holds the manifest file, as in `depth_batch` (`manifest.resolve_path`);
``--output`` / ``output_override`` stays relative to the working directory.

Segmentation: ``seg.method`` thresholds everything, the cell counts included
(``count_seg_method`` in run_config.json). ``min_size`` differs by readout:

  - area / coverage and the 2D-vs-3D bias use ``seg.min_size``, or
    `viability.AREA_MIN_SIZE` when the ``seg`` block gives none;
  - the counts use ``seg.min_size`` when given (so the default ``seg`` block's 32
    applies to them too), else `viability.COUNT_MIN_SIZE`; the ``cc`` count then
    also drops labelled objects below `viability.COUNT_MIN_SIZE`;
  - the regime estimate (``regime_*`` columns, and ``auto``'s choice) always uses
    `viability.COUNT_MIN_SIZE`, whatever ``seg.min_size`` is.

``seg.opening`` (default ``"ball"``) is the salt-noise opening of every mask. Only
the 3D mask of the 2D-vs-3D bias depends on it beyond 2D (``"ball"`` and ``"disk"``
are the same on the projections): ``"ball"`` erases cells thinner than 3 z-slices,
so for a monolayer at a coarse z-step set ``"disk"``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .io import load_volume
from . import stats
from .depth_contrasts import mann_whitney_contrasts, write_contrasts_csv
from .manifest import (
    check_groups,
    is_int,
    is_number,
    not_strings,
    require_groups,
    resolve_paths,
    unknown_keys,
)
from . import report
from . import segment
from . import viability as viab

DEFAULT_SEG = {"method": viab.DEFAULT_SEG_METHOD, "min_size": viab.AREA_MIN_SIZE}
# The two maximum-intensity-projection definitions in the per-stack table.
MIP_COLUMNS = {
    "live_cov_frac": "fraction of the intensity MIP segmented live: each channel is "
    "max-projected over z, then thresholded (likewise dead_cov_frac, "
    "viability_area and the cell counts)",
    "live_frac_mip": "fraction of the MIP of the 3D live mask: the stack is "
    "thresholded in 3D, then the mask is max-projected "
    "(viability.viability_2d_vs_3d; overestimate_mip = live_frac_mip "
    "/ live_frac_3d)",
}
# Viability runs analyse every stack; there is no auxiliary stack role.
_ROLES = ("primary",)
# Every field a manifest, group, stack and seg block may have (anything else is refused).
_FIELDS = ("title", "live_channel", "dead_channel", "seg", "count_method", "output_dir", "groups")
_GROUP_FIELDS = ("name", "stacks")
_STACK_FIELDS = ("label", "path", "role")
_SEG_FIELDS = ("method", "min_size", "threshold_scale", "opening")
# The metrics of the two-group contrast (console and contrasts.csv).
_CONTRAST_METRICS = ("live_cov_frac", "live_density_permm2", "viability_area")
# Every file a run writes; an earlier run's copies are removed when a run starts.
_OUTPUTS = (
    "viability_per_stack.csv",
    "viability_group_summary.csv",
    "contrasts.csv",
    "run_config.json",
)


def load_manifest(path: Path) -> dict:
    """Read a viability manifest (see the module docstring), filling defaults and
    resolving its paths against the manifest's folder (`manifest.resolve_paths`)."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    require_groups(m)
    m.setdefault("live_channel", None)  # None: resolved from channel names
    m.setdefault("dead_channel", None)
    m.setdefault("seg", dict(DEFAULT_SEG))
    m.setdefault("count_method", "all")
    m.setdefault("output_dir", "viability_out")
    m = resolve_paths(m, path)
    _validate(m)
    return m


def _validate(m: dict) -> None:
    """Check a viability manifest before any image is loaded; raise listing every problem."""
    problems = unknown_keys("manifest", m, _FIELDS) + not_strings(m, ("title", "output_dir"))
    for key in ("live_channel", "dead_channel"):
        v = m[key]
        if v is not None and (not is_int(v) or v < 0):
            problems.append(f"{key} must be a non-negative integer, got {v!r}")
    if m["live_channel"] is not None and m["live_channel"] == m["dead_channel"]:
        problems.append(f"live_channel and dead_channel are both {m['live_channel']}")
    methods = ("all", "auto", *viab.COUNT_METHODS)
    if m["count_method"] not in methods:
        problems.append(f"count_method must be one of {methods}, got {m['count_method']!r}")
    problems += _check_seg(m["seg"])
    problems += check_groups(
        m["groups"], roles=_ROLES, group_keys=_GROUP_FIELDS, stack_keys=_STACK_FIELDS
    )
    if problems:
        raise ValueError("invalid viability manifest:\n  - " + "\n  - ".join(problems))


def _check_seg(seg) -> list[str]:
    """Problems with the manifest's ``seg`` block (keyword arguments of `segment.binarize`)."""
    if not isinstance(seg, dict):
        return [f"seg must be an object of segmentation settings, got {seg!r}"]
    problems = unknown_keys("seg", seg, _SEG_FIELDS)
    if "method" in seg and seg["method"] not in segment.METHODS:
        problems.append(f"seg.method must be one of {segment.METHODS}, got {seg['method']!r}")
    ms = seg.get("min_size", 0)
    if not is_int(ms) or ms < 0:
        problems.append(f"seg.min_size must be a non-negative integer, got {ms!r}")
    if seg.get("opening", "ball") not in segment.OPENINGS:
        problems.append(
            f"seg.opening must be one of {tuple(segment.OPENINGS)}, got {seg['opening']!r}"
        )
    ts = seg.get("threshold_scale", 1.0)
    if not is_number(ts) or ts <= 0:
        problems.append(f"seg.threshold_scale must be a positive number, got {ts!r}")
    return problems


# `live_dead_by_count` keys written under another column name; every other key is
# written as is, except ``viability`` (equals ``consensus``, not repeated).
_COUNT_COLUMNS = {
    "consensus": "viability_count_consensus",
    "spread": "viability_count_spread",
    "method_live": "count_method_live",
    "method_dead": "count_method_dead",
    "reason_live": "count_reason_live",
    "reason_dead": "count_reason_dead",
}


def analyse_stack(path: Path, cfg: dict) -> dict:
    """All viability readouts for one two-channel stack."""
    vol, meta = load_volume(Path(path))  # (C, Z, Y, X)
    if vol.ndim != 4:  # pragma: no cover — load_volume always canonicalises to (C,Z,Y,X)
        raise ValueError(f"{path.name}: expected 4D (C,Z,Y,X), got shape {vol.shape}")
    names = meta["channel_names"]
    li, di = viab.live_dead_channels(names, cfg["live_channel"], cfg["dead_channel"])
    live = vol[li].astype(np.float32)  # (Z, Y, X)
    dead = vol[di].astype(np.float32)
    seg = cfg["seg"]

    # surface culture -> analyse the maximum-intensity projection (quasi-monolayer)
    live_mip, dead_mip = live.max(0), dead.max(0)
    nz, ny, nx = live.shape
    py, px = (float(v) for v in meta.get("voxel_size_um", (1.0, 1.0, 1.0))[-2:])  # µm/pixel
    fov_mm2 = (px * nx) * (py * ny) / 1e6

    # live_dead_by_count's `method` is the COUNT mode; the threshold goes as `threshold`
    seg_count = {k: v for k, v in seg.items() if k != "method"}
    area = viab.live_dead_fractions(live_mip, dead_mip, **seg)  # area / coverage
    counts = viab.live_dead_by_count(
        live_mip,
        dead_mip,
        method=cfg["count_method"],
        threshold=seg.get("method", viab.DEFAULT_SEG_METHOD),
        **seg_count,
    )
    regime = viab.choose_count_method(live_mip, **seg)  # transparent regime
    bias = viab.viability_2d_vs_3d(live, **seg)  # MIP vs full 3D

    row = {
        "live_channel_name": names[li],
        "dead_channel_name": names[di],
        "px_um": px,
        "voxel_size_source": meta.get("voxel_size_source", ""),
        "n_z": nz,
        "n_xy": nx,
        "fov_mm2": fov_mm2,
        # area / coverage on the intensity MIP (segment the projection; MIP_COLUMNS)
        "live_cov_frac": area["live_fraction"],
        "dead_cov_frac": area["dead_fraction"],
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
    row.update({_COUNT_COLUMNS.get(k, k): v for k, v in counts.items() if k != "viability"})
    # density from the single count, or (method "all": no single count) the CC count
    n_live_ref = counts["n_live"] if counts["n_live"] == counts["n_live"] else counts["n_live_cc"]
    row["live_density_permm2"] = n_live_ref / fov_mm2 if fov_mm2 > 0 else float("nan")
    return row


def run(manifest_path: Path, output_override: str | None = None, overwrite: bool = False) -> Path:
    """Analyse every stack in a manifest; write per-stack, group-summary and contrast
    CSVs, then ``run_config.json`` (last, replacing the "incomplete" marker).

    The output folder must be empty of its outputs or hold an earlier viability
    run's (`report.claim_output_dir`); ``overwrite`` replaces another run's files
    anyway."""
    m = load_manifest(Path(manifest_path))
    out = Path(output_override or m["output_dir"])
    # removing the earlier outputs first keeps a failed run from leaving an
    # earlier run's tables next to its "incomplete" marker; a completed run
    # writes every name again, so none is ever stale (no report.stale_notices)
    report.claim_output_dir(out, "viability", [out / n for n in _OUTPUTS], overwrite)
    cfg = {k: m[k] for k in ("live_channel", "dead_channel", "seg", "count_method")}

    rows: list[dict] = []
    for grp in m["groups"]:
        gname = grp["name"]
        for st in grp["stacks"]:
            print(f"  [{gname}] {st['label']}")
            r = analyse_stack(Path(st["path"]), cfg)
            rows.append({"group": gname, "stack": st["label"], **r})
    report.mark_output_dir(out, "viability")  # only now: a failed run leaves no new folder

    # per-stack CSV (load_manifest guarantees every group has stacks)
    fields = list(rows[0].keys())
    with open(out / "viability_per_stack.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    groups = [g["name"] for g in m["groups"]]
    _write_group_summary(out / "viability_group_summary.csv", rows, groups)
    tests = contrasts(rows, groups)
    write_contrasts_csv(out / "contrasts.csv", tests)

    _print_summary(rows, groups, tests)
    count_seg = m["seg"].get("method", viab.DEFAULT_SEG_METHOD)  # counts use seg.method too
    report.write_run_config(
        out,
        {"manifest": Path(manifest_path), "count_seg_method": count_seg},
        nested={"resolved": m, "mip_columns": MIP_COLUMNS, "files_read": _files_read(m, out)},
        tool="viability",
    )
    print(f"\nOutputs written to {out}/")
    return out


def _files_read(m: dict, out: Path) -> dict:
    """The absolute paths a run read and wrote (manifest paths already resolved
    against the manifest's folder), as `depth_batch` records them in ``run_config.json``."""
    return {
        "stacks": [str(Path(st["path"]).resolve()) for g in m["groups"] for st in g["stacks"]],
        "output_dir": str(out.resolve()),
    }


def _describe(rows: list[dict], group: str, metrics: list[str]) -> dict:
    """`stats.describe` of each metric over one group's stacks (arrays over ``metrics``)."""
    vals = np.array([[r[mk] for mk in metrics] for r in rows if r["group"] == group], float)
    return stats.describe(vals, axis=0)


def _write_group_summary(path: Path, rows: list[dict], groups: list[str]) -> None:
    """Mean and SEM (`stats.describe`: NaN SEM for n < 2) of every numeric column, by group."""
    metrics = [
        k for k in rows[0] if k not in ("group", "stack") and not isinstance(rows[0][k], str)
    ]  # numeric columns only
    with open(path, "w", newline="", encoding="utf-8") as f:
        sw = csv.writer(f)
        sw.writerow(["group", "n", "metric", "mean", "sem"])
        for g in groups:
            d = _describe(rows, g, metrics)
            for i, mk in enumerate(metrics):
                sw.writerow([g, int(d["n"][i]), mk, f"{d['mean'][i]:.5g}", f"{d['sem'][i]:.5g}"])


def _by_group(rows: list[dict], groups: list[str], metric: str) -> dict[str, list[float]]:
    """``metric``'s non-NaN values per group, in manifest order."""
    return {
        g: [r[metric] for r in rows if r["group"] == g and r[metric] == r[metric]] for g in groups
    }


def contrasts(rows: list[dict], groups: list[str]) -> list[dict]:
    """The two-group statistics of a run, as `depth_contrasts.mann_whitney_contrasts`
    rows (the contrasts.csv columns; ``window`` empty): two-sided Mann-Whitney with
    Cliff's delta on each of ``live_cov_frac``, ``live_density_permm2`` and
    ``viability_area`` (NaN stacks dropped; a metric with an empty group is skipped),
    BH-FDR across those metrics. None unless there are exactly two groups."""
    entries = []
    for mk in _CONTRAST_METRICS:
        bg = _by_group(rows, groups, mk)
        if len(bg) == 2 and all(bg.values()):
            entries.append((mk, "", bg))
    return mann_whitney_contrasts(entries, across="metrics")


def _print_summary(rows: list[dict], groups: list[str], tests: list[dict]) -> None:
    print("\n=== Group means (mean +/- SEM) ===")
    key_metrics = [
        "live_cov_frac",
        "dead_cov_frac",
        "viability_area",
        "viability_count_consensus",
        "live_density_permm2",
        "overestimate_mip",
    ]
    by_group = {g: _describe(rows, g, key_metrics) for g in groups}
    for i, mk in enumerate(key_metrics):
        parts = [
            f"{g}: {d['mean'][i]:.3g}±{d['sem'][i]:.2g} (n={int(d['n'][i])})"
            for g, d in by_group.items()
        ]
        print(f"  {mk:<26} " + "   ".join(parts))
    _print_contrast(rows, groups, tests)


def _print_contrast(rows: list[dict], groups: list[str], tests: list[dict]) -> None:
    """Print the `contrasts` rows (the same rows contrasts.csv holds)."""
    if len(groups) != 2:
        print(f"\n(two-group contrast skipped: {len(groups)} group(s); it compares exactly two)")
        return
    print(
        "\n=== Two-group contrast (Mann-Whitney U, two-sided; Cliff's delta; "
        "BH-FDR q across these tests) ==="
    )
    for c in tests:
        a, b = _by_group(rows, groups, c["metric"]).values()
        fold = float(np.mean(b) / np.mean(a)) if np.mean(a) else float("nan")
        print(
            f"  {c['metric']:<26} {c['group_a']} vs {c['group_b']}: U={c['U']:.0f}, "
            f"p={c['p']:.4g}, q={c['q']:.4g}, delta={c['cliffs_delta']:+.2f}, "
            f"fold={fold:.2f}x (n={c['n_a']},{c['n_b']})"
        )
