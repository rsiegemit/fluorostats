"""Batch depth-penetration analysis for confocal z-stacks — manifest driven.

Turns a set of z-stacks (grouped by condition) into the quantitative
outputs needed to compare how far a fluorescent probe penetrates a
material: background-subtracted mean intensity vs depth, surface-
normalised intensity vs depth, and area-under-curve over one or more
physical depth windows. Writes tidy CSVs (ready for Prism/replotting) and
publication figures.

This is the batch orchestration layer over :mod:`fluorostats.depth` (the
pure per-stack functions). Nothing here is specific to any one experiment:
all inputs — file paths, grouping, channel, background blanks,
normalisation slices, AUC windows — come from a JSON manifest, so the same
tool serves any probe-penetration / permeability study.

Run it from the command line::

    fluorostats depth <manifest.json> [--output-dir DIR]

or from Python::

    from fluorostats.depth_batch import run
    run("manifest.json")

Manifest schema (all fields except groups/stacks are optional)::

    {
      "title": "FITC-dextran 2000 kDa penetration",
      "signal_label": "FITC-dextran",  # probe/signal name on figure axes/titles (default "Signal")
      "channel": 0,                # channel index for 4D stacks
      "reducer": "mean",           # "mean" | "median" (per-slice spatial stat)
      "n_surface": 3,              # slices averaged for surface normalisation
      "fit_offset": false,         # true → 3-parameter exp fit with additive floor
      "auc_windows_um": [[0, 100], [0, 200], "full"],  # one or more windows
      "output_dir": "results/permeability",
      "groups": [
        {
          "name": "GelMA",
          "color": "#3F6AB3",              # optional; else fluorostats palette
          "background": "/path/blank.oib",  # optional no-fluo control
          "stacks": [
            {"label": "GelMA deep 1", "path": "/path/a.oib"},
            {"label": "GelMA deep 2", "path": "/path/b.oib"},
            {"label": "GelMA short", "path": "/path/c.oib", "role": "aux"}
          ]
        }
      ]
    }

``auc_windows_um`` accepts ``[z0, z1]`` pairs and/or the string ``"full"``
(each stack's entire acquired range); the singular ``auc_window_um`` is
still accepted for back-compat. ``role`` defaults to ``"primary"``; ``aux``
stacks are exported and drawn faintly but excluded from the group mean/SEM
curves and AUC statistics, so short or differently-acquired stacks can ride
along without biasing the matched comparison.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import depth, stats
from .depth import parse_window, window_bounds, window_label, window_title
from .depth_figures import (group_color, plot_auc, plot_depth_curves,
                            plot_retention_multiwindow)
from .io import load_volume
from .style import apply_style

# Window helpers and figures live in `depth` / `depth_figures`; re-exported here
# so ``from fluorostats.depth_batch import plot_auc, parse_window`` keeps working.
__all__ = [
    "load_manifest", "analyse_stack", "aggregate_group", "run",
    "write_profiles_csv", "write_auc_csv", "write_group_summary_csv",
    "parse_window", "window_label", "window_title", "window_bounds",
    "plot_depth_curves", "plot_auc", "plot_retention_multiwindow",
]


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    with open(path) as fh:
        m = json.load(fh)
    m.setdefault("title", "Depth-penetration analysis")
    m.setdefault("signal_label", "Signal")
    m.setdefault("channel", 0)
    m.setdefault("reducer", "mean")
    m.setdefault("n_surface", 3)
    m.setdefault("fit_offset", False)
    # Accept singular `auc_window_um` (back-compat) or plural `auc_windows_um`.
    if "auc_windows_um" not in m:
        m["auc_windows_um"] = [m.get("auc_window_um", [0.0, 100.0])]
    m["windows"] = [parse_window(w) for w in m["auc_windows_um"]]
    if "groups" not in m or not m["groups"]:
        raise ValueError("manifest must define at least one group with stacks")
    return m


# ---------------------------------------------------------------------------
# Core per-stack computation
# ---------------------------------------------------------------------------

def _load_profile(path: Path, cfg: dict) -> tuple[depth.DepthProfile, float]:
    """Load a stack and return its raw depth profile and z-step (µm)."""
    vol, meta = load_volume(path)
    vz = float(meta.get("voxel_size_um", (1.0, 1.0, 1.0))[0])
    return depth.intensity_depth_profile(vol, vz, channel=cfg["channel"],
                                         reducer=cfg["reducer"]), vz


def analyse_stack(path: Path, blank: depth.DepthProfile | None, cfg: dict) -> dict:
    """Load one stack and compute raw/subtracted/normalised profiles + AUC + penetration (λ) fit."""
    raw, vz = _load_profile(path, cfg)
    subtracted = depth.subtract_background(raw, blank) if blank is not None else raw
    normalized = depth.normalize_to_surface(subtracted, n_surface=cfg["n_surface"])

    max_depth = float(raw.depth_um[-1])
    auc = {}
    for w in cfg["windows"]:
        z0, z1 = window_bounds(w, max_depth)
        auc[window_label(w)] = {
            "absolute": depth.auc_depth(raw.depth_um, subtracted.mean, z_min=z0, z_max=z1),
            "normalized": depth.auc_depth(raw.depth_um, normalized.normalized, z_min=z0, z_max=z1),
            "covered_um": float(min(z1, max_depth) - z0),
            "width_um": float(z1 - z0),
        }

    # Penetration constant λ from a single-exponential fit of the absolute
    # (background-subtracted) profile — λ is gain-independent so it is fit on
    # the absolute, not surface-normalised, curve. Added alongside AUC, not a
    # replacement; carries r_squared/fit_ok so a poor fit is never hidden.
    fit = depth.fit_penetration_depth(
        raw.depth_um, subtracted.mean, offset=cfg.get("fit_offset", False)
    )

    return {
        "depth_um": raw.depth_um,
        "raw_mean": raw.mean,
        "bg_subtracted": subtracted.mean,
        "normalized": normalized.normalized,
        "surface_reference": normalized.surface_reference,
        "voxel_z_um": vz,
        "n_slices": int(raw.depth_um.size),
        "max_depth_um": max_depth,
        "auc": auc,
        "fit": fit,
    }


def _common_grid(profiles: list[dict]) -> np.ndarray:
    """Shared depth axis for group aggregation (finest step, full overlap range).

    Spans 0 to the shallowest stack's max depth so every stack contributes
    at every grid point (no extrapolation). Independent of the AUC window —
    AUC is computed per stack on full-resolution data.
    """
    steps = [np.median(np.diff(p["depth_um"])) for p in profiles if p["depth_um"].size >= 2]
    step = float(np.median(steps)) if steps else 1.0   # guard single-slice stacks (np.diff -> [])
    top = min(p["max_depth_um"] for p in profiles)
    n = int(round(top / step)) + 1
    return np.arange(n) * step


def aggregate_group(primary: list[dict]) -> dict | None:
    """Mean ± SEM of bg-subtracted and normalised curves over primary stacks."""
    if not primary:
        return None
    grid = _common_grid(primary)
    sub = np.vstack([np.interp(grid, p["depth_um"], p["bg_subtracted"]) for p in primary])
    nrm = np.vstack([np.interp(grid, p["depth_um"], p["normalized"]) for p in primary])
    n = len(primary)

    def sem(a):
        return a.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(a.shape[1])

    # Penetration constant λ over primary stacks. Only fits that converged,
    # cleared the R² threshold, AND landed within the acquired depth range
    # (fit_ok) contribute to the group λ AND to the reported mean R², so the two
    # describe the same set of accepted fits; degenerate/out-of-range fits are
    # excluded rather than dragging the mean, and their count is reported.
    ok_fits = [p["fit"] for p in primary if p["fit"].fit_ok]
    ok_lams = [f.lambda_um for f in ok_fits]
    ok_r2 = [f.r_squared for f in ok_fits]
    lam_arr = np.asarray(ok_lams, dtype=np.float64)
    lam_mean = float(lam_arr.mean()) if lam_arr.size else float("nan")
    lam_sem = (float(lam_arr.std(ddof=1) / np.sqrt(lam_arr.size))
               if lam_arr.size > 1 else 0.0)
    return {
        "depth_um": grid,
        "sub_mean": sub.mean(axis=0), "sub_sem": sem(sub),
        "norm_mean": nrm.mean(axis=0), "norm_sem": sem(nrm),
        "n": n,
        "lambda_mean_um": lam_mean,
        "lambda_sem_um": lam_sem,
        "n_lambda_ok": int(lam_arr.size),
        "r_squared_mean": float(np.mean(ok_r2)) if ok_r2 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Output: CSVs
# ---------------------------------------------------------------------------

def write_profiles_csv(out: Path, rows: list[dict]) -> None:
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "stack", "role", "depth_um",
                    "raw_mean", "bg_subtracted", "normalized"])
        for r in rows:
            g, s, role, res = r["group"], r["stack"], r["role"], r["result"]
            for i, z in enumerate(res["depth_um"]):
                w.writerow([g, s, role, f"{z:.4g}",
                            f"{res['raw_mean'][i]:.6g}",
                            f"{res['bg_subtracted'][i]:.6g}",
                            f"{res['normalized'][i]:.6g}"])


def write_auc_csv(out: Path, rows: list[dict], windows) -> None:
    labels = [window_label(w) for w in windows]
    header = ["group", "stack", "role", "voxel_z_um", "n_slices",
              "max_depth_um", "surface_intensity_subtracted",
              "lambda_um", "lambda_I0", "lambda_offset",
              "lambda_r_squared", "lambda_rmse", "lambda_fit_ok"]
    for lab in labels:
        header += [f"auc_absolute_{lab}", f"auc_normalized_{lab}",
                   f"window_covered_um_{lab}"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            res = r["result"]
            fit = res["fit"]
            row = [r["group"], r["stack"], r["role"],
                   f"{res['voxel_z_um']:g}", res["n_slices"],
                   f"{res['max_depth_um']:.4g}", f"{res['surface_reference']:.6g}",
                   f"{fit.lambda_um:.6g}", f"{fit.I0:.6g}",
                   "" if fit.offset is None else f"{fit.offset:.6g}",
                   f"{fit.r_squared:.6g}", f"{fit.rmse:.6g}", int(fit.fit_ok)]
            for lab in labels:
                a = res["auc"][lab]
                row += [f"{a['absolute']:.6g}", f"{a['normalized']:.6g}",
                        f"{a['covered_um']:.4g}"]
            w.writerow(row)


def write_group_summary_csv(out: Path, aggregates: dict) -> None:
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "n_primary_stacks", "depth_um",
                    "bg_subtracted_mean", "bg_subtracted_sem",
                    "normalized_mean", "normalized_sem",
                    "lambda_mean_um", "lambda_sem_um", "n_lambda_ok",
                    "r_squared_mean"])
        for g, agg in aggregates.items():
            if agg is None:
                continue
            # Group λ is a single value per group (repeated per depth row so the
            # long-form CSV stays self-contained for Prism/pandas).
            lam = (f"{agg['lambda_mean_um']:.6g}", f"{agg['lambda_sem_um']:.6g}",
                   agg["n_lambda_ok"], f"{agg['r_squared_mean']:.6g}")
            for i, z in enumerate(agg["depth_um"]):
                w.writerow([g, agg["n"], f"{z:.4g}",
                            f"{agg['sub_mean'][i]:.6g}", f"{agg['sub_sem'][i]:.6g}",
                            f"{agg['norm_mean'][i]:.6g}", f"{agg['norm_sem'][i]:.6g}",
                            *lam])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(manifest_path: Path, output_override: str | None = None) -> Path:
    """Analyse every stack in a manifest; write CSVs, figures and a console report."""
    m = load_manifest(manifest_path)
    out_dir = Path(output_override or m.get("output_dir", "depth_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    cfg = {k: m[k] for k in ("channel", "reducer", "n_surface", "windows", "fit_offset")}
    windows = cfg["windows"]
    rows, colors, per_group_primary = _analyse_groups(m["groups"], cfg)
    aggregates = {g: aggregate_group(ps) for g, ps in per_group_primary.items()}

    write_profiles_csv(out_dir / "depth_profiles_long.csv", rows)
    write_auc_csv(out_dir / "auc_per_stack.csv", rows, windows)
    write_group_summary_csv(out_dir / "group_depth_summary.csv", aggregates)
    _write_figures(rows, aggregates, colors, windows, m["signal_label"], out_dir)

    _print_summary(rows, windows)
    _print_lambda_summary(rows)
    print(f"\nOutputs written to {out_dir}/")
    return out_dir


def _analyse_groups(groups: list[dict], cfg: dict):
    """Analyse every stack, group by group. Returns rows, colours, primaries."""
    rows: list[dict] = []
    colors: dict[str, str] = {}
    per_group_primary: dict[str, list[dict]] = {}
    for grp in groups:
        gname = grp["name"]
        colors[gname] = group_color(gname, grp.get("color"))
        bg = grp.get("background")
        blank = _load_blank(bg, cfg) if bg else None
        per_group_primary.setdefault(gname, [])
        for st in grp["stacks"]:
            role = st.get("role", "primary")
            print(f"  [{gname}/{role}] {st['label']}")
            res = analyse_stack(Path(st["path"]), blank, cfg)
            rows.append({"group": gname, "stack": st["label"], "role": role, "result": res})
            if role == "primary":
                per_group_primary[gname].append(res)
    return rows, colors, per_group_primary


def _load_blank(bg, cfg: dict) -> depth.DepthProfile:
    """Blank (no-fluorophore) profile from one path or a list of replicate paths.

    Replicates are averaged over their common depth range, with the
    across-replicate SEM; empty/truncated trailing slices (mean ~0) are dropped.
    """
    bg_paths = [bg] if isinstance(bg, str) else list(bg)
    profs = [_load_profile(Path(bp), cfg)[0] for bp in bg_paths]
    if len(profs) == 1:
        blank = profs[0]
    else:
        n = min(len(p.mean) for p in profs)
        stack = np.stack([p.mean[:n] for p in profs])
        blank = depth.DepthProfile(profs[0].depth_um[:n], stack.mean(0),
                                   stack.std(0, ddof=1) / len(profs) ** 0.5,
                                   profs[0].n_pixels)
    valid = blank.mean > 0
    if not valid.all():
        blank = depth.DepthProfile(blank.depth_um[valid], blank.mean[valid],
                                   blank.sem[valid], blank.n_pixels)
    return blank


def _write_figures(rows, aggregates, colors, windows, sig: str, out_dir: Path) -> None:
    """Depth-vs curves (absolute + normalised), per-window AUC plots, retention summary."""
    # The narrowest fixed window is shaded on the depth-vs curves.
    fixed = [w for w in windows if w != ("full",)]
    shade_window = min(fixed, key=lambda w: w[1]) if fixed else None
    plot_depth_curves(rows, aggregates, colors, "bg_subtracted",
                      f"Mean {sig} intensity (a.u., blank-subtracted)",
                      f"{sig} intensity vs depth",
                      out_dir / "fig_depth_absolute", auc_window=shade_window)
    plot_depth_curves(rows, aggregates, colors, "normalized",
                      "Normalised intensity (surface = 1)",
                      f"Normalised {sig} intensity vs depth",
                      out_dir / "fig_depth_normalized", auc_window=shade_window)
    for w in windows:
        lab = window_label(w)
        plot_auc(rows, colors, w, out_dir / f"fig_auc_normalized_{lab}", use_normalized=True)
        plot_auc(rows, colors, w, out_dir / f"fig_auc_absolute_{lab}", use_normalized=False)
    if len(windows) > 1:
        plot_retention_multiwindow(rows, colors, windows, out_dir / "fig_auc_retention_multiwindow",
                                   title=f"{sig} retained vs depth window")


def _print_welch(label: str, by_group: dict[str, list[float]]) -> None:
    """Print a descriptive two-group Welch contrast (``by_group`` has 2 groups)."""
    (ga, va), (gb, vb) = list(by_group.items())
    r = stats.welch_t(va, vb)
    print(f"  {label} {ga} vs {gb}: t={r['t']:.3f}, p={r['p']:.4f} "
          f"(n={len(va)},{len(vb)} — underpowered, descriptive only)")


def _print_summary(rows, windows) -> None:
    for w in windows:
        lab = window_label(w)
        print(f"\n=== AUC {window_title(w)} (normalised) ===")
        by_group: dict[str, list[float]] = {}
        for r in rows:
            if r["role"] == "primary":
                by_group.setdefault(r["group"], []).append(r["result"]["auc"][lab]["normalized"])
        for g, vals in by_group.items():
            m = np.mean(vals)
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            print(f"  {g:8s} n={len(vals)}  mean={m:8.2f}  sd={sd:6.2f}  "
                  f"values={[round(v, 1) for v in vals]}")
        if len(by_group) == 2:
            _print_welch("Welch t-test", by_group)


def _print_lambda_summary(rows) -> None:
    """Per-group penetration constant λ and a descriptive Welch contrast."""
    print("\n=== Penetration constant λ (single-exponential fit, absolute profile) ===")
    by_group: dict[str, list[float]] = {}
    for r in rows:
        fit = r["result"]["fit"]
        flag = "" if fit.fit_ok else "  [fit_ok=False]"
        print(f"  [{r['group']}/{r['role']}] {r['stack']}: "
              f"λ={fit.lambda_um:7.2f} µm  R²={fit.r_squared:.4f}{flag}")
        if r["role"] == "primary" and fit.fit_ok:
            by_group.setdefault(r["group"], []).append(fit.lambda_um)
    for g, vals in by_group.items():
        m = np.mean(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"  {g:8s} n={len(vals)}  λ mean={m:7.2f} µm  SEM={sem:5.2f}  "
              f"values={[round(v, 1) for v in vals]}")
    if len(by_group) == 2:
        _print_welch("Welch t-test on λ,", by_group)
