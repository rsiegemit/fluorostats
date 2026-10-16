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
tool serves any probe-penetration / permeability study. With exactly two
groups, the Mann-Whitney contrasts (normalised AUC per window, BH-corrected
across windows, and λ) are printed and written to ``contrasts.csv``
(:mod:`fluorostats.depth_contrasts`).

Run it from the command line::

    fluorostats depth <manifest.json> [--output DIR]

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
      "fit_r2_threshold": 0.85,    # minimum R² for fit_ok (default depth.FIT_R2_THRESHOLD)
      "parametric": false,         # true → also print Welch t-tests (Mann-Whitney always)
      "auc_windows_um": [[0, 100], [0, 200], "full"],  # one or more windows
      "output_dir": "results/permeability",
      "groups": [
        {
          "name": "Control",
          "color": "#3F6AB3",              # optional; else fluorostats palette
          "background": "/path/blank.oib",  # optional no-fluo control
          "stacks": [
            {"label": "Control 1", "path": "/path/a.oib"},
            {"label": "Control 2", "path": "/path/b.oib"},
            {"label": "Control short", "path": "/path/c.oib", "role": "aux"}
          ]
        }
      ]
    }

``auc_windows_um`` accepts ``[z0, z1]`` pairs and/or the string ``"full"``
(each stack's entire acquired range); the singular ``auc_window_um`` was removed
and is refused. ``role`` defaults to ``"primary"``; ``aux``
stacks are exported and drawn faintly but excluded from the group mean/SEM
curves and AUC statistics, so short or differently-acquired stacks can ride
along without biasing the matched comparison.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
from matplotlib.colors import is_color_like

from . import depth, depth_figures, stats
from .io import load_volume
from .manifest import check_depth, require_groups
from . import pubfig, report
from .depth_contrasts import contrasts, print_auc_summary, print_lambda_summary, write_contrasts_csv

__all__ = [
    "load_manifest",
    "analyse_stack",
    "aggregate_group",
    "run",
    "write_profiles_csv",
    "write_auc_csv",
    "write_group_summary_csv",
]

# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    """Read a depth manifest (see the module docstring), filling defaults and parsing windows."""
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    require_groups(m)
    m.setdefault("title", "Depth-penetration analysis")
    m.setdefault("signal_label", "Signal")
    m.setdefault("channel", 0)
    m.setdefault("reducer", "mean")
    m.setdefault("n_surface", 3)
    m.setdefault("fit_offset", False)
    m.setdefault("fit_r2_threshold", depth.FIT_R2_THRESHOLD)
    m.setdefault("parametric", False)
    # default [0, 100] µm: no benchmark chose it (REU-5, open)
    m.setdefault("auc_windows_um", [[0.0, 100.0]])
    _validate(m)
    m["windows"] = [depth.parse_window(w) for w in m["auc_windows_um"]]
    return m


def _validate(m: dict) -> None:
    """Check a depth manifest before any image is loaded; raise listing every problem."""
    problems = check_depth(m)
    for grp in m["groups"]:
        c = grp.get("color") if isinstance(grp, dict) else None
        if c is not None and not is_color_like(c):
            problems.append(f"group {grp.get('name')!r}: color {c!r} is not a matplotlib colour")
    if problems:
        raise ValueError("invalid depth manifest:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Core per-stack computation
# ---------------------------------------------------------------------------


def _load_profile(path: Path, cfg: dict) -> tuple[depth.DepthProfile, float, str]:
    """Load a stack: raw depth profile, z-step (µm) and where the calibration came from."""
    vol, meta = load_volume(path)
    if cfg["channel"] >= vol.shape[0]:
        raise ValueError(
            f"{Path(path).name}: channel {cfg['channel']} is out of range; the "
            f"stack has {vol.shape[0]} channel(s)"
        )
    vz = float(meta.get("voxel_size_um", (1.0, 1.0, 1.0))[0])
    return (
        depth.intensity_depth_profile(vol, vz, channel=cfg["channel"], reducer=cfg["reducer"]),
        vz,
        meta.get("voxel_size_source", ""),
    )


def analyse_stack(path: Path, blank: depth.DepthProfile | None, cfg: dict) -> dict:
    """Load one stack and compute raw/subtracted/normalised profiles + AUC + penetration (λ) fit."""
    raw, vz, vz_source = _load_profile(path, cfg)
    need = 3 if cfg["fit_offset"] else 2
    if raw.depth_um.size < need:
        raise ValueError(
            f"{Path(path).name}: has {raw.depth_um.size} z-slice(s); the λ fit "
            f"needs at least {need} (3 with fit_offset)"
        )
    subtracted = depth.subtract_background(raw, blank) if blank is not None else raw
    try:
        normalized = depth.normalize_to_surface(subtracted, n_surface=cfg["n_surface"])
    except ValueError as e:
        raise ValueError(f"{Path(path).name}: {e}") from e
    norm_curve = normalized.normalized
    assert norm_curve is not None  # normalize_to_surface always sets it

    max_depth = float(raw.depth_um[-1])
    auc = {}
    for w in cfg["windows"]:
        z0, z1 = depth.window_bounds(w, max_depth)
        covered = max(min(z1, max_depth) - z0, 0.0)
        if z1 > max_depth:
            warnings.warn(
                f"{Path(path).name}: AUC window {depth.window_title(w)} reaches past the "
                f"stack's {max_depth:g} µm; only {covered:g} µm of it "
                "is covered (see window_covered_um).",
                RuntimeWarning,
                stacklevel=2,
            )
        auc[depth.window_label(w)] = {
            "absolute": depth.auc_depth(raw.depth_um, subtracted.mean, z_min=z0, z_max=z1),
            "normalized": depth.auc_depth(raw.depth_um, norm_curve, z_min=z0, z_max=z1),
            "covered_um": float(covered),
            "width_um": float(z1 - z0),
        }

    # Penetration constant λ from a single-exponential fit of the absolute
    # (background-subtracted) profile — λ is gain-independent so it is fit on
    # the absolute, not surface-normalised, curve. Added alongside AUC, not a
    # replacement; carries r_squared/fit_ok so a poor fit is never hidden.
    fit = depth.fit_penetration_depth(
        raw.depth_um,
        subtracted.mean,
        offset=cfg["fit_offset"],
        r2_threshold=cfg["fit_r2_threshold"],
    )

    return {
        "depth_um": raw.depth_um,
        "raw_mean": raw.mean,
        "bg_subtracted": subtracted.mean,
        "normalized": normalized.normalized,
        "surface_reference": normalized.surface_reference,
        "voxel_z_um": vz,
        "voxel_size_source": vz_source,
        "n_slices": int(raw.depth_um.size),
        "max_depth_um": max_depth,
        "auc": auc,
        "fit": fit,
    }


def _common_grid(profiles: list[dict]) -> np.ndarray:
    """Shared depth axis for group aggregation.

    The step is the median of the stacks' (median) z-steps. The grid starts at 0 and
    stops at the last whole step within the shallowest stack's max depth, so every
    stack contributes at every grid point (no extrapolation). Independent of the AUC
    window — AUC is computed per stack on full-resolution data.
    """
    steps = [np.median(np.diff(p["depth_um"])) for p in profiles if p["depth_um"].size >= 2]
    step = float(np.median(steps)) if steps else 1.0  # guard single-slice stacks (np.diff -> [])
    top = min(p["max_depth_um"] for p in profiles)
    n = int(np.floor(top / step + 1e-9)) + 1  # 1e-9: a whole number of steps keeps its end
    return np.arange(n) * step


def aggregate_group(primary: list[dict]) -> dict | None:
    """Mean ± SEM of bg-subtracted and normalised curves over primary stacks."""
    if not primary:
        return None
    grid = _common_grid(primary)
    # per depth over the stacks; SEM NaN for one stack (no measured spread)
    sub = stats.describe(
        [np.interp(grid, p["depth_um"], p["bg_subtracted"]) for p in primary], axis=0
    )
    nrm = stats.describe([np.interp(grid, p["depth_um"], p["normalized"]) for p in primary], axis=0)

    # Penetration constant λ over primary stacks. Only fits that converged,
    # cleared the R² threshold, AND landed within the acquired depth range
    # (fit_ok) contribute to the group λ AND to the reported mean R², so the two
    # describe the same set of accepted fits; degenerate/out-of-range fits are
    # excluded rather than dragging the mean, and their count is reported.
    ok_fits = [p["fit"] for p in primary if p["fit"].fit_ok]
    ok_lams = [f.lambda_um for f in ok_fits]
    ok_r2 = [f.r_squared for f in ok_fits]
    lam = stats.describe(ok_lams)
    return {
        "depth_um": grid,
        "sub_mean": sub["mean"],
        "sub_sem": sub["sem"],
        "norm_mean": nrm["mean"],
        "norm_sem": nrm["sem"],
        "n": len(primary),
        "lambda_mean_um": lam["mean"],
        "lambda_sem_um": lam["sem"],
        "n_lambda_ok": lam["n"],
        "r_squared_mean": float(np.mean(ok_r2)) if ok_r2 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Output: CSVs
# ---------------------------------------------------------------------------


def write_profiles_csv(out: Path, rows: list[dict]) -> None:
    """Long-form per-stack depth profiles: raw, background-subtracted and normalised."""
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["group", "stack", "role", "depth_um", "raw_mean", "bg_subtracted", "normalized"]
        )
        for r in rows:
            g, s, role, res = r["group"], r["stack"], r["role"], r["result"]
            for i, z in enumerate(res["depth_um"]):
                w.writerow(
                    [
                        g,
                        s,
                        role,
                        f"{z:.4g}",
                        f"{res['raw_mean'][i]:.6g}",
                        f"{res['bg_subtracted'][i]:.6g}",
                        f"{res['normalized'][i]:.6g}",
                    ]
                )


def write_auc_csv(out: Path, rows: list[dict], windows: list[tuple]) -> None:
    """One row per stack: acquisition, λ fit, absolute/normalised AUC per window,
    ``voxel_size_source`` (which voxel axes were read from the file vs defaulted) and
    ``window_complete_<window>`` (1 when the stack covers the whole window, 0 when
    its AUC is over only ``window_covered_um`` of it)."""
    labels = [depth.window_label(w) for w in windows]
    header = (
        "group stack role voxel_z_um n_slices max_depth_um surface_intensity_subtracted "
        "lambda_um lambda_I0 lambda_offset lambda_r_squared lambda_rmse lambda_fit_ok"
    ).split()
    for lab in labels:
        header += [f"auc_absolute_{lab}", f"auc_normalized_{lab}", f"window_covered_um_{lab}"]
    # appended last so existing column positions hold
    header += ["voxel_size_source"] + [f"window_complete_{lab}" for lab in labels]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            res = r["result"]
            fit = res["fit"]
            row = [
                r["group"],
                r["stack"],
                r["role"],
                f"{res['voxel_z_um']:g}",
                res["n_slices"],
                f"{res['max_depth_um']:.4g}",
                f"{res['surface_reference']:.6g}",
                f"{fit.lambda_um:.6g}",
                f"{fit.I0:.6g}",
                "" if fit.offset is None else f"{fit.offset:.6g}",
                f"{fit.r_squared:.6g}",
                f"{fit.rmse:.6g}",
                int(fit.fit_ok),
            ]
            for lab in labels:
                a = res["auc"][lab]
                row += [f"{a['absolute']:.6g}", f"{a['normalized']:.6g}", f"{a['covered_um']:.4g}"]
            complete = [
                int(res["auc"][lab]["covered_um"] >= res["auc"][lab]["width_um"]) for lab in labels
            ]
            w.writerow(row + [res["voxel_size_source"], *complete])


def write_group_summary_csv(out: Path, aggregates: dict) -> None:
    """Long-form group mean ± SEM profiles, with each group's λ repeated per depth row."""
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "group",
                "n_primary_stacks",
                "depth_um",
                "bg_subtracted_mean",
                "bg_subtracted_sem",
                "normalized_mean",
                "normalized_sem",
                "lambda_mean_um",
                "lambda_sem_um",
                "n_lambda_ok",
                "r_squared_mean",
            ]
        )
        for g, agg in aggregates.items():
            if agg is None:
                continue
            # Group λ is a single value per group (repeated per depth row so the
            # long-form CSV stays self-contained for Prism/pandas).
            lam = (
                f"{agg['lambda_mean_um']:.6g}",
                f"{agg['lambda_sem_um']:.6g}",
                agg["n_lambda_ok"],
                f"{agg['r_squared_mean']:.6g}",
            )
            for i, z in enumerate(agg["depth_um"]):
                w.writerow(
                    [
                        g,
                        agg["n"],
                        f"{z:.4g}",
                        f"{agg['sub_mean'][i]:.6g}",
                        f"{agg['sub_sem'][i]:.6g}",
                        f"{agg['norm_mean'][i]:.6g}",
                        f"{agg['norm_sem'][i]:.6g}",
                        *lam,
                    ]
                )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(manifest_path: Path, output_override: str | None = None, overwrite: bool = False) -> Path:
    """Analyse every stack in a manifest; write CSVs, figures and a console report.

    The output folder must be new or hold an earlier depth run
    (`report.claim_output_dir`); ``overwrite`` replaces another run's files anyway.
    """
    m = load_manifest(manifest_path)
    out_dir = Path(output_override or m.get("output_dir", "depth_output"))
    previous = report.claim_output_dir(out_dir, "depth", _owned_outputs(out_dir), overwrite)
    pubfig.apply_style("house")

    cfg = {
        k: m[k]
        for k in ("channel", "reducer", "n_surface", "windows", "fit_offset", "fit_r2_threshold")
    }
    windows = cfg["windows"]
    rows, colors, per_group_primary = _analyse_groups(m["groups"], cfg)
    aggregates = {g: aggregate_group(ps) for g, ps in per_group_primary.items()}
    report.mark_output_dir(out_dir, "depth")  # only now: a failed run leaves no new folder

    write_profiles_csv(out_dir / "depth_profiles_long.csv", rows)
    write_auc_csv(out_dir / "auc_per_stack.csv", rows, windows)
    write_group_summary_csv(out_dir / "group_depth_summary.csv", aggregates)
    tests = contrasts(rows, windows)
    write_contrasts_csv(out_dir / "contrasts.csv", tests)
    _write_figures(rows, aggregates, colors, cfg, m["signal_label"], out_dir)

    print_auc_summary(rows, windows, m["parametric"], tests)
    print_lambda_summary(rows, m["parametric"], tests)
    # The full record, last: it replaces the "incomplete" marker.
    nested = {"resolved": m, "files_read": _files_read(m["groups"], out_dir)}
    report.write_run_config(out_dir, {"manifest": Path(manifest_path)}, nested, tool="depth")
    for line in report.stale_notices(previous):
        print(line)
    print(f"\nOutputs written to {out_dir}/")
    return out_dir


def _files_read(groups: list[dict], out_dir: Path) -> dict:
    """The absolute paths a run read and wrote (manifest paths resolve against the
    working directory), so ``run_config.json`` names the files, not only the text."""
    backgrounds = [g["background"] for g in groups if g.get("background")]
    return {
        "stacks": [str(Path(st["path"]).resolve()) for g in groups for st in g["stacks"]],
        "backgrounds": [
            str(Path(bp).resolve())
            for bg in backgrounds
            for bp in ([bg] if isinstance(bg, str) else bg)
        ],
        "output_dir": str(out_dir.resolve()),
    }


def _owned_outputs(out_dir: Path) -> list[Path]:
    """Every file a run writes, which an earlier run may have left: the CSVs, every
    figure name `_write_figures` can write (any window) and ``run_config.json``."""
    tables = ("depth_profiles_long", "auc_per_stack", "group_depth_summary", "contrasts")
    figs = [
        out_dir / f"{stem}.{ext}"
        for ext in ("png", "pdf")
        for stem in ("fig_depth_absolute", "fig_depth_normalized", "fig_auc_retention_multiwindow")
    ]
    for kind in ("normalized", "absolute"):
        figs += out_dir.glob(f"fig_auc_{kind}_*.png")
        figs += out_dir.glob(f"fig_auc_{kind}_*.pdf")
    return [*(out_dir / f"{t}.csv" for t in tables), *figs, out_dir / "run_config.json"]


def _analyse_groups(groups: list[dict], cfg: dict):
    """Analyse every stack, group by group. Returns rows, colours, primaries."""
    rows: list[dict] = []
    colors = depth_figures.group_colours(groups)
    per_group_primary: dict[str, list[dict]] = {}
    for grp in groups:
        gname = grp["name"]
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

    Replicates are averaged slice by slice over the first n slices, n being the
    shortest replicate, with the across-replicate SEM. Every slice whose mean is
    <= 0 (empty or truncated; trailing or mid-stack) is then dropped, so the
    background is interpolated across mid-stack gaps and held at its last value
    past the end. Both are announced with counts; a blank with no slice left is a
    ValueError naming it.
    """
    bg_paths = [bg] if isinstance(bg, str) else list(bg)
    profs = [_load_profile(Path(bp), cfg)[0] for bp in bg_paths]
    blank = profs[0] if len(profs) == 1 else _average_blanks(profs)
    valid = blank.mean > 0
    names = ", ".join(Path(bp).name for bp in bg_paths)
    if not valid.any():
        raise ValueError(f"blank {names}: no slice has a mean > 0, so there is no background")
    if not valid.all():
        n_drop = int((~valid).sum())
        trailing = int(np.argmax(valid[::-1]))
        warnings.warn(
            f"blank {names}: dropped {n_drop} "
            f"of {valid.size} slices with mean <= 0 ({n_drop - trailing} mid-stack, "
            f"{trailing} trailing).",
            RuntimeWarning,
            stacklevel=2,
        )
        blank = depth.DepthProfile(
            blank.depth_um[valid], blank.mean[valid], blank.sem[valid], blank.n_pixels
        )
    return blank


def _average_blanks(profs: list[depth.DepthProfile]) -> depth.DepthProfile:
    """Slice-by-slice mean (and SEM) of replicate blanks over the shortest one's slices."""
    steps = {float(np.median(np.diff(p.depth_um))) for p in profs if p.depth_um.size > 1}
    if len(steps) > 1:
        warnings.warn(
            f"replicate blanks have different z-steps {sorted(steps)} µm; they are "
            "averaged slice by slice on the first blank's depth axis.",
            RuntimeWarning,
            stacklevel=3,
        )
    lengths = sorted({len(p.mean) for p in profs})
    n = lengths[0]
    if len(lengths) > 1:
        warnings.warn(
            f"replicate blanks have {lengths} slices; they are averaged over the "
            f"first {n} (the shortest).",
            RuntimeWarning,
            stacklevel=3,
        )
    d = stats.describe([p.mean[:n] for p in profs], axis=0)
    return depth.DepthProfile(profs[0].depth_um[:n], d["mean"], d["sem"], profs[0].n_pixels)


def _write_figures(rows, aggregates, colors, cfg: dict, sig: str, out_dir: Path) -> None:
    """Depth-vs curves (absolute + normalised), per-window AUC plots, retention summary."""
    windows = cfg["windows"]
    # The narrowest fixed window is shaded on the depth-vs curves.
    fixed = [w for w in windows if w != ("full",)]
    shade_window = min(fixed, key=lambda w: w[1]) if fixed else None
    model = {"n_surface": cfg["n_surface"], "fit_offset": cfg["fit_offset"]}
    depth_figures.plot_depth_curves(
        rows,
        aggregates,
        colors,
        "bg_subtracted",
        f"Mean {sig} intensity (a.u., blank-subtracted)",
        f"{sig} intensity vs depth",
        out_dir / "fig_depth_absolute",
        auc_window=shade_window,
        **model,
    )
    depth_figures.plot_depth_curves(
        rows,
        aggregates,
        colors,
        "normalized",
        "Normalised intensity (surface = 1)",
        f"Normalised {sig} intensity vs depth",
        out_dir / "fig_depth_normalized",
        auc_window=shade_window,
        **model,
    )
    for w in windows:
        lab = depth.window_label(w)
        out_norm, out_abs = (
            out_dir / f"fig_auc_normalized_{lab}",
            out_dir / f"fig_auc_absolute_{lab}",
        )
        depth_figures.plot_auc(rows, colors, w, out_norm, use_normalized=True)
        depth_figures.plot_auc(rows, colors, w, out_abs, use_normalized=False)
    if len(windows) > 1:
        depth_figures.plot_retention_multiwindow(
            rows,
            colors,
            windows,
            out_dir / "fig_auc_retention_multiwindow",
            title=f"{sig} retained vs depth window",
        )
