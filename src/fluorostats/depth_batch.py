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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import depth  # noqa: E402
from .io import load_volume  # noqa: E402
from .style import apply_style, PALETTE, material_color, lighten  # noqa: E402


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    with open(path) as fh:
        m = json.load(fh)
    m.setdefault("title", "Depth-penetration analysis")
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


def parse_window(w):
    """Normalise an AUC window spec to ('full',) or (z0, z1) floats.

    ``"full"`` integrates each stack over its own entire acquired range;
    a ``[z0, z1]`` pair is a fixed physical window shared by all stacks.
    """
    if isinstance(w, str) and w.lower() == "full":
        return ("full",)
    return (float(w[0]), float(w[1]))


def window_label(w) -> str:
    return "full" if w == ("full",) else f"{w[0]:g}_{w[1]:g}um"


def window_title(w) -> str:
    return "full depth" if w == ("full",) else f"{w[0]:g}–{w[1]:g} µm"


def window_bounds(w, max_depth_um: float):
    """Resolve a window spec to concrete (z0, z1) for a stack of given depth."""
    if w == ("full",):
        return 0.0, float(max_depth_um)
    return w[0], w[1]


# ---------------------------------------------------------------------------
# Core per-stack computation
# ---------------------------------------------------------------------------

def analyse_stack(path: Path, blank: depth.DepthProfile | None, cfg: dict) -> dict:
    """Load one stack and compute raw/subtracted/normalised profiles + AUC + penetration (λ) fit."""
    vol, meta = load_volume(path)
    vz = float(meta.get("voxel_size_um", (1.0, 1.0, 1.0))[0])

    raw = depth.intensity_depth_profile(
        vol, vz, channel=cfg["channel"], reducer=cfg["reducer"]
    )
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
    step = float(np.median([np.median(np.diff(p["depth_um"])) for p in profiles]))
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
    sem = lambda a: a.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(a.shape[1])
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
# Output: figures
# ---------------------------------------------------------------------------

def _group_color(name: str, override: str | None) -> str:
    return override or material_color(name)


def plot_depth_curves(rows, aggregates, colors, ykey, ylabel, title, out_base,
                      auc_window=None):
    """One line per group (mean ± SEM band) + faint individual stacks."""
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    if auc_window is not None:
        ax.axvspan(auc_window[0], auc_window[1], color=PALETTE["highlight"],
                   alpha=0.08, lw=0, zorder=0)
        ax.axvline(auc_window[1], color=PALETTE["muted"], lw=0.8, ls=":", zorder=0)
    for g, agg in aggregates.items():
        c = colors[g]
        # individual stacks
        for r in rows:
            if r["group"] != g:
                continue
            res = r["result"]
            faint = r["role"] != "primary"
            ax.plot(res["depth_um"], res[ykey], color=c,
                    lw=0.8, alpha=0.35,
                    ls="--" if faint else "-", zorder=1)
        if agg is None:
            continue
        mean = agg["sub_mean"] if ykey == "bg_subtracted" else agg["norm_mean"]
        sem = agg["sub_sem"] if ykey == "bg_subtracted" else agg["norm_sem"]
        ax.fill_between(agg["depth_um"], mean - sem, mean + sem,
                        color=lighten(c, 0.25), alpha=0.35, lw=0, zorder=2)
        lam = agg.get("lambda_mean_um", float("nan"))
        lab = f"{g} (n={agg['n']})"
        if ykey == "bg_subtracted" and np.isfinite(lam):
            # Overlay the single-exponential with the group λ (fit on absolute
            # profile). I0 = near-surface group mean so the curve is anchored.
            lam_sem = agg.get("lambda_sem_um", 0.0)
            r2 = agg.get("r_squared_mean", float("nan"))
            n_ok = agg.get("n_lambda_ok", 0)
            i0 = float(np.mean(mean[:3]))
            ax.plot(agg["depth_um"], i0 * np.exp(-agg["depth_um"] / lam),
                    color=c, lw=1.1, ls=(0, (4, 2)), alpha=0.9, zorder=4)
            # Show how many stacks yielded an in-range λ, so a single-stack or
            # partly-degenerate group λ is never read as a confident group value.
            lab += f"\n  λ={lam:.0f}±{lam_sem:.0f} µm, R²={r2:.2f} ({n_ok}/{agg['n']} in range)"
        ax.plot(agg["depth_um"], mean, color=c, lw=2.2, zorder=3, label=lab)
    ax.set_xlabel("Depth (µm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.margins(x=0)
    _save(fig, out_base)


def plot_auc(rows, colors, window, out_base, use_normalized=True):
    """Dot + mean/SEM bar of AUC per group for one window (primary stacks)."""
    metric = "normalized" if use_normalized else "absolute"
    lab = window_label(window)
    groups = list(colors.keys())
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    xpos = {g: i for i, g in enumerate(groups)}
    for g in groups:
        vals = [r["result"]["auc"][lab][metric] for r in rows
                if r["group"] == g and r["role"] == "primary"]
        if not vals:
            continue
        c = colors[g]
        x = xpos[g]
        m = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.bar(x, m, width=0.55, color=lighten(c, 0.28), edgecolor=c, lw=1.4, zorder=1)
        if sem:
            ax.errorbar(x, m, yerr=sem, color=c, capsize=4, lw=1.4, zorder=2)
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, color=c,
                   edgecolor="white", s=42, lw=0.8, zorder=3)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(groups)
    unit = "normalised" if use_normalized else "absolute"
    ax.set_ylabel(f"AUC {window_title(window)} ({unit})")
    ax.set_title(f"Signal retained over {window_title(window)}")
    ax.margins(x=0.25)
    _save(fig, out_base)


def plot_retention_multiwindow(rows, colors, windows, out_base):
    """Grouped bars: mean retained fraction (AUC_norm / window width) per window.

    Dividing normalised AUC by window width gives the average fraction of
    surface signal retained across that depth window (0-1), so windows of
    different sizes sit on one comparable axis and the widening GelMA/hybrid
    gap is visible at a glance.
    """
    groups = list(colors.keys())
    labels = [window_label(w) for w in windows]
    titles = [window_title(w) for w in windows]
    nW = len(windows)
    x = np.arange(nW)
    width = 0.8 / max(len(groups), 1)
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * nW, 3.8))
    for gi, g in enumerate(groups):
        c = colors[g]
        means, sems, ptsx, ptsy = [], [], [], []
        for wi, lab in enumerate(labels):
            fr = [r["result"]["auc"][lab]["normalized"] / r["result"]["auc"][lab]["width_um"]
                  for r in rows if r["group"] == g and r["role"] == "primary"]
            xc = x[wi] + (gi - (len(groups) - 1) / 2) * width
            means.append(np.mean(fr))
            sems.append(np.std(fr, ddof=1) / np.sqrt(len(fr)) if len(fr) > 1 else 0.0)
            ptsx += list(np.full(len(fr), xc) + np.linspace(-0.06, 0.06, len(fr)))
            ptsy += list(fr)
        xcs = x + (gi - (len(groups) - 1) / 2) * width
        ax.bar(xcs, means, width=width, color=lighten(c, 0.28), edgecolor=c,
               lw=1.3, label=g, zorder=1)
        ax.errorbar(xcs, means, yerr=sems, fmt="none", ecolor=c, capsize=3, lw=1.2, zorder=2)
        ax.scatter(ptsx, ptsy, color=c, edgecolor="white", s=30, lw=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_ylabel("Mean retained fraction (surface = 1)")
    ax.set_title("FITC-dextran retained vs depth window")
    ax.set_ylim(0, 1.05)
    ax.legend()
    _save(fig, out_base)


def _save(fig, out_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(manifest_path: Path, output_override: str | None = None) -> Path:
    m = load_manifest(manifest_path)
    out_dir = Path(output_override or m.get("output_dir", "depth_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    cfg = {k: m[k] for k in ("channel", "reducer", "n_surface", "windows", "fit_offset")}
    windows = cfg["windows"]
    # The narrowest fixed window is shaded on the depth-vs curves.
    fixed = [w for w in windows if w != ("full",)]
    shade_window = min(fixed, key=lambda w: w[1]) if fixed else None

    rows: list[dict] = []
    colors: dict[str, str] = {}
    per_group_primary: dict[str, list[dict]] = {}

    for grp in m["groups"]:
        gname = grp["name"]
        colors[gname] = _group_color(gname, grp.get("color"))
        blank = None
        bg = grp.get("background")
        if bg:
            from fluorostats.depth import DepthProfile
            # accept a single blank path or a list of replicate blanks (averaged)
            bg_paths = [bg] if isinstance(bg, str) else list(bg)
            profs = []
            for bp in bg_paths:
                bvol, bmeta = load_volume(Path(bp))
                bvz = float(bmeta.get("voxel_size_um", (1.0, 1.0, 1.0))[0])
                profs.append(depth.intensity_depth_profile(
                    bvol, bvz, channel=cfg["channel"], reducer=cfg["reducer"]))
            if len(profs) == 1:
                blank = profs[0]
            else:
                # average replicate blanks over the common depth range; blank SEM
                # becomes the across-replicate SEM of the mean profile
                import numpy as _np
                n = min(len(p.mean) for p in profs)
                stack = _np.stack([p.mean[:n] for p in profs])
                blank = DepthProfile(profs[0].depth_um[:n], stack.mean(0),
                                     stack.std(0, ddof=1) / len(profs) ** 0.5,
                                     profs[0].n_pixels)
            # drop empty/truncated trailing slices (mean ~0) from the blank
            valid = blank.mean > 0
            if not valid.all():
                blank = DepthProfile(blank.depth_um[valid], blank.mean[valid],
                                     blank.sem[valid], blank.n_pixels)
        per_group_primary.setdefault(gname, [])
        for st in grp["stacks"]:
            role = st.get("role", "primary")
            print(f"  [{gname}/{role}] {st['label']}")
            res = analyse_stack(Path(st["path"]), blank, cfg)
            rows.append({"group": gname, "stack": st["label"], "role": role, "result": res})
            if role == "primary":
                per_group_primary[gname].append(res)

    aggregates = {g: aggregate_group(ps) for g, ps in per_group_primary.items()}

    # CSVs
    write_profiles_csv(out_dir / "depth_profiles_long.csv", rows)
    write_auc_csv(out_dir / "auc_per_stack.csv", rows, windows)
    write_group_summary_csv(out_dir / "group_depth_summary.csv", aggregates)

    # Depth-vs figures (shared across windows)
    plot_depth_curves(rows, aggregates, colors, "bg_subtracted",
                      "Mean FITC intensity (a.u., blank-subtracted)",
                      "FITC intensity vs depth",
                      out_dir / "fig_depth_absolute", auc_window=shade_window)
    plot_depth_curves(rows, aggregates, colors, "normalized",
                      "Normalised intensity (surface = 1)",
                      "Normalised FITC intensity vs depth",
                      out_dir / "fig_depth_normalized", auc_window=shade_window)

    # One AUC dot/bar plot per window (normalised + absolute)
    for w in windows:
        lab = window_label(w)
        plot_auc(rows, colors, w, out_dir / f"fig_auc_normalized_{lab}", use_normalized=True)
        plot_auc(rows, colors, w, out_dir / f"fig_auc_absolute_{lab}", use_normalized=False)

    # Combined multi-window retention summary
    if len(windows) > 1:
        plot_retention_multiwindow(rows, colors, windows, out_dir / "fig_auc_retention_multiwindow")

    _print_summary(rows, windows)
    _print_lambda_summary(rows)
    print(f"\nOutputs written to {out_dir}/")
    return out_dir


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
            (ga, va), (gb, vb) = list(by_group.items())
            try:
                from scipy import stats as ss
                t, p = ss.ttest_ind(va, vb, equal_var=False)
                print(f"  Welch t-test {ga} vs {gb}: t={t:.3f}, p={p:.4f} "
                      f"(n={len(va)},{len(vb)} — underpowered, descriptive only)")
            except Exception:
                pass


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
        (ga, va), (gb, vb) = list(by_group.items())
        try:
            from scipy import stats as ss
            t, p = ss.ttest_ind(va, vb, equal_var=False)
            print(f"  Welch t-test on λ, {ga} vs {gb}: t={t:.3f}, p={p:.4f} "
                  f"(n={len(va)},{len(vb)} — underpowered, descriptive only)")
        except Exception:
            pass
