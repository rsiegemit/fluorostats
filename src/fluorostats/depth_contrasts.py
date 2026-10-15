"""Two-group statistics of a depth-penetration run (`depth_batch`).

With exactly two groups, `contrasts` computes the two-sided Mann-Whitney contrasts
(primary stacks' normalised AUC per window, BH-corrected across the windows, and
the accepted λ fits); the console summary prints them and `write_contrasts_csv`
writes them to ``contrasts.csv``. A Welch t-test is printed only when the manifest
asks for parametric tests (EXP-2).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from . import stats
from .depth import window_label, window_title


def _auc_by_group(rows, w) -> dict[str, list[float]]:
    """Primary stacks' normalised AUC over window ``w``, by group (in manifest order)."""
    lab = window_label(w)
    by_group: dict[str, list[float]] = {}
    for r in rows:
        if r["role"] == "primary":
            by_group.setdefault(r["group"], []).append(r["result"]["auc"][lab]["normalized"])
    return by_group


def _n_partial(rows, w) -> int:
    """Primary stacks with an AUC over window ``w`` that covers only part of it."""
    lab = window_label(w)
    aucs = [r["result"]["auc"][lab] for r in rows if r["role"] == "primary"]
    return sum(a["normalized"] == a["normalized"] and a["covered_um"] < a["width_um"] for a in aucs)


def _lambda_by_group(rows) -> dict[str, list[float]]:
    """Accepted (``fit_ok``) λ of primary stacks, by group (in manifest order)."""
    by_group: dict[str, list[float]] = {}
    for r in rows:
        if r["role"] == "primary" and r["result"]["fit"].fit_ok:
            by_group.setdefault(r["group"], []).append(r["result"]["fit"].lambda_um)
    return by_group


def _mann_whitney_row(metric: str, window: str, by_group: Mapping[str, Sequence[float]]) -> dict:
    """Two-sided Mann-Whitney between the two groups of ``by_group``, as a contrasts row."""
    (ga, va), (gb, vb) = by_group.items()
    mw = stats.mann_whitney(va, vb)
    return {
        "metric": metric,
        "window": window,
        "group_a": ga,
        "group_b": gb,
        "n_a": mw["n_x"],
        "n_b": mw["n_y"],
        "U": mw["u"],
        "p": mw["p"],
        "cliffs_delta": mw["cliffs_delta"],
        "test": "Mann-Whitney U, two-sided",
        "reason": mw["reason"],
        "n_partial": 0,
    }


def mann_whitney_contrasts(
    entries: Sequence[tuple[str, str, Mapping[str, Sequence[float]]]], across: str | None = None
) -> list[dict]:
    """Two-group contrast rows for any driver: one per ``(metric, window, by_group)``
    entry whose ``by_group`` (group name -> values, in order) has exactly two groups.

    With ``across`` (a noun such as ``"windows"``) the entries are one family and
    their p-values are BH-corrected together ("Benjamini-Hochberg FDR across n
    <across>", n counting the entries with a p-value); without it each row is its own test (q = p). ``window`` is free text
    (empty when it does not apply). ``reason`` is `stats.mann_whitney`'s: empty when
    tested, else why p is NaN (e.g. "n<2": a group below `stats.MIN_N`, left out of
    the BH family). ``n_partial`` (values measured over only part of their window)
    is 0 here; `contrasts` sets it for depth AUC rows. Rows have the
    `write_contrasts_csv` columns.
    """
    rows = [_mann_whitney_row(metric, window, bg) for metric, window, bg in entries if len(bg) == 2]
    if across is None:
        return [{**c, "q": c["p"], "correction": "none (single test; q = p)"} for c in rows]
    qs = stats.bh_fdr([c["p"] for c in rows])
    n_tested = sum(c["p"] == c["p"] for c in rows)  # bh_fdr leaves NaN p out
    correction = f"Benjamini-Hochberg FDR across {n_tested} {across}"
    return [{**c, "q": float(q), "correction": correction} for c, q in zip(rows, qs)]


def contrasts(rows: list[dict], windows: list[tuple]) -> list[dict]:
    """The two-group statistics of a depth run (none unless there are exactly two groups).

    One row per AUC window on the primary stacks' normalised AUC, BH-corrected across
    the windows, and one on accepted λ fits (a single test: q = p). Columns: metric,
    window, group_a, group_b, n_a, n_b, U, p, q, cliffs_delta, test, correction, reason,
    n_partial. ``n_partial`` counts the tested stacks whose depth covers the window
    only partly (their AUC is over the covered part: see ``window_covered_um`` in
    ``auc_per_stack.csv``); 0 for λ. The console summary prints these same rows;
    `write_contrasts_csv` writes them.
    """
    auc = [("auc_normalized", window_label(w), _auc_by_group(rows, w)) for w in windows]
    partial = {window_label(w): _n_partial(rows, w) for w in windows}
    auc_rows = mann_whitney_contrasts(auc, across="windows")
    auc_rows = [{**c, "n_partial": partial[c["window"]]} for c in auc_rows]
    return auc_rows + mann_whitney_contrasts([("lambda_um", "", _lambda_by_group(rows))])


_CONTRAST_COLUMNS = [
    "metric",
    "window",
    "group_a",
    "group_b",
    "n_a",
    "n_b",
    "U",
    "p",
    "q",
    "cliffs_delta",
    "test",
    "correction",
    "reason",
    "n_partial",
]


def write_contrasts_csv(out: Path, tests: list[dict]) -> None:
    """One row per contrast (`contrasts` / `mann_whitney_contrasts`); only the header
    when there is none."""
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CONTRAST_COLUMNS)
        for c in tests:
            w.writerow(
                [f"{c[k]:.6g}" if isinstance(c[k], float) else c[k] for k in _CONTRAST_COLUMNS]
            )


def _print_welch(label: str, by_group: dict[str, list[float]]) -> None:
    """Print a descriptive two-group Welch contrast (``by_group`` has 2 groups); only
    when the manifest asks for parametric tests, the default being non-parametric (EXP-2)."""
    (ga, va), (gb, vb) = [(g, [v for v in vals if not np.isnan(v)]) for g, vals in by_group.items()]
    r = stats.welch_t(va, vb)
    print(
        f"  {label} {ga} vs {gb}: t={r['t']:.3f}, p={r['p']:.4f} "
        f"(n={len(va)},{len(vb)} — underpowered, descriptive only)"
    )


def _print_mann_whitney(label: str, c: dict, q: str = "") -> None:
    """Print a `contrasts` row: two-sided Mann-Whitney with Cliff's δ (and BH q when given)."""
    print(
        f"  {label} {c['group_a']} vs {c['group_b']}: U={c['U']:.1f}, p={c['p']:.4f}, "
        f"Cliff's δ={c['cliffs_delta']:+.2f}{q} (n={c['n_a']},{c['n_b']})"
    )


def _print_skipped(n_groups: int, which: str = "") -> None:
    """Say why no contrast is printed: it needs exactly two groups (``which`` qualifies them)."""
    print(
        f"  (contrasts skipped: {n_groups} group{'s' * (n_groups != 1)}{which}; "
        "a two-group contrast needs exactly 2)"
    )


def print_auc_summary(
    rows: list[dict], windows: list[tuple], parametric: bool, tests: list[dict]
) -> None:
    """Per-window normalised AUC by group; with two groups, the Mann-Whitney contrasts
    of ``tests`` (preceded by Welch when ``parametric``)."""
    auc = {c["window"]: c for c in tests if c["metric"] == "auc_normalized"}
    for w in windows:
        by_group = _auc_by_group(rows, w)
        print(f"\n=== AUC {window_title(w)} (normalised) ===")
        for g, vals in by_group.items():
            d = stats.describe(vals)
            print(
                f"  {g:8s} n={d['n']}  mean={d['mean']:8.2f}  sd={d['sd']:6.2f}  "
                f"values={[round(v, 1) for v in vals]}"
            )
        c = auc.get(window_label(w))
        if c is None:
            _print_skipped(len(by_group))
            continue
        if parametric:
            _print_welch("Welch t-test (parametric, requested)", by_group)
        _print_mann_whitney("Mann-Whitney", c, f", BH q={c['q']:.4f} across {len(auc)} windows")
        if c["n_partial"]:
            print(f"  ({c['n_partial']} stack(s) cover only part of it; see window_covered_um)")


def print_lambda_summary(rows: list[dict], parametric: bool, tests: list[dict]) -> None:
    """Per-group penetration constant λ and its Mann-Whitney contrast from ``tests``
    (Welch too when ``parametric``)."""
    print("\n=== Penetration constant λ (single-exponential fit, absolute profile) ===")
    for r in rows:
        fit = r["result"]["fit"]
        flag = "" if fit.fit_ok else "  [fit_ok=False]"
        print(
            f"  [{r['group']}/{r['role']}] {r['stack']}: "
            f"λ={fit.lambda_um:7.2f} µm  R²={fit.r_squared:.4f}{flag}"
        )
    by_group = _lambda_by_group(rows)
    for g, vals in by_group.items():
        d = stats.describe(vals)
        print(
            f"  {g:8s} n={d['n']}  λ mean={d['mean']:7.2f} µm  SEM={d['sem']:5.2f}  "
            f"values={[round(v, 1) for v in vals]}"
        )
    lam = [c for c in tests if c["metric"] == "lambda_um"]
    if not lam:
        _print_skipped(len(by_group), " with an accepted λ fit")
        return
    if parametric:
        _print_welch("Welch t-test (parametric, requested) on λ,", by_group)
    _print_mann_whitney("Mann-Whitney on λ,", lam[0])
