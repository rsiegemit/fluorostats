"""Tests for the two-group contrasts that fluorostats.depth_batch.run prints and writes
(Mann-Whitney by default, Welch only when the manifest asks for it; EXP-2)."""

import json

import numpy as np
import pandas as pd

from depth_data import decay_stack, flat_stack
from fluorostats import depth_batch


def _two_group_manifest(tmp_path, **fields):
    """2 groups x 2 stacks, every λ fit accepted (λ ≈ 5-6 and 12-13 µm in a 39 µm stack)."""
    blank = tmp_path / "blank.tif"
    flat_stack(blank, value=2.0, n_z=40, yx=8)
    groups = []
    for name, lam in (("fast", 5.0), ("slow", 12.0)):
        stacks = []
        for i in range(2):
            p = tmp_path / f"{name}_{i}.tif"
            # replicates differ slightly so the rank tests see no ties
            decay_stack(p, lam=lam + i, n_z=40)
            stacks.append({"label": f"{name}{i}", "path": str(p)})
        groups.append({"name": name, "background": str(blank), "stacks": stacks})
    m = {
        "channel": 0,
        "reducer": "mean",
        "n_surface": 2,
        "auc_windows_um": [[0, 10], "full"],
        "output_dir": str(tmp_path / "out"),
        "groups": groups,
        **fields,
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    return mpath


def test_run_two_groups_prints_mann_whitney_contrasts_and_no_welch_by_default(tmp_path, capsys):
    # Regression (EXP-2): a parametric Welch t-test was printed by default.
    out = depth_batch.run(_two_group_manifest(tmp_path))
    assert (out / "auc_per_stack.csv").exists()
    text = capsys.readouterr().out
    assert "Welch" not in text
    # n=2 vs 2, complete separation: exact two-sided p = 2/C(4,2) = 1/3, δ = -1
    assert (
        text.count(
            "Mann-Whitney fast vs slow: U=0.0, p=0.3333, Cliff's δ=-1.00, "
            "BH q=0.3333 across 2 windows (n=2,2)"
        )
        == 2
    )
    assert "Mann-Whitney on λ, fast vs slow: U=0.0, p=0.3333, Cliff's δ=-1.00 (n=2,2)" in text


def test_run_prints_welch_only_when_the_manifest_asks_for_it(tmp_path, capsys):
    depth_batch.run(_two_group_manifest(tmp_path, parametric=True))
    text = capsys.readouterr().out
    assert text.count("Welch t-test (parametric, requested) fast vs slow: t=") == 2
    assert "(n=2,2 — underpowered, descriptive only)" in text
    assert "Welch t-test (parametric, requested) on λ, fast vs slow: t=" in text
    assert text.count("Mann-Whitney fast vs slow: U=0.0") == 2  # still reported


def test_run_writes_the_printed_contrasts_to_contrasts_csv(tmp_path, capsys):
    # The two-group statistics were printed only; contrasts.csv now holds the same
    # numbers, one row per AUC window (BH across windows) and one for λ.
    out = depth_batch.run(_two_group_manifest(tmp_path))
    text = capsys.readouterr().out
    c = pd.read_csv(out / "contrasts.csv", keep_default_na=False)
    assert list(c.columns) == [
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
    assert (c["reason"] == "").all() and (c["n_partial"] == 0).all()
    assert c[["metric", "window"]].values.tolist() == [
        ["auc_normalized", "0_10um"],
        ["auc_normalized", "full"],
        ["lambda_um", ""],
    ]
    assert (c["group_a"] == "fast").all() and (c["group_b"] == "slow").all()
    assert (c[["n_a", "n_b"]] == 2).all().all()
    assert np.allclose(c["p"].astype(float), 1 / 3) and np.allclose(c["q"].astype(float), 1 / 3)
    assert (c["U"].astype(float) == 0).all() and (c["cliffs_delta"].astype(float) == -1).all()
    assert (c["test"] == "Mann-Whitney U, two-sided").all()
    assert c["correction"].tolist() == ["Benjamini-Hochberg FDR across 2 windows"] * 2 + [
        "none (single test; q = p)"
    ]
    # the printed line and the table are one computation
    assert f"p={float(c['p'][0]):.4f}, Cliff's δ={float(c['cliffs_delta'][0]):+.2f}" in text


def test_contrasts_csv_has_only_a_header_without_two_groups(tmp_path):
    s0 = tmp_path / "s0.tif"
    decay_stack(s0)
    m = {
        "auc_windows_um": [[0, 4]],
        "output_dir": str(tmp_path / "out"),
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}],
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    out = depth_batch.run(mpath)
    assert (out / "contrasts.csv").read_text().splitlines() == [
        "metric,window,group_a,group_b,n_a,n_b,U,p,q,cliffs_delta,test,correction,reason,n_partial"
    ]


def test_run_with_three_groups_says_contrasts_were_skipped(tmp_path, capsys):
    stacks = {}
    for g, lam in (("a", 5.0), ("b", 8.0), ("c", 12.0)):
        p = tmp_path / f"{g}.tif"
        decay_stack(p, lam=lam, n_z=40)
        stacks[g] = [{"label": g, "path": str(p)}]
    m = {
        "auc_windows_um": [[0, 10]],
        "output_dir": str(tmp_path / "out"),
        "groups": [{"name": g, "stacks": st} for g, st in stacks.items()],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    depth_batch.run(mpath)
    text = capsys.readouterr().out
    assert "Welch" not in text and "Mann-Whitney" not in text
    assert "contrasts skipped: 3 groups; a two-group contrast needs exactly 2" in text
    assert "contrasts skipped: 3 groups with an accepted λ fit; a two-group" in text
