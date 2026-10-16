"""Tests for the manifest-driven CLI commands (depth, viability) in fluorostats.cli."""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from depth_data import decay_stack
from fluorostats.cli import cli


def _blob_stack_npy(path, n_live=5, n_dead=5, z=4, size=64, seed=0):
    """Write a 2-channel (C,Z,Y,X) .npy viability stack."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((2, z, size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]

    def stamp(ch, n):
        for _ in range(n):
            cy, cx = rng.integers(8, size - 8, 2)
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0**2))
            vol[ch, z // 2] += 4000 * blob

    stamp(0, n_live)
    stamp(1, n_dead)
    vol += rng.normal(30, 5, vol.shape)
    np.save(path, np.clip(vol, 0, None))


def test_cli_depth_cmd(tmp_path):
    """depth <manifest> should exit 0 and produce AUC csv."""
    blank = tmp_path / "blank.tif"
    decay_stack(blank, lam=1e9, i0=0.0)
    stacks = []
    for i in range(2):
        p = tmp_path / f"s{i}.tif"
        decay_stack(p)
        stacks.append({"label": f"s{i}", "path": str(p)})
    manifest = {
        "channel": 0,
        "reducer": "mean",
        "n_surface": 2,
        "auc_windows_um": [[0, 8]],
        "output_dir": str(tmp_path / "depth_out"),
        "groups": [{"name": "A", "background": str(blank), "stacks": stacks}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    res = CliRunner().invoke(cli, ["depth", str(mpath)])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "depth_out" / "auc_per_stack.csv").exists()


def test_cli_depth_cmd_output_override(tmp_path):
    """depth <manifest> --output DIR should write to the override dir."""
    blank = tmp_path / "blank.tif"
    decay_stack(blank, lam=1e9, i0=0.0)
    stacks = [{"label": "s0", "path": str(blank)}]
    manifest = {
        "channel": 0,
        "reducer": "mean",
        "n_surface": 2,
        "auc_windows_um": [[0, 8]],
        "output_dir": str(tmp_path / "ignored"),
        "groups": [{"name": "A", "stacks": stacks}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    override = tmp_path / "override_out"

    res = CliRunner().invoke(cli, ["depth", str(mpath), "--output", str(override)])
    assert res.exit_code == 0, res.output
    assert (override / "auc_per_stack.csv").exists()
    assert not (tmp_path / "ignored").exists()


def test_cli_viability_cmd(tmp_path):
    """viability <manifest> should exit 0 and produce per-stack csv."""
    groups = []
    for gname in ("cond_A", "cond_B"):
        entries = []
        for i in range(2):
            p = tmp_path / f"{gname}_{i}.npy"
            _blob_stack_npy(p, seed=i)
            entries.append({"label": f"{gname} #{i}", "path": str(p)})
        groups.append({"name": gname, "stacks": entries})

    manifest = {
        "title": "test",
        "live_channel": 0,
        "dead_channel": 1,
        "seg": {"method": "otsu", "min_size": 5},
        "count_method": "all",
        "output_dir": str(tmp_path / "vout"),
        "groups": groups,
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    res = CliRunner().invoke(cli, ["viability", str(mpath)])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "vout" / "viability_per_stack.csv").exists()


def test_cli_viability_cmd_output_override(tmp_path):
    """viability <manifest> --output DIR should write to the override dir."""
    p = tmp_path / "s0.npy"
    _blob_stack_npy(p, seed=0)
    manifest = {
        "title": "test",
        "live_channel": 0,
        "dead_channel": 1,
        "seg": {"method": "otsu", "min_size": 5},
        "count_method": "all",
        "output_dir": str(tmp_path / "ignored"),
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(p)}]}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    override = tmp_path / "vout_override"

    res = CliRunner().invoke(cli, ["viability", str(mpath), "--output", str(override)])
    assert res.exit_code == 0, res.output
    assert (override / "viability_per_stack.csv").exists()
    assert not (tmp_path / "ignored").exists()


@pytest.mark.parametrize("cmd", ["depth", "viability"])
@pytest.mark.parametrize(
    "text, message",
    [
        ('{"groups": [], "typo_field": 1}', "groups"),
        ('{"groups": [', "Expecting"),  # JSON syntax error
    ],
)
def test_a_bad_manifest_is_an_error_message_not_a_traceback(tmp_path, cmd, text, message):
    # Regression (C1): manifest errors surfaced as a 20-line Python traceback.
    m = tmp_path / "m.json"
    m.write_text(text)
    res = CliRunner().invoke(cli, [cmd, str(m), "--output", str(tmp_path / "out")])
    assert res.exit_code == 1 and isinstance(res.exception, SystemExit)
    assert res.output.startswith("Error: ") and message in res.output


def test_a_depth_run_that_fails_leaves_no_output_folder(tmp_path):
    # Regression (C3): the output folder was created before the first stack failed.
    s0 = tmp_path / "s0.tif"
    decay_stack(s0, n_z=1)  # one slice: too few for the λ fit
    m = tmp_path / "m.json"
    m.write_text(
        json.dumps({"groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}]})
    )
    res = CliRunner().invoke(cli, ["depth", str(m), "--output", str(tmp_path / "out")])
    assert res.exit_code == 1 and "s0.tif: has 1 z-slice" in res.output
    assert not (tmp_path / "out").exists()


def test_cli_viability_refuses_a_foreign_folder_unless_overwrite(tmp_path):
    # M1 for viability: the command could not pass --overwrite to viability_batch.run.
    p = tmp_path / "s0.npy"
    _blob_stack_npy(p, seed=0)
    manifest = {
        "live_channel": 0,
        "dead_channel": 1,
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(p)}]}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    out = tmp_path / "out"
    out.mkdir()
    (out / "contrasts.csv").write_text("mine")
    args = ["viability", str(mpath), "--output", str(out)]
    res = CliRunner().invoke(cli, args)
    assert res.exit_code == 1 and "--overwrite" in res.output
    assert (out / "contrasts.csv").read_text() == "mine"
    res = CliRunner().invoke(cli, args + ["--overwrite"])
    assert res.exit_code == 0, res.output
    assert json.loads((out / "run_config.json").read_text())["tool"] == "viability"
