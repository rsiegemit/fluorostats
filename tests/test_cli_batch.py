"""Tests for the manifest-driven CLI commands (depth, viability) in fluorostats.cli."""

import json

import numpy as np
import tifffile
from click.testing import CliRunner

from fluorostats.cli import cli


def _decay_stack(path, lam=20.0, n_z=16, yx=8, bg=5.0, i0=150.0):
    """Write a synthetic z-stack whose per-slice mean is i0*exp(-z/lam)+bg."""
    z = np.arange(n_z)[:, None, None]
    vol = np.broadcast_to((i0 * np.exp(-z / lam) + bg).astype(np.float32), (n_z, yx, yx))
    tifffile.imwrite(path, np.ascontiguousarray(vol))


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
    _decay_stack(blank, lam=1e9, i0=0.0)
    stacks = []
    for i in range(2):
        p = tmp_path / f"s{i}.tif"
        _decay_stack(p)
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
    _decay_stack(blank, lam=1e9, i0=0.0)
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
