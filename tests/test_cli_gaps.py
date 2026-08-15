"""Tests covering uncovered branches in fluorostats.cli.

Targets:
  - 47-49:  depth CLI command
  - 69-71:  viability CLI command
  - 116-117: quant3d no-files early return
  - 144:    quant3d --bg-radius > 0
  - 155:    quant3d without --no-skeleton
  - 235-236: quant2d no-files early return
  - 308:    _write_plots_and_pvalues early return (single condition / --no-plots)
  - 348-351: _get_condition grandparent / filename modes
  - 368-371: _parse_channel None, int, str
"""
import json

import numpy as np
import pytest
from click.testing import CliRunner

from fluorostats.cli import cli, _get_condition, _parse_channel

tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# Helpers shared with test_depth_batch.py / test_viability_batch.py
# ---------------------------------------------------------------------------

def _decay_stack(path, lam=20.0, n_z=16, yx=8, bg=5.0, i0=150.0):
    """Write a synthetic z-stack whose per-slice mean is i0*exp(-z/lam)+bg."""
    z = np.arange(n_z)[:, None, None]
    vol = np.broadcast_to(
        (i0 * np.exp(-z / lam) + bg).astype(np.float32), (n_z, yx, yx)
    )
    tifffile.imwrite(path, np.ascontiguousarray(vol))


def _blob_stack_npy(path, n_live=5, n_dead=5, z=4, size=64, seed=0):
    """Write a 2-channel (C,Z,Y,X) .npy viability stack."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((2, z, size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]

    def stamp(ch, n):
        for _ in range(n):
            cy, cx = rng.integers(8, size - 8, 2)
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))
            vol[ch, z // 2] += 4000 * blob

    stamp(0, n_live)
    stamp(1, n_dead)
    vol += rng.normal(30, 5, vol.shape)
    np.save(path, np.clip(vol, 0, None))


def _blob_3d(seed):
    """Simple 3D volume for quant3d tests."""
    vol = np.random.default_rng(seed).integers(0, 40, (8, 48, 48)).astype(np.uint16)
    vol[2:6, 12:36, 12:36] = 900
    return vol


def _blob_2d(seed):
    """Simple 2D image for quant2d tests."""
    img = np.random.default_rng(seed).integers(0, 40, (64, 64)).astype(np.uint8)
    img[16:48, 16:48] = 220
    return img


# ---------------------------------------------------------------------------
# depth CLI command  (lines 47-49)
# ---------------------------------------------------------------------------

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
        "channel": 0, "reducer": "mean", "n_surface": 2,
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
        "channel": 0, "reducer": "mean", "n_surface": 2,
        "auc_windows_um": [[0, 8]],
        "output_dir": str(tmp_path / "ignored"),
        "groups": [{"name": "A", "stacks": stacks}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    override = tmp_path / "override_out"

    res = CliRunner().invoke(cli, ["depth", str(mpath), "--output", str(override)])
    assert res.exit_code == 0, res.output
    assert override.exists()


# ---------------------------------------------------------------------------
# viability CLI command  (lines 69-71)
# ---------------------------------------------------------------------------

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
        "title": "test", "live_channel": 0, "dead_channel": 1,
        "seg": {"method": "otsu", "min_size": 5}, "count_method": "all",
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
        "title": "test", "live_channel": 0, "dead_channel": 1,
        "seg": {"method": "otsu", "min_size": 5}, "count_method": "all",
        "output_dir": str(tmp_path / "ignored"),
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(p)}]}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    override = tmp_path / "vout_override"

    res = CliRunner().invoke(cli, ["viability", str(mpath), "--output", str(override)])
    assert res.exit_code == 0, res.output
    assert override.exists()


# ---------------------------------------------------------------------------
# quant3d / quant2d no-files early return  (lines 116-117, 235-236)
# ---------------------------------------------------------------------------

def test_quant3d_empty_input_dir(tmp_path):
    """quant3d on a dir with no volume files should print message and return."""
    empty = tmp_path / "empty"
    empty.mkdir()
    res = CliRunner().invoke(cli, [
        "quant3d", "--input", str(empty), "--output", str(tmp_path / "out"),
    ])
    assert res.exit_code == 0
    assert "No volume files found" in res.output


def test_quant2d_empty_input_dir(tmp_path):
    """quant2d on a dir with no image files should print message and return."""
    empty = tmp_path / "empty"
    empty.mkdir()
    res = CliRunner().invoke(cli, [
        "quant2d", "--input", str(empty), "--output", str(tmp_path / "out"),
    ])
    assert res.exit_code == 0
    assert "No image files found" in res.output


# ---------------------------------------------------------------------------
# quant3d with --bg-radius > 0  (line 144)
# ---------------------------------------------------------------------------

def test_quant3d_bg_radius(tmp_path):
    """quant3d with --bg-radius should run background_subtract and exit 0."""
    d = tmp_path / "in" / "cond"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "s0.tif", _blob_3d(0))
    res = CliRunner().invoke(cli, [
        "quant3d",
        "--input", str(tmp_path / "in"),
        "--output", str(tmp_path / "out"),
        "--bg-radius", "5",
        "--no-skeleton",
        "--no-plots",
    ])
    assert res.exit_code == 0, res.output


# ---------------------------------------------------------------------------
# quant3d without --no-skeleton  (line 155)
# ---------------------------------------------------------------------------

def test_quant3d_with_skeleton(tmp_path):
    """quant3d without --no-skeleton exercises the skeleton_metrics branch."""
    d = tmp_path / "in" / "cond"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "s0.tif", _blob_3d(1))
    res = CliRunner().invoke(cli, [
        "quant3d",
        "--input", str(tmp_path / "in"),
        "--output", str(tmp_path / "out"),
        "--no-plots",
    ])
    assert res.exit_code == 0, res.output


# ---------------------------------------------------------------------------
# _write_plots_and_pvalues early return when single condition  (line 308)
# ---------------------------------------------------------------------------

def test_quant3d_single_condition_no_plots(tmp_path):
    """Single-condition run with --no-plots hits the early return in _write_plots_and_pvalues."""
    d = tmp_path / "in" / "GelMA"
    d.mkdir(parents=True)
    for i in range(3):
        tifffile.imwrite(d / f"s{i}.tif", _blob_3d(i))
    res = CliRunner().invoke(cli, [
        "quant3d",
        "--input", str(tmp_path / "in"),
        "--output", str(tmp_path / "out"),
        "--no-skeleton",
        "--no-plots",
    ])
    assert res.exit_code == 0, res.output


def test_quant2d_single_condition_no_plots(tmp_path):
    """quant2d single condition triggers _write_plots_and_pvalues early return."""
    d = tmp_path / "in" / "GelMA"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "s0.tif", _blob_2d(0))
    res = CliRunner().invoke(cli, [
        "quant2d",
        "--input", str(tmp_path / "in"),
        "--output", str(tmp_path / "out"),
        "--no-plots",
    ])
    assert res.exit_code == 0, res.output


# ---------------------------------------------------------------------------
# _get_condition: grandparent and filename modes  (lines 348-351)
# ---------------------------------------------------------------------------

def test_get_condition_grandparent(tmp_path):
    """grandparent mode returns parent.parent.name."""
    p = tmp_path / "GrandCond" / "subfolder" / "img.tif"
    p.parent.mkdir(parents=True)
    p.touch()
    assert _get_condition(p, "grandparent") == "GrandCond"


def test_get_condition_filename(tmp_path):
    """filename mode returns file stem."""
    p = tmp_path / "somegroup" / "replicate01.tif"
    p.parent.mkdir()
    p.touch()
    assert _get_condition(p, "filename") == "replicate01"


def test_cli_quant3d_condition_from_grandparent(tmp_path):
    """--condition-from grandparent works end-to-end."""
    # structure: in/parent/cond_A/s0.tif
    d = tmp_path / "in" / "cond_A" / "sub"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "s0.tif", _blob_3d(0))
    res = CliRunner().invoke(cli, [
        "quant3d",
        "--input", str(tmp_path / "in"),
        "--output", str(tmp_path / "out"),
        "--condition-from", "grandparent",
        "--no-skeleton",
        "--no-plots",
    ])
    assert res.exit_code == 0, res.output


def test_cli_quant3d_condition_from_filename(tmp_path):
    """--condition-from filename works end-to-end."""
    d = tmp_path / "in" / "sub"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "replicate01.tif", _blob_3d(2))
    res = CliRunner().invoke(cli, [
        "quant3d",
        "--input", str(tmp_path / "in"),
        "--output", str(tmp_path / "out"),
        "--condition-from", "filename",
        "--no-skeleton",
        "--no-plots",
    ])
    assert res.exit_code == 0, res.output


# ---------------------------------------------------------------------------
# _parse_channel: None, int, str  (lines 368-371)
# ---------------------------------------------------------------------------

def test_parse_channel_none():
    assert _parse_channel(None) is None


def test_parse_channel_int_string():
    assert _parse_channel("1") == 1
    assert isinstance(_parse_channel("1"), int)


def test_parse_channel_name_string():
    result = _parse_channel("green")
    assert result == "green"
    assert isinstance(result, str)


def test_cli_quant3d_channel_int(tmp_path):
    """--channel 0 (integer) is accepted by the CLI."""
    d = tmp_path / "in" / "cond"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "s0.tif", _blob_3d(3))
    res = CliRunner().invoke(cli, [
        "quant3d",
        "--input", str(tmp_path / "in"),
        "--output", str(tmp_path / "out"),
        "--channel", "0",
        "--no-skeleton",
        "--no-plots",
    ])
    assert res.exit_code == 0, res.output


def test_cli_quant3d_channel_name(tmp_path):
    """--channel with a name string reaches _parse_channel's str branch (returns str not int).

    The channel may not match in the test tiff; we verify the code path through
    _parse_channel is exercised (returns a str) and the CLI doesn't crash on the
    argument parse itself — the ValueError from mismatched channel name is a
    runtime error, not a CLI parse error.
    """
    # _parse_channel("Ch1") returns "Ch1" — test the str-return path directly
    result = _parse_channel("Ch1")
    assert isinstance(result, str)
    assert result == "Ch1"

    # Also confirm the CLI accepts a string value for --channel (no argument-parse error)
    d = tmp_path / "in" / "cond"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "s0.tif", _blob_3d(4))
    res = CliRunner().invoke(cli, [
        "quant3d",
        "--input", str(tmp_path / "in"),
        "--output", str(tmp_path / "out"),
        "--channel", "Ch1",   # matches the auto-named channel in a single-channel tiff
        "--no-skeleton",
        "--no-plots",
    ])
    assert res.exit_code == 0, res.output
