"""Tests for fluorostats.cli_paths: input discovery (files, folder volumes, condition
labels) and output-folder hygiene, alone and through the quant commands."""

import json

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner

from cli_data import blob_2d, blob_3d, make_dataset
from fluorostats.cli import cli
from fluorostats.cli_paths import get_condition


@pytest.mark.parametrize(
    "cmd,message", [("quant3d", "No volume files found"), ("quant2d", "No image files found")]
)
def test_empty_input_dir_is_a_usage_error_and_leaves_earlier_outputs(tmp_path, cmd, message):
    # Regression: an empty input exited 0, so a rerun "succeeded" next to the previous
    # run's tables.
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "per_file.csv").write_text("earlier run")
    res = CliRunner().invoke(cli, [cmd, "--input", str(empty), "--output", str(out)])
    assert res.exit_code == 2
    assert message in res.output
    assert (out / "per_file.csv").read_text() == "earlier run"
    assert not (out / "run_config.json").exists()


def test_get_condition_grandparent(tmp_path):
    """grandparent mode returns parent.parent.name."""
    p = tmp_path / "GrandCond" / "subfolder" / "img.tif"
    p.parent.mkdir(parents=True)
    p.touch()
    assert get_condition(p, "grandparent") == "GrandCond"


def test_get_condition_filename(tmp_path):
    """filename mode returns file stem."""
    p = tmp_path / "somegroup" / "replicate01.tif"
    p.parent.mkdir()
    p.touch()
    assert get_condition(p, "filename") == "replicate01"


def test_cli_quant3d_condition_from_grandparent(tmp_path):
    """--condition-from grandparent works end-to-end."""
    # structure: in/parent/cond_A/s0.tif
    d = tmp_path / "in" / "cond_A" / "sub"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "s0.tif", blob_3d(0))
    res = CliRunner().invoke(
        cli,
        [
            "quant3d",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--condition-from",
            "grandparent",
            "--no-skeleton",
            "--no-plots",
        ],
    )
    assert res.exit_code == 0, res.output
    assert pd.read_csv(tmp_path / "out" / "per_file.csv")["condition"].tolist() == ["cond_A"]


def test_cli_quant3d_condition_from_filename(tmp_path):
    """--condition-from filename works end-to-end."""
    d = tmp_path / "in" / "sub"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "replicate01.tif", blob_3d(2))
    res = CliRunner().invoke(
        cli,
        [
            "quant3d",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--condition-from",
            "filename",
            "--no-skeleton",
            "--no-plots",
        ],
    )
    assert res.exit_code == 0, res.output
    assert pd.read_csv(tmp_path / "out" / "per_file.csv")["condition"].tolist() == ["replicate01"]


@pytest.mark.parametrize("cmd", ["quant2d", "quant3d"])
@pytest.mark.parametrize("sub", [".", "out", "GelMA/out"])
def test_quant_refuses_output_inside_input(tmp_path, cmd, sub):
    # Regression: an output folder inside the input folder was accepted, so a rerun
    # picked up the tool's own overlay/plot PNGs as input images.
    make_dataset(tmp_path / "in", blob_2d)
    out = tmp_path / "in" / sub
    res = CliRunner().invoke(cli, [cmd, "--input", str(tmp_path / "in"), "--output", str(out)])
    assert res.exit_code == 2
    assert "inside the input folder" in res.output
    assert not (out / "per_file.csv").exists()


def test_rerun_removes_only_stale_outputs_it_owns(tmp_path):
    # Regression: plots and pvalues.csv from an earlier run survived a rerun that did
    # not rewrite them, leaving stale results next to new ones.

    make_dataset(tmp_path / "in", blob_2d)
    out = tmp_path / "out"
    args = ["quant2d", "--input", str(tmp_path / "in"), "--output", str(out), "--no-overlays"]
    assert CliRunner().invoke(cli, args).exit_code == 0
    plot_dir = out / "plots"
    assert (plot_dir / "pvalues.csv").exists() and (plot_dir / "area_fraction_bar.png").exists()
    (plot_dir / "my_notes.txt").write_text("not the tool's")
    (plot_dir / "custom.png").write_bytes(b"x")

    # one replicate per condition: box plots are rewritten, bars and p-values are not
    for cond in ("GelMA", "Hybrid"):
        for i in (1, 2):
            (tmp_path / "in" / cond / f"s{i}.tif").unlink()
    res = CliRunner().invoke(cli, args)
    assert res.exit_code == 0, res.output
    assert not (plot_dir / "pvalues.csv").exists()
    assert not list(plot_dir.glob("*_bar.png"))
    assert (plot_dir / "area_fraction_box.png").exists()
    assert "pvalues.csv" in res.output and "area_fraction_bar.png" in res.output
    assert "area_fraction_box.png" not in res.output  # rewritten, not removed
    assert (plot_dir / "my_notes.txt").exists() and (plot_dir / "custom.png").exists()
    assert len(pd.read_csv(out / "per_file.csv")) == 2

    # --no-plots: every plot the tool named goes; foreign files stay
    res = CliRunner().invoke(cli, args + ["--no-plots"])
    assert res.exit_code == 0, res.output
    assert sorted(p.name for p in plot_dir.iterdir()) == ["custom.png", "my_notes.txt"]


@pytest.mark.parametrize(
    "cmd,maker,extra", [("quant2d", blob_2d, []), ("quant3d", blob_3d, ["--no-skeleton"])]
)
def test_overlays_of_same_named_files_in_different_conditions_are_kept(tmp_path, cmd, maker, extra):
    make_dataset(tmp_path / "in", maker)
    res = CliRunner().invoke(
        cli,
        [
            cmd,
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--no-plots",
            *extra,
        ],
    )
    assert res.exit_code == 0, res.output
    made = sorted(
        p.relative_to(tmp_path / "out" / "overlays").as_posix()
        for p in (tmp_path / "out" / "overlays").rglob("*.png")
    )
    assert made == [f"{c}/s{i}_overlay.png" for c in ("GelMA", "Hybrid") for i in range(3)]


def test_rerun_removes_stale_overlays_and_writes_run_config_last(tmp_path, monkeypatch):
    # Regression: overlays of files no longer in the input survived a rerun, and
    # run_config.json was written before measuring, so a run that failed part-way left a
    # folder claiming the new configuration over the old run's tables.
    from fluorostats import cli as cli_mod

    make_dataset(tmp_path / "in", blob_2d)
    out = tmp_path / "out"
    args = ["quant2d", "--input", str(tmp_path / "in"), "--output", str(out), "--no-plots"]
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert (out / "overlays" / "GelMA" / "s2_overlay.png").exists()
    (out / "overlays" / "GelMA" / "notes.png").write_bytes(b"x")  # not the tool's

    (tmp_path / "in" / "GelMA" / "s2.tif").unlink()
    res = CliRunner().invoke(cli, args)
    assert res.exit_code == 0, res.output
    assert not (out / "overlays" / "GelMA" / "s2_overlay.png").exists()
    assert "s2_overlay.png (not produced by this run)" in res.output
    assert "s1_overlay.png" not in res.output  # rewritten, not removed
    assert (out / "overlays" / "GelMA" / "notes.png").exists()
    assert json.loads((out / "run_config.json").read_text())["threshold"] == "li"

    def fail(*a, **k):
        raise RuntimeError("measurement failed")

    monkeypatch.setattr(cli_mod, "_measure_image", fail)
    res = CliRunner().invoke(cli, args)
    assert isinstance(res.exception, RuntimeError)
    assert not (out / "run_config.json").exists()


def test_quant3d_measures_a_keyence_folder_as_one_volume(tmp_path):
    # Regression: inputs were found by suffix only, so every *_CHF slice of a Keyence
    # BZ-X folder was measured as its own sample and counted as a replicate.
    for cond, tubes in (("GelMA", ("tube1", "tube2")), ("Hybrid", ("tube3",))):
        for ti, tube in enumerate(tubes):
            d = tmp_path / "in" / cond / tube
            d.mkdir(parents=True)
            vol = blob_3d(ti)
            for zi in range(vol.shape[0]):
                tifffile.imwrite(
                    d / f"Image_Z{zi + 1:03d}_CHF.bz.ome.tif", np.stack([vol[zi], vol[zi] // 2])
                )
    tifffile.imwrite(tmp_path / "in" / "Hybrid" / "plain.tif", blob_3d(5))
    res = CliRunner().invoke(
        cli,
        [
            "quant3d",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--no-skeleton",
            "--no-plots",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Found 4 volume file(s)" in res.output
    df = pd.read_csv(tmp_path / "out" / "per_file.csv")
    assert df[["file", "condition"]].values.tolist() == [
        ["tube1", "GelMA"],
        ["tube2", "GelMA"],
        ["plain.tif", "Hybrid"],
        ["tube3", "Hybrid"],
    ]
    assert (df.loc[df["file"] != "plain.tif", "shape"] == "(2, 8, 48, 48)").all()
    assert (tmp_path / "out" / "overlays" / "GelMA" / "tube1_overlay.png").exists()


@pytest.mark.parametrize("cmd", ["quant2d", "quant3d"])
def test_condition_modes_come_from_one_registry(cmd):
    from fluorostats.cli import CONDITION_FROM

    option = next(p for p in cli.commands[cmd].params if p.name == "condition_from")
    assert (
        list(option.type.choices) == list(CONDITION_FROM) == ["parent", "grandparent", "filename"]
    )
