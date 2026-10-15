"""Tests for the output-folder handling of the quant commands in fluorostats.cli:
empty inputs, an output inside the input, and what a rerun removes (the rules live
in `cli_paths`)."""

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from cli_data import blob_2d, blob_3d, make_dataset
from fluorostats.cli import cli


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


@pytest.mark.parametrize("other", ["quant3d", None])
def test_quant_refuses_a_folder_another_run_wrote_unless_overwrite(tmp_path, other):
    # Regression (M1): quant2d replaced another tool's (or the user's) per_file.csv and
    # run_config.json without notice.
    make_dataset(tmp_path / "in", blob_2d)
    out = tmp_path / "out"
    out.mkdir()
    (out / "per_file.csv").write_text("theirs")
    if other:
        (out / "run_config.json").write_text(f'{{"tool": "{other}"}}')
    args = ["quant2d", "--input", str(tmp_path / "in"), "--output", str(out), "--no-plots"]
    res = CliRunner().invoke(cli, args)
    assert res.exit_code == 2 and "--overwrite" in res.output
    assert (out / "per_file.csv").read_text() == "theirs"
    res = CliRunner().invoke(cli, args + ["--overwrite"])
    assert res.exit_code == 0, res.output
    assert len(pd.read_csv(out / "per_file.csv")) == 6
    assert json.loads((out / "run_config.json").read_text())["tool"] == "quant2d"


def test_a_file_that_cannot_be_measured_is_listed_and_the_run_exits_1(tmp_path):
    # Regression (vendor report): one unreadable file aborted quant2d/quant3d with a
    # traceback and no tables.
    make_dataset(tmp_path / "in", blob_2d)
    (tmp_path / "in" / "GelMA" / "broken.tif").write_bytes(b"not a tiff")
    out = tmp_path / "out"
    args = ["quant2d", "--input", str(tmp_path / "in"), "--output", str(out), "--no-plots"]
    res = CliRunner().invoke(cli, args)
    assert res.exit_code == 1
    assert "1 of 7 file(s) could not be measured" in res.output and "broken.tif" in res.output
    failed = pd.read_csv(out / "failures.csv")
    assert failed[["file", "condition"]].values.tolist() == [["broken.tif", "GelMA"]]
    assert len(pd.read_csv(out / "per_file.csv")) == 6
    assert json.loads((out / "run_config.json").read_text())["tool"] == "quant2d"

    (tmp_path / "in" / "GelMA" / "broken.tif").unlink()
    res = CliRunner().invoke(cli, args)  # fixed input: the stale failures.csv goes
    assert res.exit_code == 0, res.output
    assert not (out / "failures.csv").exists() and "failures.csv" in res.output


def test_a_quant_run_that_writes_nothing_leaves_no_output_folder(tmp_path, monkeypatch):
    # Regression (C3): the output folder was created before anything was written.
    from fluorostats import cli as cli_mod

    def fail(*a, **k):
        raise RuntimeError("reporting failed")

    monkeypatch.setattr(cli_mod, "_report_and_plot", fail)
    make_dataset(tmp_path / "in", blob_2d)
    out = tmp_path / "new" / "out"
    args = ["quant2d", "--input", str(tmp_path / "in"), "--output", str(out), "--no-overlays"]
    assert isinstance(CliRunner().invoke(cli, args).exception, RuntimeError)
    assert not (tmp_path / "new").exists()
