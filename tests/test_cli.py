"""Tests for fluorostats.cli. The smoke tests exercise the quant2d / quant3d pipelines end to end.

These drive io -> preprocess -> segment -> metrics -> report -> qc -> plots in
one pass, so they guard the whole command path (previously 0% covered)."""

import json

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner

from cli_data import blob_2d, blob_3d, make_dataset
from fluorostats.cli import cli, _parse_channel


def test_cli_quant2d(tmp_path):
    make_dataset(tmp_path / "in", blob_2d)
    res = CliRunner().invoke(
        cli, ["quant2d", "--input", str(tmp_path / "in"), "--output", str(tmp_path / "out")]
    )
    assert res.exit_code == 0, res.output
    df = pd.read_csv(tmp_path / "out" / "per_file.csv")
    assert len(df) == 6 and df["area_fraction"].between(0, 1).all()
    assert set(pd.read_csv(tmp_path / "out" / "per_condition.csv")["condition"]) == {
        "GelMA",
        "Hybrid",
    }
    assert (tmp_path / "out" / "plots" / "pvalues.csv").exists()


def test_cli_quant3d(tmp_path):
    make_dataset(tmp_path / "in", blob_3d)
    res = CliRunner().invoke(
        cli,
        [
            "quant3d",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--no-skeleton",
        ],
    )
    assert res.exit_code == 0, res.output
    assert list((tmp_path / "out").rglob("*.csv"))
    # each row records which channel was measured and where its calibration came from
    df = pd.read_csv(tmp_path / "out" / "per_file.csv")
    assert {"voxel_size_source", "channel_used"} <= set(df.columns)
    assert df["voxel_size_source"].str.contains("default").all()  # synthetic TIFFs
    assert df["channel_used"].iloc[0].startswith("0")


def test_cli_formats():
    res = CliRunner().invoke(cli, ["formats"])
    assert res.exit_code == 0
    assert "[x] .tif" in res.output and "pip install" in res.output


# ---------------------------------------------------------------------------
# quant3d with --bg-radius > 0  (line 144)
# ---------------------------------------------------------------------------


def test_quant3d_bg_radius(tmp_path):
    """quant3d with --bg-radius should run background_subtract and exit 0."""
    d = tmp_path / "in" / "cond"
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
            "--bg-radius",
            "5",
            "--no-skeleton",
            "--no-plots",
        ],
    )
    assert res.exit_code == 0, res.output
    assert json.loads((tmp_path / "out" / "run_config.json").read_text())["bg_radius"] == 5
    assert len(pd.read_csv(tmp_path / "out" / "per_file.csv")) == 1


# ---------------------------------------------------------------------------
# quant3d without --no-skeleton  (line 155)
# ---------------------------------------------------------------------------


def test_quant3d_with_skeleton(tmp_path):
    """quant3d without --no-skeleton exercises the skeleton_metrics branch."""
    d = tmp_path / "in" / "cond"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "s0.tif", blob_3d(1))
    res = CliRunner().invoke(
        cli,
        [
            "quant3d",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--no-plots",
        ],
    )
    assert res.exit_code == 0, res.output
    row = pd.read_csv(tmp_path / "out" / "per_file.csv").iloc[0]
    # measured (a number, possibly 0), not left blank as with --no-skeleton
    assert row[["total_length_um", "n_branches", "n_junctions"]].notna().all()


# ---------------------------------------------------------------------------
# _write_plots_and_pvalues early return when single condition  (line 308)
# ---------------------------------------------------------------------------


def test_quant3d_single_condition_no_plots(tmp_path):
    """Single-condition run with --no-plots hits the early return in _write_plots_and_pvalues."""
    d = tmp_path / "in" / "GelMA"
    d.mkdir(parents=True)
    for i in range(3):
        tifffile.imwrite(d / f"s{i}.tif", blob_3d(i))
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
    assert not (tmp_path / "out" / "plots").exists()
    assert pd.read_csv(tmp_path / "out" / "per_condition.csv")["condition"].tolist() == ["GelMA"]


def test_quant2d_single_condition_no_plots(tmp_path):
    """quant2d single condition triggers _write_plots_and_pvalues early return."""
    d = tmp_path / "in" / "GelMA"
    d.mkdir(parents=True)
    tifffile.imwrite(d / "s0.tif", blob_2d(0))
    res = CliRunner().invoke(
        cli,
        [
            "quant2d",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--no-plots",
        ],
    )
    assert res.exit_code == 0, res.output
    assert not (tmp_path / "out" / "plots").exists()
    assert pd.read_csv(tmp_path / "out" / "per_condition.csv")["condition"].tolist() == ["GelMA"]


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
    tifffile.imwrite(d / "s0.tif", blob_3d(3))
    res = CliRunner().invoke(
        cli,
        [
            "quant3d",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--channel",
            "0",
            "--no-skeleton",
            "--no-plots",
        ],
    )
    assert res.exit_code == 0, res.output
    assert pd.read_csv(tmp_path / "out" / "per_file.csv")["channel_used"].tolist() == ["0 (Ch1)"]


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
    tifffile.imwrite(d / "s0.tif", blob_3d(4))
    res = CliRunner().invoke(
        cli,
        [
            "quant3d",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--channel",
            "Ch1",  # matches the auto-named channel in a single-channel tiff
            "--no-skeleton",
            "--no-plots",
        ],
    )
    assert res.exit_code == 0, res.output
    assert pd.read_csv(tmp_path / "out" / "per_file.csv")["channel_used"].tolist() == ["0 (Ch1)"]


@pytest.mark.parametrize(
    "opt,val",
    [("--threshold-scale", "0"), ("--min-size", "-1"), ("--sigma", "-0.5"), ("--bg-radius", "-3")],
)
def test_cli_rejects_out_of_range_options(tmp_path, opt, val):
    (tmp_path / "in").mkdir()
    for cmd in ("quant2d", "quant3d"):
        res = CliRunner().invoke(
            cli, [cmd, "--input", str(tmp_path / "in"), "--output", str(tmp_path / "out"), opt, val]
        )
        assert res.exit_code == 2 and "Invalid value" in res.output


def test_no_skeleton_leaves_skeleton_columns_blank_not_zero(tmp_path):
    # Regression: --no-skeleton wrote 0 into the skeleton columns, which were then
    # summarised, plotted and Mann-Whitney-tested as if they had been measured.

    make_dataset(tmp_path / "in", blob_3d)
    res = CliRunner().invoke(
        cli,
        [
            "quant3d",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--no-skeleton",
            "--no-overlays",
        ],
    )
    assert res.exit_code == 0, res.output
    df = pd.read_csv(tmp_path / "out" / "per_file.csv")
    for col in (
        "total_length_um",
        "n_branches",
        "n_junctions",
        "n_junction_nodes",
        "mean_branch_length_um",
    ):
        assert df[col].isna().all()
    assert df["volume_fraction"].notna().all()
    plots_dir = tmp_path / "out" / "plots"
    assert list(plots_dir.glob("volume_fraction_*"))
    assert not list(plots_dir.glob("total_length_um_*"))
    metrics = pd.read_csv(plots_dir / "pvalues.csv")["metric"].tolist()
    assert "volume_fraction" in metrics and "total_length_um" not in metrics


def test_quant2d_records_calibration_crop_box_and_um2_cluster_areas(tmp_path, monkeypatch):
    # quant2d dropped the loader's pixel size and the auto-crop box, so the µm² cluster
    # areas never reached any output.
    from fluorostats import io as iq_io

    make_dataset(tmp_path / "in", blob_2d)
    load_image = iq_io.load_image

    def calibrated(path):
        arr, meta = load_image(path)
        return arr, {**meta, "voxel_size_um": (0.5, 0.5), "voxel_size_source": "y:file,x:file"}

    args = ["quant2d", "--input", str(tmp_path / "in"), "--no-overlays", "--no-plots"]
    assert CliRunner().invoke(cli, args + ["--output", str(tmp_path / "raw")]).exit_code == 0
    raw = pd.read_csv(tmp_path / "raw" / "per_file.csv")
    assert (raw["voxel_size_um"] == "(1.0, 1.0)").all()
    assert (raw["voxel_size_source"] == "y:default,x:default").all()
    assert (raw["mean_cluster_area_um2"] == raw["mean_cluster_area_px"]).all()
    assert raw["auto_crop_box"].str.fullmatch(r"\[\d+:\d+, \d+:\d+\]").all()

    monkeypatch.setattr(iq_io, "load_image", calibrated)
    res = CliRunner().invoke(cli, args + ["--output", str(tmp_path / "cal"), "--no-auto-crop"])
    assert res.exit_code == 0, res.output
    cal = pd.read_csv(tmp_path / "cal" / "per_file.csv")
    assert (cal["voxel_size_um"] == "(0.5, 0.5)").all()
    assert (cal["voxel_size_source"] == "y:file,x:file").all()
    assert np.allclose(cal["mean_cluster_area_um2"], 0.25 * cal["mean_cluster_area_px"])
    assert np.allclose(cal["median_cluster_area_um2"], 0.25 * cal["median_cluster_area_px"])
    assert cal["auto_crop_box"].isna().all()  # not cropped
    # the existing columns keep their order; the record columns follow them
    from fluorostats import report

    assert list(cal.columns[:-5]) == [
        "file",
        "condition",
        "replicate",
        "shape",
        "channel_used",
        *report.METRICS_2D,
    ]


@pytest.mark.parametrize("cmd,default", [("quant2d", "[default: 15"), ("quant3d", "[default: 0")])
def test_help_states_defaults_and_result_changing_options(cmd, default):
    text = " ".join(CliRunner().invoke(cli, [cmd, "--help"]).output.split())
    assert "Background subtraction disk radius in pixels (0=off" in text and default in text
    for opt in ("--channel", "--min-size", "--sigma", "--bg-radius", "--threshold-scale"):
        help_line = text.split(opt, 1)[1].split(" --", 1)[0]
        assert "CAN AFFECT RESULTS" in help_line, opt


@pytest.mark.parametrize(
    "args, expected",
    [([], "ball"), (["--opening", "disk"], "disk"), (["--opening", "none"], "none")],
)
def test_quant3d_opening_reaches_segmentation(tmp_path, monkeypatch, args, expected):
    """--opening picks segment_channel's salt-noise opening (default ball) and is recorded."""
    from fluorostats import segment

    seen = []
    real = segment.segment_channel
    monkeypatch.setattr(
        segment, "segment_channel", lambda img, **k: seen.append(k["opening"]) or real(img, **k)
    )
    d = tmp_path / "in" / "cond"
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
            "--no-skeleton",
            "--no-plots",
            "--no-overlays",
            *args,
        ],
    )
    assert res.exit_code == 0, res.output
    assert seen == [expected]
    assert json.loads((tmp_path / "out" / "run_config.json").read_text())["opening"] == expected


def test_quant3d_opening_choices_and_help():
    from fluorostats.segment import OPENINGS

    res = CliRunner().invoke(cli, ["quant3d", "--help"])
    text = " ".join(res.output.split())
    assert "--opening" in text and "CAN AFFECT RESULTS" in text.split("--opening")[1]
    assert all(name in text.split("--opening")[1][:400] for name in OPENINGS)
    bad = CliRunner().invoke(cli, ["quant3d", "--input", ".", "--output", "o", "--opening", "cube"])
    assert bad.exit_code == 2
