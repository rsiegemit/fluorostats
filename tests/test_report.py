"""Tests for fluorostats.report: the per-file / per-condition CSVs and run_config.json."""

import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fluorostats
from fluorostats import report


def _rows():
    return [
        {
            "file": f"{c}{i}.tif",
            "condition": c,
            "replicate": i,
            "area_fraction": v,
            "n_components": n,
        }
        for c, vals in (("a", [(0.1, 2), (0.3, 4), (0.5, 9)]), ("b", [(0.2, 1), (np.nan, 3)]))
        for i, (v, n) in enumerate(vals, 1)
    ]


def test_per_file_csv_has_the_mode_columns_in_order(tmp_path):
    df = report.write_per_file_csv(_rows(), tmp_path / "sub" / "pf.csv", mode="2d")
    on_disk = pd.read_csv(tmp_path / "sub" / "pf.csv")
    assert list(df.columns) == list(on_disk.columns)
    assert list(df.columns[:3]) == ["file", "condition", "replicate"]
    assert list(df.columns).index("area_fraction") < list(df.columns).index("n_components")
    assert df["coverage_tile_cv"].isna().all()  # absent metric -> blank column
    df3 = report.write_per_file_csv(_rows(), tmp_path / "pf3.csv", mode="3d")
    assert "volume_fraction" in df3.columns and "area_fraction" not in df3.columns


def test_per_condition_csv_count_mean_std_median(tmp_path):
    df = report.write_per_file_csv(_rows(), tmp_path / "pf.csv", mode="2d")
    summary = report.write_per_condition_csv(df, tmp_path / "pc.csv", mode="2d")
    a = summary.set_index("condition").loc["a"]
    assert a["area_fraction_count"] == 3
    assert a["area_fraction_mean"] == pytest.approx(0.3)
    assert a["area_fraction_std"] == pytest.approx(0.2)  # sample SD (ddof=1)
    assert a["area_fraction_median"] == pytest.approx(0.3)
    assert a["n_components_median"] == 4
    b = summary.set_index("condition").loc["b"]
    assert b["area_fraction_count"] == 1  # NaN is not counted
    assert np.isnan(b["area_fraction_std"])
    pd.testing.assert_frame_equal(summary, pd.read_csv(tmp_path / "pc.csv"), check_dtype=False)
    # metrics absent from the table are left out rather than reported as empty
    assert not any(c.startswith("volume_fraction") for c in summary.columns)


@pytest.mark.parametrize("writer", ["write_per_file_csv", "write_per_condition_csv"])
def test_unknown_mode_is_refused(tmp_path, writer):
    # Regression: any mode other than "3d" (e.g. a typo "3D") silently gave 2D columns.
    arg = _rows() if writer == "write_per_file_csv" else pd.DataFrame(_rows())
    with pytest.raises(ValueError, match="Unknown report mode '3D'"):
        getattr(report, writer)(arg, tmp_path / "x.csv", mode="3D")
    assert not (tmp_path / "x.csv").exists()


def test_environment_records_versions_and_readers():
    env = report.environment()
    assert env["packages"]["numpy"] == np.__version__
    assert env["packages"]["pandas"] == pd.__version__
    assert set(env["optional_readers"]) == set(report.OPTIONAL_READERS)
    assert all(isinstance(v, bool) for v in env["optional_readers"].values())
    assert env["python"].count(".") == 2


def test_environment_records_none_for_an_uninstalled_package(monkeypatch):
    real = importlib.metadata.version

    def version(name):
        if name == "skan":
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    env = report.environment()
    assert env["packages"]["skan"] is None
    assert env["packages"]["numpy"] == np.__version__


def test_environment_records_the_code_that_ran_not_the_installed_dist(monkeypatch):
    # Regression: the fluorostats version came from the installed distribution, so a
    # run from a source checkout over an old install recorded the old version.
    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda n: "0.1.0" if n == "fluorostats" else real(n)
    )
    env = report.environment()
    assert env["packages"]["fluorostats"] == fluorostats.__version__
    # From a git checkout the commit is its HEAD; from an archive or an installed
    # package (no git, or the package not tracked) it is None (M7: the suite ran
    # only inside a checkout).
    commit = env["fluorostats_commit"]
    assert (commit and commit.removesuffix("-dirty")) == _tracked_head(Path(report.__file__).parent)


def _tracked_head(package_dir: Path) -> str | None:
    """Short HEAD of the git checkout that tracks ``package_dir``; None when none does."""
    try:
        for args in (
            ["ls-files", "--error-unmatch", "__init__.py"],
            ["rev-parse", "--short", "HEAD"],
        ):
            run = subprocess.run(["git", *args], cwd=package_dir, capture_output=True, text=True)
            if run.returncode:
                return None
    except OSError:  # no git on this machine
        return None
    return run.stdout.strip()


def test_git_commit_is_none_outside_a_checkout_and_flags_local_changes(tmp_path, monkeypatch):
    assert report._git_commit(tmp_path) is None  # not a git checkout
    monkeypatch.setattr(report.subprocess, "run", _raise(OSError("no git")))
    assert report._git_commit(Path(report.__file__).parent) is None
    monkeypatch.setattr(
        report.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(
            args, 0, "M x.py\n" if "status" in args else "abc1234\n"
        ),
    )
    assert report._git_commit(tmp_path) == "abc1234-dirty"


def _raise(exc):
    def run(*args, **kwargs):
        raise exc

    return run


def test_write_run_config_keeps_scalars_and_paths_and_nested(tmp_path):
    params = {
        "input_dir": Path("/data/in"),
        "sigma": 1.5,
        "n": 3,
        "flag": True,
        "channel": None,
        "name": "x",
        "files": [Path("a")],
        "obj": object(),
    }
    path = report.write_run_config(tmp_path, params, nested={"window": (0, 20), "p": Path("q")})
    assert path == tmp_path / "run_config.json"
    cfg = json.loads(path.read_text())
    assert {k: cfg[k] for k in ("input_dir", "sigma", "n", "flag", "channel", "name")} == {
        "input_dir": "/data/in",
        "sigma": 1.5,
        "n": 3,
        "flag": True,
        "channel": None,
        "name": "x",
    }
    assert "files" not in cfg and "obj" not in cfg  # non-scalars dropped
    assert cfg["window"] == [0, 20] and cfg["p"] == "q"  # nested kept, str fallback
    assert cfg["_environment"] == json.loads(json.dumps(report.environment()))


def test_write_run_config_records_the_tool_first(tmp_path):
    cfg = json.loads(report.write_run_config(tmp_path, {"a": 1}, tool="depth").read_text())
    assert next(iter(cfg)) == "tool" and cfg["tool"] == "depth"


def _record(folder, **cfg):
    (folder / "run_config.json").write_text(json.dumps(cfg))


def test_claim_output_dir_removes_an_earlier_run_of_the_same_tool(tmp_path):
    _record(tmp_path, tool="depth")
    owned = [tmp_path / "a.csv", tmp_path / "b.png", tmp_path / "run_config.json"]
    owned[0].write_text("old")
    (tmp_path / "mine.txt").write_text("user")
    assert report.claim_output_dir(tmp_path, "depth", owned) == sorted(owned[::2])
    assert not owned[0].exists() and (tmp_path / "mine.txt").exists()
    marker = {"tool": "depth", "status": "incomplete"}  # M3: the folder stays the tool's
    assert json.loads((tmp_path / "run_config.json").read_text()) == marker
    new = tmp_path / "new"
    assert report.claim_output_dir(new, "depth", [new / p.name for p in owned]) == []
    assert not new.exists()  # created by the run's first write, not here


def test_mark_output_dir_creates_the_folder_and_the_record_completes_it(tmp_path):
    out = tmp_path / "a" / "out"
    report.mark_output_dir(out, "quant2d")
    marker = {"tool": "quant2d", "status": "incomplete"}
    assert json.loads((out / "run_config.json").read_text()) == marker
    cfg = json.loads(report.write_run_config(out, {"x": 1}, tool="quant2d").read_text())
    assert list(cfg)[:2] == ["tool", "status"] and cfg["status"] == "complete"


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"tool": "viability"}, "written by fluorostats viability"),
        ({"_environment": {}}, "an earlier fluorostats version .* --overwrite once"),
    ],
)
def test_claim_output_dir_refuses_a_folder_another_run_wrote(tmp_path, record, expected):
    # Regression (M1): a depth run deleted and overwrote another tool's (or a
    # pre-record run's) outputs and left one run_config naming only itself.
    _record(tmp_path, **record)
    (tmp_path / "contrasts.csv").write_text("theirs")
    owned = [tmp_path / "contrasts.csv", tmp_path / "run_config.json"]
    with pytest.raises(ValueError, match=expected):
        report.claim_output_dir(tmp_path, "depth", owned)
    assert (tmp_path / "contrasts.csv").read_text() == "theirs"


def test_claim_output_dir_refuses_owned_names_without_a_record(tmp_path):
    # Regression (M1): a user's contrasts.csv was replaced without notice.
    (tmp_path / "contrasts.csv").write_text("mine")
    owned = [tmp_path / "contrasts.csv", tmp_path / "x.csv", tmp_path / "run_config.json"]
    with pytest.raises(ValueError, match="holds contrasts.csv but no run_config.json"):
        report.claim_output_dir(tmp_path, "depth", owned)
    with pytest.warns(RuntimeWarning, match="replacing contrasts.csv, not written by"):
        assert report.claim_output_dir(tmp_path, "depth", owned, overwrite=True) == owned[:1]
    assert not (tmp_path / "contrasts.csv").exists()


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '{"tool": 3}'])
def test_claim_output_dir_treats_an_unreadable_record_as_unrecorded(tmp_path, text):
    (tmp_path / "run_config.json").write_text(text)
    with pytest.raises(ValueError, match="an unrecorded run"):
        report.claim_output_dir(tmp_path, "depth", [tmp_path / "run_config.json"])


@pytest.mark.parametrize("overwrite", [False, True])
def test_claim_output_dir_refuses_a_folder_at_an_output_name(tmp_path, overwrite):
    # Regression (L6): a folder named like an output crashed with PermissionError.
    (tmp_path / "contrasts.csv").mkdir()
    owned = [tmp_path / "contrasts.csv", tmp_path / "run_config.json"]
    with pytest.raises(ValueError, match="contrasts.csv is a folder"):
        report.claim_output_dir(tmp_path, "depth", owned, overwrite=overwrite)
    assert (tmp_path / "contrasts.csv").is_dir()


def test_claim_output_dir_removes_a_symlink_at_an_output_name_not_its_target(tmp_path):
    # Regression (L7): a dangling symlink was kept, so the run wrote through it to a
    # file outside the output folder.
    out = tmp_path / "out"
    out.mkdir()
    _record(out, tool="depth")
    (tmp_path / "kept.csv").write_text("theirs")
    (out / "a.csv").symlink_to(tmp_path / "kept.csv")
    (out / "b.csv").symlink_to(tmp_path / "absent.csv")
    owned = [out / "a.csv", out / "b.csv", out / "run_config.json"]
    assert report.claim_output_dir(out, "depth", owned) == owned
    assert not any(p.is_symlink() for p in owned)
    assert (tmp_path / "kept.csv").read_text() == "theirs"
    assert not (tmp_path / "absent.csv").exists()


def test_claim_output_dir_names_files_relative_to_the_folder(tmp_path):
    # Regression (L8): same-named overlays in different subfolders were listed by
    # base name only, so they read as duplicates.
    owned = [tmp_path / c / "s0_overlay.png" for c in ("A", "B")]
    for p in owned:
        p.parent.mkdir()
        p.write_bytes(b"x")
    with pytest.raises(ValueError, match="holds A/s0_overlay.png, B/s0_overlay.png but no"):
        report.claim_output_dir(tmp_path, "quant2d", owned)
    with pytest.warns(RuntimeWarning, match="replacing A/s0_overlay.png, B/s0_overlay.png,"):
        report.claim_output_dir(tmp_path, "quant2d", owned, overwrite=True)
