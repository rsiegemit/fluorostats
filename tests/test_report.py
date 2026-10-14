"""Tests for fluorostats.report: the per-file / per-condition CSVs and run_config.json."""
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fluorostats import report


def _rows():
    return [{"file": f"{c}{i}.tif", "condition": c, "replicate": i,
             "area_fraction": v, "n_components": n}
            for c, vals in (("a", [(0.1, 2), (0.3, 4), (0.5, 9)]), ("b", [(0.2, 1), (np.nan, 3)]))
            for i, (v, n) in enumerate(vals, 1)]


def test_per_file_csv_has_the_mode_columns_in_order(tmp_path):
    df = report.write_per_file_csv(_rows(), tmp_path / "sub" / "pf.csv", mode="2d")
    on_disk = pd.read_csv(tmp_path / "sub" / "pf.csv")
    assert list(df.columns) == list(on_disk.columns)
    assert list(df.columns[:3]) == ["file", "condition", "replicate"]
    assert list(df.columns).index("area_fraction") < list(df.columns).index("n_components")
    assert df["coverage_tile_cv"].isna().all()                    # absent metric -> blank column
    df3 = report.write_per_file_csv(_rows(), tmp_path / "pf3.csv", mode="3d")
    assert "volume_fraction" in df3.columns and "area_fraction" not in df3.columns


def test_per_condition_csv_count_mean_std_median(tmp_path):
    df = report.write_per_file_csv(_rows(), tmp_path / "pf.csv", mode="2d")
    summary = report.write_per_condition_csv(df, tmp_path / "pc.csv", mode="2d")
    a = summary.set_index("condition").loc["a"]
    assert a["area_fraction_count"] == 3
    assert a["area_fraction_mean"] == pytest.approx(0.3)
    assert a["area_fraction_std"] == pytest.approx(0.2)          # sample SD (ddof=1)
    assert a["area_fraction_median"] == pytest.approx(0.3)
    assert a["n_components_median"] == 4
    b = summary.set_index("condition").loc["b"]
    assert b["area_fraction_count"] == 1                          # NaN is not counted
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
    # Running from source (PYTHONPATH=src) has no installed fluorostats distribution.
    real = importlib.metadata.version

    def version(name):
        if name == "fluorostats":
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    env = report.environment()
    assert env["packages"]["fluorostats"] is None
    assert env["packages"]["numpy"] == np.__version__


def test_write_run_config_keeps_scalars_and_paths_and_nested(tmp_path):
    params = {"input_dir": Path("/data/in"), "sigma": 1.5, "n": 3, "flag": True,
              "channel": None, "name": "x", "files": [Path("a")], "obj": object()}
    path = report.write_run_config(tmp_path, params, nested={"window": (0, 20), "p": Path("q")})
    assert path == tmp_path / "run_config.json"
    cfg = json.loads(path.read_text())
    assert {k: cfg[k] for k in ("input_dir", "sigma", "n", "flag", "channel", "name")} == {
        "input_dir": "/data/in", "sigma": 1.5, "n": 3, "flag": True, "channel": None, "name": "x"}
    assert "files" not in cfg and "obj" not in cfg                 # non-scalars dropped
    assert cfg["window"] == [0, 20] and cfg["p"] == "q"            # nested kept, str fallback
    assert cfg["_environment"] == json.loads(json.dumps(report.environment()))
