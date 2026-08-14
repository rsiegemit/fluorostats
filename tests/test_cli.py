"""CLI smoke tests — exercise the quant2d / quant3d pipelines end to end.

These drive io -> preprocess -> segment -> metrics -> report -> qc -> plots in
one pass, so they guard the whole command path (previously 0% covered)."""
import numpy as np
import pytest
from click.testing import CliRunner

from fluorostats.cli import cli

tifffile = pytest.importorskip("tifffile")


def _blob_2d(seed):
    img = np.random.default_rng(seed).integers(0, 40, (64, 64)).astype(np.uint8)
    img[16:48, 16:48] = 220                     # bright foreground patch
    return img


def _blob_3d(seed):
    vol = np.random.default_rng(seed).integers(0, 40, (8, 48, 48)).astype(np.uint16)
    vol[2:6, 12:36, 12:36] = 900
    return vol


def _make_dataset(root, maker):
    for ci, cond in enumerate(("GelMA", "Hybrid")):
        d = root / cond
        d.mkdir(parents=True)
        for i in range(3):                      # 3 replicates -> plots + p-values run
            tifffile.imwrite(d / f"s{i}.tif", maker(10 * ci + i))


def test_cli_quant2d(tmp_path):
    _make_dataset(tmp_path / "in", _blob_2d)
    res = CliRunner().invoke(cli, ["quant2d", "--input", str(tmp_path / "in"),
                                   "--output", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output
    assert list((tmp_path / "out").rglob("*.csv"))          # per-file / per-condition written


def test_cli_quant3d(tmp_path):
    _make_dataset(tmp_path / "in", _blob_3d)
    res = CliRunner().invoke(cli, ["quant3d", "--input", str(tmp_path / "in"),
                                   "--output", str(tmp_path / "out"), "--no-skeleton"])
    assert res.exit_code == 0, res.output
    assert list((tmp_path / "out").rglob("*.csv"))


def test_cli_formats():
    res = CliRunner().invoke(cli, ["formats"])
    assert res.exit_code == 0
