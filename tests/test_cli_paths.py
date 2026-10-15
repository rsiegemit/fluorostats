"""Tests for fluorostats.cli_paths: condition labels from paths (the quant commands
using cli_paths end to end are tested in test_cli.py and test_cli_outputs.py)."""

from fluorostats.cli_paths import get_condition


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
