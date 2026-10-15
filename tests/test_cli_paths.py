"""Tests for fluorostats.cli_paths: condition labels from paths (the quant commands
using cli_paths end to end are tested in test_cli.py and test_cli_outputs.py)."""

import pytest

from fluorostats.cli_paths import find_files, get_condition


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


@pytest.mark.parametrize("name", ["Ctrl_1.ome.tif", "Ctrl_1.ome.tiff", "Ctrl_1.OME.TIF"])
def test_get_condition_filename_drops_the_whole_ome_suffix(tmp_path, name):
    # Regression (C4): Path.stem strips only ".tif", so the label was "Ctrl_1.ome".
    assert get_condition(tmp_path / name, "filename") == "Ctrl_1"


def test_find_files_skips_the_planes_of_an_oif_companion_folder(tmp_path):
    # Regression (F8): the per-plane TIFFs in "<name>.oif.files" were measured as
    # samples of their own (a spurious condition "Z-03.oif.files" with 18 replicates).
    (tmp_path / "Z-03.oif").write_bytes(b"")
    planes = tmp_path / "Z-03.oif.files"
    planes.mkdir()
    for i in range(3):
        (planes / f"s_C001Z00{i}.tif").write_bytes(b"")
    orphan = tmp_path / "other.oif.files"  # no other.oif beside it: its files stay
    orphan.mkdir()
    (orphan / "a.tif").write_bytes(b"")
    found = find_files(tmp_path, {".oif", ".tif"})
    assert found == [tmp_path / "Z-03.oif", orphan / "a.tif"]
