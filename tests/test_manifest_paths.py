"""Relative paths in a manifest file resolve against the manifest's folder, not the
working directory (`manifest.resolve_path`); a dict manifest keeps the working
directory."""

import json

import numpy as np
import pytest

from depth_data import decay_stack, flat_stack
from fluorostats import depth_batch, viability_batch
from fluorostats.manifest import check_groups, resolve_path


def _write(path, m):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m), encoding="utf-8")
    return path


def _depth_manifest(folder, **over):
    decay_stack(folder / "s0.tif")
    flat_stack(folder / "blank.tif")
    m = {
        "output_dir": "out",
        "groups": [
            {
                "name": "A",
                "background": "blank.tif",
                "stacks": [{"label": "s0", "path": "s0.tif"}],
            }
        ],
        **over,
    }
    return _write(folder / "m.json", m)


def test_a_depth_manifest_runs_from_another_working_directory(tmp_path, monkeypatch):
    """(a) stack, background and output_dir are found next to the manifest."""
    sub = tmp_path / "sub"
    sub.mkdir()
    mp = _depth_manifest(sub)
    monkeypatch.chdir(tmp_path)
    out = depth_batch.run(mp)
    assert out.resolve() == (sub / "out").resolve()
    cfg = json.loads((sub / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["files_read"] == {
        "stacks": [str((sub / "s0.tif").resolve())],
        "backgrounds": [str((sub / "blank.tif").resolve())],
        "output_dir": str((sub / "out").resolve()),
    }
    assert not (tmp_path / "out").exists()


def test_a_viability_manifest_runs_from_another_working_directory(tmp_path, monkeypatch):
    """(a) for the viability driver."""
    sub = tmp_path / "sub"
    sub.mkdir()
    rng = np.random.default_rng(0)
    np.save(sub / "s.npy", rng.normal(30, 5, (2, 3, 32, 32)).astype(np.float32))
    m = {
        "live_channel": 0,
        "dead_channel": 1,
        "count_method": "cc",
        "output_dir": "o",
        "groups": [{"name": "g", "stacks": [{"label": "s", "path": "s.npy"}]}],
    }
    mp = _write(sub / "m.json", m)
    monkeypatch.chdir(tmp_path)
    out = viability_batch.run(mp)
    assert out.resolve() == (sub / "o").resolve()
    cfg = json.loads((sub / "o" / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["files_read"]["stacks"] == [str((sub / "s.npy").resolve())]


def test_the_cli_output_option_stays_relative_to_the_working_directory(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    mp = _depth_manifest(sub)
    monkeypatch.chdir(tmp_path)
    assert depth_batch.run(mp, "cli_out").resolve() == (tmp_path / "cli_out").resolve()


@pytest.mark.parametrize("loader", [depth_batch.load_manifest, viability_batch.load_manifest])
def test_a_path_found_only_in_the_working_directory_is_an_error(tmp_path, monkeypatch, loader):
    """(b) no silent fallback to the working directory: the error names both places."""
    (tmp_path / "s0.tif").write_bytes(b"")
    m = {"groups": [{"name": "A", "stacks": [{"label": "s0", "path": "s0.tif"}]}]}
    mp = _write(tmp_path / "sub" / "m.json", m)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="resolved against the manifest's folder") as e:
        loader(mp)
    msg = str(e.value)
    assert str(tmp_path / "sub" / "s0.tif") in msg
    assert str((tmp_path / "s0.tif").resolve()) in msg


def test_a_background_found_only_in_the_working_directory_is_an_error(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    mp = _depth_manifest(sub)
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["groups"][0]["background"] = ["blank.tif", "cwd_blank.tif"]
    _write(mp, m)
    (tmp_path / "cwd_blank.tif").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="cwd_blank.tif"):
        depth_batch.load_manifest(mp)


def test_a_path_missing_everywhere_names_the_manifest_relative_path(tmp_path, monkeypatch):
    m = {"groups": [{"name": "A", "stacks": [{"label": "s0", "path": "nope.tif"}]}]}
    mp = _write(tmp_path / "sub" / "m.json", m)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="invalid depth manifest") as e:
        depth_batch.load_manifest(mp)
    assert f"stack {tmp_path / 'sub' / 'nope.tif'} does not exist" in str(e.value)


def test_absolute_paths_are_unchanged(tmp_path, monkeypatch):
    """(c)"""
    stack = tmp_path / "data" / "s0.tif"
    stack.parent.mkdir()
    decay_stack(stack)
    out = tmp_path / "abs_out"
    m = {
        "output_dir": str(out),
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(stack)}]}],
    }
    mp = _write(tmp_path / "sub" / "m.json", m)
    monkeypatch.chdir(tmp_path / "data")
    loaded = depth_batch.load_manifest(mp)
    assert loaded["groups"][0]["stacks"][0]["path"] == str(stack)
    assert loaded["output_dir"] == str(out)
    assert resolve_path(mp, str(stack)) == str(stack)


def test_dict_manifests_keep_the_working_directory(tmp_path, monkeypatch):
    """(d) with no manifest file (manifest_path None) a path is used as given, so the
    checks on a dict manifest find it relative to the working directory."""
    (tmp_path / "s0.tif").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert resolve_path(None, "s0.tif") == "s0.tif"
    groups = [{"name": "A", "stacks": [{"label": "s0", "path": "s0.tif"}]}]
    assert check_groups(groups) == []


def test_resolve_path_joins_a_relative_path_to_the_manifest_folder(tmp_path):
    mp = tmp_path / "sub" / "m.json"
    assert resolve_path(mp, "a/b.tif") == str(tmp_path / "sub" / "a" / "b.tif")


def test_a_manifest_in_the_working_directory_is_not_ambiguous(tmp_path, monkeypatch):
    """Both rules name the same file: no error, the path is kept as written."""
    mp = _depth_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    m = depth_batch.load_manifest("m.json")
    assert m["groups"][0]["stacks"][0]["path"] == "s0.tif"
    assert resolve_path(mp, "s0.tif") == str(tmp_path / "s0.tif")
