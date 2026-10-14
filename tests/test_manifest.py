import json

import pytest

from fluorostats import depth_batch, viability_batch
from fluorostats.manifest import check_groups


def _write(tmp_path, m):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    return p


def test_check_groups_reports_every_problem(tmp_path):
    ok = tmp_path / "a.tif"
    ok.write_bytes(b"")
    groups = [
        {"stacks": [{"path": str(ok), "label": "a", "role": "Primary"}]},       # no name, bad role
        {"name": "g", "stacks": []},                                             # no stacks
        {"name": "h", "background": [str(tmp_path / "nope.tif")],
         "stacks": [{"path": str(tmp_path / "missing.tif"), "label": "m"}, {"path": str(ok)}]},
    ]
    problems = check_groups(groups, roles=("primary", "aux"))
    text = "\n".join(problems)
    for expected in ("group 0 has no 'name'", "role must be one of", "'g' has no stacks",
                     "background", "missing.tif does not exist", "needs 'path' and 'label'"):
        assert expected in text
    assert check_groups([{"name": "x", "stacks": [{"path": str(ok), "label": "x"}]}]) == []


def test_depth_manifest_rejected_before_loading(tmp_path):
    ok = tmp_path / "a.tif"
    ok.write_bytes(b"")
    m = {"channel": -1, "reducer": "max", "n_surface": 0,
         "auc_windows_um": [[50, 10], "whole"],
         "groups": [{"name": "g", "stacks": [{"path": str(ok), "label": "a", "role": "x"}]}]}
    with pytest.raises(ValueError) as e:
        depth_batch.load_manifest(_write(tmp_path, m))
    msg = str(e.value)
    for expected in ("channel must be", "reducer must be", "n_surface must be",
                     "[50, 10]", "'whole'", "role must be"):
        assert expected in msg


def test_viability_manifest_rejected_before_loading(tmp_path):
    m = {"live_channel": 1, "dead_channel": 1, "count_method": "best",
         "groups": [{"name": "d1", "stacks": [{"path": str(tmp_path / "x.tif"), "label": "x"}]}]}
    with pytest.raises(ValueError) as e:
        viability_batch.load_manifest(_write(tmp_path, m))
    msg = str(e.value)
    for expected in ("both 1", "count_method must be", "x.tif does not exist"):
        assert expected in msg
    m2 = dict(m, live_channel="0", dead_channel=-2, count_method="cc", groups=[])
    with pytest.raises(ValueError, match="non-empty 'groups'"):
        viability_batch.load_manifest(_write(tmp_path, m2))
    m3 = dict(m, live_channel="0", dead_channel=-2, count_method="cc",
              groups=m["groups"])
    with pytest.raises(ValueError, match="live_channel must be.*\\n.*dead_channel must be"):
        viability_batch.load_manifest(_write(tmp_path, m3))


def test_batch_runs_record_resolved_manifest_and_environment(tmp_path):
    import numpy as np
    import tifffile

    z = np.arange(8)[:, None, None]
    for name in ("a", "b"):
        tifffile.imwrite(tmp_path / f"{name}.tif",
                         np.broadcast_to(100 * np.exp(-z / 4.0) + 5, (8, 6, 6)).astype(np.float32))
    m = {"auc_windows_um": [[0, 4]],
         "groups": [{"name": "g", "stacks": [{"path": str(tmp_path / "a.tif"), "label": "a"},
                                             {"path": str(tmp_path / "b.tif"), "label": "b"}]}]}
    out = depth_batch.run(_write(tmp_path, m), output_override=str(tmp_path / "out"))
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["manifest"].endswith("m.json")
    assert cfg["resolved"]["reducer"] == "mean"            # defaults are recorded too
    assert cfg["resolved"]["windows"] == [[0.0, 4.0]]
    assert "numpy" in cfg["_environment"]["packages"]
