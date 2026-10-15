import json

import pytest

from fluorostats import depth_batch, viability_batch
from fluorostats.manifest import check_groups, not_strings


def _write(tmp_path, m):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    return p


def test_check_groups_reports_every_problem(tmp_path):
    ok = tmp_path / "a.tif"
    ok.write_bytes(b"")
    groups = [
        {"stacks": [{"path": str(ok), "label": "a", "role": "Primary"}]},  # no name, bad role
        {"name": "g", "stacks": []},  # no stacks
        {
            "name": "h",
            "background": [str(tmp_path / "nope.tif")],
            "stacks": [{"path": str(tmp_path / "missing.tif"), "label": "m"}, {"path": str(ok)}],
        },
    ]
    problems = check_groups(groups, roles=("primary", "aux"))
    text = "\n".join(problems)
    for expected in (
        "group 0 has no 'name'",
        "role must be one of",
        "'g' has no stacks",
        "background",
        "missing.tif does not exist",
        "needs 'path' and 'label'",
    ):
        assert expected in text
    assert check_groups([{"name": "x", "stacks": [{"path": str(ok), "label": "x"}]}]) == []


def test_depth_manifest_rejected_before_loading(tmp_path):
    ok = tmp_path / "a.tif"
    ok.write_bytes(b"")
    m = {
        "channel": -1,
        "reducer": "max",
        "n_surface": 0,
        "auc_windows_um": [[50, 10], "whole"],
        "groups": [{"name": "g", "stacks": [{"path": str(ok), "label": "a", "role": "x"}]}],
    }
    with pytest.raises(ValueError) as e:
        depth_batch.load_manifest(_write(tmp_path, m))
    msg = str(e.value)
    for expected in (
        "channel must be",
        "reducer must be",
        "n_surface must be",
        "[50, 10]",
        "'whole'",
        "role must be",
    ):
        assert expected in msg


def test_viability_manifest_rejected_before_loading(tmp_path):
    m = {
        "live_channel": 1,
        "dead_channel": 1,
        "count_method": "best",
        "groups": [{"name": "d1", "stacks": [{"path": str(tmp_path / "x.tif"), "label": "x"}]}],
    }
    with pytest.raises(ValueError) as e:
        viability_batch.load_manifest(_write(tmp_path, m))
    msg = str(e.value)
    for expected in ("both 1", "count_method must be", "x.tif does not exist"):
        assert expected in msg
    m2 = dict(m, live_channel="0", dead_channel=-2, count_method="cc", groups=[])
    with pytest.raises(ValueError, match="non-empty 'groups'"):
        viability_batch.load_manifest(_write(tmp_path, m2))
    m3 = dict(m, live_channel="0", dead_channel=-2, count_method="cc", groups=m["groups"])
    with pytest.raises(ValueError, match="live_channel must be.*\\n.*dead_channel must be"):
        viability_batch.load_manifest(_write(tmp_path, m3))


def test_batch_runs_record_resolved_manifest_and_environment(tmp_path):
    import numpy as np
    import tifffile

    z = np.arange(8)[:, None, None]
    for name in ("a", "b"):
        tifffile.imwrite(
            tmp_path / f"{name}.tif",
            np.broadcast_to(100 * np.exp(-z / 4.0) + 5, (8, 6, 6)).astype(np.float32),
        )
    m = {
        "auc_windows_um": [[0, 4]],
        "groups": [
            {
                "name": "g",
                "stacks": [
                    {"path": str(tmp_path / "a.tif"), "label": "a"},
                    {"path": str(tmp_path / "b.tif"), "label": "b"},
                ],
            }
        ],
    }
    out = depth_batch.run(_write(tmp_path, m), output_override=str(tmp_path / "out"))
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["manifest"].endswith("m.json")
    assert cfg["resolved"]["reducer"] == "mean"  # defaults are recorded too
    assert cfg["resolved"]["windows"] == [[0.0, 4.0]]
    assert "numpy" in cfg["_environment"]["packages"]


def test_check_groups_rejects_duplicate_group_names(tmp_path):
    """Two groups with one name used to be merged silently into one group."""
    ok = tmp_path / "a.tif"
    ok.write_bytes(b"")
    st = [{"path": str(ok), "label": "a"}]
    problems = check_groups(
        [{"name": "g", "stacks": st}, {"name": "h", "stacks": st}, {"name": "g", "stacks": st}]
    )
    assert problems == ["group name 'g' is used by more than one group; names must be unique"]


def test_check_groups_rejects_duplicate_stack_labels_within_a_group(tmp_path):
    # Regression: two stacks with one label in a group were both analysed and became
    # indistinguishable rows in every table.
    ok = tmp_path / "a.tif"
    ok.write_bytes(b"")
    st = {"path": str(ok), "label": "a"}
    problems = check_groups(
        [{"name": "g", "stacks": [st, st, {**st, "label": "b"}]}, {"name": "h", "stacks": [st]}]
    )  # the same label elsewhere is fine
    assert problems == [
        "group 'g': stack label 'a' is used by more than one stack; "
        "labels must be unique within a group"
    ]


def test_check_groups_rejects_unknown_group_and_stack_fields_when_given(tmp_path):
    ok = tmp_path / "a.tif"
    ok.write_bytes(b"")
    groups = [
        {"name": "g", "colour": "red", "stacks": [{"path": str(ok), "label": "a", "lable": "x"}]}
    ]
    assert check_groups(groups) == []  # not checked unless asked
    problems = check_groups(groups, group_keys=("name", "stacks"), stack_keys=("path", "label"))
    assert problems == [
        "group 'g': unknown field 'colour' (allowed: name, stacks)",
        "group 'g': stack 'a': unknown field 'lable' (allowed: path, label)",
    ]


def test_unknown_keys_names_the_place_and_the_allowed_fields():
    from fluorostats.manifest import unknown_keys

    assert unknown_keys("manifest", {"chanel": 1, "title": "t"}, ("title", "channel")) == [
        "manifest: unknown field 'chanel' (allowed: title, channel)"
    ]
    assert unknown_keys("manifest", {"title": "t"}, ("title",)) == []


@pytest.mark.parametrize("loader", [depth_batch.load_manifest, viability_batch.load_manifest])
@pytest.mark.parametrize(
    "m, expected",
    [
        ([{"name": "g"}], "must be a JSON object"),
        ({"groups": {"name": "g"}}, "non-empty 'groups' list"),
        ({"groups": "g"}, "non-empty 'groups' list"),
    ],
)
def test_manifest_of_the_wrong_shape_is_reported_not_a_traceback(tmp_path, loader, m, expected):
    # Regression: these raised AttributeError / TypeError from inside the loader.
    with pytest.raises(ValueError, match=expected):
        loader(_write(tmp_path, m))


def test_check_groups_reports_fields_of_the_wrong_type(tmp_path):
    # Regression: a group or stack that is not an object, a numeric background or
    # path crashed with a traceback, and a list name was accepted.
    ok = tmp_path / "a.tif"
    ok.write_bytes(b"")
    st = {"path": str(ok), "label": "a"}
    groups = [
        "g0",
        {"name": ["A"], "stacks": [st]},
        {"name": "b", "background": 5, "stacks": [st]},
        {"name": "c", "background": [str(ok), 7], "stacks": [st]},
        {"name": "d", "stacks": "a.tif"},
        {
            "name": "e",
            "stacks": ["a.tif", {"path": 3, "label": "x"}, {"path": str(ok), "label": []}],
        },
    ]
    assert check_groups(groups) == [
        "group 0 must be an object with 'name' and 'stacks', got 'g0'",
        "group 1: name must be a non-empty string, got ['A']",
        "group 'b': background must be a path or a list of paths, got 5",
        "group 'c': background must be a path or a list of paths, got [" + repr(str(ok)) + ", 7]",
        "group 'd': stacks must be a list of stack objects, got 'a.tif'",
        "group 'e': every stack must be an object with 'path' and 'label', got 'a.tif'",
        "group 'e': stack path must be a string, got 3",
        "group 'e': stack label must be a string or a number, got []",
    ]


def test_depth_manifest_with_a_non_object_group_is_reported(tmp_path):
    m = {"groups": ["g"]}
    with pytest.raises(ValueError, match="group 0 must be an object"):
        depth_batch.load_manifest(_write(tmp_path, m))


def test_check_groups_reports_a_repeated_numeric_label_zero(tmp_path):
    # Regression (L5): repeats were found only among truthy values, so label 0 twice passed.
    ok = tmp_path / "a.tif"
    ok.write_bytes(b"")
    st = {"path": str(ok), "label": 0}
    problems = check_groups([{"name": "g", "stacks": [st, st]}])
    assert problems == [
        "group 'g': stack label 0 is used by more than one stack; "
        "labels must be unique within a group"
    ]


def test_not_strings_names_each_present_field_that_is_not_text():
    m = {"title": "t", "output_dir": 5, "signal_label": None}
    assert not_strings(m, ("title", "output_dir", "signal_label", "absent")) == [
        "output_dir must be a string, got 5",
        "signal_label must be a string, got None",
    ]
