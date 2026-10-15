"""Tests for fluorostats.depth_batch.load_manifest: the fields it accepts, and invalid
manifests refused up front."""

import json

import pytest

from depth_data import decay_stack, flat_stack
from fluorostats.depth_batch import load_manifest


def test_load_manifest_singular_auc_window(tmp_path):
    """auc_window_um (singular) should be accepted and wrapped into windows."""
    blank = tmp_path / "blank.tif"
    flat_stack(blank)
    s0 = tmp_path / "s0.tif"
    decay_stack(s0)
    m = {
        "auc_window_um": [0, 10],  # singular (back-compat), NOT auc_windows_um
        "output_dir": str(tmp_path / "out"),
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    loaded = load_manifest(mpath)
    assert loaded["windows"] == [(0.0, 10.0)]


def test_load_manifest_no_groups_raises(tmp_path):
    m = {"auc_windows_um": [[0, 10]], "groups": []}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="at least one group"):
        load_manifest(mpath)


def test_load_manifest_missing_groups_key_raises(tmp_path):
    m = {"auc_windows_um": [[0, 10]]}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="at least one group"):
        load_manifest(mpath)


def _manifest_errors(tmp_path, **fields):
    s0 = tmp_path / "s0.tif"
    s0.write_bytes(b"")  # never read: validation comes first
    m = {"groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}], **fields}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="invalid depth manifest") as e:
        load_manifest(mpath)
    return str(e.value)


@pytest.mark.parametrize("field", ["channel", "n_surface"])
def test_load_manifest_rejects_booleans_as_integers(tmp_path, field):
    assert f"{field} must be" in _manifest_errors(tmp_path, **{field: True})


def test_load_manifest_requires_a_bool_for_parametric(tmp_path):
    msg = _manifest_errors(tmp_path, parametric="false")
    assert "parametric must be true or false, got 'false'" in msg


@pytest.mark.parametrize(
    "windows, expected",
    [
        ([0, 100], "a single window is [[0, 100]]"),  # flat pair: used to be a bare TypeError
        ("full", "auc_windows_um must be a list"),
        ([], "auc_windows_um must be a list"),
        ([[0, "a"]], "entry [0, 'a'] must be 'full' or a [z0, z1] pair of numbers"),
        ([[0, None]], "entry [0, None] must be"),
        ([[0, 10, 20]], "entry [0, 10, 20] must be"),
        ([[False, 10]], "entry [False, 10] must be"),
        ([{"z0": 0}], "entry {'z0': 0} must be"),
    ],
)
def test_load_manifest_rejects_malformed_windows(tmp_path, windows, expected):
    msg = _manifest_errors(tmp_path, auc_windows_um=windows)
    assert expected in msg
    assert "e.g. [[0, 100], 'full']" in msg  # names the expected form


def test_load_manifest_rejects_malformed_singular_window(tmp_path):
    assert "entry 'deep' must be" in _manifest_errors(tmp_path, auc_window_um="deep")


def test_load_manifest_refuses_both_window_fields(tmp_path):
    # Regression: auc_window_um was silently ignored when auc_windows_um was given too.
    msg = _manifest_errors(tmp_path, auc_window_um=[0, 50], auc_windows_um=[[0, 100]])
    assert "give auc_windows_um or the older auc_window_um, not both" in msg


@pytest.mark.parametrize(
    "windows, label", [([[0, 100], [0, 100.0]], "0_100um"), (["full", [0, 5], "FULL"], "full")]
)
def test_load_manifest_refuses_repeated_windows(tmp_path, windows, label):
    # Regression: two equal windows shared one label, so their figures overwrote each
    # other and the window counted twice in the BH correction.
    msg = _manifest_errors(tmp_path, auc_windows_um=windows)
    assert f"AUC window {label} is given more than once" in msg


def test_load_manifest_rejects_bad_group_color(tmp_path):
    s0 = tmp_path / "s0.tif"
    s0.write_bytes(b"")
    m = {
        "groups": [
            {"name": "A", "color": "not-a-colour", "stacks": [{"label": "s0", "path": str(s0)}]},
            {"name": "B", "color": "#3F6AB3", "stacks": [{"label": "s1", "path": str(s0)}]},
            {"name": "C", "color": None, "stacks": [{"label": "s2", "path": str(s0)}]},
        ]
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError) as e:
        load_manifest(mpath)
    assert str(e.value).endswith("group 'A': color 'not-a-colour' is not a matplotlib colour")


def test_load_manifest_rejects_unknown_fields_at_every_level(tmp_path):
    # Regression: "chanel": 1 was ignored and the run measured channel 0.
    s0 = tmp_path / "s0.tif"
    s0.write_bytes(b"")
    m = {
        "chanel": 1,
        "groups": [
            {
                "name": "A",
                "colour": "red",
                "stacks": [{"label": "s0", "path": str(s0), "rol": "aux"}],
            }
        ],
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError) as e:
        load_manifest(mpath)
    msg = str(e.value)
    assert "manifest: unknown field 'chanel' (allowed: title, signal_label, channel," in msg
    assert "group 'A': unknown field 'colour' (allowed: name, color, background, stacks)" in msg
    assert "group 'A': stack 's0': unknown field 'rol' (allowed: label, path, role)" in msg


@pytest.mark.parametrize("field", ["fit_offset", "parametric"])
def test_load_manifest_requires_real_bools(tmp_path, field):
    # Regression: "fit_offset": "false" is a non-empty string, so it fitted the offset.
    assert f"{field} must be true or false, got 'false'" in _manifest_errors(
        tmp_path, **{field: "false"}
    )


@pytest.mark.parametrize("value", [0.0, 1, 0.5])
def test_load_manifest_accepts_an_r2_threshold_in_0_1(tmp_path, value):
    s0 = tmp_path / "s0.tif"
    s0.write_bytes(b"")
    m = {
        "fit_r2_threshold": value,
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}],
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    assert load_manifest(mpath)["fit_r2_threshold"] == value


@pytest.mark.parametrize("value", [1.5, -0.1, "0.9", True])
def test_load_manifest_rejects_a_bad_r2_threshold(tmp_path, value):
    assert "fit_r2_threshold must be a number in [0, 1]" in _manifest_errors(
        tmp_path, fit_r2_threshold=value
    )
