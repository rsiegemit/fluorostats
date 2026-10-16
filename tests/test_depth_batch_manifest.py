"""Tests for fluorostats.depth_batch.load_manifest: the fields it accepts, and invalid
manifests refused up front."""

import json

import pytest

from fluorostats.depth_batch import load_manifest


def test_load_manifest_rejects_the_removed_singular_window_key(tmp_path):
    """auc_window_um (singular) was removed: refused, naming its replacement, not ignored."""
    msg = _manifest_errors(tmp_path, auc_window_um=[0, 10])
    assert "'auc_window_um' was removed; use 'auc_windows_um'" in msg
    assert "unknown field 'auc_window_um'" not in msg  # one problem, not two


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


def test_load_manifest_keeps_windows_apart_beyond_six_digits(tmp_path):
    # Regression (D5): `%g` labels kept 6 significant digits, so distinct windows were
    # refused as "given more than once".
    s0 = tmp_path / "s0.tif"
    s0.write_bytes(b"")
    m = {
        "auc_windows_um": [[0, 1234567], [0, 1234568]],
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    assert len(load_manifest(mpath)["windows"]) == 2


@pytest.mark.parametrize("field", ["output_dir", "title", "signal_label"])
def test_load_manifest_requires_text_fields_to_be_strings(tmp_path, field):
    # Regression (D4): "output_dir": 5 passed validation and crashed run() with TypeError.
    assert f"{field} must be a string, got 5" in _manifest_errors(tmp_path, **{field: 5})
