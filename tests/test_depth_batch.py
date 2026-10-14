"""Smoke test for the depth-penetration batch driver (fluorostats.depth_batch)."""
import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from fluorostats import depth, depth_batch
from fluorostats.depth_batch import (
    load_manifest,
    _common_grid,
    _load_blank,
    write_group_summary_csv,
    aggregate_group,
)


def _smoke_decay_stack(path, lam, n_z=20, yx=8, bg=10.0, i0=200.0):
    """Write a synthetic z-stack whose per-slice mean is i0*exp(-z/lam)+bg."""
    z = np.arange(n_z)[:, None, None]
    vol = np.broadcast_to((i0 * np.exp(-z / lam) + bg).astype(np.float32), (n_z, yx, yx))
    tifffile.imwrite(path, np.ascontiguousarray(vol))


def test_depth_batch_run(tmp_path):
    blank = tmp_path / "blank.tif"
    _smoke_decay_stack(blank, lam=1e9, i0=0.0)                       # flat background
    manifest = {
        "channel": 0, "reducer": "mean", "n_surface": 2,
        "auc_windows_um": [[0, 10], "full"],
        "output_dir": str(tmp_path / "out"),
        "groups": [],
    }
    for gname, lam in [("fast", 8.0), ("slow", 40.0)]:
        stacks = []
        for i in range(2):
            p = tmp_path / f"{gname}_{i}.tif"
            _smoke_decay_stack(p, lam=lam)
            stacks.append({"label": f"{gname} {i}", "path": str(p)})
        manifest["groups"].append({"name": gname, "background": str(blank), "stacks": stacks})
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    with pytest.warns(RuntimeWarning, match="no z/y/x calibration"):
        depth_batch.run(mpath)

    out = tmp_path / "out"
    assert (out / "auc_per_stack.csv").exists()
    assert (out / "group_depth_summary.csv").exists()

    ps = pd.read_csv(out / "auc_per_stack.csv")
    assert len(ps) == 4                                        # 2 groups x 2 stacks
    # plain TIFFs carry no calibration: the table records that the 1 µm z-step is a default
    assert list(ps.columns[-1:]) == ["voxel_size_source"]
    assert (ps["voxel_size_source"] == "z:default,y:default,x:default").all()
    assert set(ps["group"]) == {"fast", "slow"}
    # slower decay retains signal deeper -> larger surface-normalised full-depth AUC
    assert ps[ps.group == "slow"].auc_normalized_full.mean() > \
           ps[ps.group == "fast"].auc_normalized_full.mean()


# ---------------------------------------------------------------------------
# Helper: write a synthetic decay tiff
# ---------------------------------------------------------------------------

def _decay_stack(path, lam=20.0, n_z=16, yx=8, bg=5.0, i0=150.0):
    """Write a synthetic z-stack whose per-slice mean is i0*exp(-z/lam)+bg."""
    z = np.arange(n_z)[:, None, None]
    vol = np.broadcast_to(
        (i0 * np.exp(-z / lam) + bg).astype(np.float32), (n_z, yx, yx)
    )
    tifffile.imwrite(path, np.ascontiguousarray(vol))


def _flat_stack(path, value=5.0, n_z=16, yx=8):
    """Write a spatially-flat stack (used as blank)."""
    vol = np.full((n_z, yx, yx), value, dtype=np.float32)
    tifffile.imwrite(path, vol)


def _zero_tail_blank(path, n_z=16, yx=8, good_slices=8):
    """Blank with last slices == 0 to exercise the trim-trailing-zeros branch."""
    vol = np.zeros((n_z, yx, yx), dtype=np.float32)
    vol[:good_slices] = 5.0   # only first half has signal
    tifffile.imwrite(path, vol)


# ---------------------------------------------------------------------------
# load_manifest — back-compat singular auc_window_um  (line 88)
# ---------------------------------------------------------------------------

def test_load_manifest_singular_auc_window(tmp_path):
    """auc_window_um (singular) should be accepted and wrapped into windows."""
    blank = tmp_path / "blank.tif"
    _flat_stack(blank)
    s0 = tmp_path / "s0.tif"
    _decay_stack(s0)
    m = {
        "auc_window_um": [0, 10],   # singular (back-compat), NOT auc_windows_um
        "output_dir": str(tmp_path / "out"),
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    loaded = load_manifest(mpath)
    assert loaded["windows"] == [(0.0, 10.0)]


# ---------------------------------------------------------------------------
# load_manifest — no groups raises ValueError  (line 91)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _common_grid — single-slice stack guard  (line 186)
# ---------------------------------------------------------------------------

def test_common_grid_single_slice_stack():
    """A stack with only 1 slice gives depth_um of size 1; np.diff -> []; guard fires."""
    # Simulate a 1-slice profile (depth_um has 1 element)
    single = {"depth_um": np.array([0.0]), "bg_subtracted": np.array([5.0]),
               "normalized": np.array([1.0]), "max_depth_um": 0.0}
    multi = {"depth_um": np.array([0.0, 2.0, 4.0]), "bg_subtracted": np.array([5.0, 4.0, 3.0]),
              "normalized": np.array([1.0, 0.8, 0.6]), "max_depth_um": 4.0}
    # The single-slice member has no step; the grid uses the other's 2 µm step and
    # stops at the shallowest stack (0 µm).
    np.testing.assert_array_equal(_common_grid([single, multi]), [0.0])
    np.testing.assert_array_equal(_common_grid([multi]), [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(_common_grid([single]), [0.0])   # no steps at all -> 1 µm


# ---------------------------------------------------------------------------
# write_group_summary_csv — agg is None branch  (line 272)
# ---------------------------------------------------------------------------

def test_write_group_summary_csv_skips_none_agg(tmp_path):
    """Groups with agg=None are silently skipped."""
    out = tmp_path / "summary.csv"
    aggregates = {"A": None, "B": None}
    write_group_summary_csv(out, aggregates)
    assert out.exists()
    content = out.read_text()
    # Only the header row, no data rows
    lines = [line for line in content.splitlines() if line.strip()]
    assert len(lines) == 1   # header only


# ---------------------------------------------------------------------------
# aggregate_group — no primary stacks returns None  (indirectly tests line 272)
# ---------------------------------------------------------------------------

def test_aggregate_group_empty_returns_none():
    assert aggregate_group([]) is None


# ---------------------------------------------------------------------------
# Full run with multiple replicate blanks  (lines 458-461)
# ---------------------------------------------------------------------------

def test_run_with_replicate_blanks(tmp_path):
    """background as a list of paths triggers the replicate-averaging branch."""
    blank1 = tmp_path / "blank1.tif"
    blank2 = tmp_path / "blank2.tif"
    _flat_stack(blank1, value=5.0)
    _flat_stack(blank2, value=6.0)

    s0 = tmp_path / "s0.tif"
    _decay_stack(s0)

    manifest = {
        "channel": 0, "reducer": "mean", "n_surface": 2,
        "auc_windows_um": [[0, 8]],
        "output_dir": str(tmp_path / "out"),
        "groups": [{
            "name": "A",
            "background": [str(blank1), str(blank2)],   # list of blanks
            "stacks": [{"label": "s0", "path": str(s0)}],
        }],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    out = depth_batch.run(mpath)
    # the decay stack minus the averaged 5.5 blank, normalised to the first 2 slices
    s0_curve = 150 * np.exp(-np.arange(16) / 20.0) + 5.0 - 5.5
    expected = depth.auc_depth(np.arange(16.0), s0_curve / s0_curve[:2].mean(), z_max=8.0)
    ps = pd.read_csv(out / "auc_per_stack.csv")
    assert ps["auc_normalized_0_8um"].iloc[0] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# _load_blank — replicate lengths and dropped non-positive slices are announced
# ---------------------------------------------------------------------------

_CFG = {"channel": 0, "reducer": "mean"}


def test_load_blank_announces_replicates_cut_to_the_shortest(tmp_path):
    _flat_stack(tmp_path / "b1.tif", value=5.0, n_z=16)
    _flat_stack(tmp_path / "b2.tif", value=7.0, n_z=10)
    with pytest.warns(RuntimeWarning, match=r"replicate blanks have \[10, 16\] slices.*first 10"):
        blank = _load_blank([str(tmp_path / "b1.tif"), str(tmp_path / "b2.tif")], _CFG)
    np.testing.assert_allclose(blank.mean, np.full(10, 6.0))


def test_load_blank_announces_dropped_mid_stack_slices(tmp_path):
    vol = np.full((12, 8, 8), 5.0, dtype=np.float32)
    vol[4] = 0.0                                          # one empty slice mid-stack
    vol[9:] = 0.0                                         # three trailing
    tifffile.imwrite(tmp_path / "b.tif", vol)
    with pytest.warns(RuntimeWarning, match="dropped 4 of 12 slices with mean <= 0 "
                                           r"\(1 mid-stack, 3 trailing\)"):
        blank = _load_blank(str(tmp_path / "b.tif"), _CFG)
    np.testing.assert_array_equal(blank.depth_um, [0, 1, 2, 3, 5, 6, 7, 8])


# ---------------------------------------------------------------------------
# Full run — blank with trailing zero slices (trim branch)
# ---------------------------------------------------------------------------

def test_run_blank_trailing_zeros_trimmed(tmp_path):
    """Blank with trailing zero slices triggers the valid.all() False branch."""
    blank = tmp_path / "blank_zeros.tif"
    _zero_tail_blank(blank, n_z=16, good_slices=8)

    s0 = tmp_path / "s0.tif"
    _decay_stack(s0)

    manifest = {
        "channel": 0, "reducer": "mean", "n_surface": 2,
        "auc_windows_um": [[0, 8]],
        "output_dir": str(tmp_path / "out"),
        "groups": [{
            "name": "A",
            "background": str(blank),
            "stacks": [{"label": "s0", "path": str(s0)}],
        }],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    with pytest.warns(RuntimeWarning, match=r"dropped 8 of 16 slices.*\(0 mid-stack, 8 trailing\)"):
        out = depth_batch.run(mpath)
    # The blank is 5 over 0-7 µm and held at 5 beyond, so the stack is exactly its decay.
    s0_curve = 150 * np.exp(-np.arange(16) / 20.0)
    expected = depth.auc_depth(np.arange(16.0), s0_curve / s0_curve[:2].mean(), z_max=8.0)
    ps = pd.read_csv(out / "auc_per_stack.csv")
    assert ps["auc_normalized_0_8um"].iloc[0] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# _print_summary and _print_lambda_summary Welch t-test branches  (lines 531-532, 552-559)
# These are hit by a full run with 2 groups where >=2 primary stacks each have fit_ok.
# ---------------------------------------------------------------------------

def test_run_two_groups_with_fit_ok_hits_ttest_branches(tmp_path):
    """Run with 2 groups, 2 stacks each where both groups have fit_ok=True -> λ t-test branch fires.

    λ must be WITHIN the acquired depth range for fit_ok to be True.
    With n_z=40 slices and default 1µm voxel spacing, max depth = 39µm.
    Use lam=5 and lam=12, both < 39µm.
    """
    blank = tmp_path / "blank.tif"
    _flat_stack(blank, value=2.0, n_z=40, yx=8)

    stacks_a, stacks_b = [], []
    for i in range(2):
        pa = tmp_path / f"a_{i}.tif"
        pb = tmp_path / f"b_{i}.tif"
        # lam=5 and lam=12 are both well within 39µm depth range
        _decay_stack(pa, lam=5.0, n_z=40)
        _decay_stack(pb, lam=12.0, n_z=40)
        stacks_a.append({"label": f"a{i}", "path": str(pa)})
        stacks_b.append({"label": f"b{i}", "path": str(pb)})

    manifest = {
        "channel": 0, "reducer": "mean", "n_surface": 2,
        "auc_windows_um": [[0, 10], "full"],
        "output_dir": str(tmp_path / "out"),
        "groups": [
            {"name": "fast", "background": str(blank), "stacks": stacks_a},
            {"name": "slow", "background": str(blank), "stacks": stacks_b},
        ],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    out = depth_batch.run(mpath)
    assert (out / "auc_per_stack.csv").exists()
    assert (out / "group_depth_summary.csv").exists()


def test_run_announces_partial_window_and_mismatched_blanks(tmp_path):
    def ij(path, value, spacing, n_z=6):
        tifffile.imwrite(path, np.full((n_z, 1, 6, 6), value, np.uint16), imagej=True,
                         metadata={"spacing": spacing, "axes": "ZCYX"})

    ij(tmp_path / "b1.tif", 5, 2.0)
    ij(tmp_path / "b2.tif", 6, 4.0)                      # different z-step
    z = np.arange(6)[:, None, None]
    tifffile.imwrite(tmp_path / "s.tif",
                     (100 * np.exp(-z / 3.0) + 10)[:, None].repeat(6, 2).repeat(6, 3).astype(np.uint16),
                     imagej=True, metadata={"spacing": 2.0, "axes": "ZCYX"})
    m = {"auc_windows_um": [[0, 50]], "output_dir": str(tmp_path / "out"),          # stack is 10 µm deep
         "groups": [{"name": "A", "background": [str(tmp_path / "b1.tif"), str(tmp_path / "b2.tif")],
                     "stacks": [{"label": "s", "path": str(tmp_path / "s.tif")}]}]}
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    with pytest.warns(RuntimeWarning) as rec:
        depth_batch.run(mpath)
    msgs = " | ".join(str(w.message) for w in rec)
    assert "different z-steps" in msgs
    assert "reaches past the stack" in msgs
    ps = pd.read_csv(tmp_path / "out" / "auc_per_stack.csv")
    assert ps["voxel_size_source"].tolist() == ["z:file,y:default,x:default"]


# ---------------------------------------------------------------------------
# load_manifest — invalid input is rejected before any image loads (ERR-1)
# ---------------------------------------------------------------------------

def _manifest_errors(tmp_path, **fields):
    s0 = tmp_path / "s0.tif"
    s0.write_bytes(b"")                                   # never read: validation comes first
    m = {"groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}], **fields}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="invalid depth manifest") as e:
        load_manifest(mpath)
    return str(e.value)


@pytest.mark.parametrize("field", ["channel", "n_surface"])
def test_load_manifest_rejects_booleans_as_integers(tmp_path, field):
    assert f"{field} must be" in _manifest_errors(tmp_path, **{field: True})


@pytest.mark.parametrize("windows, expected", [
    ([0, 100], "a single window is [[0, 100]]"),          # flat pair: used to be a bare TypeError
    ("full", "auc_windows_um must be a list"),
    ([], "auc_windows_um must be a list"),
    ([[0, "a"]], "entry [0, 'a'] must be 'full' or a [z0, z1] pair of numbers"),
    ([[0, None]], "entry [0, None] must be"),
    ([[0, 10, 20]], "entry [0, 10, 20] must be"),
    ([[False, 10]], "entry [False, 10] must be"),
    ([{"z0": 0}], "entry {'z0': 0} must be"),
])
def test_load_manifest_rejects_malformed_windows(tmp_path, windows, expected):
    msg = _manifest_errors(tmp_path, auc_windows_um=windows)
    assert expected in msg
    assert "e.g. [[0, 100], 'full']" in msg                # names the expected form


def test_load_manifest_rejects_malformed_singular_window(tmp_path):
    assert "entry 'deep' must be" in _manifest_errors(tmp_path, auc_window_um="deep")


def test_load_manifest_rejects_bad_group_color(tmp_path):
    s0 = tmp_path / "s0.tif"
    s0.write_bytes(b"")
    m = {"groups": [{"name": "A", "color": "not-a-colour", "stacks": [{"label": "s0", "path": str(s0)}]},
                    {"name": "B", "color": "#3F6AB3", "stacks": [{"label": "s1", "path": str(s0)}]},
                    {"name": "C", "color": None, "stacks": [{"label": "s2", "path": str(s0)}]}]}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError) as e:
        load_manifest(mpath)
    assert str(e.value).endswith("group 'A': color 'not-a-colour' is not a matplotlib colour")


def test_window_covered_um_is_clamped_at_zero(tmp_path):
    """A window starting past the stack end covers 0 µm, not a negative depth."""
    s0 = tmp_path / "s0.tif"
    _decay_stack(s0, n_z=11)                               # 0-10 µm at the 1 µm default
    cfg = {"channel": 0, "reducer": "mean", "n_surface": 2, "fit_offset": False,
           "windows": [(5.0, 20.0), (50.0, 60.0)]}
    with pytest.warns(RuntimeWarning, match="only 0 µm of it"):
        res = depth_batch.analyse_stack(s0, None, cfg)
    assert res["auc"]["5_20um"]["covered_um"] == 5.0
    assert res["auc"]["50_60um"]["covered_um"] == 0.0
    assert res["auc"]["50_60um"]["normalized"] == 0.0
