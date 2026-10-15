"""Smoke test for the depth-penetration batch driver (fluorostats.depth_batch)."""

import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from depth_data import decay_stack, flat_stack
from fluorostats import depth, depth_batch
from fluorostats.depth_batch import (
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
    _smoke_decay_stack(blank, lam=1e9, i0=0.0)  # flat background
    manifest = {
        "channel": 0,
        "reducer": "mean",
        "n_surface": 2,
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
    assert len(ps) == 4  # 2 groups x 2 stacks
    # plain TIFFs carry no calibration: the table records that the 1 µm z-step is a default
    assert list(ps.columns[-3:]) == [
        "voxel_size_source",
        "window_complete_0_10um",
        "window_complete_full",
    ]
    assert (ps[["window_complete_0_10um", "window_complete_full"]] == 1).all().all()
    assert (ps["voxel_size_source"] == "z:default,y:default,x:default").all()
    assert set(ps["group"]) == {"fast", "slow"}
    # slower decay retains signal deeper -> larger surface-normalised full-depth AUC
    assert (
        ps[ps.group == "slow"].auc_normalized_full.mean()
        > ps[ps.group == "fast"].auc_normalized_full.mean()
    )


# ---------------------------------------------------------------------------
# Helper: a blank whose last slices are zero
# ---------------------------------------------------------------------------


def _zero_tail_blank(path, n_z=16, yx=8, good_slices=8):
    """Blank with last slices == 0 to exercise the trim-trailing-zeros branch."""
    vol = np.zeros((n_z, yx, yx), dtype=np.float32)
    vol[:good_slices] = 5.0  # only first half has signal
    tifffile.imwrite(path, vol)


# ---------------------------------------------------------------------------
# _common_grid — single-slice stack guard  (line 186)
# ---------------------------------------------------------------------------


def test_common_grid_single_slice_stack():
    """A stack with only 1 slice gives depth_um of size 1; np.diff -> []; guard fires."""
    # Simulate a 1-slice profile (depth_um has 1 element)
    single = {
        "depth_um": np.array([0.0]),
        "bg_subtracted": np.array([5.0]),
        "normalized": np.array([1.0]),
        "max_depth_um": 0.0,
    }
    multi = {
        "depth_um": np.array([0.0, 2.0, 4.0]),
        "bg_subtracted": np.array([5.0, 4.0, 3.0]),
        "normalized": np.array([1.0, 0.8, 0.6]),
        "max_depth_um": 4.0,
    }
    # The single-slice member has no step; the grid uses the other's 2 µm step and
    # stops at the shallowest stack (0 µm).
    np.testing.assert_array_equal(_common_grid([single, multi]), [0.0])
    np.testing.assert_array_equal(_common_grid([multi]), [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(_common_grid([single]), [0.0])  # no steps at all -> 1 µm


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
    assert len(lines) == 1  # header only


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
    flat_stack(blank1, value=5.0)
    flat_stack(blank2, value=6.0)

    s0 = tmp_path / "s0.tif"
    decay_stack(s0)

    manifest = {
        "channel": 0,
        "reducer": "mean",
        "n_surface": 2,
        "auc_windows_um": [[0, 8]],
        "output_dir": str(tmp_path / "out"),
        "groups": [
            {
                "name": "A",
                "background": [str(blank1), str(blank2)],  # list of blanks
                "stacks": [{"label": "s0", "path": str(s0)}],
            }
        ],
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
    flat_stack(tmp_path / "b1.tif", value=5.0, n_z=16)
    flat_stack(tmp_path / "b2.tif", value=7.0, n_z=10)
    with pytest.warns(RuntimeWarning, match=r"replicate blanks have \[10, 16\] slices.*first 10"):
        blank = _load_blank([str(tmp_path / "b1.tif"), str(tmp_path / "b2.tif")], _CFG)
    np.testing.assert_allclose(blank.mean, np.full(10, 6.0))


def test_load_blank_announces_dropped_mid_stack_slices(tmp_path):
    vol = np.full((12, 8, 8), 5.0, dtype=np.float32)
    vol[4] = 0.0  # one empty slice mid-stack
    vol[9:] = 0.0  # three trailing
    tifffile.imwrite(tmp_path / "b.tif", vol)
    with pytest.warns(
        RuntimeWarning,
        match="dropped 4 of 12 slices with mean <= 0 "
        r"\(1 mid-stack, 3 trailing\)",
    ):
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
    decay_stack(s0)

    manifest = {
        "channel": 0,
        "reducer": "mean",
        "n_surface": 2,
        "auc_windows_um": [[0, 8]],
        "output_dir": str(tmp_path / "out"),
        "groups": [
            {
                "name": "A",
                "background": str(blank),
                "stacks": [{"label": "s0", "path": str(s0)}],
            }
        ],
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


def test_run_announces_partial_window_and_mismatched_blanks(tmp_path):
    def ij(path, value, spacing, n_z=6):
        tifffile.imwrite(
            path,
            np.full((n_z, 1, 6, 6), value, np.uint16),
            imagej=True,
            metadata={"spacing": spacing, "axes": "ZCYX"},
        )

    ij(tmp_path / "b1.tif", 5, 2.0)
    ij(tmp_path / "b2.tif", 6, 4.0)  # different z-step
    z = np.arange(6)[:, None, None]
    tifffile.imwrite(
        tmp_path / "s.tif",
        (100 * np.exp(-z / 3.0) + 10)[:, None].repeat(6, 2).repeat(6, 3).astype(np.uint16),
        imagej=True,
        metadata={"spacing": 2.0, "axes": "ZCYX"},
    )
    m = {
        "auc_windows_um": [[0, 50]],
        "output_dir": str(tmp_path / "out"),  # stack is 10 µm deep
        "groups": [
            {
                "name": "A",
                "background": [str(tmp_path / "b1.tif"), str(tmp_path / "b2.tif")],
                "stacks": [{"label": "s", "path": str(tmp_path / "s.tif")}],
            }
        ],
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    with pytest.warns(RuntimeWarning) as rec:
        depth_batch.run(mpath)
    msgs = " | ".join(str(w.message) for w in rec)
    assert "different z-steps" in msgs
    assert "reaches past the stack" in msgs
    ps = pd.read_csv(tmp_path / "out" / "auc_per_stack.csv")
    assert ps["voxel_size_source"].tolist() == ["z:file-unit-unknown,y:default,x:default"]


def test_window_covered_um_is_clamped_at_zero(tmp_path):
    """A window starting past the stack end covers 0 µm, not a negative depth."""
    s0 = tmp_path / "s0.tif"
    decay_stack(s0, n_z=11)  # 0-10 µm at the 1 µm default
    cfg = {
        "channel": 0,
        "reducer": "mean",
        "n_surface": 2,
        "fit_offset": False,
        "fit_r2_threshold": 0.85,
        "windows": [(5.0, 20.0), (50.0, 60.0)],
    }
    with pytest.warns(RuntimeWarning, match="only 0 µm of it"):
        res = depth_batch.analyse_stack(s0, None, cfg)
    assert res["auc"]["5_20um"]["covered_um"] == 5.0
    assert res["auc"]["50_60um"]["covered_um"] == 0.0
    assert np.isnan(res["auc"]["50_60um"]["normalized"])


def test_aggregate_group_single_stack_has_no_spread():
    """One primary stack has no measured spread: curve SEMs and λ SEM are NaN, not 0."""
    z = np.arange(0.0, 10.0)
    fit = depth.PenetrationFit(
        lambda_um=5.0, I0=1.0, offset=None, r_squared=0.99, rmse=0.0, n_points=z.size, fit_ok=True
    )
    agg = aggregate_group(
        [
            {
                "depth_um": z,
                "max_depth_um": 9.0,
                "bg_subtracted": np.exp(-z / 5),
                "normalized": np.exp(-z / 5),
                "fit": fit,
            }
        ]
    )
    assert np.isnan(agg["sub_sem"]).all() and np.isnan(agg["norm_sem"]).all()
    assert agg["lambda_mean_um"] == 5.0 and np.isnan(agg["lambda_sem_um"])


def test_rerun_removes_stale_figures_and_writes_run_config_last(tmp_path, capsys, monkeypatch):
    # Regression: figures of windows dropped from the manifest survived a rerun, and
    # run_config.json was written before analysis, so a failed run left the new
    # configuration next to the old run's tables.
    s0 = tmp_path / "s0.tif"
    decay_stack(s0)
    out = tmp_path / "out"
    m = {
        "auc_windows_um": [[0, 4], [0, 8]],
        "output_dir": str(out),
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}],
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    depth_batch.run(mpath)
    assert (out / "fig_auc_normalized_0_4um.png").exists()
    assert (out / "fig_auc_retention_multiwindow.pdf").exists()
    (out / "notes.png").write_bytes(b"x")  # not the tool's
    capsys.readouterr()

    mpath.write_text(json.dumps({**m, "auc_windows_um": [[0, 8]]}))
    depth_batch.run(mpath)
    text = capsys.readouterr().out
    gone = ["fig_auc_absolute_0_4um", "fig_auc_normalized_0_4um", "fig_auc_retention_multiwindow"]
    for stem in gone:
        for ext in ("png", "pdf"):
            assert not (out / f"{stem}.{ext}").exists()
            assert f"{stem}.{ext} (not produced by this run)" in text
    assert "fig_auc_normalized_0_8um.png (not" not in text  # rewritten, not removed
    assert (out / "fig_auc_normalized_0_8um.png").exists() and (out / "notes.png").exists()
    assert json.loads((out / "run_config.json").read_text())["resolved"]["auc_windows_um"] == [
        [0, 8]
    ]

    def fail(*a, **k):
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(depth_batch, "analyse_stack", fail)
    with pytest.raises(RuntimeError, match="analysis failed"):
        depth_batch.run(mpath)
    assert not (out / "run_config.json").exists()


def test_common_grid_never_extends_past_the_shallowest_stack():
    # Regression: the point count was rounded, so a shallowest depth that is not a
    # whole number of steps put a grid point past it (np.interp then held the last
    # value: extrapolation the docstring ruled out).
    def prof(n, step):
        z = np.arange(n) * step
        return {"depth_um": z, "max_depth_um": float(z[-1])}

    short = {"depth_um": np.linspace(0, 38, 11), "max_depth_um": 38.0}  # 3.8 µm steps
    grid = _common_grid([prof(16, 4.0), prof(16, 4.0), short])  # median step 4 µm
    np.testing.assert_array_equal(grid, np.arange(10) * 4.0)  # 0-36 µm, not 0-40
    # a whole number of steps keeps its last point despite float rounding
    np.testing.assert_allclose(_common_grid([prof(11, 0.3)])[-1], 3.0)
    assert len(_common_grid([prof(11, 0.3)])) == 11


def test_r2_threshold_decides_fit_ok(tmp_path):
    # The acceptance threshold was hard-wired at 0.85; it is now a manifest field.
    s0 = tmp_path / "s0.tif"
    z = np.arange(30)[:, None, None]
    noise = np.random.default_rng(0).normal(0, 12, (30, 1, 1))
    tifffile.imwrite(
        s0, np.broadcast_to(150 * np.exp(-z / 6.0) + noise + 20, (30, 8, 8)).astype(np.float32)
    )
    cfg = {
        "channel": 0,
        "reducer": "mean",
        "n_surface": 2,
        "fit_offset": False,
        "windows": [(0.0, 10.0)],
    }
    r2 = depth_batch.analyse_stack(s0, None, {**cfg, "fit_r2_threshold": 0.0})["fit"].r_squared
    assert 0.5 < r2 < 0.99
    for thr, ok in ((r2 - 0.01, True), (r2 + 0.01, False)):
        fit = depth_batch.analyse_stack(s0, None, {**cfg, "fit_r2_threshold": thr})["fit"]
        assert fit.fit_ok is ok


# ---------------------------------------------------------------------------
# Problems found only once a stack is read name the file (ERR-1)
# ---------------------------------------------------------------------------


def _cfg(**kw):
    return {
        "channel": 0,
        "reducer": "mean",
        "n_surface": 2,
        "fit_offset": False,
        "fit_r2_threshold": 0.85,
        "windows": [(0.0, 10.0)],
        **kw,
    }


def test_out_of_range_channel_names_the_file(tmp_path):
    # Regression: a bare IndexError ("index 3 is out of bounds for axis 0 with size 1").
    s0 = tmp_path / "s0.tif"
    decay_stack(s0)
    with pytest.raises(
        ValueError, match="s0.tif: channel 3 is out of range; the stack has 1 channel"
    ):
        depth_batch.analyse_stack(s0, None, _cfg(channel=3))


@pytest.mark.parametrize("n_z,fit_offset,need", [(1, False, 2), (2, True, 3)])
def test_too_few_slices_names_the_file(tmp_path, n_z, fit_offset, need):
    s0 = tmp_path / "s0.tif"
    tifffile.imwrite(
        s0, np.full((n_z, 1, 8, 8), 50.0, np.float32), imagej=True, metadata={"axes": "ZCYX"}
    )
    with pytest.raises(ValueError, match=f"s0.tif: has {n_z} z-slice.*needs at least {need}"):
        depth_batch.analyse_stack(s0, None, _cfg(fit_offset=fit_offset))


def test_non_positive_surface_reference_names_the_file(tmp_path):
    s0 = tmp_path / "s0.tif"
    flat_stack(s0, value=5.0)
    blank = depth.intensity_depth_profile(np.full((16, 8, 8), 9.0), 1.0)  # brighter than the stack
    with pytest.raises(
        ValueError, match=r"s0.tif: near-surface reference is 0 \(non-positive or NaN\)"
    ):
        depth_batch.analyse_stack(s0, blank, _cfg())


def test_load_blank_with_no_positive_slice_names_the_blank(tmp_path):
    # Regression (D3): every slice <= 0 left an empty profile and numpy's bare
    # "array of sample points is empty" later, naming no file.
    flat_stack(tmp_path / "b.tif", value=-5.0)
    with pytest.raises(ValueError, match="blank b.tif: no slice has a mean > 0"):
        _load_blank(str(tmp_path / "b.tif"), _CFG)


def test_auc_csv_flags_a_window_the_stack_covers_only_partly(tmp_path):
    # Regression (M2): a clamped AUC sat in auc_per_stack.csv like a complete one.
    s0 = tmp_path / "s0.tif"
    decay_stack(s0, n_z=11)  # 0-10 µm
    cfg = {**_cfg(), "windows": [(0.0, 5.0), (5.0, 20.0)]}
    with pytest.warns(RuntimeWarning, match="only 5 µm of it"):
        res = depth_batch.analyse_stack(s0, None, cfg)
    depth_batch.write_auc_csv(
        tmp_path / "auc.csv",
        [{"group": "A", "stack": "s0", "role": "primary", "result": res}],
        cfg["windows"],
    )
    ps = pd.read_csv(tmp_path / "auc.csv")
    assert list(ps.columns[-2:]) == ["window_complete_0_5um", "window_complete_5_20um"]
    assert ps.iloc[0, -2:].tolist() == [1, 0]


def test_run_refuses_a_folder_holding_files_it_did_not_write(tmp_path):
    # Regression (M1): a depth run replaced the user's contrasts.csv without notice;
    # its own earlier tables were not removed before a rerun.
    s0 = tmp_path / "s0.tif"
    decay_stack(s0)
    out = tmp_path / "out"
    out.mkdir()
    (out / "contrasts.csv").write_text("mine")
    m = {
        "output_dir": str(out),
        "groups": [{"name": "A", "stacks": [{"label": "s0", "path": str(s0)}]}],
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="holds contrasts.csv but no run_config.json"):
        depth_batch.run(mpath)
    assert (out / "contrasts.csv").read_text() == "mine"
    with pytest.warns(RuntimeWarning, match="replacing contrasts.csv"):
        depth_batch.run(mpath, overwrite=True)
    assert json.loads((out / "run_config.json").read_text())["tool"] == "depth"
    (out / "depth_profiles_long.csv").write_text("stale")
    depth_batch.run(mpath)  # a rerun of its own folder rewrites its tables
    assert (out / "depth_profiles_long.csv").read_text().startswith("group,stack")
