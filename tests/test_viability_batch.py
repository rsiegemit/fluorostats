"""Smoke + behaviour tests for the manifest-driven Live/Dead viability driver."""

import json
from pathlib import Path

import numpy as np
import pytest

from fluorostats import viability_batch


def _blob_stack(n_live, n_dead, z=4, size=96, seed=0):
    """Synthetic 2-channel (C,Z,Y,X) stack with n_live green + n_dead red blobs."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((2, z, size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]

    def stamp(ch, n):
        for _ in range(n):
            cy, cx = rng.integers(8, size - 8, 2)
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0**2))
            vol[ch, z // 2] += 4000 * blob

    stamp(0, n_live)  # live channel
    stamp(1, n_dead)  # dead channel
    vol += rng.normal(30, 5, vol.shape)  # mild background
    return np.clip(vol, 0, None)


def _manifest(tmp: Path) -> Path:
    groups = []
    specs = {
        "sparse": [(5, 5, 1), (6, 4, 2), (4, 6, 3)],
        "dense": [(40, 5, 4), (44, 6, 5), (38, 4, 6)],
    }
    for gname, stacks in specs.items():
        entries = []
        for nl, nd, seed in stacks:
            p = tmp / f"{gname}_{seed}.npy"
            np.save(p, _blob_stack(nl, nd, seed=seed))
            entries.append({"label": f"{gname} #{seed}", "path": str(p)})
        groups.append({"name": gname, "stacks": entries})
    man = {
        "title": "synthetic",
        "live_channel": 0,
        "dead_channel": 1,
        "seg": {"method": "otsu", "min_size": 5},
        "count_method": "all",
        "output_dir": str(tmp / "out"),
        "groups": groups,
    }
    mp = tmp / "manifest.json"
    mp.write_text(json.dumps(man))
    return mp


def test_viability_batch_runs_and_recovers_coverage_gradient(tmp_path):
    out = viability_batch.run(_manifest(tmp_path))
    per = out / "viability_per_stack.csv"
    summ = out / "viability_group_summary.csv"
    assert per.exists() and summ.exists()

    import csv

    rows = list(csv.DictReader(open(per)))
    assert len(rows) == 6
    sparse = np.mean([float(r["live_cov_frac"]) for r in rows if r["group"] == "sparse"])
    dense = np.mean([float(r["live_cov_frac"]) for r in rows if r["group"] == "dense"])
    # more blobs -> higher live coverage (monotone direction recovered)
    assert dense > sparse

    # count-based viability present and finite for every stack
    for r in rows:
        assert r["viability_count_consensus"] not in ("", "nan")
        assert float(r["viability_area"]) == float(r["viability_area"])  # not NaN


def test_single_count_method_path(tmp_path):
    mp = _manifest(tmp_path)
    man = json.loads(mp.read_text())
    man["count_method"] = "cc"
    man["output_dir"] = str(tmp_path / "out_cc")
    mp.write_text(json.dumps(man))
    out = viability_batch.run(mp)
    import csv

    rows = list(csv.DictReader(open(out / "viability_per_stack.csv")))
    assert all("n_live" in r for r in rows)  # single-method path emits n_live
    # the same columns as count_method="all"; what cc did not compute is NaN
    (tmp_path / "a").mkdir()
    all_rows = list(
        csv.DictReader(
            open(viability_batch.run(_manifest(tmp_path / "a")) / "viability_per_stack.csv")
        )
    )
    assert list(rows[0]) == list(all_rows[0])
    assert all(r["viability_count_spread"] == "nan" for r in rows)  # one method ran
    assert all(r["n_live_watershed"] == "nan" and r["n_live_cc"] == r["n_live"] for r in rows)
    assert all(r["count_method_live"] == "cc" for r in rows)


def test_viability_batch_manifest_errors(tmp_path):
    def run_man(man, name):
        mp = tmp_path / f"{name}.json"
        mp.write_text(json.dumps(man))
        return viability_batch.run(mp)

    base = {
        "live_channel": 0,
        "dead_channel": 1,
        "seg": {"method": "otsu", "min_size": 3},
        "count_method": "cc",
    }
    with pytest.raises(ValueError):  # empty groups (guard)
        run_man({**base, "groups": []}, "empty")

    with pytest.raises(ValueError, match="has no stacks"):  # rejected by load_manifest
        run_man(
            {**base, "output_dir": str(tmp_path / "o2"), "groups": [{"name": "g", "stacks": []}]},
            "nostacks",
        )


def _cfg(**kw):
    return {
        "live_channel": 0,
        "dead_channel": 1,
        "seg": {"method": "otsu", "min_size": 5},
        "count_method": "cc",
        **kw,
    }


def test_fov_area_uses_y_and_x_pixel_sizes(monkeypatch):
    """Non-square pixels: the field of view is (dx * nx) * (dy * ny)."""
    vol = _blob_stack(5, 5)[:, :, :, :80]  # ny = 96, nx = 80
    meta = {
        "voxel_size_um": (2.0, 0.5, 1.0),
        "voxel_size_source": "file",
        "channel_names": ["Ch1", "Ch2"],
    }
    monkeypatch.setattr(viability_batch, "load_volume", lambda p: (vol, meta))
    row = viability_batch.analyse_stack(Path("x.npy"), _cfg())
    assert row["fov_mm2"] == pytest.approx((1.0 * 80) * (0.5 * 96) / 1e6)


def _one_stack_manifest(tmp_path, **over):
    p = tmp_path / "s.npy"
    np.save(p, _blob_stack(3, 3))
    man = {
        "live_channel": 0,
        "dead_channel": 1,
        "count_method": "cc",
        "output_dir": str(tmp_path / "o"),
        "groups": [{"name": "g", "stacks": [{"label": "s", "path": str(p)}]}],
        **over,
    }
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(man))
    return mp


def test_manifest_rejects_aux_role(tmp_path):
    mp = _one_stack_manifest(tmp_path)
    man = json.loads(mp.read_text())
    man["groups"][0]["stacks"][0]["role"] = "aux"
    mp.write_text(json.dumps(man))
    with pytest.raises(ValueError, match="role must be one of"):
        viability_batch.load_manifest(mp)


@pytest.mark.parametrize(
    "seg, msg",
    [
        ({"method": "bogus"}, "seg.method must be one of"),
        ({"min_size": -1}, "seg.min_size must be a non-negative integer"),
        ({"min_size": 2.5}, "seg.min_size must be a non-negative integer"),
        ({"threshold_scale": 0}, "seg.threshold_scale must be a positive number"),
        ({"sigma": 1.0}, "seg: unknown field 'sigma'"),
        ("otsu", "seg must be an object"),
    ],
)
def test_manifest_rejects_bad_seg_block(tmp_path, seg, msg):
    with pytest.raises(ValueError, match=msg):
        viability_batch.load_manifest(_one_stack_manifest(tmp_path, seg=seg))


def test_manifest_accepts_full_seg_block(tmp_path):
    seg = {"method": "li", "min_size": 0, "threshold_scale": 0.8}
    assert viability_batch.load_manifest(_one_stack_manifest(tmp_path, seg=seg))["seg"] == seg


def test_per_stack_csv_records_voxel_size_source(tmp_path):
    import csv

    out = viability_batch.run(_one_stack_manifest(tmp_path))
    rows = list(csv.DictReader(open(out / "viability_per_stack.csv")))
    assert rows[0]["voxel_size_source"] == "z:default,y:default,x:default"
    summary = (out / "viability_group_summary.csv").read_text()
    assert "voxel_size_source" not in summary


def test_count_segmentation_follows_seg_method(tmp_path, monkeypatch):
    """seg.method thresholds the cell counts too (it used to be dropped for counts)."""
    seen = []
    real = viability_batch.viab.live_dead_by_count
    monkeypatch.setattr(
        viability_batch.viab,
        "live_dead_by_count",
        lambda *a, **k: seen.append(k.get("threshold")) or real(*a, **k),
    )
    out = viability_batch.run(_one_stack_manifest(tmp_path, seg={"method": "li", "min_size": 5}))
    cfg = json.loads((out / "run_config.json").read_text())
    assert seen == ["li"] and cfg["count_seg_method"] == "li"


def test_two_group_contrast_reports_effect_size_and_bh(tmp_path, capsys):
    viability_batch.run(_manifest(tmp_path))
    text = capsys.readouterr().out
    assert "Cliff's delta" in text and "BH-FDR" in text
    lines = [ln for ln in text.splitlines() if "sparse vs dense" in ln]
    assert len(lines) == 3
    assert all("delta=" in ln and "q=" in ln for ln in lines)


def test_contrast_skipped_notice_for_one_group(tmp_path, capsys):
    viability_batch.run(_one_stack_manifest(tmp_path))
    assert "two-group contrast skipped: 1 group" in capsys.readouterr().out


def test_manifest_rejects_bool_channel(tmp_path):
    mp = _one_stack_manifest(tmp_path, live_channel=False, dead_channel=True)
    with pytest.raises(ValueError, match="live_channel must be a non-negative integer"):
        viability_batch.load_manifest(mp)


def test_single_stack_group_has_no_sem(tmp_path, capsys):
    """One stack has no measured spread: the group SEM is NaN, not 0."""
    import csv

    out = viability_batch.run(_one_stack_manifest(tmp_path))
    rows = list(csv.DictReader(open(out / "viability_group_summary.csv")))
    assert rows and all(r["sem"] == "nan" for r in rows)
    assert {r["n"] for r in rows} <= {"0", "1"} and any(r["n"] == "1" for r in rows)
    assert "±nan (n=1)" in capsys.readouterr().out


def test_the_two_mip_columns_are_the_documented_definitions(tmp_path):
    # live_cov_frac segments the intensity MIP; live_frac_mip projects the 3D mask.
    import csv
    from fluorostats import viability
    from fluorostats.segment import binarize

    out = viability_batch.run(_one_stack_manifest(tmp_path))
    row = next(csv.DictReader(open(out / "viability_per_stack.csv")))
    vol = _blob_stack(3, 3)
    seg = viability_batch.DEFAULT_SEG
    live = vol[0].astype(np.float32)
    intensity_mip = viability.live_dead_fractions(live.max(0), **seg)["live_fraction"]
    mask_mip = binarize(live, **seg).max(0).mean()
    assert float(row["live_cov_frac"]) == pytest.approx(intensity_mip)
    assert float(row["live_frac_mip"]) == pytest.approx(mask_mip)
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["mip_columns"] == viability_batch.MIP_COLUMNS


def test_channels_resolved_from_names_and_recorded(tmp_path, monkeypatch):
    # No channels in the manifest: the live/dead dyes are found by name (here in
    # reverse order) and both names are written to the table.
    import csv

    real = viability_batch.load_volume

    def swapped(path):
        vol, meta = real(path)
        return vol[::-1], {**meta, "channel_names": ["PI", "Calcein-AM"]}

    monkeypatch.setattr(viability_batch, "load_volume", swapped)
    mp = _one_stack_manifest(tmp_path)
    man = json.loads(mp.read_text())
    del man["live_channel"], man["dead_channel"]
    mp.write_text(json.dumps(man))
    out = viability_batch.run(mp)
    row = next(csv.DictReader(open(out / "viability_per_stack.csv")))
    assert (row["live_channel_name"], row["dead_channel_name"]) == ("Calcein-AM", "PI")
    monkeypatch.setattr(viability_batch, "load_volume", real)
    (tmp_path / "r").mkdir()
    ref_out = viability_batch.run(_one_stack_manifest(tmp_path / "r"))
    ref = next(csv.DictReader(open(ref_out / "viability_per_stack.csv")))
    assert row["live_cov_frac"] == ref["live_cov_frac"]  # the same live signal measured


def test_unnamed_channels_fall_back_to_0_1_with_a_warning(tmp_path):
    import csv

    mp = _one_stack_manifest(tmp_path)
    man = json.loads(mp.read_text())
    del man["live_channel"], man["dead_channel"]
    mp.write_text(json.dumps(man))
    with pytest.warns(RuntimeWarning, match="no channel name identifies the live"):
        out = viability_batch.run(mp)
    row = next(csv.DictReader(open(out / "viability_per_stack.csv")))
    assert (row["live_channel_name"], row["dead_channel_name"]) == ("Ch1", "Ch2")


def test_manifest_rejects_duplicate_labels_and_unknown_fields(tmp_path):
    # Strict like depth: a misspelt or unused field (e.g. a group "background",
    # which viability never subtracts) is refused instead of silently ignored.
    mp = _one_stack_manifest(tmp_path)
    man = json.loads(mp.read_text())
    st = man["groups"][0]["stacks"][0]
    man["groups"][0]["stacks"].append({**st, "colour": "red"})
    man["groups"][0]["background"] = st["path"]
    man["live_chanel"] = 0
    mp.write_text(json.dumps(man))
    with pytest.raises(ValueError) as err:
        viability_batch.load_manifest(mp)
    msg = str(err.value)
    assert msg.count("stack label 's' is used by more than one stack") == 1
    assert "group 'g': unknown field 'background'" in msg
    assert "manifest: unknown field 'live_chanel'" in msg
    assert "stack 's': unknown field 'colour'" in msg


def test_contrasts_csv_has_the_depth_columns(tmp_path):
    from fluorostats import stats
    from fluorostats.depth_batch import write_contrasts_csv
    import csv

    out = viability_batch.run(_manifest(tmp_path))
    rows = list(csv.DictReader(open(out / "contrasts.csv")))
    per = list(csv.DictReader(open(out / "viability_per_stack.csv")))
    hdr = tmp_path / "hdr.csv"
    write_contrasts_csv(hdr, [])
    assert list(rows[0]) == hdr.read_text().strip().split(",")
    assert [r["metric"] for r in rows] == ["live_cov_frac", "live_density_permm2", "viability_area"]
    ps = []
    for r in rows:
        a = [float(p[r["metric"]]) for p in per if p["group"] == "sparse"]
        b = [float(p[r["metric"]]) for p in per if p["group"] == "dense"]
        mw = stats.mann_whitney(a, b)
        assert (r["group_a"], r["group_b"], r["n_a"], r["n_b"]) == ("sparse", "dense", "3", "3")
        assert float(r["p"]) == pytest.approx(mw["p"], rel=1e-5)
        assert float(r["cliffs_delta"]) == pytest.approx(mw["cliffs_delta"], rel=1e-5)
        ps.append(mw["p"])
    q = stats.bh_fdr(ps)
    assert [float(r["q"]) for r in rows] == pytest.approx(list(q), rel=1e-5)
    assert rows[0]["correction"] == "Benjamini-Hochberg FDR across 3 metrics"


def test_one_group_writes_a_header_only_contrasts_csv(tmp_path):
    out = viability_batch.run(_one_stack_manifest(tmp_path))
    assert len((out / "contrasts.csv").read_text().strip().splitlines()) == 1


def test_a_failed_rerun_leaves_no_earlier_outputs_or_run_config(tmp_path, monkeypatch):
    # Earlier outputs are removed at the start and the full run_config.json is written
    # last, so a run that fails part-way leaves only the "incomplete" marker.
    mp = _one_stack_manifest(tmp_path)
    out = viability_batch.run(mp)
    names = (
        "viability_per_stack.csv",
        "viability_group_summary.csv",
        "contrasts.csv",
        "run_config.json",
    )
    assert all((out / n).exists() for n in names)

    def boom(path, cfg):
        raise RuntimeError("unreadable stack")

    monkeypatch.setattr(viability_batch, "analyse_stack", boom)
    with pytest.raises(RuntimeError, match="unreadable stack"):
        viability_batch.run(mp)
    assert not any((out / n).exists() for n in names[:-1])
    marker = {"tool": "viability", "status": "incomplete"}  # M3: still this tool's folder
    assert json.loads((out / "run_config.json").read_text()) == marker
    monkeypatch.undo()
    assert json.loads((viability_batch.run(mp) / "run_config.json").read_text())["status"] == (
        "complete"
    )


def test_manifest_seg_opening_is_validated(tmp_path):
    seg = {"method": "otsu", "opening": "disk"}
    assert viability_batch.load_manifest(_one_stack_manifest(tmp_path, seg=seg))["seg"] == seg
    with pytest.raises(ValueError, match="seg.opening must be one of"):
        viability_batch.load_manifest(_one_stack_manifest(tmp_path, seg={"opening": "cube"}))


def test_seg_opening_disk_measures_single_slice_cells_in_3d(tmp_path):
    """The synthetic cells lie in one z-slice: the default "ball" opening erases them
    from the 3D mask; "disk" keeps them."""
    m = json.loads(_manifest(tmp_path).read_text())
    m["seg"]["opening"] = "disk"
    stack = m["groups"][1]["stacks"][0]["path"]
    row = viability_batch.analyse_stack(Path(stack), {**m, "seg": m["seg"]})
    assert row["live_frac_3d"] > 0


def test_coverage_is_reported_as_a_fraction_only(tmp_path):
    """EXP-1: coverage columns are fractions; the removed _pct duplicates are gone."""
    import csv

    out = viability_batch.run(_one_stack_manifest(tmp_path))
    row = next(csv.DictReader(open(out / "viability_per_stack.csv")))
    for ch in ("live", "dead"):
        frac = float(row[f"{ch}_cov_frac"])
        assert 0 < frac < 1
        assert f"{ch}_cov_pct" not in row


def test_run_reads_and_writes_text_as_utf8(tmp_path):
    """A non-ASCII group / label (e.g. "µ") must not depend on the platform's default
    encoding: every text open names UTF-8 (no EncodingWarning under -X
    warn_default_encoding)."""
    import subprocess
    import sys

    mp = _one_stack_manifest(tmp_path)
    man = json.loads(mp.read_text(encoding="utf-8"))
    man["groups"][0]["name"] = "Gel 5 µm"
    man["groups"][0]["stacks"][0]["label"] = "Zelle é"
    mp.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    code = (
        "import warnings, sys; from pathlib import Path; "
        "from fluorostats import viability_batch; "
        "warnings.filterwarnings('error', category=EncodingWarning, module='fluorostats.viability_batch'); "
        "viability_batch.run(Path(sys.argv[1]))"
    )
    res = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-c", code, str(mp)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert res.returncode == 0, res.stderr[-800:]
    rows = (tmp_path / "o" / "viability_per_stack.csv").read_text(encoding="utf-8")
    assert "Gel 5 µm" in rows and "Zelle é" in rows


def test_run_config_records_the_absolute_stack_paths_read(tmp_path, monkeypatch):
    """Relative stack paths still resolve against the working directory; the record
    names the absolute files that were read."""
    mp = _one_stack_manifest(tmp_path)
    man = json.loads(mp.read_text(encoding="utf-8"))
    man["groups"][0]["stacks"][0]["path"] = "s.npy"
    mp.write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out = viability_batch.run(mp)
    cfg = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["files_read"] == {  # the same record as depth's
        "stacks": [str((tmp_path / "s.npy").resolve())],
        "output_dir": str((tmp_path / "o").resolve()),
    }
    assert cfg["resolved"]["groups"][0]["stacks"][0]["path"] == "s.npy"  # as given


def test_run_refuses_a_folder_another_tool_or_the_user_wrote(tmp_path, capsys):
    """M1: a viability run must not silently replace files it did not write."""
    mp = _one_stack_manifest(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    (out / "contrasts.csv").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match="holds contrasts.csv but no run_config.json"):
        viability_batch.run(mp)
    assert (out / "contrasts.csv").read_text(encoding="utf-8") == "mine"
    with pytest.warns(RuntimeWarning, match="replacing contrasts.csv"):
        viability_batch.run(mp, overwrite=True)
    cfg = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["tool"] == "viability"
    viability_batch.run(mp)  # a rerun of its own folder rewrites its tables
    (out / "run_config.json").write_text(json.dumps({"tool": "depth"}), encoding="utf-8")
    with pytest.raises(ValueError, match="depth"):
        viability_batch.run(mp)


@pytest.mark.parametrize("key", ["title", "output_dir"])
def test_manifest_rejects_non_string_text_fields(tmp_path, key):
    with pytest.raises(ValueError, match=f"{key} must be a string"):
        viability_batch.load_manifest(_one_stack_manifest(tmp_path, **{key: 5}))


def test_a_failed_run_leaves_no_new_output_folder(tmp_path, monkeypatch):
    """C3: the output folder is created only once every stack has been analysed."""

    def boom(path, cfg):
        raise RuntimeError("unreadable stack")

    monkeypatch.setattr(viability_batch, "analyse_stack", boom)
    with pytest.raises(RuntimeError, match="unreadable stack"):
        viability_batch.run(_one_stack_manifest(tmp_path))
    assert not (tmp_path / "o").exists()
