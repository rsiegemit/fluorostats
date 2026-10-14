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
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))
            vol[ch, z // 2] += 4000 * blob
    stamp(0, n_live)   # live channel
    stamp(1, n_dead)   # dead channel
    vol += rng.normal(30, 5, vol.shape)          # mild background
    return np.clip(vol, 0, None)


def _manifest(tmp: Path) -> Path:
    groups = []
    specs = {"sparse": [(5, 5, 1), (6, 4, 2), (4, 6, 3)],
             "dense": [(40, 5, 4), (44, 6, 5), (38, 4, 6)]}
    for gname, stacks in specs.items():
        entries = []
        for nl, nd, seed in stacks:
            p = tmp / f"{gname}_{seed}.npy"
            np.save(p, _blob_stack(nl, nd, seed=seed))
            entries.append({"label": f"{gname} #{seed}", "path": str(p)})
        groups.append({"name": gname, "stacks": entries})
    man = {"title": "synthetic", "live_channel": 0, "dead_channel": 1,
           "seg": {"method": "otsu", "min_size": 5}, "count_method": "all",
           "output_dir": str(tmp / "out"), "groups": groups}
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
    sparse = np.mean([float(r["live_cov_pct"]) for r in rows if r["group"] == "sparse"])
    dense = np.mean([float(r["live_cov_pct"]) for r in rows if r["group"] == "dense"])
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
    assert all("n_live" in r for r in rows)     # single-method path emits n_live


def test_viability_batch_manifest_errors(tmp_path):
    def run_man(man, name):
        mp = tmp_path / f"{name}.json"
        mp.write_text(json.dumps(man))
        return viability_batch.run(mp)

    base = {"live_channel": 0, "dead_channel": 1, "seg": {"method": "otsu", "min_size": 3},
            "count_method": "cc"}
    with pytest.raises(ValueError):                      # empty groups (guard)
        run_man({**base, "groups": []}, "empty")

    with pytest.raises(ValueError, match="has no stacks"):   # rejected by load_manifest
        run_man({**base, "output_dir": str(tmp_path / "o2"),
                 "groups": [{"name": "g", "stacks": []}]}, "nostacks")


def _cfg(**kw):
    return {"live_channel": 0, "dead_channel": 1, "seg": {"method": "otsu", "min_size": 5},
            "count_method": "cc", **kw}


def test_fov_area_uses_y_and_x_pixel_sizes(monkeypatch):
    """Non-square pixels: the field of view is (dx * nx) * (dy * ny)."""
    vol = _blob_stack(5, 5)[:, :, :, :80]                      # ny = 96, nx = 80
    meta = {"voxel_size_um": (2.0, 0.5, 1.0), "voxel_size_source": "file"}
    monkeypatch.setattr(viability_batch, "load_volume", lambda p: (vol, meta))
    row = viability_batch.analyse_stack(Path("x.npy"), _cfg())
    assert row["fov_mm2"] == pytest.approx((1.0 * 80) * (0.5 * 96) / 1e6)


def _one_stack_manifest(tmp_path, **over):
    p = tmp_path / "s.npy"
    np.save(p, _blob_stack(3, 3))
    man = {"live_channel": 0, "dead_channel": 1, "count_method": "cc",
           "output_dir": str(tmp_path / "o"),
           "groups": [{"name": "g", "stacks": [{"label": "s", "path": str(p)}]}], **over}
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


@pytest.mark.parametrize("seg, msg", [
    ({"method": "bogus"}, "seg.method must be one of"),
    ({"min_size": -1}, "seg.min_size must be a non-negative integer"),
    ({"min_size": 2.5}, "seg.min_size must be a non-negative integer"),
    ({"threshold_scale": 0}, "seg.threshold_scale must be a positive number"),
    ({"sigma": 1.0}, "seg has unknown keys"),
    ("otsu", "seg must be an object"),
])
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
    monkeypatch.setattr(viability_batch.viab, "live_dead_by_count",
                        lambda *a, **k: seen.append(k.get("threshold")) or real(*a, **k))
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
