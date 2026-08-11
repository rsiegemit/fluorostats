"""Smoke + behaviour tests for the manifest-driven Live/Dead viability driver."""
import json
from pathlib import Path

import numpy as np

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
