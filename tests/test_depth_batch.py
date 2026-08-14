"""Smoke test for the depth-penetration batch driver (fluorostats.depth_batch)."""
import json

import numpy as np
import pandas as pd
import pytest

from fluorostats import depth_batch

tifffile = pytest.importorskip("tifffile")


def _decay_stack(path, lam, n_z=20, yx=8, bg=10.0, i0=200.0):
    """Write a synthetic z-stack whose per-slice mean is i0*exp(-z/lam)+bg."""
    z = np.arange(n_z)[:, None, None]
    vol = np.broadcast_to((i0 * np.exp(-z / lam) + bg).astype(np.float32), (n_z, yx, yx))
    tifffile.imwrite(path, np.ascontiguousarray(vol))


def test_depth_batch_run(tmp_path):
    blank = tmp_path / "blank.tif"
    _decay_stack(blank, lam=1e9, i0=0.0)                       # flat background
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
            _decay_stack(p, lam=lam)
            stacks.append({"label": f"{gname} {i}", "path": str(p)})
        manifest["groups"].append({"name": gname, "background": str(blank), "stacks": stacks})
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))

    depth_batch.run(mpath)

    out = tmp_path / "out"
    assert (out / "auc_per_stack.csv").exists()
    assert (out / "group_depth_summary.csv").exists()

    ps = pd.read_csv(out / "auc_per_stack.csv")
    assert len(ps) == 4                                        # 2 groups x 2 stacks
    assert set(ps["group"]) == {"fast", "slow"}
    # slower decay retains signal deeper -> larger surface-normalised full-depth AUC
    assert ps[ps.group == "slow"].auc_normalized_full.mean() > \
           ps[ps.group == "fast"].auc_normalized_full.mean()
