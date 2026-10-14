"""Produce results/b_vascular_phantom_3d.csv — vessel metrics on a 3D tube-tree phantom.

Cylinders of radius 3 voxels are swept along a bifurcating 3D tree whose total
centreline length and segment count are known exactly; fluorostats skeleton
length, branch count and volume fraction are scored against them. No data needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS
from fluorostats.metrics_3d import volume_fraction
from fluorostats.skeleton import skeleton_metrics


def draw_cyl(vol, p0, p1, radius):
    """Sweep a ball of ``radius`` from p0 to p1 into ``vol``; returns the segment length."""
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    length = np.linalg.norm(p1 - p0)
    zz, yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
    ball = (zz ** 2 + yy ** 2 + xx ** 2) <= radius ** 2
    for t in np.linspace(0, 1, int(length * 2) + 2):
        c = np.round(p0 + t * (p1 - p0)).astype(int)
        sl = tuple(slice(max(0, a), min(s, a + 2 * radius + 1)) for a, s in zip(c - radius, vol.shape))
        bsl = tuple(slice(max(0, -a), max(0, -a) + (s.stop - s.start)) for a, s in zip(c - radius, sl))
        vol[sl] |= ball[bsl]
    return length


def phantom(depth, size=140, radius=3):
    vol = np.zeros((size, size, size), bool)
    acc = {"segments": 0, "length": 0.0}

    def rec(p, vec, ln, d):
        if d > depth or ln < 8:
            return
        e = p + vec * ln
        acc["length"] += draw_cyl(vol, p, e, radius)
        acc["segments"] += 1
        if d < depth:
            for dth in (-0.5, 0.5):
                nv = vec.copy()
                nv[1] = vec[1] * np.cos(dth) - vec[2] * np.sin(dth)
                nv[2] = vec[1] * np.sin(dth) + vec[2] * np.cos(dth)
                rec(e, nv / np.linalg.norm(nv), ln * 0.7, d + 1)

    rec(np.array([size // 2, size // 2, 20.]), np.array([0, 0, 1.]), 34, 0)
    return vol, acc["length"], acc["segments"]


def main():
    rows = []
    for depth in [1, 2]:
        vol, true_len, true_seg = phantom(depth)
        true_vf = vol.mean()
        m = skeleton_metrics(vol, voxel_size_um=(1.0, 1.0, 1.0))
        meas_len = m["total_length_um"]
        vf = volume_fraction(vol)
        rows.append({"phantom": f"tree_d{depth}", "true_centerline_len": round(true_len, 1),
                     "fs_skeleton_len": round(float(meas_len), 1),
                     "len_err_pct": round(abs(meas_len - true_len) / true_len * 100, 1),
                     "true_segments": true_seg, "fs_branches": m["n_branches"],
                     "true_vf": round(true_vf, 4), "fs_vf": round(float(vf), 4),
                     "vf_err_pct": round(abs(vf - true_vf) / true_vf * 100, 1)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b_vascular_phantom_3d.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
