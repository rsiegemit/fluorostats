"""Produce results/b_skeleton_tree.csv — skeleton branch/junction counts on synthetic trees.

A planar bifurcating tree of known depth is drawn into a 3D volume; the true
branch and bifurcation counts follow from the construction. No data needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS
from fluorostats.skeleton import skeleton_metrics


def draw_line(vol, p0, p1):
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    n = int(np.abs(p1 - p0).max()) + 1
    for t in np.linspace(0, 1, n * 2):
        z, y, x = np.round(p0 + t * (p1 - p0)).astype(int)
        vol[z, y, x] = True


def build_tree(depth, length=40, shrink=0.62):
    """Binary tree at fixed z; returns (volume, true #branches, true #bifurcations)."""
    size = 260
    vol = np.zeros((60, size, size), bool)
    z = 30
    counts = {"branches": 0, "bifurcations": 0}

    def rec(p, ang, ln, d):
        if d > depth or ln < 5:
            return
        end = (p[0] + ln * np.cos(ang), p[1] + ln * np.sin(ang))
        draw_line(vol, (z, int(p[1]), int(p[0])), (z, int(end[1]), int(end[0])))
        counts["branches"] += 1
        if d < depth:
            counts["bifurcations"] += 1
            rec(end, ang - 0.5, ln * shrink, d + 1)
            rec(end, ang + 0.5, ln * shrink, d + 1)

    rec((size // 2, size - 20), -np.pi / 2, length, 0)
    return vol, counts["branches"], counts["bifurcations"]


def main():
    rows = []
    for depth in [2, 3, 4]:
        vol, nb, nbif = build_tree(depth)
        m = skeleton_metrics(vol, prune=True, min_branch_length_um=4)
        rows.append({"tree_depth": depth, "true_branches": nb, "fs_n_branches": m["n_branches"],
                     "true_bifurcations": nbif, "fs_n_junction_nodes": m["n_junction_nodes"],
                     "branch_match": abs(m["n_branches"] - nb) <= 1,
                     "bifurcation_match": abs(m["n_junction_nodes"] - nbif) <= 1})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b_skeleton_tree.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
