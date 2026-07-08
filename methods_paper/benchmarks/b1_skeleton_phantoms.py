"""B1 (skeleton) — correctness anchor on line phantoms with known geometry.

fluorostats and Fiji AnalyzeSkeleton both build on the Lee-1994 thinning
algorithm, so on clean thin structures they should agree with the analytic
ground truth. This validates:
    - total_length_um : geometric length (primary, unambiguous)
    - n_branches      : number of skeleton segments (exact for simple topologies)

n_junctions in fluorostats counts junction-to-junction branches (skan
branch_type==2), NOT the number of junction NODES — a convention worth
pinning down in the paper. We report it descriptively here.

Voxel size = (1,1,1) so length in um == length in voxels. Needs no external data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fluorostats.metrics_3d import skeleton_metrics

OUT = Path(__file__).resolve().parent / "results"
N = 96
C = N // 2
VOX = (1.0, 1.0, 1.0)


def _line(v, p0, p1):
    """Draw a 1-voxel axis-aligned line segment [p0, p1] inclusive."""
    p0 = np.array(p0); p1 = np.array(p1)
    steps = int(np.abs(p1 - p0).max())
    for t in np.linspace(0, 1, steps + 1):
        z, y, x = np.round(p0 + t * (p1 - p0)).astype(int)
        v[z, y, x] = True


def straight_line(L=40):
    v = np.zeros((N, N, N), bool)
    _line(v, (C, C, C - L // 2), (C, C, C + L // 2))
    return v, {"length": L, "n_branches": 1}


def bent_L(a=30):
    v = np.zeros((N, N, N), bool)
    _line(v, (C, C, C), (C, C, C + a))   # arm 1 along x
    _line(v, (C, C, C), (C, C + a, C))   # arm 2 along y (shares corner)
    return v, {"length": 2 * a, "n_branches": 1}  # one bent path, no junction


def two_lines(L=30):
    v = np.zeros((N, N, N), bool)
    _line(v, (C, C - 10, C - L // 2), (C, C - 10, C + L // 2))
    _line(v, (C, C + 10, C - L // 2), (C, C + 10, C + L // 2))
    return v, {"length": 2 * L, "n_branches": 2}


def y_junction(a=25):
    v = np.zeros((N, N, N), bool)
    _line(v, (C, C, C), (C, C, C + a))         # +x
    _line(v, (C, C, C), (C, C + a, C - a // 2))  # up-left
    _line(v, (C, C, C), (C, C - a, C - a // 2))  # down-left
    return v, {"length": None, "n_branches": 3}


def plus_cross(a=25):
    v = np.zeros((N, N, N), bool)
    _line(v, (C, C, C - a), (C, C, C + a))  # x axis
    _line(v, (C, C - a, C), (C, C + a, C))  # y axis
    return v, {"length": 4 * a, "n_branches": 4}


CASES = [
    ("straight_line", straight_line),
    ("bent_L", bent_L),
    ("two_lines", two_lines),
    ("y_junction", y_junction),
    ("plus_cross", plus_cross),
]


def main():
    rows = []
    ok = True
    for name, gen in CASES:
        vol, truth = gen()
        m = skeleton_metrics(vol, voxel_size_um=VOX)
        length = m["total_length_um"]
        nb = m["n_branches"]
        # n_branches is exact ground truth
        nb_ok = (nb == truth["n_branches"])
        # length within 8% (skeletonization can trim ~1 voxel per branch end)
        if truth["length"] is not None:
            len_err = abs(length - truth["length"]) / truth["length"] * 100
            len_ok = len_err < 8.0
        else:
            len_err = float("nan"); len_ok = True
        passed = nb_ok and len_ok
        ok &= passed
        rows.append({
            "phantom": name,
            "expected_length": truth["length"],
            "fluorostats_length": round(length, 2),
            "length_err_pct": round(len_err, 2) if truth["length"] else None,
            "expected_n_branches": truth["n_branches"],
            "fluorostats_n_branches": nb,
            "n_branches_pass": nb_ok,
            "n_junctions_JtoJ": m["n_junctions"],
            "PASS": passed,
        })
        flag = "PASS" if passed else "FAIL"
        el = f"{len_err:.1f}%" if truth["length"] else "n/a"
        print(f"  [{flag}] {name:14s} length={length:6.1f} (exp {truth['length']}, "
              f"err {el})  n_branches={nb} (exp {truth['n_branches']})", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "b1_skeleton_phantoms.csv", index=False)
    print(f"\nSaved {OUT / 'b1_skeleton_phantoms.csv'}")
    print(f"OVERALL: {'ALL SKELETON PHANTOMS PASS' if ok else 'SOME FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
