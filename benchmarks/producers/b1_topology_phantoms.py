"""Produce results/b1_topology_phantoms.csv — connectivity metrics vs analytic topology.

Phantoms with known Euler number (BoneJ-style validation, Doube 2010): solid
ball (chi=1), N disjoint balls (chi=N), solid torus (chi=0), hollow ball (chi=2).
``fluorostats.metrics_3d.connectivity_metrics`` must match exactly. No data needed.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import write_table
from fluorostats.metrics_3d import connectivity_metrics


def _ball(shape, center, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - center[0]) ** 2 + (yy - center[1]) ** 2
            + (xx - center[2]) ** 2) <= radius ** 2


def solid_ball(n=64, r=20):
    return _ball((n, n, n), (n // 2, n // 2, n // 2), r)


def disjoint_balls(k, n=128):
    """k non-touching balls on a cubic lattice; radius leaves a gap between neighbours."""
    v = np.zeros((n, n, n), bool)
    g = int(np.ceil(k ** (1 / 3)))
    step = n // (g + 1)
    r = max(4, step // 2 - 3)
    centres = [(step * (iz + 1), step * (iy + 1), step * (ix + 1))
               for iz in range(g) for iy in range(g) for ix in range(g)][:k]
    for c in centres:
        v |= _ball(v.shape, c, r)
    return v


def solid_torus(n=80, R=22, r=8):
    zz, yy, xx = np.ogrid[:n, :n, :n]
    c = n // 2
    q = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) - R
    return (q ** 2 + (zz - c) ** 2) <= r ** 2


def hollow_ball(n=64, r_out=24, r_in=14):
    c = (n // 2, n // 2, n // 2)
    return _ball((n, n, n), c, r_out) & ~_ball((n, n, n), c, r_in)


# name, volume, expected Euler, expected n_components, expected LCC fraction
CASES = [
    ("solid_ball", solid_ball(), 1, 1, 1.0),
    ("2_disjoint_balls", disjoint_balls(2), 2, 2, 0.5),
    ("3_disjoint_balls", disjoint_balls(3), 3, 3, 1 / 3),
    ("5_disjoint_balls", disjoint_balls(5), 5, 5, 1 / 5),
    ("solid_torus", solid_torus(), 0, 1, 1.0),
    ("hollow_ball", hollow_ball(), 2, 1, 1.0),
]


def main():
    rows = []
    for name, vol, exp_euler, exp_ncomp, exp_lcc in CASES:
        m = connectivity_metrics(vol)
        euler_ok = int(m["euler_number"]) == exp_euler
        ncomp_ok = int(m["n_components"]) == exp_ncomp
        lcc_ok = abs(m["largest_component_fraction"] - exp_lcc) < 0.02
        rows.append({
            "phantom": name,
            "expected_euler": exp_euler, "fluorostats_euler": int(m["euler_number"]),
            "euler_pass": euler_ok,
            "expected_n_comp": exp_ncomp, "fluorostats_n_comp": int(m["n_components"]),
            "ncomp_pass": ncomp_ok,
            "expected_LCC": round(exp_lcc, 4),
            "fluorostats_LCC": round(m["largest_component_fraction"], 4),
            "lcc_pass": lcc_ok,
            "PASS": euler_ok and ncomp_ok and lcc_ok,
        })
    df = pd.DataFrame(rows)
    write_table(df, "b1_topology_phantoms", {"phantoms": [c[0] for c in CASES], "lcc_tolerance": 0.02})
    print(df.to_string(index=False))
    return 0 if df["PASS"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
