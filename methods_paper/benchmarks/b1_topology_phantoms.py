"""B1 (topology) — correctness anchor on phantoms with known Euler number.

Reproduces the validation strategy of BoneJ (Doube 2010): objects with
analytically known topology. fluorostats' connectivity metrics must match
the ground truth EXACTLY (zero-error pass criterion).

Euler number convention (skimage, 3D, connectivity=3):
    chi = (#components) - (#tunnels/handles) + (#cavities)

Phantoms:
    solid ball          -> chi = 1,   1 component,   LCC fraction = 1
    N disjoint balls     -> chi = N,   N components,  LCC fraction = 1/N
    solid torus          -> chi = 0,   1 component,   1 tunnel
    hollow ball (shell)  -> chi = 2,   1 component,   1 cavity

Needs no external data — fully reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fluorostats.metrics_3d import connectivity_metrics

OUT = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Phantom generators
# ---------------------------------------------------------------------------

def _ball(shape, center, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - center[0]) ** 2 + (yy - center[1]) ** 2
            + (xx - center[2]) ** 2) <= radius ** 2


def solid_ball(n=64, r=20):
    v = np.zeros((n, n, n), bool)
    v |= _ball(v.shape, (n // 2, n // 2, n // 2), r)
    return v


def disjoint_balls(k, n=128, r=None):
    """k non-touching balls in a cubic lattice; radius auto-fits spacing."""
    v = np.zeros((n, n, n), bool)
    # lattice big enough to hold k cells
    g = int(np.ceil(k ** (1 / 3)))
    step = n // (g + 1)
    if r is None:
        r = max(4, step // 2 - 3)  # guarantee a gap between neighbours
    placed = 0
    for iz in range(g):
        for iy in range(g):
            for ix in range(g):
                if placed >= k:
                    break
                c = (step * (iz + 1), step * (iy + 1), step * (ix + 1))
                v |= _ball(v.shape, c, r)
                placed += 1
    return v


def solid_torus(n=80, R=22, r=8):
    """Solid torus: chi = 0 (one tunnel)."""
    zz, yy, xx = np.ogrid[:n, :n, :n]
    cz, cy, cx = n // 2, n // 2, n // 2
    # torus around the z-axis in the xy-plane
    q = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - R
    return (q ** 2 + (zz - cz) ** 2) <= r ** 2


def hollow_ball(n=64, r_out=24, r_in=14):
    """Spherical shell: one component, one enclosed cavity -> chi = 2."""
    v = np.zeros((n, n, n), bool)
    c = (n // 2, n // 2, n // 2)
    v |= _ball(v.shape, c, r_out) & ~_ball(v.shape, c, r_in)
    return v


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

CASES = [
    # name, volume, expected euler, expected n_components, expected LCC fraction (or None)
    ("solid_ball", solid_ball(), 1, 1, 1.0),
    ("2_disjoint_balls", disjoint_balls(2), 2, 2, 0.5),
    ("3_disjoint_balls", disjoint_balls(3), 3, 3, 1 / 3),
    ("5_disjoint_balls", disjoint_balls(5), 5, 5, 1 / 5),
    ("solid_torus", solid_torus(), 0, 1, 1.0),
    ("hollow_ball", hollow_ball(), 2, 1, 1.0),
]


def main():
    rows = []
    all_pass = True
    for name, vol, exp_euler, exp_ncomp, exp_lcc in CASES:
        m = connectivity_metrics(vol)
        euler_ok = int(m["euler_number"]) == exp_euler
        ncomp_ok = int(m["n_components"]) == exp_ncomp
        lcc_ok = abs(m["largest_component_fraction"] - exp_lcc) < 0.02
        passed = euler_ok and ncomp_ok and lcc_ok
        all_pass &= passed
        rows.append({
            "phantom": name,
            "expected_euler": exp_euler, "fluorostats_euler": int(m["euler_number"]),
            "euler_pass": euler_ok,
            "expected_n_comp": exp_ncomp, "fluorostats_n_comp": int(m["n_components"]),
            "ncomp_pass": ncomp_ok,
            "expected_LCC": round(exp_lcc, 4),
            "fluorostats_LCC": round(m["largest_component_fraction"], 4),
            "lcc_pass": lcc_ok,
            "PASS": passed,
        })
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name:18s} euler {m['euler_number']:+d} (exp {exp_euler:+d})  "
              f"n_comp {m['n_components']} (exp {exp_ncomp})  "
              f"LCC {m['largest_component_fraction']:.3f} (exp {exp_lcc:.3f})",
              flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "b1_topology_phantoms.csv", index=False)
    print(f"\nSaved {OUT / 'b1_topology_phantoms.csv'}")
    print(f"OVERALL: {'ALL PHANTOMS PASS (zero-error)' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
