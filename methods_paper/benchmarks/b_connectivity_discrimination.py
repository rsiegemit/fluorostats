#!/usr/bin/env python3.13
"""
Benchmark: connectivity-metric discrimination of a fragmentation gradient.

We build synthetic 3D volumes that span a controlled gradient from FRAGMENTED
(N isolated blobs) to CONNECTED (one spanning network). The gradient is driven
by a single known parameter -- the number of bridges added between neighbouring
blobs. At 0 bridges the structure is maximally fragmented; once every adjacent
pair is bridged the structure is a single percolating network.

Five connectivity measures are compared on how well they track this known
gradient (Spearman rho vs. fragmentation level):

  1. fluorostats largest_component_fraction   (metrics_3d.connectivity_metrics)
  2. fluorostats euler_number                  (metrics_3d.connectivity_metrics)
  3. fluorostats n_components                  (metrics_3d.connectivity_metrics)
  4. spanning-cluster / percolation indicator  (implemented here via label_3d
     + per-component bounding-box span across all three axes)
  5. mean component size / fragmentation index (implemented here)

Run:  python3.13 b_connectivity_discrimination.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import fluorostats.metrics_3d as m3
from fluorostats.objects import label_3d


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---------------------------------------------------------------------------
# Synthetic structure generation
# ---------------------------------------------------------------------------

# A 3x3x3 = 27-blob lattice living inside a single volume. Blobs are spheres on
# a regular grid. "Bridges" are cylinders joining axis-adjacent blob centres.
GRID = 3                       # blobs per axis -> GRID**3 total blobs
SPACING = 14                   # voxel spacing between blob centres
RADIUS = 4                     # blob radius (voxels)
BRIDGE_R = 1.6                 # bridge cylinder radius (voxels)
MARGIN = 6                     # empty border around the lattice


def _grid_centres() -> list[tuple[int, int, int]]:
    centres = []
    for i in range(GRID):
        for j in range(GRID):
            for k in range(GRID):
                centres.append(
                    (
                        MARGIN + RADIUS + i * SPACING,
                        MARGIN + RADIUS + j * SPACING,
                        MARGIN + RADIUS + k * SPACING,
                    )
                )
    return centres


def _all_adjacent_pairs(centres: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """Index pairs of blobs that are neighbours along a single lattice axis."""
    pairs = []
    for a in range(len(centres)):
        for b in range(a + 1, len(centres)):
            d = np.abs(np.array(centres[a]) - np.array(centres[b]))
            # exactly one axis differs by one spacing, others identical
            if sorted(d.tolist()) == [0, 0, SPACING]:
                pairs.append((a, b))
    return pairs


def _shape() -> tuple[int, int, int]:
    extent = 2 * MARGIN + 2 * RADIUS + (GRID - 1) * SPACING + 1
    return (extent, extent, extent)


def _draw_sphere(vol: np.ndarray, centre, r: float) -> None:
    zz, yy, xx = np.ogrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    cz, cy, cx = centre
    mask = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    vol[mask] = True


def _draw_cylinder(vol: np.ndarray, p0, p1, r: float) -> None:
    """Rasterise a capsule (cylinder + rounded caps) between two points."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    seg = p1 - p0
    seg_len2 = float(seg @ seg)
    zz, yy, xx = np.mgrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    pts = np.stack([zz, yy, xx], axis=-1).astype(float)
    rel = pts - p0
    t = np.clip((rel @ seg) / seg_len2, 0.0, 1.0)
    proj = p0 + t[..., None] * seg
    dist2 = np.sum((pts - proj) ** 2, axis=-1)
    vol[dist2 <= r ** 2] = True


def make_volume(frac_bridges: float, seed: int = 0) -> np.ndarray:
    """
    Build a lattice of blobs and bridge a `frac_bridges` fraction of adjacent
    pairs (0.0 -> fully fragmented, 1.0 -> fully connected network).

    A fixed random order is used so that increasing frac_bridges is a strict
    superset of bridges -- the gradient is monotone by construction.
    """
    centres = _grid_centres()
    vol = np.zeros(_shape(), dtype=bool)
    for c in centres:
        _draw_sphere(vol, c, RADIUS)

    pairs = _all_adjacent_pairs(centres)
    order = np.random.RandomState(seed).permutation(len(pairs))
    n_bridge = int(round(frac_bridges * len(pairs)))
    for idx in order[:n_bridge]:
        a, b = pairs[idx]
        _draw_cylinder(vol, centres[a], centres[b], BRIDGE_R)
    return vol


# ---------------------------------------------------------------------------
# Measures implemented locally (not part of fluorostats)
# ---------------------------------------------------------------------------

def spanning_indicator(mask: np.ndarray) -> float:
    """
    Percolation indicator: 1.0 if any single component's bounding box spans the
    full foreground extent along all three axes, else the best per-axis span
    fraction achieved by any one component (continuous 0..1 so it can be ranked).
    """
    labels, n = label_3d(mask, min_size=0)
    if n == 0:
        return 0.0
    fg = np.argwhere(mask)
    global_span = fg.max(axis=0) - fg.min(axis=0) + 1  # extent occupied by ALL fg
    best = 0.0
    for lab in range(1, n + 1):
        coords = np.argwhere(labels == lab)
        span = coords.max(axis=0) - coords.min(axis=0) + 1
        frac = float(np.min(span / global_span))  # weakest axis governs spanning
        best = max(best, frac)
    return best


def mean_component_size(mask: np.ndarray) -> float:
    """Mean voxel count per connected component (fragmentation index)."""
    labels, n = label_3d(mask, min_size=0)
    if n == 0:
        return 0.0
    counts = np.bincount(labels.ravel())[1:]  # drop background
    return float(counts.mean())


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

@dataclass
class Row:
    frag_level: float          # 0 = connected end, 1 = fragmented end
    frac_bridges: float
    seed: int
    largest_component_fraction: float
    euler_number: float
    n_components: float
    spanning_indicator: float
    mean_component_size: float


def run(seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    bridge_levels = np.linspace(0.0, 1.0, 11)  # 0%..100% of adjacent pairs bridged
    rows: list[Row] = []
    for seed in seeds:
        for fb in bridge_levels:
            vol = make_volume(float(fb), seed=seed)
            fs = m3.connectivity_metrics(vol)
            # Fragmentation level: high when few bridges. Define as 1 - frac_bridges
            # so that rho is reported against *fragmentation* (the intuitive axis).
            rows.append(
                Row(
                    frag_level=1.0 - float(fb),
                    frac_bridges=float(fb),
                    seed=seed,
                    largest_component_fraction=fs["largest_component_fraction"],
                    euler_number=float(fs["euler_number"]),
                    n_components=float(fs["n_components"]),
                    spanning_indicator=spanning_indicator(vol),
                    mean_component_size=mean_component_size(vol),
                )
            )
    return pd.DataFrame([r.__dict__ for r in rows])


MEASURES = [
    "largest_component_fraction",
    "euler_number",
    "n_components",
    "spanning_indicator",
    "mean_component_size",
]

# Human-readable note on each measure's origin.
SOURCE = {
    "largest_component_fraction": "fluorostats",
    "euler_number": "fluorostats",
    "n_components": "fluorostats",
    "spanning_indicator": "implemented (label_3d + bbox span)",
    "mean_component_size": "implemented (label_3d + component sizes)",
}


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for measure in MEASURES:
        rho, p = spearmanr(df["frag_level"], df[measure])
        out.append(
            {
                "measure": measure,
                "source": SOURCE[measure],
                "spearman_rho_vs_fragmentation": round(float(rho), 4),
                "abs_rho": round(abs(float(rho)), 4),
                "p_value": float(p),
            }
        )
    res = pd.DataFrame(out).sort_values("abs_rho", ascending=False).reset_index(drop=True)
    return res


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no tabulate dep)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *body])


def write_summary(df: pd.DataFrame, corr: pd.DataFrame) -> str:
    lines = []
    lines.append("# Connectivity-metric discrimination benchmark\n")
    lines.append(
        "Synthetic 3D lattice of 27 spherical blobs (3x3x3). A known fraction of "
        "adjacent blob pairs is bridged with cylinders, sweeping from fully "
        "**fragmented** (0 bridges) to a fully **connected** spanning network "
        "(all adjacent pairs bridged). Gradient = fragmentation level "
        "(1 - fraction bridged). 11 levels x 5 seeds = 55 volumes.\n"
    )
    lines.append("## Spearman correlation vs. known fragmentation level\n")
    lines.append("Ranked by |rho|. Sign shows direction (positive = rises with fragmentation).\n")
    lines.append(_md_table(corr))
    lines.append("")

    best = corr.iloc[0]
    lines.append("\n## Verdict\n")
    lines.append(
        f"Best discriminator: **{best['measure']}** ({best['source']}), "
        f"|rho| = {best['abs_rho']:.3f}, rho = {best['spearman_rho_vs_fragmentation']:+.3f}.\n"
    )
    # Honest breakdown
    perfect = corr[corr["abs_rho"] >= 0.99]["measure"].tolist()
    weak = corr[corr["abs_rho"] < 0.7]["measure"].tolist()
    if perfect:
        lines.append(
            "Measures that track the gradient essentially monotonically "
            f"(|rho| >= 0.99): {', '.join(perfect)}.\n"
        )
    if weak:
        lines.append(
            "Measures that track it only weakly here (|rho| < 0.70): "
            f"{', '.join(weak)}.\n"
        )
    return "\n".join(lines)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = run()
    corr = correlations(df)

    raw_path = os.path.join(RESULTS_DIR, "connectivity_discrimination_raw.csv")
    corr_path = os.path.join(RESULTS_DIR, "connectivity_discrimination_correlations.csv")
    md_path = os.path.join(RESULTS_DIR, "connectivity_discrimination_summary.md")

    df.to_csv(raw_path, index=False)
    corr.to_csv(corr_path, index=False)
    summary = write_summary(df, corr)
    with open(md_path, "w") as fh:
        fh.write(summary)

    print(f"Wrote {raw_path}")
    print(f"Wrote {corr_path}")
    print(f"Wrote {md_path}\n")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
