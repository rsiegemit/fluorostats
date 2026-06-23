"""End-to-end multi-condition analysis using only FluoroStats.

Reproduces the kind of analysis we ran on the Extrusion bioprinting
project (GelMA vs Hybrid × Day 1/7/14 × Top/Middle/Bottom) using only
the FluoroStats library — no project-specific helpers.

Pipeline:

  1.  Per-stack quantification:
        - segmentation pipeline (denoise → background subtract → otsu)
        - 3D metrics (VF, connectivity, skeleton)
        - FOV-normalised densities (zoom-invariant)
        - intensity-based morphometry (lateral Gini/CV, depth span)
        - per-object metrics (nuclei size + density when applicable)

  2.  Statistics:
        - stratified Mann-Whitney + BH-FDR across (day × region × metric)
        - bootstrap fold-change CIs for the headline metrics
        - Stouffer pooling across strata or modalities
        - Scheirer-Ray-Hare 2-way ANOVA for interaction effects
        - bootstrap power curve for sample-size planning

  3.  Visualisation:
        - effect-size heatmap with FDR stars
        - forest plot of fold-change CIs
        - condition strip plot (markers differentiate batches)
        - 3D isosurface render on physical-µm grid

Adapt `DATA_ROOT` and the discovery logic to your own folder layout —
everything below is library code, not paper-specific glue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fluorostats.io import load_volume
from fluorostats.preprocess import select_green_channel, denoise, background_subtract
from fluorostats.segment import binarize
from fluorostats.metrics_3d import (
    volume_fraction, connectivity_metrics, skeleton_metrics,
    normalise_skeleton_metrics,
)
from fluorostats.morphometry import lateral_homogeneity, depth_span
from fluorostats.objects import (
    label_3d, equivalent_diameters_um, object_density_per_mm3,
    object_centroids, centroid_homogeneity,
)
from fluorostats.stats import (
    stratified_mann_whitney, bootstrap_fold_change_ci,
    stouffer_combine, scheirer_ray_hare,
)
from fluorostats.power import fdr_power_curve
from fluorostats.plots import (
    effect_size_heatmap, forest_plot, condition_strip,
)
from fluorostats.render3d import save_isosurface


# ---------------------------------------------------------------------------
# 1. Per-stack quantification
# ---------------------------------------------------------------------------

def quantify_stack(path: Path) -> dict:
    """Run the full per-stack metric set on one volume."""
    arr, meta = load_volume(path)
    vol = select_green_channel(arr, meta["channel_names"])
    smoothed = denoise(vol, sigma=1.0)
    bg = background_subtract(smoothed, radius=15)
    mask = binarize(bg, method="otsu", threshold_scale=0.9, min_size=64)

    vsz = meta["voxel_size_um"]
    metrics = {"file": path.name, "shape": vol.shape, "voxel_size_um": vsz,
               "volume_fraction": volume_fraction(mask)}
    metrics.update(connectivity_metrics(mask))
    metrics.update(skeleton_metrics(mask, voxel_size_um=vsz))
    metrics.update(normalise_skeleton_metrics(metrics, mask.shape, vsz))
    metrics.update(lateral_homogeneity(vol, tiles=8))
    metrics.update(depth_span(vol, voxel_size_um=vsz))
    return metrics


def quantify_objects(path: Path, channel_index: int = 0) -> dict:
    """Per-object metrics for a channel (e.g. DAPI nuclei)."""
    arr, meta = load_volume(path)
    vol = arr[channel_index] if arr.ndim == 4 else arr
    smoothed = denoise(vol, sigma=1.0)
    bg = background_subtract(smoothed, radius=15)
    mask = binarize(bg, method="otsu", threshold_scale=0.9, min_size=64)

    labels, n = label_3d(mask, min_size=64)
    diams = equivalent_diameters_um(labels, meta["voxel_size_um"])
    density = object_density_per_mm3(n, mask.shape, meta["voxel_size_um"])
    centroids = object_centroids(labels)
    hom = centroid_homogeneity(centroids, mask.shape, tiles=8)
    return {
        "file": path.name,
        "n_objects": n,
        "median_diameter_um": float(np.median(diams)) if diams.size else 0.0,
        "density_per_mm3": density,
        **hom,
    }


# ---------------------------------------------------------------------------
# 2. Statistics
# ---------------------------------------------------------------------------

def run_statistics(df: pd.DataFrame) -> dict:
    """Stratified FDR, bootstrap CIs, Stouffer pooling, SRH interaction."""
    metric_cols = [
        "volume_fraction", "length_density_um_per_mm3",
        "junction_density_per_mm3", "largest_component_fraction",
        "lateral_gini", "lateral_cv",
    ]

    fdr_table = stratified_mann_whitney(
        df, value_cols=metric_cols, group_col="material",
        group_a="GelMA", group_b="Hybrid",
        strata=["day", "region"],
    )

    headline = df[(df.day == 14) & (df.region == "middle")]
    g = headline[headline.material == "GelMA"]
    h = headline[headline.material == "Hybrid"]
    cis = pd.DataFrame([
        {"metric": m,
         **bootstrap_fold_change_ci(g[m].values, h[m].values, n_boot=5000)}
        for m in metric_cols
    ])

    pooled = stouffer_combine(
        fdr_table[(fdr_table.region == "middle") &
                  (fdr_table.metric == "volume_fraction")]["p"].values
    )

    srh = scheirer_ray_hare(df, value_col="lateral_gini",
                             factor_a="material", factor_b="day")

    return {"fdr": fdr_table, "fold_ci": cis,
            "stouffer_vf_pooled": pooled, "srh_lateral_gini": srh}


def plan_sample_size(df: pd.DataFrame) -> pd.DataFrame:
    """Power curve to decide how many replicates the immuno arm needs."""
    sub = df[(df.day == 14) & (df.region == "middle")]
    a_samples = {m: sub[sub.material == "GelMA"][m].values
                 for m in ["volume_fraction", "length_density_um_per_mm3"]}
    b_samples = {m: sub[sub.material == "Hybrid"][m].values
                 for m in ["volume_fraction", "length_density_um_per_mm3"]}
    return fdr_power_curve(a_samples, b_samples,
                            ns=[4, 6, 8, 10, 12, 15, 20], n_sims=500)


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------

def make_figures(df: pd.DataFrame, stats: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fdr = stats["fdr"].copy()
    fdr["stratum"] = fdr["region"].astype(str) + "_D" + fdr["day"].astype(str)
    effect_size_heatmap(fdr, out_dir / "heatmap.png",
                        row_col="metric", col_col="stratum",
                        value_col="cliffs_delta", sig_col="sig_q05",
                        title="Hybrid vs GelMA — Cliff's delta (* = BH q<0.05)")

    fc = stats["fold_ci"].rename(columns={"metric": "label"})
    forest_plot(fc, out_dir / "fold_change_forest.png",
                title="Day-14 middle: Hybrid / GelMA (95% bootstrap CI)")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    condition_strip(df[df.region == "middle"], x_col="day", y_col="volume_fraction",
                    hue_col="material", marker_col="batch", ax=ax)
    fig.tight_layout(); fig.savefig(out_dir / "middle_vf_strip.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver — adapt `discover_stacks` to your folder layout
# ---------------------------------------------------------------------------

def discover_stacks(root: Path) -> pd.DataFrame:
    """Walk `root` and label each .oib stack with material/day/region/batch.

    Replace with your own labelling logic. The downstream pipeline only
    cares about the column names, not how they were derived.
    """
    rows = []
    for path in root.rglob("*.oib"):
        name = path.name.lower()
        material = "GelMA" if "gel" in name and "hybrid" not in name else "Hybrid"
        day = next((int(d) for d in (1, 7, 14) if f"{d}d" in name), 14)
        region = next((r for r in ("top", "middle", "bottom") if r in name), "na")
        batch = "new" if "new" in name else "original"
        rows.append({"path": path, "material": material, "day": day,
                     "region": region, "batch": batch})
    return pd.DataFrame(rows)


def main(data_root: Path, out_dir: Path) -> None:
    stacks = discover_stacks(data_root)
    per_file = []
    for _, row in stacks.iterrows():
        m = quantify_stack(row["path"])
        m.update({"material": row["material"], "day": row["day"],
                  "region": row["region"], "batch": row["batch"]})
        per_file.append(m)
    df = pd.DataFrame(per_file)
    df.to_csv(out_dir / "per_file.csv", index=False)

    stats = run_statistics(df)
    stats["fdr"].to_csv(out_dir / "fdr.csv", index=False)
    stats["fold_ci"].to_csv(out_dir / "fold_ci.csv", index=False)
    stats["srh_lateral_gini"].to_csv(out_dir / "srh.csv", index=False)

    power = plan_sample_size(df)
    power.to_csv(out_dir / "power_curve.csv", index=False)

    make_figures(df, stats, out_dir / "plots")

    # Optional 3D render of one representative Hybrid Day-14 middle stack
    rep = stacks.query("material == 'Hybrid' and day == 14 and region == 'middle'").iloc[0]
    arr, meta = load_volume(rep["path"])
    vol = select_green_channel(arr, meta["channel_names"])
    mask = binarize(background_subtract(denoise(vol), radius=15),
                    method="otsu", threshold_scale=0.9, min_size=64)
    save_isosurface(mask, out_dir / "plots" / "rep_iso.png",
                    voxel_size_um=meta["voxel_size_um"],
                    color="#d62728", downsample=(1, 4, 4), scalebar_um=100)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
