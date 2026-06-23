<p align="center">
  <img src="https://raw.githubusercontent.com/rsiegemit/fluorostats/main/logo.png" alt="FluoroStats" width="500">
</p>

<p align="center">
  <strong>Turn fluorescence microscopy images into publication-ready quantitative data.</strong>
</p>

---

FluoroStats is a Python CLI tool and library that converts microscopy images — confocal z-stacks, widefield fluorescence, any major microscope format — into the numbers, statistics, and figures needed for publication. No ImageJ macros, no manual thresholding, no spreadsheet wrangling.

## Why FluoroStats?

Fluorescence microscopy experiments often produce clear qualitative differences between conditions — more cells, denser networks, better coverage. But manuscripts require quantitative evidence: volume fractions, connectivity metrics, error bars, and p-values.

FluoroStats bridges that gap with a single command:

```bash
fluorostats quant3d --input ./my_confocal_data/ --output ./results/
```

This produces:
- **Per-file CSV** with volume fraction, connectivity metrics, and skeleton analysis
- **Per-condition summary** with mean, std, and median across replicates
- **QC overlays** for visual verification of segmentation accuracy
- **Publication plots** — bar charts with SEM and boxplots with individual data points
- **Statistical comparisons** — Mann-Whitney U p-values between all condition pairs

## What Can It Quantify?

### 3D Confocal Z-Stacks

Well suited for bioprinted constructs, tissue sections, organoids, and spheroids — any volumetric data where cell presence and network structure matter.

**Metrics:**
- **Volume fraction** — percentage of the volume containing cells
- **Euler number** — topological measure of network interconnectedness (more negative = more loops and tunnels)
- **Largest component fraction** — whether the structure forms one connected network (e.g., 97%) or many scattered clusters (e.g., 19%)
- **Skeleton length, branches, junctions** — extent and branching complexity of the cell network

### 2D Fluorescence Images

Well suited for endothelial coverage, monolayer confluence, wound healing assays, and other planar cell coverage measurements.

**Metrics:**
- **Area fraction** — percentage of the surface covered by cells
- **Cluster count and sizes** — whether cells form one confluent sheet or many scattered patches
- **Largest component fraction** — degree of coverage fragmentation

## Getting Started

### Install

```bash
pip install fluorostats
```

For microscope-specific proprietary formats, add the corresponding extra:

```bash
pip install fluorostats[olympus]    # .oib, .oif files
pip install fluorostats[zeiss]      # .czi files
pip install fluorostats[nikon]      # .nd2 files
pip install fluorostats[leica]      # .lif files
pip install fluorostats[all]        # all formats
```

TIFF, OME-TIFF, PNG, JPEG, and BMP are supported out of the box.

### Organize Data

Each experimental condition should be in its own folder, with replicate files inside:

```
my_experiment/
  GelMA/
    sample1.oib
    sample2.oib
    sample3.oib
  Hybrid/
    sample1.oib
    sample2.oib
    sample3.oib
  Control/
    sample1.oib
    sample2.oib
    sample3.oib
```

### Run

```bash
# 3D confocal data
fluorostats quant3d --input ./my_experiment/ --output ./results_3d/

# 2D fluorescence images
fluorostats quant2d --input ./endothelial_images/ --output ./results_2d/
```

### Review Results

Start with the `overlays/` folder — each image gets a QC overlay (grayscale intensity + magenta segmentation mask) for immediate visual verification.

Then inspect:
- `per_file.csv` — measurements for every file
- `per_condition.csv` — summary statistics grouped by condition
- `plots/summary_panel.png` — all metrics in one composite figure
- `plots/pvalues.csv` — statistical comparisons (available when replicates are present)

## Tuning

Defaults are optimized for typical fluorescence microscopy, but two parameters are worth adjusting for specific datasets:

**Threshold method** — `otsu` (default for 3D) works well for bright, high-contrast signal. `li` (default for 2D) is better suited to dim or diffuse signal such as endothelial monolayers.

```bash
fluorostats quant3d --input ./data/ --output ./results/ --threshold li
```

**Threshold scale** — scales the computed threshold by a multiplicative factor. Lower values capture more dim signal. The 3D default of 0.9 slightly favors sensitivity over specificity.

```bash
fluorostats quant3d --input ./data/ --output ./results/ --threshold-scale 0.8
```

A recommended workflow: run with defaults, review the overlays, then adjust if needed.

## Supported Formats

```bash
fluorostats formats   # check availability on the current system
```

| Format | Microscope | Install |
|--------|-----------|---------|
| `.tif` `.tiff` | Universal / OME-TIFF / ImageJ | included |
| `.png` `.jpg` `.bmp` | Exported snapshots | included |
| `.npy` | NumPy arrays | included |
| `.oib` `.oif` | Olympus FluoView | `fluorostats[olympus]` |
| `.czi` | Zeiss ZEN | `fluorostats[zeiss]` |
| `.nd2` | Nikon NIS-Elements | `fluorostats[nikon]` |
| `.lif` | Leica LAS X | `fluorostats[leica]` |

## Python API

FluoroStats can also be used as a library for custom analysis workflows:

```python
from fluorostats.io import load_auto
from fluorostats.preprocess import select_green_channel, denoise
from fluorostats.segment import binarize
from fluorostats.metrics_3d import volume_fraction, connectivity_metrics, skeleton_metrics

# Load any supported format (auto-detected)
arr, meta = load_auto("my_sample.czi")

# Extract the fluorescence channel (auto-detects 488nm, FITC, GFP, etc.)
green = select_green_channel(arr, meta["channel_names"])
green = denoise(green)

# Segment
mask = binarize(green, method="otsu", threshold_scale=0.9)

# Measure
print(f"Volume fraction: {volume_fraction(mask):.1%}")
print(f"Connectivity: {connectivity_metrics(mask)}")
print(f"Skeleton: {skeleton_metrics(mask, meta['voxel_size_um'])}")
```

## Advanced Analysis (v0.2)

Beyond the per-file metrics, FluoroStats now exposes the toolkit needed to run a full statistical analysis directly from the library — homogeneity, density normalisation, multi-group statistics, power planning, and publication-style 3D renders.

### Spatial homogeneity and depth (no segmentation needed)

```python
from fluorostats.morphometry import (
    lateral_homogeneity, depth_profile, depth_span, depth_centroid,
)

hom = lateral_homogeneity(green, tiles=8)        # Gini + CV across an 8x8 XY grid
prof = depth_profile(green)                       # mean intensity vs z slice
span = depth_span(green, voxel_size_um=meta["voxel_size_um"])
cent = depth_centroid(green, voxel_size_um=meta["voxel_size_um"])
```

Useful for "is the signal uniformly distributed?" and "how deep do cells penetrate?" questions without committing to a binary mask.

### Per-object measurements

```python
from fluorostats.objects import (
    label_3d, equivalent_diameters_um, object_density_per_mm3, centroid_homogeneity,
)

labels, n = label_3d(mask, min_size=64)
diams_um = equivalent_diameters_um(labels, meta["voxel_size_um"])
density = object_density_per_mm3(n, mask.shape, meta["voxel_size_um"])
centroids = object_centroids(labels)
spatial = centroid_homogeneity(centroids, mask.shape, tiles=8)
```

Right tool for nuclei sizing (DAPI), cluster-size distributions, and centroid-based homogeneity checks.

### FOV-normalised densities (digital-zoom-safe)

```python
from fluorostats.metrics_3d import normalise_skeleton_metrics

skel = skeleton_metrics(mask, meta["voxel_size_um"])
skel = normalise_skeleton_metrics(skel, mask.shape, meta["voxel_size_um"])
# adds length_density_um_per_mm3, junction_density_per_mm3, branch_density_per_mm3
```

Use whenever stacks have different voxel sizes — counts/lengths per FOV are not comparable, but per-mm³ densities are.

### Multi-group statistics

```python
from fluorostats.stats import (
    stratified_mann_whitney, bootstrap_fold_change_ci,
    stouffer_combine, scheirer_ray_hare, cliffs_delta, bh_fdr,
)

# Stratified Mann-Whitney + BH-FDR across (region × metric)
stats_df = stratified_mann_whitney(
    df, value_cols=["volume_fraction", "length_density_um_per_mm3"],
    group_col="material", group_a="GelMA", group_b="Hybrid",
    strata=["day", "region"],
)

# Distribution-free fold-change interval
ci = bootstrap_fold_change_ci(gelma_vf, hybrid_vf, n_boot=5000)
# {"fold_change_median": 9.5, "ci_low": 2.5, "ci_high": 38.0, ...}

# Pool independent evidence (e.g. across modalities)
pooled = stouffer_combine([p_live_dead, p_immuno])

# Non-parametric 2-way ANOVA on ranks (Scheirer-Ray-Hare)
anova = scheirer_ray_hare(df, value_col="lateral_gini",
                          factor_a="material", factor_b="day")
```

### Bootstrap power analysis

```python
from fluorostats.power import bootstrap_power, power_curve, fdr_power_curve

# How many replicates do I need to clear FDR at q < 0.05?
curve = power_curve(samples_a, samples_b,
                    ns=[4, 6, 8, 10, 12, 15, 20], n_sims=1000)

# Joint power under BH-FDR across multiple metrics
multi = fdr_power_curve(samples_per_metric_a, samples_per_metric_b,
                        ns=[4, 8, 12, 20], n_sims=500)
```

### Publication 3D rendering

```python
from fluorostats.render3d import render_isosurface, render_voxel_cloud, save_isosurface

save_isosurface(
    mask, "out/iso.png",
    voxel_size_um=meta["voxel_size_um"],
    color="#d62728", downsample=(1, 4, 4), scalebar_um=100,
)
```

Marching-cubes isosurface on a physical-micrometre grid, or chunky voxel-cloud variant. Both compose into multi-panel figures via the standard matplotlib subplot machinery.

### Effect-size grids and forest plots

```python
from fluorostats.plots import effect_size_heatmap, forest_plot, modality_panel

effect_size_heatmap(stats_df, "out/heatmap.png",
                    row_col="metric", col_col="region",
                    value_col="cliffs_delta", sig_col="sig_q05")

forest_plot(bootstrap_ci_df, "out/forest.png",
            label_col="metric", center_col="fold_change_median",
            lo_col="ci_low", hi_col="ci_high", log_scale=True)

modality_panel(df, metrics=["volume_fraction", "length_density_um_per_mm3"],
               modality_col="modality", out_path="out/modality.png")
```

## All CLI Options

<details>
<summary>fluorostats quant3d</summary>

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Folder containing volume files |
| `--output` | required | Output folder for results |
| `--condition-from` | `parent` | Label source: `parent` folder, `grandparent`, or `filename` |
| `--channel` | auto | Force channel by index or name |
| `--threshold` | `otsu` | Thresholding method: `otsu` or `li` |
| `--threshold-scale` | `0.9` | Scale threshold (lower = more sensitive) |
| `--min-size` | `64` | Min object size in voxels |
| `--sigma` | `1.0` | Gaussian blur sigma |
| `--bg-radius` | `0` | Background subtraction radius (0 = off) |
| `--no-skeleton` | off | Skip skeleton analysis (faster) |
| `--no-overlays` | off | Skip QC overlay images |
| `--no-plots` | off | Skip comparison plots |

</details>

<details>
<summary>fluorostats quant2d</summary>

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Folder containing image files |
| `--output` | required | Output folder for results |
| `--condition-from` | `parent` | Label source: `parent` folder, `grandparent`, or `filename` |
| `--channel` | auto | Force channel by index or name |
| `--threshold` | `li` | Thresholding method: `otsu` or `li` |
| `--threshold-scale` | `1.0` | Scale threshold (lower = more sensitive) |
| `--min-size` | `64` | Min object size in pixels |
| `--sigma` | `1.0` | Gaussian blur sigma |
| `--bg-radius` | `15` | Background subtraction radius (0 = off) |
| `--auto-crop` | on | Auto-crop microscope software borders |
| `--no-overlays` | off | Skip QC overlay images |
| `--no-plots` | off | Skip comparison plots |

</details>

## License

MIT
