<p align="center">
  <img src="https://raw.githubusercontent.com/rsiegemit/fluorostats/main/logo_v3.png" alt="FluoroStats" width="500">
</p>

<p align="center">
  <strong>Turn fluorescence microscopy images into publication-ready quantitative data.</strong>
</p>

---

FluoroStats is a Python CLI tool and library that takes your microscopy images — confocal z-stacks, widefield fluorescence, any major microscope format — and produces the numbers, statistics, and figures you need for your paper. No ImageJ macros, no manual thresholding, no spreadsheet wrangling.

## Why FluoroStats?

You have fluorescence images from your experiment. You can *see* that one condition has more cells, or a denser network, or better coverage. But reviewers want numbers, error bars, and p-values.

FluoroStats bridges that gap:

```bash
fluorostats quant3d --input ./my_confocal_data/ --output ./results/
```

That one command gives you:
- **Per-file CSV** with volume fraction, connectivity metrics, skeleton analysis
- **Per-condition summary** with mean, std, median across replicates
- **QC overlays** so you can verify the segmentation looks right
- **Publication plots** — bar charts with SEM, boxplots with individual data points
- **Statistical comparisons** — Mann-Whitney U p-values between all condition pairs

## What Can You Quantify?

### 3D Confocal Z-Stacks

Ideal for: bioprinted constructs, tissue sections, organoids, spheroids — anything where you need to measure cell presence and network structure in a volume.

**Metrics produced:**
- **Volume fraction** — what percentage of the volume contains cells?
- **Euler number** — how interconnected is the cell network? (more negative = more loops and tunnels = more connected)
- **Largest component fraction** — is it one big connected network (97%) or many scattered clusters (19%)?
- **Skeleton length, branches, junctions** — how extensive and branched is the cell network?

### 2D Fluorescence Images

Ideal for: endothelial coverage, monolayer confluence, wound healing assays, any planar cell coverage measurement.

**Metrics produced:**
- **Area fraction** — what percentage of the surface is covered by cells?
- **Cluster count and sizes** — are cells forming one sheet or many scattered patches?
- **Largest component fraction** — is coverage confluent or fragmented?

## Getting Started

### Install

```bash
pip install fluorostats
```

Your images are from a specific microscope? Add format support:

```bash
pip install fluorostats[olympus]    # .oib, .oif files
pip install fluorostats[zeiss]      # .czi files
pip install fluorostats[nikon]      # .nd2 files
pip install fluorostats[leica]      # .lif files
pip install fluorostats[all]        # everything
```

TIFF, OME-TIFF, PNG, JPEG, and BMP work out of the box.

### Organize Your Data

Put each experimental condition in its own folder. Files inside are replicates:

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

### Check Your Results

Open the `overlays/` folder first — each image gets a QC overlay (grayscale intensity + magenta segmentation mask) so you can immediately see if the thresholding worked.

Then look at:
- `per_file.csv` — every measurement for every file
- `per_condition.csv` — summary statistics grouped by condition
- `plots/summary_panel.png` — all metrics in one figure
- `plots/pvalues.csv` — statistical comparisons (when you have replicates)

## Tuning

The defaults are optimized for typical fluorescence microscopy, but every dataset is different. The two most useful knobs:

**Threshold method** — `otsu` (default for 3D) works well for bright, distinct signal. `li` (default for 2D) is better for dim or diffuse signal like endothelial monolayers.

```bash
fluorostats quant3d --input ./data/ --output ./results/ --threshold li
```

**Threshold scale** — multiply the computed threshold by a factor. Lower = capture more dim signal. The default (0.9 for 3D) slightly favors sensitivity over specificity.

```bash
fluorostats quant3d --input ./data/ --output ./results/ --threshold-scale 0.8
```

If you're unsure, run with defaults first, check the overlays, then adjust.

## Supported Formats

```bash
fluorostats formats   # see what's available on your system
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

For custom workflows or integration into your own scripts:

```python
from fluorostats.io import load_auto
from fluorostats.preprocess import select_green_channel, denoise
from fluorostats.segment import binarize
from fluorostats.metrics_3d import volume_fraction, connectivity_metrics, skeleton_metrics

# Load any format — auto-detected
arr, meta = load_auto("my_sample.czi")

# Extract green/fluorescence channel (auto-detects 488nm, FITC, GFP, etc.)
green = select_green_channel(arr, meta["channel_names"])
green = denoise(green)

# Segment
mask = binarize(green, method="otsu", threshold_scale=0.9)

# Measure
print(f"Volume fraction: {volume_fraction(mask):.1%}")
print(f"Connectivity: {connectivity_metrics(mask)}")
print(f"Skeleton: {skeleton_metrics(mask, meta['voxel_size_um'])}")
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
