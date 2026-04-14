<p align="center">
  <img src="https://raw.githubusercontent.com/rsiegemit/fluorostats/main/logo_v2.png" alt="FluoroStats" width="500">
</p>

<p align="center">
  <em>Universal fluorescence microscopy image quantification</em>
</p>

---

Segment, measure, and compare fluorescence signals across experimental conditions — from confocal z-stacks to widefield 2D images.

## Features

- **3D confocal analysis** — volume fraction, Euler number (interconnectedness), largest component fraction, skeleton length/branches/junctions
- **2D coverage analysis** — area fraction, cluster count, largest component fraction, mean/median cluster size
- **Universal format support** — TIFF, OME-TIFF, Olympus (.oib/.oif), Zeiss (.czi), Nikon (.nd2), Leica (.lif), PNG, JPEG, BMP, NumPy
- **Auto-detection** — automatic file format detection, green/fluorescence channel selection, microscope border cropping
- **Replicate-aware** — per-file and per-condition CSVs with mean/std/median, bar+SEM plots, Mann-Whitney U p-values
- **QC overlays** — MIP + mask overlay PNGs for visual verification
- **Publication-ready plots** — boxplots, bar+SEM charts, multi-metric summary panels

## Install

```bash
# Core (TIFF, PNG, JPEG, BMP, NumPy)
pip install fluorostats

# With specific microscope format support
pip install fluorostats[olympus]    # Olympus .oib/.oif
pip install fluorostats[zeiss]      # Zeiss .czi
pip install fluorostats[nikon]      # Nikon .nd2
pip install fluorostats[leica]      # Leica .lif

# Everything
pip install fluorostats[all]
```

## Quick Start

### 3D confocal z-stacks

```bash
fluorostats quant3d \
  --input ./confocal_data/ \
  --output ./results_3d \
  --condition-from parent
```

Expects files organized by condition:
```
confocal_data/
  GelMA/rep1.oib, rep2.oib, rep3.oib
  Hybrid/rep1.oib, rep2.oib, rep3.oib
  CMCMA-rich/rep1.oib, rep2.oib, rep3.oib
```

### 2D fluorescence images

```bash
fluorostats quant2d \
  --input ./endothelial_data/ \
  --output ./results_2d \
  --condition-from parent
```

### Check supported formats

```bash
fluorostats formats
```

## Output

Each run produces:

| File | Description |
|------|-------------|
| `per_file.csv` | Per-file metrics with condition and replicate ID |
| `per_condition.csv` | Mean, std, median grouped by condition |
| `overlays/*.png` | MIP + magenta mask overlay for visual QC |
| `plots/summary_panel.png` | All metrics in one composite figure |
| `plots/*_bar.png` | Bar + SEM plots (when replicates present) |
| `plots/*_box.png` | Boxplots with individual data points |
| `plots/pvalues.csv` | Mann-Whitney U p-values (when replicates present) |
| `run_config.json` | Parameters used (for reproducibility) |

### 3D metrics

| Metric | Description |
|--------|-------------|
| `volume_fraction` | Fraction of voxels segmented as cell |
| `n_components` | Number of disconnected cell clusters |
| `euler_number` | Topological interconnectedness (more negative = more loops/tunnels) |
| `largest_component_fraction` | Fraction of cell volume in the largest connected component |
| `total_length_um` | Total skeleton branch length in um |
| `n_branches` | Number of skeleton branches |
| `n_junctions` | Number of skeleton branching points |
| `mean_branch_length_um` | Average skeleton branch length |

### 2D metrics

| Metric | Description |
|--------|-------------|
| `area_fraction` | Fraction of pixels covered by cells |
| `n_components` | Number of disconnected cell clusters |
| `largest_component_fraction` | Fraction of cell area in the largest cluster |
| `mean_cluster_area_px` | Mean cluster size in pixels |
| `median_cluster_area_px` | Median cluster size (robust to outliers) |

## CLI Options

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `--condition-from` | `parent` | How to assign condition labels: `parent`, `grandparent`, or `filename` |
| `--channel` | auto | Force channel by index or name substring |
| `--threshold` | `otsu` (3D) / `li` (2D) | Thresholding method |
| `--threshold-scale` | `0.9` (3D) / `1.0` (2D) | Scale threshold (lower = more sensitive) |
| `--min-size` | `64` | Minimum object size in pixels/voxels |
| `--sigma` | `1.0` | Gaussian blur sigma |
| `--bg-radius` | `0` (3D) / `15` (2D) | Background subtraction radius (0 = off) |

### 3D-specific

| Option | Description |
|--------|-------------|
| `--no-skeleton` | Skip skeleton analysis (faster) |

### 2D-specific

| Option | Description |
|--------|-------------|
| `--auto-crop / --no-auto-crop` | Auto-crop microscope software borders (default: on) |

## Supported Formats

| Extension | Microscope | Package | Install |
|-----------|-----------|---------|---------|
| `.tif`, `.tiff` | Universal | tifffile (core) | included |
| `.png`, `.jpg`, `.bmp` | Exported images | imageio (core) | included |
| `.npy` | NumPy arrays | numpy (core) | included |
| `.oib`, `.oif` | Olympus FluoView | oiffile | `pip install fluorostats[olympus]` |
| `.czi` | Zeiss ZEN | czifile | `pip install fluorostats[zeiss]` |
| `.nd2` | Nikon NIS-Elements | nd2 | `pip install fluorostats[nikon]` |
| `.lif` | Leica LAS X | readlif | `pip install fluorostats[leica]` |

## Python API

```python
from fluorostats.io import load_volume, load_image, load_auto
from fluorostats.preprocess import select_green_channel, denoise
from fluorostats.segment import binarize
from fluorostats.metrics_3d import volume_fraction, connectivity_metrics, skeleton_metrics
from fluorostats.metrics_2d import coverage_metrics

# Load any supported format
arr, meta = load_auto("sample.czi")
green = select_green_channel(arr, meta["channel_names"])
green = denoise(green)
mask = binarize(green, method="otsu", threshold_scale=0.9)

# 3D metrics
vf = volume_fraction(mask)
conn = connectivity_metrics(mask)
skel = skeleton_metrics(mask, voxel_size_um=meta["voxel_size_um"])

# 2D metrics
cov = coverage_metrics(mask)
```

## Tests

```bash
pip install fluorostats[dev]
pytest tests/ -v
```

## License

MIT
