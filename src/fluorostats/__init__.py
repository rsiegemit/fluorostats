"""FluoroStats — Universal fluorescence microscopy image quantification.

Public API surface, grouped by module:

  - ``io`` — load_volume (any supported format).
  - ``preprocess`` — channel selection, denoising, background subtraction.
  - ``segment`` — binarisation (Otsu, percentile, manual).
  - ``metrics_3d`` — volume_fraction, connectivity_metrics, skeleton_metrics,
    fov_volume_mm3, normalise_skeleton_metrics.
  - ``metrics_2d`` — 2D coverage/cluster metrics.
  - ``morphometry`` — intensity-only spatial homogeneity, depth profiles,
    depth span, depth centroid (no segmentation needed).
  - ``objects`` — per-object volumes, equivalent diameters, centroids,
    object density, centroid homogeneity.
  - ``stats`` — Mann-Whitney, BH-FDR, Cliff's delta, bootstrap fold-change
    CIs, Stouffer pooling, Scheirer-Ray-Hare 2-way non-parametric ANOVA.
  - ``power`` — bootstrap power and power curves.
  - ``render3d`` — isosurface mesh (light/dark, smoothed, shaded), voxel
    cloud, two-channel live/dead MIP, MIP grid, depth-coded MIP, and
    top/bottom layer split for slide-style image panels.
  - ``plots`` — bar/box/strip plots, summary panels, effect-size heatmaps,
    forest plots, modality panels.
  - ``style`` — publication matplotlib defaults, palette, and per-condition
    color lookup (light) plus the dark palette for reference-style renders.
  - ``qc`` — segmentation QC overlays.
  - ``report`` — per-condition aggregation.
"""

__version__ = "0.3.0"

from . import (  # noqa: F401
    io,
    preprocess,
    segment,
    metrics_3d,
    metrics_2d,
    morphometry,
    objects,
    stats,
    power,
    render3d,
    plots,
    style,
    qc,
    report,
)
