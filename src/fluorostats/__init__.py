"""FluoroStats — Universal fluorescence microscopy image quantification.

Public API surface, grouped by module:

  - ``io`` — load_volume (any supported format).
  - ``keyence`` — assemble a z-stack from a Keyence BZ-X folder of per-slice
    OME-TIFFs (load_keyence_stack) and parse the ``.gci`` acquisition metadata.
  - ``preprocess`` — channel selection, denoising, background subtraction.
  - ``segment`` — binarisation (Otsu, percentile, manual).
  - ``metrics_3d`` — volume_fraction, connectivity_metrics, fov_volume_mm3,
    normalise_skeleton_metrics (skeleton_metrics re-exported from ``skeleton``).
  - ``skeleton`` — skeleton_metrics (opt-in spur pruning), prune_skeleton,
    n_junction_nodes (field-standard branchpoint count).
  - ``metrics_2d`` — 2D coverage/cluster metrics.
  - ``agreement`` — method-comparison stats: Bland-Altman, Lin's CCC, ICC,
    agreement_report (for validating fluorostats against other tools).
  - ``morphometry`` — intensity-only spatial homogeneity, depth profiles,
    depth span, depth centroid (no segmentation needed).
  - ``depth`` — depth-penetration / permeability profiles (intensity-vs-depth,
    background subtraction, surface normalisation, AUC, single-exponential λ fit).
  - ``depth_batch`` / ``viability_batch`` — manifest-driven batch drivers (many
    stacks by condition → tidy CSVs + figures).
  - ``objects`` — per-object volumes, equivalent diameters, centroids,
    object density, centroid / angular / radial spatial distribution,
    watershed splitting of touching objects, and border-object clearing.
  - ``ring`` — annular cross-section morphometry: lumen area/circularity,
    wall thickness + coverage, inner/outer diameter, concentricity.
  - ``texture`` — oriented-texture and open-space descriptors for network
    signal: structure-tensor orientation coherence and mesh (pore) size.
  - ``viability`` — Live/Dead quantification: live/dead fractions, depth-resolved
    viability profile, 2D-vs-3D overestimation, attenuation correction.
  - ``validate`` — instance-segmentation metrics: instance F1 / precision /
    recall at a single IoU threshold, plus average precision (AP) averaged
    over a range of IoU thresholds.
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

__version__ = "0.8.0"

from . import (  # noqa: F401
    io,
    preprocess,
    segment,
    skeleton,
    metrics_3d,
    metrics_2d,
    morphometry,
    depth,
    depth_batch,
    keyence,
    objects,
    ring,
    texture,
    viability,
    viability_batch,
    stats,
    agreement,
    validate,
    power,
    render3d,
    plots,
    style,
    qc,
    report,
)
