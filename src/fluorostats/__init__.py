"""FluoroStats — Universal fluorescence microscopy image quantification.

Public API surface, grouped by module:

  - ``io`` — load_volume (any supported format).
  - ``keyence`` — assemble a z-stack from a Keyence BZ-X folder of per-slice
    OME-TIFFs (load_keyence_stack) and parse the ``.gci`` acquisition metadata.
  - ``preprocess`` — channel selection, denoising, background subtraction.
  - ``segment`` — binarisation by a named threshold method (THRESHOLD_METHODS:
    otsu, li, isodata, triangle, yen, mean; or auto / consensus over them),
    segment_channel (denoise, background-subtract, threshold), remove_small.
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
    stacks by condition → tidy CSVs + figures); ``depth_figures`` draws the
    depth driver's figures.
  - ``objects`` — per-object volumes, equivalent diameters, centroids, shape
    descriptors (elongation/orientation/solidity), object density, centroid /
    angular / radial spatial distribution, nearest-neighbour clustering
    (Clark-Evans), object-to-mask association (e.g. nuclei on a network),
    watershed splitting, border clearing.
  - ``ring`` — annular cross-section morphometry: lumen area/circularity,
    wall thickness + coverage, inner/outer diameter, concentricity.
  - ``texture`` — oriented-texture and open-space descriptors for network
    signal: structure-tensor orientation coherence, nematic orientation order /
    alignment-to-reference, and mesh (pore) size.
  - ``spatial`` — spatial sampling & heterogeneity: tile a field / slab a
    volume with any reducer, plus Moran's I and CV uniformity summaries.
  - ``viability`` — Live/Dead quantification: live/dead fractions, depth-resolved
    viability profile, 2D-vs-3D overestimation, attenuation correction.
  - ``validate`` — instance-segmentation metrics: instance F1 / precision /
    recall at a single IoU threshold (match_instances), plus average precision
    (AP, Kaggle DSB2018 formula) averaged over a range of IoU thresholds.
  - ``stats`` — Mann-Whitney, stratified Mann-Whitney (BH-FDR across strata),
    BH-FDR, Cliff's delta, bootstrap fold-change CIs, Stouffer pooling,
    Scheirer-Ray-Hare 2-way non-parametric ANOVA.
  - ``power`` — bootstrap power and power curves.
  - ``render3d`` — isosurface mesh (light/dark, smoothed, shaded), voxel
    cloud, two-channel live/dead MIP, MIP grid, depth-coded MIP, and
    top/bottom layer split for image panels.
  - ``plots`` — bar/box/strip plots, summary panels, effect-size heatmaps,
    forest plots, modality panels.
  - ``pubfig`` — the one figure style: named rcParams themes (compact, house),
    palettes, panel/image helpers and the figure writer; ``style`` keeps its
    earlier names as a thin layer over it.
  - ``qc`` — segmentation QC overlays.
  - ``report`` — per-condition aggregation.
"""

__version__ = "0.8.2"

import importlib

from . import (  # noqa: F401
    io,
    preprocess,
    segment,
    skeleton,
    metrics_3d,
    metrics_2d,
    morphometry,
    depth,
    keyence,
    manifest,
    objects,
    ring,
    texture,
    spatial,
    viability,
    stats,
    agreement,
    validate,
    power,
    qc,
)

# Figure and orchestration modules load on first use (PEP 562), so importing a
# measurement module never imports matplotlib.pyplot or changes its backend.
_LAZY = {"depth_batch", "depth_figures", "viability_batch", "render3d", "plots",
         "pubfig", "style", "report"}


def __getattr__(name: str):
    if name in _LAZY:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
