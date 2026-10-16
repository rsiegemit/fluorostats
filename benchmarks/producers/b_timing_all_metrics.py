"""Produce results/b_timing_all_metrics.csv — per-call runtime of every fluorostats metric and comparator.

Each operation is timed on the same inputs (first BBBC024 volume + labels, first
BBBC039 image, the Kerkhoff "700 Live; 700 Dead-1" image, seeded random samples):
mean wall-clock ms over 3-5 repeats after one warm-up. Times depend on the
machine, so a re-run reproduces relative cost, not the committed digits.

Deep-learning inference times are in ``b_seg_timing.py`` (one CPU for every
method). Comparator
implementations (the point-pattern statistics and lacunarity that
b_homogeneity_multi validates the tile Gini against) are the independent
references in ``common.py``, not library code; the skimage peak count there is the
library's own algorithm, timed for its raw cost.
Data: ``$FLUOROSTATS_DATA/{BBBC024,BBBC039,kerkhoff_lda}``.
"""
from __future__ import annotations

import glob
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy import stats as sps
from skimage import filters, restoration
from skimage.feature import peak_local_max
from skimage.filters import median
from skimage.measure import euler_number
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.segmentation import watershed

from common import (DATA, MAXIMA_MIN_DISTANCE, MAXIMA_THRESHOLD_REL, bbbc024_volumes, bbbc039_pairs,
                    clark_evans, skimage_peak_count,
                    lacunarity, morisita, quadrat_var, require, ripley_L_dev, thresholds, write_table)
from fluorostats import (agreement, metrics_2d, metrics_3d, morphometry, objects, preprocess, segment,
                         skeleton, stats, validate, viability)

SLOW = ["rolling", "average_precision", "median_filter", "watershed", "instance_f1"]  # 3 repeats, else 5


def load_inputs():
    fp, gp = bbbc024_volumes(n=1)[0]
    vol = tifffile.imread(fp).astype(np.float32)
    mask = segment.remove_small(vol > filters.threshold_otsu(vol), 20)
    mid = vol[vol.shape[0] // 2]
    kerk = [x for x in glob.glob(str(require(DATA / "kerkhoff_lda" / "synthetic") / "**" / "*.tif"),
                                 recursive=True) if "700 Live; 700 Dead-1" in x][0]
    kf = tifffile.imread(kerk).astype(np.float32)
    d = SimpleNamespace(vol=vol, gt=tifffile.imread(gp).astype(np.int32), mask=mask, lab=ndi.label(mask)[0],
                        img2=tifffile.imread(bbbc039_pairs()[0][1]).astype(np.float32),
                        slab=vol[vol.shape[0] // 2 - 8:vol.shape[0] // 2 + 8],
                        sk=skeletonize(mid > filters.threshold_otsu(mid)), live=kf[1], dead=kf[0],
                        a=np.random.default_rng(0).normal(0, 1, 200), b=np.random.default_rng(1).normal(0.5, 1, 200),
                        sdf=pd.DataFrame({"grp": ["A"] * 24 + ["B"] * 24, "st": (["s1"] * 12 + ["s2"] * 12) * 2,
                                          "v": np.random.default_rng(3).normal(0, 1, 48)}))
    d.cent = objects.object_centroids(d.lab)
    d.pts = d.cent[:, 1:] if d.cent.shape[1] == 3 else d.cent
    return d


def point_pattern_comparators(d):
    """The reference statistics b_homogeneity_multi validates the tile Gini against."""
    field = d.vol.shape[1]
    area = field * field
    return [
        ("comparator", "clark_evans NN", lambda: clark_evans(d.pts, area, len(d.pts))),
        ("comparator", "morisita index", lambda: morisita(d.pts, field)),
        ("comparator", "quadrat_variance", lambda: quadrat_var(d.pts, field)),
        ("comparator", "ripleys_K", lambda: ripley_L_dev(d.pts, area)),
        ("comparator", "lacunarity", lambda: lacunarity(d.pts, field)),
    ]


def segmentation_ops(d):
    v = d.vol
    return [
        ("fluorostats", "segment.binarize(otsu) 3D", lambda: segment.binarize(v, method="otsu", min_size=20)),
        *[("comparator", f"threshold_{name} 3D", lambda fn=fn: v > fn(v))
          for name, fn in thresholds(["otsu", "li", "isodata", "triangle", "yen", "mean"]).items()],
        ("fluorostats", "preprocess.denoise(gauss) 3D", lambda: preprocess.denoise(v, sigma=1.0)),
        ("comparator", "median_filter 3D", lambda: median(v, footprint=np.ones((3, 3, 3)))),
        ("comparator", "gaussian_sigma2 3D", lambda: ndi.gaussian_filter(v, 2.0)),
        ("fluorostats", "preprocess.background_subtract 3D-slab",
         lambda: preprocess.background_subtract(d.slab, radius=15)),
        ("comparator", "rolling_ball 3D-slab",
         lambda: np.stack([restoration.rolling_ball(s, radius=15) for s in d.slab])),
        ("comparator", "morph_opening 3D-slab", lambda: ndi.grey_opening(d.slab, size=(1, 15, 15))),
        ("fluorostats", "objects.label_3d", lambda: objects.label_3d(d.mask, min_size=20)),
        ("fluorostats", "objects.watershed_split",
         lambda: objects.watershed_split(d.mask, min_size=20, min_distance=4)),
        ("comparator", "skimage watershed 3D", lambda: watershed(-ndi.distance_transform_edt(d.mask), mask=d.mask)),
    ]


def metric_ops(d):
    v = d.vol
    otsu2 = lambda: d.img2 > filters.threshold_otsu(d.img2)  # noqa: E731 - thresholding is part of the timed call
    return [
        ("fluorostats", "metrics_3d.volume_fraction", lambda: metrics_3d.volume_fraction(d.mask)),
        ("fluorostats", "metrics_3d.connectivity_metrics", lambda: metrics_3d.connectivity_metrics(d.lab)),
        ("comparator", "euler_number(skimage) 3D", lambda: euler_number(d.mask)),
        ("fluorostats", "metrics_3d.density_per_mm3",
         lambda: metrics_3d.density_per_mm3(d.lab.max(), v.shape, (1., 1., 1.))),
        ("fluorostats", "objects.equivalent_diameters_um", lambda: objects.equivalent_diameters_um(d.lab, (1., 1., 1.))),
        ("fluorostats", "objects.count_local_maxima 3D",
         lambda: objects.count_local_maxima(v, min_distance=MAXIMA_MIN_DISTANCE, threshold_rel=MAXIMA_THRESHOLD_REL)),
        ("comparator", "peak_local_max 3D",
         lambda: peak_local_max(v, min_distance=MAXIMA_MIN_DISTANCE, threshold_abs=v.max() * MAXIMA_THRESHOLD_REL)),
        ("fluorostats", "metrics_2d.area_fraction", lambda: metrics_2d.area_fraction(otsu2())),
        ("fluorostats", "metrics_2d.coverage_metrics", lambda: metrics_2d.coverage_metrics(otsu2())),
        ("fluorostats", "morphometry.lateral_homogeneity", lambda: morphometry.lateral_homogeneity(d.mask)),
        ("fluorostats", "morphometry.depth_profile", lambda: morphometry.depth_profile(v)),
        ("fluorostats", "morphometry.depth_centroid", lambda: morphometry.depth_centroid(v)),
        ("fluorostats", "morphometry.depth_span", lambda: morphometry.depth_span(v)),
        ("fluorostats", "objects.centroid_homogeneity", lambda: objects.centroid_homogeneity(d.cent, v.shape)),
        *point_pattern_comparators(d),
        ("fluorostats", "skeleton.skeleton_metrics 2D", lambda: skeleton.skeleton_metrics(d.sk, voxel_size_um=(1., 1.))),
        ("fluorostats", "skeleton.prune_skeleton 2D", lambda: skeleton.prune_skeleton(d.sk, min_branch_length=10)),
        ("fluorostats", "skeleton.n_junction_nodes 2D", lambda: skeleton.n_junction_nodes(d.sk)),
        ("comparator", "skeletonize Lee 2D", lambda: skeletonize(d.sk)),
        ("comparator", "medial_axis 2D", lambda: medial_axis(d.sk)),
        ("comparator", "thin 2D", lambda: thin(d.sk)),
        ("comparator", "skeletonize zhang 2D", lambda: skeletonize(d.sk, method="zhang")),
    ]


def viability_stats_ops(d):
    live, dead, a, b = d.live, d.dead, d.a, d.b
    by_count = [("fluorostats", f"viability.live_dead_by_count {m}",
                 lambda m=m: viability.live_dead_by_count(live, dead, method=m))
                for m in ("cc", "maxima", "watershed", "auto", "all")]
    return [
        ("fluorostats", "viability.live_dead_fractions", lambda: viability.live_dead_fractions(live, dead, min_size=8)),
        *by_count,
        ("comparator", "skimage peak count (live + dead)", lambda: (skimage_peak_count(live), skimage_peak_count(dead))),
        ("fluorostats", "viability.attenuation_correct", lambda: viability.attenuation_correct(d.vol)),
        ("fluorostats", "stats.mann_whitney", lambda: stats.mann_whitney(a, b)),
        ("comparator", "scipy.mannwhitneyu", lambda: sps.mannwhitneyu(a, b, alternative="two-sided")),
        ("fluorostats", "stats.cliffs_delta", lambda: stats.cliffs_delta(a, b)),
        ("fluorostats", "stats.bh_fdr", lambda: stats.bh_fdr([0.01, 0.02, 0.2, 0.5])),
        ("fluorostats", "stats.bootstrap_fold_change_ci",
         lambda: stats.bootstrap_fold_change_ci(np.abs(a) + 1, np.abs(b) + 1, n_boot=1000)),
        ("fluorostats", "stats.stouffer_combine", lambda: stats.stouffer_combine([0.01, 0.04, 0.2])),
        ("comparator", "scipy.combine_pvalues", lambda: sps.combine_pvalues([0.01, 0.04, 0.2], method="stouffer")),
        ("fluorostats", "stats.stratified_mann_whitney",
         lambda: stats.stratified_mann_whitney(d.sdf, ["v"], "grp", "A", "B", strata=["st"])),
        ("fluorostats", "agreement.bland_altman", lambda: agreement.bland_altman(a, b)),
        ("fluorostats", "agreement.lins_ccc", lambda: agreement.lins_ccc(a, b)),
        ("fluorostats", "agreement.icc", lambda: agreement.icc(a, b)),
        ("comparator", "scipy.pearsonr", lambda: sps.pearsonr(a, b)),
        ("fluorostats", "validate.instance_f1", lambda: validate.instance_f1(d.lab, d.gt)),
        ("fluorostats", "validate.average_precision", lambda: validate.average_precision(d.lab, d.gt)),
    ]


def mean_ms(fn, reps):
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000


def main():
    d = load_inputs()
    rows = []
    for grp, name, fn in segmentation_ops(d) + metric_ops(d) + viability_stats_ops(d):
        reps = 3 if any(k in name for k in SLOW) else 5
        rows.append({"group": grp, "metric": name, "ms": round(mean_ms(fn, reps), 3)})
    df = pd.DataFrame(rows)
    write_table(df, "b_timing_all_metrics", {}, timing=True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
