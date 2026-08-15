"""Degenerate-input / guard-branch tests to close the last coverage gaps."""
import json

import numpy as np
import pytest

from fluorostats import (agreement, metrics_3d, morphometry, objects, power,
                         viability, validate, viability_batch, keyence, style,
                         segment, skeleton)


def test_agreement_too_few_points():
    assert np.isnan(agreement.lins_ccc(np.array([1.0]), np.array([1.0])))   # size < 2
    assert np.isnan(agreement.icc(np.array([1.0]), np.array([1.0])))        # n < 2


def test_keyence_int64_bad_value():
    assert keyence._int64_to_double("not-an-int") is None


def test_metrics3d_empty_mask():
    m = metrics_3d.connectivity_metrics(np.zeros((4, 4, 4), bool))
    assert m["largest_component_fraction"] == 0.0


def test_morphometry_degenerate():
    assert morphometry.depth_span(np.zeros((3, 4, 4)))["span_slices"] == 0
    assert np.isnan(morphometry._gini(np.array([])))
    assert morphometry._cv(np.array([2.0])) == 0.0        # single value
    assert np.isnan(morphometry._cv(np.array([])))


def test_objects_empty_and_single():
    z = np.zeros((5, 5, 5), np.int64)
    _, n = objects.watershed_split(z.astype(bool))
    assert n == 0
    assert objects.equivalent_diameters_um(z, (1.0, 1.0, 1.0)).size == 0
    assert "centroid_gini" in objects.centroid_homogeneity(np.array([[0.0, 1.0, 1.0]]), (2, 4, 4))
    assert "centroid_gini" in objects.centroid_homogeneity(np.zeros((0, 3)), (2, 4, 4))


def test_power_empty_group():
    assert np.isnan(power.bootstrap_power(np.array([]), np.array([1.0, 2.0]), n=5, n_sims=10))
    r = power.fdr_power_curve({"m": []}, {"m": [1.0, 2.0, 3.0]}, ns=[4], n_sims=10)
    assert r is not None


def test_style_material_color_substring_and_fallback():
    assert style.material_color("my GelMA sample") == style.MATERIAL_COLORS["GelMA"]  # substring
    assert style.material_color("totally unknown") == style.PALETTE["muted"]          # fallback


def test_validate_empty_gt_and_double_claim():
    pred = np.zeros((10, 10), int)
    pred[1:4, 1:4] = 1
    assert validate.match_instances(pred, np.zeros((10, 10), int))["fp"] == 1   # no GT -> all FP
    # two predictions overlapping ONE gt -> the 2nd hits an already-matched gt (continue branch)
    gt = np.zeros((10, 10), int); gt[2:8, 2:8] = 1
    pred2 = np.zeros((10, 10), int); pred2[2:5, 2:8] = 1; pred2[5:8, 2:8] = 2
    r = validate.match_instances(pred2, gt)
    assert r["tp"] <= 1


def test_viability_choose_count_method_regimes():
    size, z = 96, 3
    yy, xx = np.mgrid[0:size, 0:size]
    # clean + crowded -> maxima
    clean = np.full((z, size, size), 5.0, np.float32)
    for cy in range(12, size - 12, 8):
        for cx in range(12, size - 12, 8):
            clean[z // 2] += 5000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
    assert viability.choose_count_method(clean, method="otsu", min_size=3)["method"] in {"maxima", "cc"}
    # low SNR -> cc
    noisy = np.random.default_rng(0).normal(100, 90, (z, size, size)).astype(np.float32)
    noisy[z // 2, 40:56, 40:56] += 120
    assert viability.choose_count_method(noisy, method="otsu", min_size=3)["method"] == "cc"
    # no cells -> cc
    assert viability.choose_count_method(np.zeros((z, size, size), np.float32),
                                         method="otsu", min_size=3)["method"] == "cc"


def test_segment_constant_image_paths():
    flat = np.zeros((16, 16), np.float32)
    # some threshold algorithms raise on a constant image -> exercised except branches
    choice = segment.choose_threshold_method(flat)
    assert "method" in choice
    out = segment.binarize(flat, method="consensus")
    assert out.shape == flat.shape


def test_skeleton_prune_spur_free_and_default_length():
    line = np.zeros((3, 30, 30), bool)
    line[1, 15, 5:25] = True                     # a single straight branch, no spurs
    m = skeleton.skeleton_metrics(line, (1.0, 1.0, 1.0), prune=True)   # default min length
    assert "total_length_um" in m


def test_viability_batch_manifest_errors(tmp_path):
    def run_man(man, name):
        mp = tmp_path / f"{name}.json"
        mp.write_text(json.dumps(man))
        return viability_batch.run(mp)

    base = {"live_channel": 0, "dead_channel": 1, "seg": {"method": "otsu", "min_size": 3},
            "count_method": "cc"}
    with pytest.raises(ValueError):                      # empty groups (guard)
        run_man({**base, "groups": []}, "empty")

    with pytest.raises(ValueError):                      # no stacks analysed guard
        run_man({**base, "output_dir": str(tmp_path / "o2"),
                 "groups": [{"name": "g", "stacks": []}]}, "nostacks")
