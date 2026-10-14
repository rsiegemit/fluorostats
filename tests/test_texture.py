"""Tests for fluorostats.texture (orientation, mesh size)."""

import numpy as np
import pytest

from fluorostats import texture


# --------------------------------------------------------------------------- #
# texture.orientation_anisotropy
# --------------------------------------------------------------------------- #
def test_orientation_stripes_vs_isotropic():
    yy, xx = np.mgrid[0:128, 0:128]
    stripes = np.sin(yy / 3.0).astype(float)                 # aligned along x
    blob = np.exp(-((yy - 64) ** 2 + (xx - 64) ** 2) / (2 * 25.0 ** 2))
    a_str = texture.orientation_anisotropy(stripes)
    a_iso = texture.orientation_anisotropy(blob)
    assert a_str["coherence"] > 0.5
    assert a_iso["coherence"] < a_str["coherence"]
    assert 0.0 <= a_str["dominant_orientation_deg"] < 180.0


def test_orientation_flat_and_empty_mask():
    flat = np.zeros((32, 32))
    a = texture.orientation_anisotropy(flat)
    assert a["coherence"] == 0.0 and np.isnan(a["dominant_orientation_deg"])
    img = np.random.default_rng(0).random((32, 32))
    a2 = texture.orientation_anisotropy(img, mask=np.zeros((32, 32), bool))
    assert a2["n_pixels"] == 0


def test_orientation_mask_restricts():
    yy, xx = np.mgrid[0:64, 0:64]
    img = np.sin(xx / 2.0)
    mask = xx < 32
    a = texture.orientation_anisotropy(img, mask=mask)
    assert a["n_pixels"] == int(mask.sum())


# --------------------------------------------------------------------------- #
# texture.mesh_size
# --------------------------------------------------------------------------- #
def test_mesh_size_grid_and_scaling():
    m = np.zeros((100, 100), bool)
    m[::10, :] = True
    m[:, ::10] = True                                        # 10-px mesh grid
    s1 = texture.mesh_size(m, voxel_size_um=1.0)
    s2 = texture.mesh_size(m, voxel_size_um=2.0)
    assert s1 > 0
    assert s2 == pytest.approx(2 * s1, rel=1e-6)


def test_mesh_size_degenerate():
    assert np.isnan(texture.mesh_size(np.zeros((10, 10), bool)))   # empty
    assert np.isnan(texture.mesh_size(np.ones((10, 10), bool)))    # full


# --------------------------------------------------------------------------- #
# texture.orientation_order
# --------------------------------------------------------------------------- #
def test_orientation_order_aligned_vs_random():
    aligned = np.full(50, 30.0)
    rng = np.random.default_rng(2)
    rand = rng.uniform(0, 180, 500)
    oa = texture.orientation_order(aligned, reference_deg=30.0)
    orr = texture.orientation_order(rand)
    assert oa["order"] == pytest.approx(1.0, abs=1e-6)
    assert oa["mean_orientation_deg"] == pytest.approx(30.0, abs=1.0)
    assert oa["alignment"] == pytest.approx(1.0, abs=1e-6)       # parallel to reference
    assert orr["order"] < 0.2                                    # random -> low order
    # perpendicular reference -> alignment ≈ -1
    perp = texture.orientation_order(aligned, reference_deg=120.0)
    assert perp["alignment"] == pytest.approx(-1.0, abs=1e-6)
    assert texture.orientation_order(np.array([]))["n"] == 0
