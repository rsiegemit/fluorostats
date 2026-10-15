"""Tests for fluorostats.ring (ring morphometry)."""

import numpy as np
import pytest

from fluorostats import ring


# --------------------------------------------------------------------------- #
# ring.ring_morphometry
# --------------------------------------------------------------------------- #
def _annulus(n=200, cy=100, cx=100, r_in=30, r_out=60):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - cy, xx - cx)
    return (r <= r_out) & (r >= r_in)


def test_ring_annulus_geometry():
    m = _annulus()
    r = ring.ring_morphometry(m, voxel_size_um=(1.0, 1.0))
    assert r["lumen_present"]
    assert r["outer_diam"] == pytest.approx(120, abs=4)
    assert r["inner_diam"] == pytest.approx(60, abs=4)
    assert r["lumen_circularity"] == pytest.approx(1.0, abs=0.15)
    assert r["concentricity"] == pytest.approx(1.0, abs=0.05)  # centred lumen
    assert r["wall_coverage_frac"] == pytest.approx(1.0, abs=0.05)  # solid wall
    assert r["wall_thickness_mean"] == pytest.approx(30, abs=6)


def test_ring_scalar_voxel_size_is_isotropic():
    # Regression: a scalar voxel_size_um raised TypeError ('float' is not subscriptable).
    m = _annulus()
    assert ring.ring_morphometry(m, voxel_size_um=0.5) == ring.ring_morphometry(
        m, voxel_size_um=(0.5, 0.5)
    )


def test_ring_offcentre_lumen_lowers_concentricity():
    m = _annulus(cy=100, cx=100, r_in=0, r_out=60)  # solid disk
    # carve an off-centre hole fully enclosed by the disk
    yy, xx = np.mgrid[0:200, 0:200]
    hole = np.hypot(yy - 118, xx - 118) <= 15
    m = m & ~hole
    r = ring.ring_morphometry(m)
    assert r["lumen_present"]
    assert r["concentricity"] < 0.9


def test_ring_solid_disk_has_no_lumen():
    yy, xx = np.mgrid[0:120, 0:120]
    disk = np.hypot(yy - 60, xx - 60) <= 40
    r = ring.ring_morphometry(disk)
    assert not r["lumen_present"]
    assert r["inner_diam"] == 0.0
    assert np.isnan(r["concentricity"])


def test_ring_tiny_hole_below_floor_is_not_lumen():
    m = np.ones((60, 60), bool)
    m[30:32, 30:32] = False  # 4 px enclosed speck
    r = ring.ring_morphometry(m, min_lumen_frac=0.05)
    assert not r["lumen_present"]


def test_ring_second_object_ignored_and_empty_mask():
    m = _annulus()
    m[5:10, 5:10] = True  # stray blob -> 2 CCs
    r = ring.ring_morphometry(m)
    assert r["lumen_present"]  # largest object used
    empty = ring.ring_morphometry(np.zeros((20, 20), bool))
    assert np.isnan(empty["outer_area"]) and not empty["lumen_present"]


def test_ring_spacing_scales_area():
    m = _annulus()
    a1 = ring.ring_morphometry(m, voxel_size_um=(1.0, 1.0))["outer_area"]
    a2 = ring.ring_morphometry(m, voxel_size_um=(2.0, 2.0))["outer_area"]
    assert a2 == pytest.approx(4 * a1, rel=1e-6)


def test_ring_morphometry_refuses_a_3d_mask():
    with pytest.raises(ValueError, match=r"2D \(Y, X\) cross-section"):
        ring.ring_morphometry(np.ones((3, 16, 16), bool))


def test_ring_wall_thickness_with_a_uint8_0_255_skeleton(monkeypatch):
    """scikit-image < 0.23 can return a uint8 0/255 skeleton; it must act as a mask."""
    import skimage.morphology

    ref = ring.ring_morphometry(_annulus(), voxel_size_um=(1.0, 1.0))
    orig = skimage.morphology.skeletonize
    monkeypatch.setattr(skimage.morphology, "skeletonize", lambda m: orig(m).astype(np.uint8) * 255)
    assert ring.ring_morphometry(_annulus(), voxel_size_um=(1.0, 1.0)) == ref
