"""Tests for the geometry helpers: ring morphometry, texture, point distributions."""
import numpy as np
import pytest

from fluorostats import ring, texture, objects


# --------------------------------------------------------------------------- #
# ring.ring_morphometry
# --------------------------------------------------------------------------- #
def _annulus(n=200, cy=100, cx=100, r_in=30, r_out=60):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - cy, xx - cx)
    return (r <= r_out) & (r >= r_in)


def test_ring_annulus_geometry():
    m = _annulus()
    r = ring.ring_morphometry(m, spacing=(1.0, 1.0))
    assert r["lumen_present"]
    assert r["outer_diam"] == pytest.approx(120, abs=4)
    assert r["inner_diam"] == pytest.approx(60, abs=4)
    assert r["lumen_circularity"] == pytest.approx(1.0, abs=0.15)
    assert r["concentricity"] == pytest.approx(1.0, abs=0.05)     # centred lumen
    assert r["wall_coverage_frac"] == pytest.approx(1.0, abs=0.05)  # solid wall
    assert r["wall_thickness_mean"] == pytest.approx(30, abs=6)


def test_ring_offcentre_lumen_lowers_concentricity():
    m = _annulus(cy=100, cx=100, r_in=0, r_out=60)          # solid disk
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
    m[30:32, 30:32] = False                                  # 4 px enclosed speck
    r = ring.ring_morphometry(m, min_lumen_frac=0.05)
    assert not r["lumen_present"]


def test_ring_second_object_ignored_and_empty_mask():
    m = _annulus()
    m[5:10, 5:10] = True                                     # stray blob -> 2 CCs
    r = ring.ring_morphometry(m)
    assert r["lumen_present"]                                # largest object used
    empty = ring.ring_morphometry(np.zeros((20, 20), bool))
    assert np.isnan(empty["outer_area"]) and not empty["lumen_present"]


def test_ring_spacing_scales_area():
    m = _annulus()
    a1 = ring.ring_morphometry(m, spacing=(1.0, 1.0))["outer_area"]
    a2 = ring.ring_morphometry(m, spacing=(2.0, 2.0))["outer_area"]
    assert a2 == pytest.approx(4 * a1, rel=1e-6)


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
    s1 = texture.mesh_size(m, spacing=1.0)
    s2 = texture.mesh_size(m, spacing=2.0)
    assert s1 > 0
    assert s2 == pytest.approx(2 * s1, rel=1e-6)


def test_mesh_size_degenerate():
    assert np.isnan(texture.mesh_size(np.zeros((10, 10), bool)))   # empty
    assert np.isnan(texture.mesh_size(np.ones((10, 10), bool)))    # full


# --------------------------------------------------------------------------- #
# objects.angular_homogeneity / radial_distribution
# --------------------------------------------------------------------------- #
def test_angular_homogeneity_uniform_vs_clustered():
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ring_pts = np.column_stack([100 + 50 * np.sin(ang), 100 + 50 * np.cos(ang)])
    uni = objects.angular_homogeneity(ring_pts, (100, 100))
    arc = np.linspace(0, 0.4, 60)                            # tight arc
    clus = np.column_stack([100 + 50 * np.sin(arc), 100 + 50 * np.cos(arc)])
    cl = objects.angular_homogeneity(clus, (100, 100))
    assert uni["resultant_length"] < 0.1
    assert cl["resultant_length"] > 0.9
    assert cl["angular_gini"] > uni["angular_gini"]
    assert objects.angular_homogeneity(np.zeros((0, 2)), (0, 0))["n"] == 0


def test_radial_distribution_shell_and_degenerate():
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    at_r = np.column_stack([100 + 50 * np.sin(ang), 100 + 50 * np.cos(ang)])
    r = objects.radial_distribution(at_r, (100, 100), n_bins=10)
    assert r["mean_norm_radius"] == pytest.approx(1.0, abs=0.05)
    assert r["fractions"].sum() == pytest.approx(1.0)
    assert objects.radial_distribution(np.zeros((0, 2)), (0, 0))["n"] == 0
    # all points at the centre -> r_max <= 0 branch
    z = objects.radial_distribution(np.array([[5.0, 5.0], [5.0, 5.0]]), (5, 5))
    assert z["mean_norm_radius"] == 0.0
    # explicit r_max
    assert objects.radial_distribution(at_r, (100, 100), r_max=100.0)["r_max"] == 100.0
