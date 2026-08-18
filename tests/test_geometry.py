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


# --------------------------------------------------------------------------- #
# objects.object_shape_metrics
# --------------------------------------------------------------------------- #
def test_object_shape_metrics_elongation_and_scaling():
    lab = np.zeros((40, 40), int)
    lab[5:8, 5:35] = 1          # long thin horizontal bar -> elongated
    lab[20:30, 20:30] = 2       # square -> ~round
    s = objects.object_shape_metrics(lab)
    assert s["elongation"][0] > 3.0
    assert s["elongation"][1] < 1.5
    assert 0.0 <= s["orientation_deg"][0] < 180.0
    assert s["solidity"][1] == pytest.approx(1.0, abs=0.05)
    # µm scaling multiplies axis lengths by the mean pixel pitch
    s2 = objects.object_shape_metrics(lab, voxel_size_um=(2.0, 2.0))
    assert s2["major_axis"][0] == pytest.approx(2.0 * s["major_axis"][0], rel=1e-6)


def test_object_shape_metrics_empty_and_ndim_guard():
    e = objects.object_shape_metrics(np.zeros((10, 10), int))
    assert e["elongation"].size == 0
    with pytest.raises(ValueError):
        objects.object_shape_metrics(np.zeros((3, 3, 3), int))


# --------------------------------------------------------------------------- #
# objects.object_mask_association
# --------------------------------------------------------------------------- #
def test_object_mask_association_on_and_off_structure():
    mask = np.zeros((50, 50), bool)
    mask[25, :] = True                                  # a horizontal strand
    on = np.array([[25.0, 10.0], [25.0, 40.0]])         # sit on the strand
    off = np.array([[5.0, 10.0], [45.0, 40.0]])         # 20 px away
    r_on = objects.object_mask_association(on, mask)
    r_off = objects.object_mask_association(off, mask)
    assert r_on["median_distance"] == pytest.approx(0.0)
    assert r_on["frac_within"] == 1.0
    assert r_off["median_distance"] > 15 and r_off["frac_within"] == 0.0
    # µm scaling + explicit threshold
    r = objects.object_mask_association(off, mask, voxel_size_um=(2.0, 2.0), max_dist_um=50.0)
    assert r["frac_within"] == 1.0                       # 20 px * 2 µm = 40 µm < 50
    # 3D centroids (z,y,x) against a 2D mask use the last two columns
    p3 = np.array([[7.0, 25.0, 10.0]])
    assert objects.object_mask_association(p3, mask)["median_distance"] == pytest.approx(0.0)


def test_object_mask_association_degenerate():
    r = objects.object_mask_association(np.zeros((0, 2)), np.ones((5, 5), bool))
    assert r["n"] == 0 and np.isnan(r["median_distance"])
    r2 = objects.object_mask_association(np.array([[1.0, 1.0]]), np.zeros((5, 5), bool))
    assert np.isnan(r2["frac_within"])                   # empty mask


# --------------------------------------------------------------------------- #
# objects.nearest_neighbor_stats
# --------------------------------------------------------------------------- #
def test_nearest_neighbor_clustered_vs_regular():
    rng = np.random.default_rng(0)
    # two tight clusters -> clustered (R < 1)
    clus = np.vstack([rng.normal([10, 10], 1.0, (40, 2)),
                      rng.normal([90, 90], 1.0, (40, 2))])
    # regular lattice -> dispersed (R > 1)
    gy, gx = np.mgrid[0:10, 0:10]
    grid = np.column_stack([gy.ravel() * 10.0, gx.ravel() * 10.0])
    rc = objects.nearest_neighbor_stats(clus)
    rg = objects.nearest_neighbor_stats(grid)
    assert rc["clark_evans_R"] < 0.9 and rc["pattern"] == "clustered"
    assert rg["clark_evans_R"] > 1.1 and rg["pattern"] == "dispersed"
    assert rc["mean_nn_dist"] > 0


def test_nearest_neighbor_3d_and_degenerate():
    rng = np.random.default_rng(1)
    p3 = rng.uniform(0, 50, (60, 3))
    r = objects.nearest_neighbor_stats(p3, voxel_size_um=(2.0, 0.5, 0.5))
    assert np.isfinite(r["clark_evans_R"]) and r["n"] == 60      # 3D branch
    assert objects.nearest_neighbor_stats(np.zeros((1, 2)))["pattern"] == "n/a"   # <2 points
    # unsupported dimensionality -> R is NaN but spacing still returned
    r4 = objects.nearest_neighbor_stats(rng.uniform(0, 1, (5, 4)))
    assert np.isnan(r4["clark_evans_R"]) and r4["mean_nn_dist"] > 0


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
