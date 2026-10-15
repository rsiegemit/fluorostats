#!/usr/bin/env python3
"""Example: end-to-end cell-laden tube analysis from a Keyence BZ-X z-stack.

Built on ``fluorostats.keyence`` (stack + metadata) and the segmentation / object /
skeleton toolkit. A worked template for two-channel (nuclei + cell marker) 3D
constructs; adapt the channel roles and metrics to your own assay.

Input: a folder of Keyence `*_CHF*.ome.tif` z-slices (all channels inside each file;
here C0 = DAPI nuclei, C1 = green channel — phalloidin / F-actin cytoskeleton in this
dataset) plus `Image .gci`. One folder = one stack/position.
Only these two file types are needed — the 8-bit `Overlay` / `Preview` exports are
derivable, so requesting just `*_CHF*` + `.gci` keeps datasets ~60% smaller.

Computes, per stack:
  - nuclei: 3D count, density (nuclei/mm^3), size distribution, spatial homogeneity (Gini)
  - z / wall: cell signal vs depth -> wall-band depth, thickness (FWHM), span
  - lumen / coverage: cellular MIP mask scored by ``fluorostats.ring.ring_morphometry``
    (lumen area, wall-relative coverage)
  - cytoskeleton network (green channel): coverage, skeleton junction density,
    overlap with nuclei — apt for the F-actin/phalloidin filament network here
  - intensity: DAPI/GFP means (GAIN-SENSITIVE -> only compare within one exposure setting)

Optionally geometry-aware (`geometry=`), using the general fluorostats helpers:
  - "cross_section" (tube cut across): the rest of the ring morphometry
    — lumen circularity, inner/outer diameter, wall thickness, concentricity —
    plus circumferential/radial cell distribution
    (fluorostats.objects.angular_homogeneity / radial_distribution).
  - "wall" (wall en face): F-actin orientation coherence + network mesh size
    (fluorostats.texture) and the through-wall (outer-third) nuclei fraction.

Always (geometry-independent) via `analyze_organization`, exercising the general
spatial toolkit: nuclei point pattern (objects.nearest_neighbor_stats), nuclei-on-
network association (objects.object_mask_association), nucleus shape + nucleus↔network
alignment (objects.object_shape_metrics + texture.orientation_order), cell-density
spatial heterogeneity (spatial.tile_point_density + spatial_heterogeneity), and a
through-depth network gradient (spatial.slab_reduce).

Writes CSVs + figures to an output folder. `batch()` runs many folders and writes a
combined cross-sample table (use count/structure metrics for cross-formulation
comparison; flag intensity metrics as gain-dependent).

Usage:
    python keyence_tube_analysis.py <input_folder> <output_folder> [downsample]
    # or import: analyze_folder(in_dir, out_dir, geometry="cross_section")
    #            batch({name: folder, ...}, out_dir, geometry="wall")
"""
from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_holes

from fluorostats.segment import binarize, remove_small
from fluorostats.metrics_3d import fov_volume_mm3
from fluorostats.objects import centroid_homogeneity
from fluorostats.pubfig import imnorm
from fluorostats.skeleton import n_junction_nodes
from fluorostats.stats import coefficient_of_variation
from fluorostats import keyence, ring, texture, objects, spatial

# Calibration assumed when the .gci metadata lacks it (announced with a warning).
DEFAULT_PX_UM = 0.4
DEFAULT_Z_STEP_UM = 2.5


# --------------------------------------------------------------------------- #
# Loading — delegates to the reusable fluorostats.keyence reader
# --------------------------------------------------------------------------- #
def load_stack(folder, downsample: int = 3):
    """DAPI/GFP z-stack + metadata via fluorostats.keyence (px/z-step from .gci)."""
    arr, meta = keyence.load_keyence_stack(folder, downsample)
    if arr.shape[0] < 2:
        raise ValueError(f"expected >=2 channels (DAPI, GFP), got {arr.shape[0]}")
    meta = dict(meta)
    for key, default in (("px_um", DEFAULT_PX_UM), ("z_step_um", DEFAULT_Z_STEP_UM)):
        if not meta.get(key):
            warnings.warn(f"{folder}: no {key} in the Keyence metadata; assuming {default} µm, so every "
                          "calibrated length, area, volume and density uses it.", RuntimeWarning, stacklevel=2)
            meta[key] = default
    meta["vox_z_um"] = meta.get("vox_z_um") or meta["z_step_um"]
    meta["vox_xy_um"] = meta.get("vox_xy_um") or round(meta["px_um"] * downsample, 4)
    return arr[0], arr[1], meta


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def analyze_nuclei(dapi, vox):
    """Otsu foreground split into nuclei by ``objects.watershed_split`` (robust to
    touching nuclei in dense tissue); nucleus centres are the object centroids.
    The split's distance transform is in voxels, so it assumes roughly isotropic voxels."""
    vz, vy, vx = vox
    bw = binarize(dapi, method="otsu", min_size=max(8, int(round(30 / (vy * vx)))))
    # min_distance ~ the smallest nucleus radius (~4 µm) in pixels
    lab, n = objects.watershed_split(bw, min_size=max(8, int(round(30 / (vy * vx)))),
                                     min_distance=max(2, int(round(4 / vx))))
    peaks = objects.object_centroids(lab)
    imaged_mm3 = fov_volume_mm3(dapi.shape, vox)
    out = {"nuclei_count": int(n), "density_per_mm3": round(objects.object_density_per_mm3(n, dapi.shape, vox), 1),
           "imaged_volume_mm3": round(imaged_mm3, 4), "dapi_area_fraction_pct": round(100 * bw.mean(), 2)}
    if n:
        vol_um3 = objects.object_volumes_voxels(lab) * float(np.prod(vox))
        out.update({"nucleus_vol_um3_median": round(float(np.median(vol_um3)), 1),
                    "nucleus_diam_um_median": round(float(np.median(objects.equivalent_diameters_um(lab, vox))), 1),
                    "nucleus_size_cv_pct": round(100 * coefficient_of_variation(vol_um3, ddof=1), 1)})
        gini = centroid_homogeneity(peaks, dapi.shape).get("centroid_gini")
        out["spatial_gini"] = round(float(gini), 3) if gini is not None else None
    return out, lab, bw, peaks


def analyze_zprofile(dapi, gfp, vox):
    vz = vox[0]
    dth = np.percentile(dapi, 95)
    gth = np.percentile(gfp, 95)
    z = np.arange(dapi.shape[0]) * vz
    df = pd.DataFrame({
        "depth_um": z,
        "dapi_area_pct": (dapi > dth).reshape(dapi.shape[0], -1).mean(1) * 100,
        "gfp_area_pct": (gfp > gth).reshape(gfp.shape[0], -1).mean(1) * 100,
        "dapi_mean": dapi.reshape(dapi.shape[0], -1).mean(1),
        "gfp_mean": gfp.reshape(gfp.shape[0], -1).mean(1),
    })
    # wall band from DAPI %area: peak + FWHM
    y = df.dapi_area_pct.values
    peak_i = int(y.argmax())
    half = y[peak_i] / 2
    above = np.where(y >= half)[0]
    span_idx = np.where(y > 1)[0]
    band = {"wall_peak_depth_um": round(float(z[peak_i]), 1),
            "wall_fwhm_um": round(float((above[-1] - above[0]) * vz), 1) if len(above) else 0.0,
            "cell_span_um": round(float((span_idx[-1] - span_idx[0]) * vz), 1) if len(span_idx) else 0.0}
    return df, band


def analyze_wall_lumen(dapi, vox, band_slices):
    """Cellular MIP over the wall band -> ``ring.ring_morphometry`` (lumen + coverage).

    Returns the summary metrics, the MIP, the cell mask, the full ring morphometry
    dict (reused by ``analyze_geometry``) and the radial DAPI profile.
    """
    vz, vy, vx = vox
    mip = dapi[band_slices].max(0)
    mask = mip > np.percentile(mip, 80)
    mask = remove_small(mask, 64)
    mask = ndi.binary_closing(mask, iterations=3)
    rm = ring.ring_morphometry(mask, voxel_size_um=(vy, vx))
    cov = rm["wall_coverage_frac"]
    # radial DAPI profile from cellular centroid (dip at centre => lumen)
    cy, cx = ndi.center_of_mass(ndi.binary_fill_holes(mask))
    yy, xx = np.indices(mip.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    radial = ndi.mean(mip.astype(float), r, index=np.arange(0, r.max(), max(1, int(10 / vx))))
    return {"wall_coverage_pct": round(cov * 100, 1) if cov == cov else float("nan"),
            "lumen_area_um2": round(rm["lumen_area"], 0),
            "lumen_present": rm["lumen_present"]}, mip, mask, rm, radial


def analyze_gfp(gfp, dapi_bw, vox):
    vz, vy, vx = vox
    mip = gfp.max(0)
    mask = mip > np.percentile(mip, 85)
    mask = remove_small_holes(remove_small(mask, 64), max_size=64)
    skel = skeletonize(mask)
    # junction nodes (adjacent junction pixels merged), the AngioTool/REAVER definition
    junctions = n_junction_nodes(skel)
    area_mm2 = mip.size * vx * vy / 1e6
    dapi_cov = dapi_bw.max(0)
    overlap = 100 * (mask & dapi_cov).sum() / max(mask.sum(), 1)
    return {"gfp_coverage_pct": round(100 * mask.mean(), 1),
            "gfp_skeleton_junctions_per_mm2": round(junctions / max(area_mm2, 1e-9), 0),
            "gfp_on_nuclei_overlap_pct": round(overlap, 1)}, mip, mask, skel


def analyze_organization(dapi, gfp, peaks, bw, gmip, gmask, vox):
    """Geometry-independent spatial-organisation metrics, via the general
    fluorostats helpers — applies to any 2-channel construct:

      - nuclei point pattern (``objects.nearest_neighbor_stats``): clustered / random / regular
      - nuclei on the network (``objects.object_mask_association``): co-localised fraction
      - nucleus shape + nucleus↔network alignment (``objects.object_shape_metrics``,
        ``texture.orientation_order`` vs the network's dominant orientation)
      - cell-density spatial heterogeneity (``spatial.tile_point_density`` +
        ``spatial.spatial_heterogeneity``): tile-to-tile CV + Moran's I
      - through-depth network gradient (``spatial.slab_reduce``): outer vs inner coverage
    """
    vz, vy, vx = vox
    ny, nx = dapi.shape[1:]
    out: dict = {}

    if len(peaks) >= 2:
        nn = objects.nearest_neighbor_stats(peaks, voxel_size_um=vox)
        out["nn_dist_um"] = round(nn["mean_nn_dist"], 1)
        out["clark_evans_R"] = round(nn["clark_evans_R"], 3)
        out["nuclei_pattern"] = nn["pattern"]

    assoc = objects.object_mask_association(peaks, gmask, voxel_size_um=(vy, vx))
    if assoc["frac_within"] == assoc["frac_within"]:
        out["nuclei_on_network_frac"] = round(assoc["frac_within"], 3)
        out["nuclei_to_network_um"] = round(assoc["median_distance"], 1)

    # per-nucleus shape (2D, touching nuclei split) + alignment to the F-actin network
    lab2d, _ = objects.watershed_split(bw.max(0), min_size=max(8, int(round(20 / (vy * vx)))),
                                       min_distance=max(2, int(round(4 / vx))))
    shape = objects.object_shape_metrics(lab2d, voxel_size_um=(vy, vx))
    if shape["elongation"].size:
        out["nucleus_elongation_median"] = round(float(np.nanmedian(shape["elongation"])), 2)
        net = texture.orientation_anisotropy(gmip, mask=gmask)
        oo = texture.orientation_order(shape["orientation_deg"],
                                       reference_deg=net["dominant_orientation_deg"])
        out["nucleus_order"] = round(oo["order"], 3)                       # self-alignment
        out["nucleus_vs_network_alignment"] = round(oo["alignment"], 3)    # -1..1

    # spatial heterogeneity of cell density (tile the FOV)
    den_grid = spatial.tile_point_density(peaks, (ny, nx), grid=(5, 5),
                                          per_area=True, pixel_area=vy * vx)
    het = spatial.spatial_heterogeneity(den_grid)
    if het["cv"] == het["cv"]:
        out["density_tile_cv"] = round(het["cv"], 3)
    if het["morans_i"] == het["morans_i"]:
        out["density_morans_i"] = round(het["morans_i"], 3)

    # through-depth network coverage gradient (outer surface -> inner)
    thr = float(np.percentile(gfp, 85))
    _, cov = spatial.slab_reduce(gfp, lambda s: float((s > thr).mean()), n_slabs=6)
    valid = cov[~np.isnan(cov)]
    if valid.size >= 2:
        out["network_depth_gradient"] = round(float(valid[0] - valid[-1]), 4)
    return out


def analyze_geometry(geometry, peaks, cmask, rm, gmip, gmask, vox):
    """Geometry-aware metrics via the general fluorostats helpers.

    geometry="cross_section" -> the ring morphometry ``rm`` of ``cmask`` (lumen
      area and coverage are already in the summary) + circumferential/radial cell
      distribution (fluorostats.objects);
    geometry="wall" -> F-actin orientation/mesh (fluorostats.texture) +
      through-wall (depth) nuclei distribution.
    Returns a flat metric dict (empty if geometry is None).
    """
    vz, vy, vx = vox
    out: dict = {}
    if geometry == "cross_section":
        cv = rm["wall_thickness_cv"]
        out.update({
            "lumen_circularity": rm["lumen_circularity"],
            "inner_diam_um": rm["inner_diam"],
            "outer_diam_um": rm["outer_diam"],
            "wall_thickness_um": rm["wall_thickness_mean"],
            "wall_thickness_cv_pct": cv * 100 if cv == cv else float("nan"),
            "concentricity": rm["concentricity"],
        })
        filled = ndi.binary_fill_holes(cmask)
        if filled.any() and len(peaks):
            center = ndi.center_of_mass(filled)                      # (y, x)
            ah = objects.angular_homogeneity(peaks, center)
            rd = objects.radial_distribution(peaks, center, n_bins=3)
            out["circumferential_gini"] = ah["angular_gini"]
            out["circumferential_R"] = ah["resultant_length"]
            out["cells_inner_third_frac"] = float(rd["fractions"][0])
            out["cells_outer_third_frac"] = float(rd["fractions"][-1])
    elif geometry == "wall":
        oa = texture.orientation_anisotropy(gmip, mask=gmask)
        out["factin_coherence"] = oa["coherence"]
        out["factin_mesh_size_um"] = texture.mesh_size(gmask, voxel_size_um=(vy, vx))
        # through-wall depth distribution: shallowest (objective-facing = outer)
        # third of the nucleus z-range vs the rest.
        if len(peaks):
            z = peaks[:, 0].astype(float)
            zlo, zhi = float(z.min()), float(z.max())
            if zhi > zlo:
                out["nuclei_outer_third_frac"] = float((z <= zlo + (zhi - zlo) / 3).mean())
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, out, name, tag):
    fig.tight_layout()
    fig.savefig(out["_dir"] / f"{name}_{tag}.png", dpi=140)
    plt.close(fig)


def _no_ticks(axes):
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])


def _figure_overview(out, name, dapi, gfp, zdf, band):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    ax[0].imshow(imnorm(dapi.max(0), hi=99.7), cmap="gray")
    ax[0].set_title("DAPI (nuclei) MIP")
    ax[1].imshow(imnorm(gfp.max(0), hi=99.7), cmap="gray")
    ax[1].set_title("GFP MIP")
    mg = np.dstack([imnorm(dapi.max(0), hi=99.7), imnorm(gfp.max(0), hi=99.7), imnorm(dapi.max(0), hi=99.7) * 0.3])
    ax[2].imshow(np.clip(mg, 0, 1))
    ax[2].set_title("merge (R=DAPI G=GFP)")
    _no_ticks(ax[:3])
    ax[3].plot(zdf.depth_um, zdf.dapi_area_pct, color="#0072B2", label="DAPI %area (cells)")
    ax[3].plot(zdf.depth_um, zdf.gfp_area_pct, color="#009E73", label="GFP %area")
    ax[3].axvline(band["wall_peak_depth_um"], ls="--", color="k", lw=0.8)
    ax[3].set_xlabel("depth (µm)")
    ax[3].set_ylabel("% area")
    ax[3].legend(fontsize=7)
    ax[3].set_title("cells vs depth")
    fig.suptitle(f"{name} — {out['nuclei_count']} nuclei, {out['density_per_mm3']:.0f}/mm³, wall peak {band['wall_peak_depth_um']:.0f}µm", fontsize=10)
    _save(fig, out, name, "A_overview")


def _figure_nuclei(out, name, lab, peaks):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
    vol_um3 = objects.object_volumes_voxels(lab) * float(np.prod(out["_vox"]))
    if vol_um3.size:
        ax[0].hist(vol_um3.clip(max=np.percentile(vol_um3, 99)), bins=40, color="#0072B2")
    ax[0].set_xlabel("nucleus volume (µm³)")
    ax[0].set_ylabel("count")
    ax[0].set_title("nucleus size dist")
    ax[1].scatter(peaks[:, 2], peaks[:, 1], s=3, c="#0072B2", alpha=0.5)
    ax[1].set_aspect("equal")
    ax[1].invert_yaxis()
    _no_ticks(ax[1:])
    ax[1].set_title(f"nucleus centroids (Gini {out.get('spatial_gini')})")
    _save(fig, out, name, "B_nuclei")


def _figure_wall(out, name, mip_d, cmask, radial):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].imshow(imnorm(mip_d, hi=99.7), cmap="gray")
    ax[0].set_title("cellular MIP (wall band)")
    ov = np.dstack([imnorm(mip_d, hi=99.7)] * 3)
    ov[cmask] = [0.1, 0.5, 0.9]
    ax[1].imshow(ov)
    ax[1].set_title(f"cell mask (blue)\ncoverage {out['wall_coverage_pct']}%  lumen {out['lumen_area_um2']:.0f}µm²")
    ax[2].plot(np.arange(len(radial)) * 10, radial, color="#0072B2")
    ax[2].set_xlabel("radius from centroid (µm)")
    ax[2].set_ylabel("DAPI intensity")
    ax[2].set_title("radial profile (dip=lumen)")
    _no_ticks(ax[:2])
    _save(fig, out, name, "C_wall_lumen")


def _figure_network(out, name, gmip, skel):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    ax[0].imshow(imnorm(gmip, hi=99.7), cmap="gray")
    ax[0].set_title("GFP MIP")
    ov = np.dstack([imnorm(gmip, hi=99.7)] * 3)
    ov[skel] = [0.1, 1, 0.3]
    ax[1].imshow(ov)
    ax[1].set_title(f"GFP skeleton\n{out['gfp_coverage_pct']}% cov  {out.get('gfp_skeleton_junctions_per_mm2')}/mm² junctions")
    _no_ticks(ax)
    _save(fig, out, name, "D_gfp_network")


def make_figures(out, name, dapi, gfp, lab, peaks, zdf, band, wall, gfp_res):
    (mip_d, cmask, radial) = wall
    (gmip, skel) = gfp_res
    _figure_overview(out, name, dapi, gfp, zdf, band)
    _figure_nuclei(out, name, lab, peaks)
    _figure_wall(out, name, mip_d, cmask, radial)
    _figure_network(out, name, gmip, skel)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyze_folder(in_dir, out_dir, downsample: int = 3, name: str | None = None,
                   geometry: str | None = None):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or in_dir.name.strip().replace(" ", "_")
    print(f"[{name}] loading ({downsample}x)...")
    dapi, gfp, meta = load_stack(in_dir, downsample)
    vox = (meta["vox_z_um"], meta["vox_xy_um"], meta["vox_xy_um"])
    print(f"[{name}] {dapi.shape} vox(zyx)={vox} px={meta['px_um']}µm ch={meta['channels']}")
    nuc, lab, bw, peaks = analyze_nuclei(dapi, vox)
    zdf, band = analyze_zprofile(dapi, gfp, vox)
    band_slices = np.where(zdf.dapi_area_pct.values > max(1, zdf.dapi_area_pct.max() * 0.25))[0]
    if len(band_slices) == 0:
        band_slices = np.arange(dapi.shape[0])
    wall, mip_d, cmask, rm, radial = analyze_wall_lumen(dapi, vox, band_slices)
    gfpm, gmip, gmask, skel = analyze_gfp(gfp, bw, vox)
    geom = analyze_geometry(geometry, peaks, cmask, rm, gmip, gmask, vox)
    org = analyze_organization(dapi, gfp, peaks, bw, gmip, gmask, vox)
    summary = {"sample": name, **nuc, **band, **wall, **gfpm, **geom, **org,
               "gfp_mean": round(float(gfp.mean()), 1), "dapi_mean": round(float(dapi.mean()), 1),
               "px_um": meta["px_um"], "z_step_um": meta["z_step_um"], "objective": meta["objective"],
               "n_slices": meta["n_slices_loaded"]}
    summary["_dir"] = out_dir
    summary["_vox"] = vox
    zdf.to_csv(out_dir / f"{name}_zprofile.csv", index=False)
    pd.DataFrame([{k: v for k, v in summary.items() if not k.startswith("_")}]).to_csv(out_dir / f"{name}_summary.csv", index=False)
    json.dump(meta, open(out_dir / f"{name}_meta.json", "w"), indent=2, default=str)
    make_figures(summary, name, dapi, gfp, lab, peaks, zdf, band, (mip_d, cmask, radial), (gmip, skel))
    print(f"[{name}] done -> {out_dir}")
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def batch(samples: dict, out_dir, downsample: int = 3, geometry: str | None = None):
    """Analyse every sample folder and write the cross-sample table. A sample that
    fails is kept as a row whose ``error`` column says why (ERR-3), not dropped."""
    out_dir = Path(out_dir)
    rows = []
    for name, folder in samples.items():
        try:
            rows.append({**analyze_folder(folder, Path(out_dir) / name, downsample,
                                          name=name, geometry=geometry), "error": ""})
        except Exception as e:  # one bad sample must not stop the batch; recorded in its row
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            rows.append({"sample": name, "error": f"{type(e).__name__}: {e}"})
    n_failed = sum(bool(r["error"]) for r in rows)
    if n_failed:
        warnings.warn(f"{n_failed} of {len(rows)} samples failed; see the 'error' column", RuntimeWarning, stacklevel=2)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(Path(out_dir) / "ALL_samples_summary.csv", index=False)
        print(f"\ncross-sample summary -> {Path(out_dir)/'ALL_samples_summary.csv'}")
        print(df.to_string(index=False))
    return rows


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 3:
        print("usage: keyence_tube_analysis.py <input_folder> <output_folder> [downsample]")
        sys.exit(1)
    analyze_folder(a[1], a[2], int(a[3]) if len(a) > 3 else 3)
