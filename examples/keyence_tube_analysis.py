#!/usr/bin/env python3
"""Example: end-to-end cell-laden tube analysis from a Keyence BZ-X z-stack.

Built on ``fluorostats.keyence`` (stack + metadata) and the segmentation / object /
skeleton toolkit. A worked template for two-channel (nuclei + cell marker) 3D
constructs; adapt the channel roles and metrics to your own assay.

Input: a folder of Keyence `*_CHF*.ome.tif` z-slices (all channels inside each file;
here C0 = DAPI nuclei, C1 = GFP) plus `Image .gci`. One folder = one stack/position.
Only these two file types are needed — the 8-bit `Overlay` / `Preview` exports are
derivable, so requesting just `*_CHF*` + `.gci` keeps datasets ~60% smaller.

Computes, per stack:
  - nuclei: 3D count, density (nuclei/mm^3), size distribution, spatial homogeneity (Gini)
  - z / wall: cell signal vs depth -> wall-band depth, thickness (FWHM), span
  - lumen / coverage: cellular MIP mask, largest internal void (lumen) area, coverage
  - GFP network: coverage, skeleton junction density, GFP-vs-nuclei overlap
  - intensity: DAPI/GFP means (GAIN-SENSITIVE -> only compare within one exposure setting)

Writes CSVs + figures to an output folder. `batch()` runs many folders and writes a
combined cross-sample table (use count/structure metrics for cross-formulation
comparison; flag intensity metrics as gain-dependent).

Usage:
    python keyence_tube_analysis.py <input_folder> <output_folder> [downsample]
    # or import: analyze_folder(in_dir, out_dir); batch({name: folder, ...}, out_dir)
"""
from __future__ import annotations
import sys, re, glob, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
warnings.filterwarnings("ignore")

from fluorostats.segment import binarize
from fluorostats.objects import centroid_homogeneity
from fluorostats import keyence


# --------------------------------------------------------------------------- #
# Loading — delegates to the reusable fluorostats.keyence reader
# --------------------------------------------------------------------------- #
def load_stack(folder, downsample: int = 3):
    """DAPI/GFP z-stack + metadata via fluorostats.keyence (px/z-step from .gci)."""
    arr, meta = keyence.load_keyence_stack(folder, downsample)
    if arr.shape[0] < 2:
        raise ValueError(f"expected >=2 channels (DAPI, GFP), got {arr.shape[0]}")
    meta["px_um"] = meta.get("px_um") or 0.4
    meta["z_step_um"] = meta.get("z_step_um") or 2.5
    meta["vox_z_um"] = meta.get("vox_z_um") or meta["z_step_um"]
    meta["vox_xy_um"] = meta.get("vox_xy_um") or round(meta["px_um"] * downsample, 4)
    return arr[0], arr[1], meta


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def analyze_nuclei(dapi, vox):
    """Marker-controlled watershed: detect nucleus centres (intensity local-maxima),
    then segment. Robust to touching nuclei in dense tissue."""
    vz, vy, vx = vox
    bw = binarize(dapi, method="otsu", min_size=max(8, int(round(30 / (vy * vx)))))
    nuc_r_px = max(1.5, 4.0 / vx)                                    # ~4 µm nucleus radius
    sm = ndi.gaussian_filter(dapi.astype(np.float32), sigma=(0.6, nuc_r_px * 0.6, nuc_r_px * 0.6))
    thr = float(np.percentile(dapi[bw], 30)) if bw.any() else 0.0
    peaks = peak_local_max(sm, min_distance=max(2, int(round(7 / vx))),
                           threshold_abs=thr, exclude_border=False)
    n = len(peaks)
    markers = np.zeros(dapi.shape, np.int32)
    if n:
        markers[tuple(peaks.T)] = np.arange(1, n + 1)
    lab = watershed(-sm, markers, mask=bw)
    imaged_mm3 = dapi.size * vx * vy * vz / 1e9
    out = {"nuclei_count": int(n), "density_per_mm3": round(n / imaged_mm3, 1),
           "imaged_volume_mm3": round(imaged_mm3, 4), "dapi_area_fraction_pct": round(100 * bw.mean(), 2)}
    if n:
        props = pd.DataFrame(regionprops_table(lab, properties=["area", "equivalent_diameter"]))
        vol_um3 = props["area"] * vx * vy * vz
        diam_um = props["equivalent_diameter"] * (vx * vy * vz) ** (1 / 3)
        out.update({"nucleus_vol_um3_median": round(float(vol_um3.median()), 1),
                    "nucleus_diam_um_median": round(float(diam_um.median()), 1),
                    "nucleus_size_cv_pct": round(100 * float(vol_um3.std() / max(vol_um3.mean(), 1e-9)), 1)})
        gini = centroid_homogeneity(peaks.astype(float), dapi.shape).get("centroid_gini")
        out["spatial_gini"] = round(float(gini), 3) if gini is not None else None
    return out, lab, bw, peaks


def analyze_zprofile(dapi, gfp, vox):
    vz = vox[0]
    dth = np.percentile(dapi, 95); gth = np.percentile(gfp, 95)
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
    peak_i = int(y.argmax()); half = y[peak_i] / 2
    above = np.where(y >= half)[0]
    span_idx = np.where(y > 1)[0]
    band = {"wall_peak_depth_um": round(float(z[peak_i]), 1),
            "wall_fwhm_um": round(float((above[-1] - above[0]) * vz), 1) if len(above) else 0.0,
            "cell_span_um": round(float((span_idx[-1] - span_idx[0]) * vz), 1) if len(span_idx) else 0.0}
    return df, band


def analyze_wall_lumen(dapi, vox, band_slices):
    """Cellular MIP over the wall band -> coverage + largest internal void (lumen)."""
    vz, vy, vx = vox
    mip = dapi[band_slices].max(0)
    mask = mip > np.percentile(mip, 80)
    mask = remove_small_objects(mask, 64)
    mask = ndi.binary_closing(mask, iterations=3)
    coverage = 100 * mask.mean()
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask                                   # enclosed voids
    holes = remove_small_objects(holes, 64)
    lab, n = ndi.label(holes)
    lumen_area = 0.0
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        lumen_area = float(sizes.max()) * vx * vy            # um^2
    # radial DAPI profile from cellular centroid (dip at centre => lumen)
    cy, cx = ndi.center_of_mass(filled)
    yy, xx = np.indices(mip.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    radial = ndi.mean(mip.astype(float), r, index=np.arange(0, r.max(), max(1, int(10 / vx))))
    return {"wall_coverage_pct": round(coverage, 1),
            "lumen_area_um2": round(lumen_area, 0),
            "lumen_present": bool(lumen_area > 0.02 * mip.size * vx * vy)}, mip, mask, holes, radial


def analyze_gfp(gfp, dapi_bw, vox):
    vz, vy, vx = vox
    mip = gfp.max(0)
    mask = mip > np.percentile(mip, 85)
    mask = remove_small_holes(remove_small_objects(mask, 64), 64)
    skel = skeletonize(mask)
    # junctions: skeleton pixels with >=3 neighbours
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3)), mode="constant") - 1
    junctions = int(((skel) & (nb >= 3)).sum())
    area_mm2 = mip.size * vx * vy / 1e6
    dapi_cov = dapi_bw.max(0)
    overlap = 100 * (mask & dapi_cov).sum() / max(mask.sum(), 1)
    return {"gfp_coverage_pct": round(100 * mask.mean(), 1),
            "gfp_skeleton_junctions_per_mm2": round(junctions / max(area_mm2, 1e-9), 0),
            "gfp_on_nuclei_overlap_pct": round(overlap, 1)}, mip, mask, skel


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _norm(m, lo=1, hi=99.7):
    a, b = np.percentile(m, [lo, hi]); return np.clip((m - a) / (b - a + 1e-9), 0, 1)


def make_figures(out, name, dapi, gfp, lab, peaks, zdf, band, wall, gfp_res):
    (_, mip_d, cmask, holes, radial) = wall
    (_, gmip, gmask, skel) = gfp_res
    vx = out["_vox"][2]
    # A. overview
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    ax[0].imshow(_norm(dapi.max(0)), cmap="gray"); ax[0].set_title("DAPI (nuclei) MIP")
    ax[1].imshow(_norm(gfp.max(0)), cmap="gray"); ax[1].set_title("GFP MIP")
    mg = np.dstack([_norm(dapi.max(0)), _norm(gfp.max(0)), _norm(dapi.max(0)) * 0.3]); ax[2].imshow(np.clip(mg, 0, 1)); ax[2].set_title("merge (R=DAPI G=GFP)")
    for a in ax[:3]: a.set_xticks([]); a.set_yticks([])
    ax[3].plot(zdf.depth_um, zdf.dapi_area_pct, color="#0072B2", label="DAPI %area (cells)")
    ax[3].plot(zdf.depth_um, zdf.gfp_area_pct, color="#009E73", label="GFP %area")
    ax[3].axvline(band["wall_peak_depth_um"], ls="--", color="k", lw=0.8)
    ax[3].set_xlabel("depth (µm)"); ax[3].set_ylabel("% area"); ax[3].legend(fontsize=7); ax[3].set_title("cells vs depth")
    fig.suptitle(f"{name} — {out['nuclei_count']} nuclei, {out['density_per_mm3']:.0f}/mm³, wall peak {band['wall_peak_depth_um']:.0f}µm", fontsize=10)
    fig.tight_layout(); fig.savefig(out["_dir"] / f"{name}_A_overview.png", dpi=140); plt.close(fig)
    # B. nuclei size + spatial
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
    props = pd.DataFrame(regionprops_table(lab, properties=["area"]))
    ax[0].hist((props["area"] * vx * vx * out["_vox"][0]).clip(upper=np.percentile(props["area"] * vx * vx * out["_vox"][0], 99)), bins=40, color="#0072B2")
    ax[0].set_xlabel("nucleus volume (µm³)"); ax[0].set_ylabel("count"); ax[0].set_title("nucleus size dist")
    ax[1].scatter(peaks[:, 2], peaks[:, 1], s=3, c="#0072B2", alpha=0.5)
    ax[1].set_aspect("equal"); ax[1].invert_yaxis(); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title(f"nucleus centroids (Gini {out.get('spatial_gini')})")
    fig.tight_layout(); fig.savefig(out["_dir"] / f"{name}_B_nuclei.png", dpi=140); plt.close(fig)
    # C. wall / lumen
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].imshow(_norm(mip_d), cmap="gray"); ax[0].set_title("cellular MIP (wall band)")
    ov = np.dstack([_norm(mip_d)] * 3); ov[cmask] = [0.1, 0.5, 0.9]; ov[holes] = [1, 0.3, 0.1]
    ax[1].imshow(ov); ax[1].set_title(f"cell mask (blue) / voids (red)\ncoverage {out['wall_coverage_pct']}%  lumen {out['lumen_area_um2']:.0f}µm²")
    ax[2].plot(np.arange(len(radial)) * 10, radial, color="#0072B2"); ax[2].set_xlabel("radius from centroid (µm)"); ax[2].set_ylabel("DAPI intensity"); ax[2].set_title("radial profile (dip=lumen)")
    for a in ax[:2]: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(out["_dir"] / f"{name}_C_wall_lumen.png", dpi=140); plt.close(fig)
    # D. GFP network
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    ax[0].imshow(_norm(gmip), cmap="gray"); ax[0].set_title("GFP MIP")
    ov = np.dstack([_norm(gmip)] * 3); ov[skel] = [0.1, 1, 0.3]
    ax[1].imshow(ov); ax[1].set_title(f"GFP skeleton\n{out['gfp_coverage_pct']}% cov  {out.get('gfp_skeleton_junctions_per_mm2')}/mm² junctions")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(out["_dir"] / f"{name}_D_gfp_network.png", dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyze_folder(in_dir, out_dir, downsample: int = 3, name: str | None = None):
    in_dir = Path(in_dir); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = name or in_dir.name.strip().replace(" ", "_")
    print(f"[{name}] loading ({downsample}x)...")
    dapi, gfp, meta = load_stack(in_dir, downsample)
    vox = (meta["vox_z_um"], meta["vox_xy_um"], meta["vox_xy_um"])
    print(f"[{name}] {dapi.shape} vox(zyx)={vox} px={meta['px_um']}µm ch={meta['channels']}")
    nuc, lab, bw, peaks = analyze_nuclei(dapi, vox)
    zdf, band = analyze_zprofile(dapi, gfp, vox)
    band_slices = np.where(zdf.dapi_area_pct.values > max(1, zdf.dapi_area_pct.max() * 0.25))[0]
    if len(band_slices) == 0: band_slices = np.arange(dapi.shape[0])
    wall, mip_d, cmask, holes, radial = analyze_wall_lumen(dapi, vox, band_slices)
    gfpm, gmip, gmask, skel = analyze_gfp(gfp, bw, vox)
    summary = {"sample": name, **nuc, **band, **wall, **gfpm,
               "gfp_mean": round(float(gfp.mean()), 1), "dapi_mean": round(float(dapi.mean()), 1),
               "px_um": meta["px_um"], "z_step_um": meta["z_step_um"], "objective": meta["objective"],
               "n_slices": meta["n_slices_loaded"]}
    summary["_dir"] = out_dir; summary["_vox"] = vox
    zdf.to_csv(out_dir / f"{name}_zprofile.csv", index=False)
    pd.DataFrame([{k: v for k, v in summary.items() if not k.startswith("_")}]).to_csv(out_dir / f"{name}_summary.csv", index=False)
    json.dump(meta, open(out_dir / f"{name}_meta.json", "w"), indent=2, default=str)
    make_figures(summary, name, dapi, gfp, lab, peaks, zdf, band, (None, mip_d, cmask, holes, radial), (None, gmip, gmask, skel))
    print(f"[{name}] done -> {out_dir}")
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def batch(samples: dict, out_dir, downsample: int = 3):
    out_dir = Path(out_dir); rows = []
    for name, folder in samples.items():
        try:
            rows.append(analyze_folder(folder, Path(out_dir) / name, downsample, name=name))
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(Path(out_dir) / "ALL_samples_summary.csv", index=False)
        print(f"\ncross-sample summary -> {Path(out_dir)/'ALL_samples_summary.csv'}")
        print(df.to_string(index=False))
    return rows


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 3:
        print("usage: keyence_tube_analysis.py <input_folder> <output_folder> [downsample]"); sys.exit(1)
    analyze_folder(a[1], a[2], int(a[3]) if len(a) > 3 else 3)
