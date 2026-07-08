# Volume-Fraction & Connectivity/Euler Benchmark Datasets

Sources for benchmarking **fluorostats** volume-fraction (Delesse–Glagolev voxel
counting, **B5**) and Euler-number / connectivity (vs BoneJ, **B1**) quantification.
Constraint: public, citable data only (DOI/citation). Verified by fetching where noted;
unverified items flagged explicitly.

**Bottom line (read first):** For *correctness* validation with exact, zero-error
ground truth, use **synthetic phantoms with analytically known topology and volume
fraction** (Section A). Real published datasets (Section B) are valuable for *realism*
and cross-tool agreement (vs BoneJ), but their "ground truth" is itself software-derived
(BoneJ/Scanco/CTAn), so they test *agreement*, not *correctness*. Recommended primary
approach: synthetic phantoms generated in-repo (no download), cross-checked against
`foam_ct_phantom` and one real micro-CT sample.

---

## A. SYNTHETIC PHANTOMS — analytically known topology & volume fraction (STRONGLY RECOMMENDED)

These give **exact** ground truth with **zero measurement error**, generated in-repo,
no external download, fully citable via the underlying mathematics. Best for proving
correctness of both the Euler-number pipeline (B1) and voxel-counting VF (B5).

### A1. Euler characteristic χ — analytic topology phantoms (feeds B1)

The Euler characteristic of a 3D solid is χ = β₀ − β₁ + β₂ (connected components −
handles/tunnels + enclosed cavities). Exact reference values:

| Phantom | Construction | χ (exact) | Notes |
|---|---|---|---|
| Solid ball | filled sphere | **1** | β₀=1, β₁=0, β₂=0 |
| N disjoint balls | N separated filled spheres | **N** | tests β₀ counting |
| Ball with 1 tunnel (solid torus) | torus / drilled ball | **0** | 1−1+0 |
| Ball with *k* tunnels | genus-*k* solid | **1 − k** | tests β₁ |
| Ball with 1 internal cavity (hollow shell) | sphere shell | **2** | 1−0+1 |
| Ball with *m* cavities | shell with *m* voids | **1 + m** | tests β₂ |
| Solid cube / any contractible solid | box | **1** | sanity check |

Connectivity (BoneJ convention) = 1 − χ for a single connected structure; Conn.D =
connectivity / total volume. These identities are exactly what Doube et al. used to
validate BoneJ ("creating simple connected structures and measuring their Euler
characteristics" — verified from the paper, see B3).

- **Citation for the convention/algorithm:** Odgaard A & Gundersen HJG (1993),
  "Quantification of connectivity in cancellous bone, with special emphasis on 3-D
  reconstructions," *Bone* 14:173–182. And Toriwaki J & Yonekura T (2002), "Euler number
  and connectivity indexes of a three dimensional digital picture," *Forma* 17:183–209.
  (Both verified as BoneJ's cited references.)
- **License:** N/A — you generate the voxel volumes yourself (e.g. NumPy boolean arrays
  from implicit surface tests). Ground truth is the mathematical identity above.
- **Caveat:** discretization/voxelization of curved surfaces introduces the *only* error;
  choose radii ≫ voxel size and confirm χ is exact for the discretized set (a drilled
  cube with rectangular tunnels is exactly χ=0 with no discretization error and is the
  cleanest hard-zero test).

### A2. Volume fraction — analytic VF phantoms (feeds B5)

- **Checkerboard / striped volumes:** set an exact known fraction *p* of voxels to
  foreground (e.g. p=0.5 checkerboard, or first ⌊pN⌋ voxels). VF ground truth = *p*
  **exactly** (no discretization error, since it's voxel-native). This is precisely
  BoneJ's own VF validation: "Volume Fraction was tested on binary images with known
  proportions of voxels set to foreground and background" (verified from the paper).
- **Sphere-in-box:** ball radius *r* in box side *L*; analytic VF = (4/3)πr³ / L³, with
  small known voxelization error → also validates the Delesse principle (areal fraction
  of random slices → volume fraction) since a sphere's cross-sectional area fraction is
  analytically integrable.
- **Sphere packing (foam):** VF = 1 − (packing fraction); see A3 for a tool.

### A3. `foam_ct_phantom` — synthetic foam with computable porosity (VERIFIED)

- **Citation:** Pelt DM, Hendriksen AA, Batenburg KJ (2022), "Foam-like phantoms for
  comparing tomography algorithms," *J. Synchrotron Radiation* 29(1):254–265.
  Software DOI: **10.5281/zenodo.3726909**.
- **Download / install:** GitHub `https://github.com/dmpelt/foam_ct_phantom`;
  `conda install -c conda-forge foam_ct_phantom`. **License: MIT** (verified).
- **What / ground truth:** generates 3D volumes by removing a specified number of
  **non-overlapping spheres** from a solid cylinder. Because the geometry is
  deterministic (known cylinder, known sphere set), **volume fraction/porosity is
  computable analytically** and the phantom is topologically well-defined (each disjoint
  cavity contributes to χ). Format: HDF5. Feeds **B5** (VF) and secondarily **B1**
  (cavity/χ counting).
- **Caveat:** designed for tomography-reconstruction benchmarking; you use the
  ground-truth voxel volume directly (skip the projection/recon step).

---

## B. REAL PUBLISHED DATASETS — realism & cross-tool agreement

### B1-real. Digital Rocks Portal DRP-372 — complex porous media + Minkowski functionals (VERIFIED)

- **Citation:** Santos JE, Yin Y, Jo H, et al. (2022), "A Dataset of 3D Structural and
  Simulated Transport Properties of Complex Porous Media," *Scientific Data* 9:579.
  DOI: **10.1038/s41597-022-01664-0**.
- **Dataset DOI (Digital Rocks Portal DRP-372):** **10.17612/93pd-y471**.
- **Contents:** 217 samples (256³ and 480³ voxels), binary volumes (0=void, 1=solid),
  with published **porosity, surface area, mean curvature, and Euler characteristic**
  (full Minkowski functional set) plus distance maps and simulation results.
- **Format:** HDF5 (`.mat`). **Size:** >1 TB uncompressed (grab a subset).
  **License: CC-BY 4.0** (verified). Code: `https://github.com/je-santos/Large-simulation-dataset`.
- **Feeds:** **B5** (porosity/VF against published values) and **B1** (Euler
  characteristic against published χ). This is the single best *real* dataset because χ
  and porosity are **published as ground truth**, not left for you to recompute.
- **Caveat (IMPORTANT):** the Digital Rocks Portal migrated to Digital Porous Media
  (digitalporousmedia.org) in 2025; the old `digitalrocksportal.org/projects/372` URL now
  301-redirects to a tombstone page (returned 403 on fetch). **Resolve via the DOI**
  (10.17612/93pd-y471) rather than the legacy URL. Access route on the new portal was
  not fully verifiable at fetch time — flag as *access-path-unverified, DOI-verified*.
  Their published Minkowski/χ values are computed with their own code, so this tests
  algorithmic *agreement*, not absolute correctness.

### B2. BoneJ (Doube 2010) — the reference tool you benchmark against (VERIFIED)

- **Citation:** Doube M, Kłosowski MM, Arganda-Carreras I, et al. (2010), "BoneJ: free and
  extensible bone image analysis in ImageJ," *Bone* 47(6):1076–1079.
  DOI: **10.1016/j.bone.2010.08.023**.
- **Role:** defines the connectivity = 1 − χ and BV/TV conventions fluorostats must match
  (B1, B5). Validated exactly by the analytic phantoms in Section A — so Section A
  *is* the BoneJ-equivalence test.
- **Validation datasets in the paper:** (i) known-topology structures for Euler; (ii)
  binary images with known foreground fraction for VF; (iii) a 1 cm³ elephant femoral-head
  trabecular cube independently scanned/analysed by Scanco and SkyScan for cross-vendor
  agreement. **Caveat:** the elephant-bone volume is *not* provided as a public download in
  the paper. BoneJ software + test images: `https://bonej.org` (open source).

### B3. Porcine talar subchondral trabecular bone — real micro-CT with morphometry (PARTIALLY VERIFIED)

- **Dataset citation:** Koria L, et al. (2020), "Estimating tissue-level properties of
  porcine talar subchondral bone," *J. Mechanical Behavior of Biomedical Materials*
  (PubMed 32805501; DOI 10.1016/j.jmbbm.2020.104044). Ten cylindrical trabecular samples,
  16 µm isotropic micro-CT; reported as an open-access dataset and reused by the gyroid
  study arXiv:2211.13036.
- **Feeds:** B5 (BV/TV) and B1 (Conn.D) against reported morphometry.
- **Caveat:** the **direct download URL / repository was NOT verified** — the paper
  references an open dataset but I could not confirm the exact Zenodo/Figshare/institutional
  link. **Flag as unverified download**; verify the "Data availability" statement in the
  paper before relying on it. Ground truth is BoneJ/CTAn-derived (agreement, not truth).

### B4. Digital Porous Media / Digital Rocks Portal (general repository) (VERIFIED as repository)

- **Repository:** Digital Porous Media Portal, `https://digitalporousmedia.org` (formerly
  Digital Rocks Portal). re3data: r3d100012033. Repository DOI: 10.17612/FGMN-D889.
  Hosts many CC-licensed micro-CT porous-media volumes, several with reported porosity.
- **Feeds:** B5 (porosity/VF). **Caveat:** per-dataset licenses and whether χ/connectivity
  is *published* vary — must be checked per project. Best single entry point beyond DRP-372.

---

## Recommendation & suggested benchmark design

**Primary (correctness, B1 + B5):** in-repo synthetic phantoms (A1/A2) — solid ball
(χ=1), N disjoint balls (χ=N), drilled cube / genus-k solid (χ=1−k), hollow shell (χ=2),
and known-fraction voxel volumes (VF=p exactly). Zero download, exact ground truth,
reproduces BoneJ's own validation. Use a **rectangular drilled cube** for the cleanest
hard-zero χ test (no discretization error).

**Secondary (realism / cross-tool, B1 + B5):**
1. `foam_ct_phantom` (MIT, DOI 10.5281/zenodo.3726909) — analytic-porosity foam, one line
   to install.
2. DRP-372 subset (CC-BY 4.0, DOI **10.17612/93pd-y471**) — real porous media with
   **published porosity AND Euler characteristic**; the strongest real B1/B5 anchor.

**Single best real-data URL:** DRP-372 via DOI **https://doi.org/10.17612/93pd-y471**
(binary volumes + published Minkowski functionals incl. Euler characteristic, CC-BY 4.0).

**Single best overall approach:** synthetic analytic phantoms (Section A) — exact,
zero-error, no download, and they *are* the BoneJ equivalence proof.
