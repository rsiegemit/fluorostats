# Changelog

Notable changes to fluorostats. Format follows [Keep a Changelog](https://keepachangelog.com);
versions follow [semantic versioning](https://semver.org).

## [0.8.0]

### Added
- Ships a `py.typed` marker (PEP 561) so the library's type hints are usable by
  downstream type checkers.
- **Reproducibility artifacts** for the methods paper: a top-level `benchmarks/`
  harness (figure-regeneration scripts + precomputed metric tables), a
  `benchmarks/DATA_MANIFEST.md` listing every dataset with licence and source, a
  `benchmarks/VALIDATION.md` recording published-vs-observed baselines, and a
  pinned `requirements.lock`. The benchmark harness moved from
  `examples/benchmark_figures/` to `benchmarks/`.
- **Spatial sampling & heterogeneity** (`fluorostats.spatial`, general): `tile_reduce`
  (partition a 2D field into a grid and reduce each cell with any callable),
  `tile_point_density` (bin object centroids into a grid), `slab_reduce` (split a
  volume into axial slabs for through-depth profiling), `morans_i` (spatial
  autocorrelation), and `spatial_heterogeneity` (CV + Moran's I). Turns one large
  field of view into many sub-region measurements and quantifies how uniform vs
  patchy/clustered a signal is.
- **Object spatial-pattern & shape analysis** (`fluorostats.objects`):
  `object_shape_metrics` (elongation / orientation / solidity / axis lengths per
  object), `nearest_neighbor_stats` (nearest-neighbour spacing + Clark-Evans
  clustering index: clustered vs random vs regular), and `object_mask_association`
  (distance from objects to a reference structure + co-localised fraction, e.g.
  nuclei sitting on a network).
- **Geometry & orientation helpers** (general, assay-agnostic): `ring.ring_morphometry`
  — annular cross-section morphometry (lumen area/circularity, inner/outer diameter,
  wall thickness + wall-relative coverage, concentricity, robust to an off-centre or
  partly open ring); `texture.orientation_anisotropy`, `texture.orientation_order`
  (nematic order + alignment-to-reference), and `texture.mesh_size` — orientation
  coherence, orientation alignment, and network pore/mesh size for fibrous signal;
  `objects.angular_homogeneity` / `objects.radial_distribution` — circumferential and
  radial point-distribution statistics around a centre. The `keyence_tube_analysis`
  example gains an optional `geometry=` mode wiring these into cross-section vs wall
  analyses.
- **Keyence BZ-X support** (`fluorostats.keyence`): `load_keyence_stack` assembles a
  z-stack from a folder of per-slice `*_CHF` OME-TIFFs (all channels per file), and
  `parse_gci` reads the `.gci` acquisition metadata (pixel size, z-step, channel
  names, exposures). `io.load_volume` now auto-detects a Keyence folder. Worked
  example: `examples/keyence_tube_analysis.py`.
- Manifest-driven batch drivers: `fluorostats depth` (depth-penetration / permeability)
  and `fluorostats viability` (Live/Dead) — many stacks grouped by condition, matched
  blanks, multi-window AUC, tidy CSVs + publication figures.

### Fixed
- `stats.stouffer_combine`: weights are now aligned with the surviving p-values after
  NaN removal (they were previously mis-paired, silently corrupting weighted pooling).
- `preprocess.auto_crop`: no longer trims `margin` pixels off a borderless image; the
  inward margin is applied only on edges where a border was actually detected.
- `objects.object_volumes_voxels` / `equivalent_diameters_um`: skip gaps in
  non-contiguous label images (no more spurious zero-size objects).
- `plots`: bar/box/summary colours no longer collide past 8 conditions; `boxplot` no
  longer uses the deprecated Matplotlib `labels=` kwarg; strip-plot jitter uses one RNG
  per figure (was reseeded per group, giving identical jitter).
- `depth_batch` / `viability_batch`: guard a single-slice stack and an empty manifest
  instead of crashing.
- `render3d`: voxel-cloud box aspect now reflects anisotropic voxel size and per-axis
  downsample rather than raw voxel counts; `__all__` lists every public MIP helper.

### Changed
- Figure labels are now fully data-driven rather than probe-specific: `depth_batch`
  takes an optional `signal_label` manifest field (default `"Signal"`) that names the
  probe on all axis/title text, and the `quant2d` summary panel no longer hardcodes a
  tissue type — so the general tools produce correctly-labelled figures for any
  experiment without code changes.
- Release-hardening: removed dead code, corrected all inaccurate docstrings/comments,
  de-duplicated the per-condition stats helper in `plots` and the shared quant2d/quant3d
  reporting tail in `cli`, and expanded the test suite to 354 tests at **100% line
  coverage** (vendor loaders exercised via mocks; a handful of unreachable version/lib
  fallbacks marked `# pragma: no cover`). No public API changed.
