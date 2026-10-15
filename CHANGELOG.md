# Changelog

Notable changes to fluorostats. Format follows [Keep a Changelog](https://keepachangelog.com);
versions follow [semantic versioning](https://semver.org).

## [Unreleased]

Three rounds of audits against the engineering rules (`docs/ENGINEERING.md`).
Refactors were proven behaviour-identical with a golden-output harness over every
public function, every CLI command and every figure script. The changes below that
DO alter outputs come first, each with the inputs it affects.

### Fixed (changes outputs)

Reading files
- **Leica `.lif` files load** (every one failed: the reader used a name readlif does
  not export); a negative z length (top-down acquisition) is read as its magnitude.
- **Olympus axes come from the plane file names**, checked against the mainfile
  sizes: 2-channel single-plane files were read as 1 channel x 2 slices, time series
  as z-stacks, C x Z x T files lost a channel, and a file name pattern dropped all
  but one plane silently (a size mismatch now raises). Pixel sizes written with a
  comma decimal are read.
- ImageJ: a z spacing of 0 or NaN is not calibration; the unit `inch` is converted.
  RGB channel names stay on their data; unlabelled 5D/6D arrays follow the 4D rule;
  `load_image` of a z-stack no longer parses or warns about z.
- **Axis order comes from the file's axis labels** (TIFF series axes, nd2 sizes,
  czifile axes), not from the array shape. Changes results for TIFF/ND2/CZI files
  not already in (C, Z, Y, X) order: ImageJ ZCYX hyperstacks with more than one
  channel, OME files with Z before C or a time axis, RGB TIFFs (colour samples are
  channels), single-plane multichannel files, and RGB z-stacks with one unlabelled
  axis (taken as z, with a warning). Time points, positions and scenes keep index 0
  with a warning. Files without labels keep the shape rule, which now warns when it
  guesses and refuses layouts with more than 10 channels or X < 5.
- **Channel names stay positional.** An unnamed channel no longer shifts later names
  onto the wrong channel (OME/CZI/ND2/Olympus: ["", "GFP"] was read as ["GFP",
  "Ch2"], so the green-channel pick measured the empty channel). CZI names come from
  the image's channels only (they collected every `Channel` element of the file).
- **Voxel sizes are converted from their stated unit** (OME `PhysicalSize*Unit`,
  ImageJ `unit`/`zunit`; 500 nm pixels were read as 500 µm); ImageJ x/y pixel size
  is read from the resolution tags when the file names a unit; Leica `.lif` sizes
  were 10^6 too large; ND2 axes nd2 cannot calibrate are recorded as defaulted. An
  unknown or missing unit keeps the value, warns and records `file-unit-unknown`.
- **CZI/ND2/LIF/Olympus metadata parsers keep what they read** and warn naming what
  they could not (a CZI with a non-numeric z `Distance` lost its x/y sizes silently).
- **Unnamed 4-channel inputs are measured on channel 1 (green), not channel 0**, for
  RGBA images and unnamed 4-channel volumes in `quant3d` (announced as a guess).

Segmentation and objects
- **`clear_border_labels` re-merged touching objects**; `watershed_split` merged
  objects touching only at a corner. Both keep objects apart now.
- `object_centroids` gives one row per present label (gaps in label ids made phantom
  objects); skeletonize output is treated as a mask on every scikit-image version;
  spur pruning keeps two branches at every junction.
- **`objects.watershed_split(min_size>0)` re-merged the objects it split** (the size
  filter relabelled the foreground). It now filters the watershed labels, and seeds
  at the h-maxima of the distance transform (h = `min_distance`/2) so irregular or
  elongated objects are not over-split. Watershed counts change everywhere,
  including `live_dead_by_count(method="watershed"/"all")` and the viability
  driver's watershed, consensus and spread columns.
- **One small-object rule** (`segment.remove_small`: objects smaller than `min_size`
  are removed, objects of exactly `min_size` kept) on every scikit-image version,
  used by `binarize`, `label_3d` and the benchmark producers.
- **`skeleton.n_junction_nodes` merges junction voxels with full connectivity** as
  documented (about 1% fewer nodes where branches touch diagonally).
- **`texture.orientation_anisotropy` angles use the regionprops convention** of
  `object_shape_metrics` (they were mirrored), and never read 180.
- **`objects.nearest_neighbor_stats`** uses the 2D or 3D Clark-Evans formula for the
  number of axes the points span (coplanar 3D points mixed an area with the 3D
  constant); NaN with a warning below two varying axes.

Measurements and statistics
- **Cliff's delta ignores NaN** (NaN counted as a tie); **one minimum-n rule** for
  every two-group test (a group with fewer than 2 values is untested, p NaN, not
  p = 1.0); `compute_pvalues` no longer crashes on missing or mixed-type labels.
- **The λ fit no longer depends on the intensity scale** (tiny intensities gave a
  wrong λ with `fit_ok` true) and refuses a λ below the z-step.
- Depth windows a stack covers only partly are flagged (`n_partial`), and retention
  divides by the covered depth; power estimates refuse empty inputs.
- **A value that was never measured is NaN, not 0**: a skeleton that cannot be
  measured; `largest_component_fraction` and cluster areas of an empty mask; the
  mean branch length of no branches; the regime estimate with no segmented cell;
  the count spread when one method ran; a depth AUC window entirely past a stack;
  R² of a constant depth profile; the SD/SEM of a single value (depth curves and
  λ, viability summaries, plot error bars, Bland-Altman limits); an empty
  condition's bar; density for a zero imaged volume.
- **Skeleton spur pruning compares branch length in µm** with per-axis voxel sizes.
- **An all-tied Mann-Whitney comparison has p = NaN on every scipy** (scipy < 1.18
  said 1.0).
- **Tile CVs use the population form (ddof=0) everywhere**: `coverage_tile_cv` and
  `spatial_heterogeneity` cv are about 2% lower on the default 5×5 grid.
- **One tiling rule for images and one for points** (`lateral_homogeneity` used to
  crop edge pixels; point binning used two edge rules).
- **`power`**: one resampling helper and one seeding rule; `power_curve` now equals
  `bootstrap_power` at each n, `fdr_power_curve` values change, and an empty group
  has NaN power (was 0).
- **`morphometry.depth_span`/`depth_centroid`** use the z voxel size whatever
  `z_axis` is; `depth_centroid`'s `z_*` keys are slices and the µm values are the new
  `z_*_um` keys.
- **`fluorostats viability`**: the field of view uses the y pixel size for y; counts
  use the manifest's `seg.method` (they always used Otsu); omitted live/dead channels
  are resolved from dye names (with a warning when it falls back to 0/1).
- **CLI summary-panel stars used raw p-values** while `pvalues.csv` starred BH
  q-values; both use q now.

CLI and drivers
- **A run refuses an output folder holding files it did not write** (`--overwrite` to
  replace them) and records which tool wrote the folder; files it wrote earlier are
  replaced and named.
- **One unreadable file no longer stops `quant2d`/`quant3d`**: failures go to
  `failures.csv`, the rest are measured, and the command exits 1. Errors from bad
  manifests or options print as messages, not tracebacks; a failed run leaves no
  empty folder; `quant3d` skips Olympus `.oif.files` companion folders;
  `--condition-from filename` strips `.ome.tif`; `--prune` with `--no-skeleton` is
  refused.
- **`quant2d`/`quant3d` refuse an `--output` inside `--input`**; reruns remove the
  tool's own earlier plots, overlays and (depth) figures, naming each, and every
  command writes `run_config.json` last, so a failed run leaves none. An input
  folder with nothing to measure exits 2.
- **QC overlays of same-named files in different condition folders no longer
  overwrite each other**; **fractional depth AUC windows** no longer overwrite each
  other's figures.
- **`quant3d` measures a Keyence BZ-X folder as one volume** (each slice was measured
  as its own sample).
- **Depth**: the group grid stops within the shallowest stack (it extrapolated); the
  λ overlay uses the manifest's `n_surface` and fitted floor; `window_covered_um` is
  0, not negative; the Welch t-test is printed only with `"parametric": true`.
- **Manifests are validated before any image loads**: unknown fields at any level,
  non-bool flags, `true`/`false` as channels, malformed windows, unknown colours,
  duplicate group names or stack labels, empty groups, `"role": "aux"` or a group
  `background` in a viability manifest, a bad `seg` block. Late failures (channel out
  of range, too few slices) name the file. Unknown option values
  (`attenuation_correct(reference=)`, `morans_i(connectivity=)`,
  `render_isosurface(style=)`, report `mode`, bootstrap `agg`) raise.
- **Figures**: bars use one style (some `*_bar.png` change); more than eight
  conditions get distinct colours; `render3d.mip_overlay` uses the QC overlays'
  min-max stretch.
- Importing `fluorostats` no longer imports pyplot or switches the matplotlib
  backend.

### Added
- `quant3d --opening` and the viability `seg.opening`; `viability.live_dead_channels`
  (live/dead from dye names); `live_cov_frac`/`dead_cov_frac` (fractions; the `_pct`
  columns are kept, deprecated); `run_config.json` records the tool, the code commit
  and the absolute paths read; `io.read_voxel_size`; an optional regression suite on
  real vendor files (`FLUOROSTATS_VENDOR_DATA`); a CI job at the minimum dependency
  versions.
- `voxel_size_source` (which axes came from the file, which were defaulted, copied
  or assumed) in every per-file/per-stack table; quant2d also records
  `voxel_size_um`, `auto_crop_box` and µm² cluster areas; `channel_used`.
- `contrasts.csv` from `fluorostats depth` and `fluorostats viability` (the printed
  two-group Mann-Whitney contrasts with n, U, p, BH q, Cliff's δ, test and
  correction). `pvalues.csv` names its test and correction, reports Cliff's δ and
  keeps untestable pairs with a `reason`.
- Warnings instead of silent fallbacks (defaulted calibration, positional channel
  guesses, dropped axes, layout guesses, mis-saved z-stacks, Keyence gaps and its
  assumed z unit, failed threshold algorithms, partial windows, trimmed blanks,
  unbalanced Scheirer-Ray-Hare designs, zero fold-change denominators).
- Options: `quant3d --opening ball|disk|none` (the 3D noise opening erases
  structures thinner than three slices; `disk` opens each slice), depth manifest
  `parametric` and `fit_r2_threshold`; CLI defaults shown and result-changing
  options marked.
- API: `segment.segment_channel`, `segment.remove_small`, `segment.OPENINGS`,
  `validate.dice`, `stats.describe(axis=)`, `stats.compute_pvalues`,
  `stats.significance_stars`, `stats.REDUCERS`, `stats.gini`,
  `stats.coefficient_of_variation`, `stats.welch_t`, `stats.bootstrap_mean_ci(rng=)`,
  `viability.live_ratio`, `viability.live_dead_channels`, `qc.overlay_rgb`,
  `io.read_voxel_size`, `io.FORMATS`, `io.DIRECTORY_FORMATS`, `pubfig` (one figure
  style module with themes, `save_figure`, `TEXT_WIDTH_IN`), `depth_figures`,
  `depth_contrasts`, `projections` (2D MIP panels, moved from `render3d`, which re-exports them), `cli_paths` (CLI input discovery and output hygiene), `manifest`, `deprecation`, `skeleton.UNMEASURED_METRICS`.
- Type checking (mypy) and a CI workflow (ruff lint and format, mypy, the test
  suite at 100% coverage on Python 3.10-3.13); `ruff` and `mypy` pinned in `dev`.

### Changed
- **Dependency floors raised to versions that work**: numpy >= 1.25, scikit-image
  >= 0.23, matplotlib >= 3.8, skan >= 0.12 (at the old floors 3D skeleton metrics
  were NaN and other functions failed).
- `fluorostats.style`, the `render3d` MIP panels, the `depth_batch` window/figure
  helpers, `pubfig.panel(dx=, dy=)` and the manifest's `auc_window_um` are deprecated
  and warn (use `pubfig`, `projections`, `depth`/`depth_figures`). `quant2d`/`quant3d`
  plots follow the one figure style (their colours and fonts change).
- The maxima count is described as what it is (local maxima above a fraction of the
  image maximum); no claim of agreement with the Fiji Live/Dead macro.
- Version 0.9.0.dev0 on the development branch.
- **One name for the voxel-size argument, `voxel_size_um`** (per-axis µm in array
  order; a scalar means isotropic): `ring_morphometry`/`mesh_size` (`spacing`),
  `coverage_metrics` (`pixel_size_um`), `intensity_depth_profile`
  (`voxel_size_z_um`), `prune_skeleton` (`min_branch_length_px` ->
  `min_branch_length`). The old keywords keep working with a DeprecationWarning;
  `watershed_split`'s `footprint_size` (keyword or positional) is accepted and
  ignored with a DeprecationWarning.
- **`live_dead_by_count` returns one flat set of keys in every mode** (n_live, n_dead,
  viability, consensus, spread, method/reason per channel, and per-method counts);
  `by_method` (method "all") is kept but deprecated.
- `fluorostats.style` is a thin layer over `pubfig`; `fluorostats.io` is split into
  `io`, `io_vendors` and `io_canonical`; `plots.compute_pvalues` is
  `stats.compute_pvalues`. Every existing public name keeps working.
- `keyence.load_keyence_stack` returns `voxel_size_um`; its `px_um`/`z_step_um`/
  `vox_*` keys are deprecated.
- Metric labels: `n_junctions` is "Junction-to-Junction Branches".

### Benchmarks and examples
- The external-viability peak-count comparator is named `skimage_peakcount` (it is a
  direct scikit-image peak count with the library's parameters, not the Fiji macro);
  figure and docs wording follow.
- Every locally reproducible results table is regenerated with this library
  (watershed columns change materially; others in the 3rd-4th decimal). Not
  regenerated, and marked so in `benchmarks/README.md`: cluster-only tables
  (deep-learning evaluations, `b_ve_metrics`), the wall-clock timing tables, and
  the collaborator `fig6_*` tables (raw data not deposited). Every
  producer writes `<table>.run.json` (parameters, library versions, git commit).
- Producers use the library's small-object rule; a failed threshold scores NaN; the
  two fluorostats Otsu+CC pipelines are named in the tables; new count-agreement
  and derived tables; column renames: `median_diam_vox` (BBBC024 is uncalibrated),
  `fluorostats_maxima`, `bias_vs_3D`; `b4_reaver_ranking` gains `mean_bias`.
- Figure scripts only read and plot (recorded exceptions: live determinism runs and
  display overlays); caption numbers and claims come from the tables; Figure 2(d)
  shows the default and watershed pipelines as separate lines; the Figure 4 scale
  bar is drawn from the file's calibration (it was 438 µm labelled 1 mm); scripts
  run from any directory; S1-S4 names; stale previews removed.
- Examples: `keyence_tube_analysis` uses the library's ring, watershed, diameter,
  density and junction-node measurements (its outputs change);
  `multi_condition_pipeline` creates its output folder; the depth-penetration
  example reads its caption from its run.

## [0.8.2]

### Fixed
- **Install instructions pointed at a package that isn't published.** fluorostats
  is not on PyPI, so every `pip install fluorostats[...]` in the docs and in
  runtime errors handed users a command that fails. The `src/` ones were the worse
  half: they fire the moment a `.czi` or `.oib` fails to open, so a user hit a dead
  end exactly when they needed a working instruction. The README now installs from
  source, and `io.py` / `cli.py` name the actual distribution (`oiffile`,
  `czifile`, `nd2`, `readlif`), which pip can install today.
- README citation block now leads with the concept DOI, which resolves to the
  newest release, instead of pinning a version DOI that goes stale every release.

Measurement code unchanged apart from the `__version__` string and the text of
those install messages, so no manuscript number is affected.

## [0.8.1]

### Fixed
- **Benchmark data manifest reconciled with the manuscript.** The manifest ships
  inside the Zenodo deposit, so its drift from the paper was reviewer-visible.
  Removed three datasets it over-listed (VascuSynth, S-BIAD2215, S-BIAD2920) that
  were downloaded during exploration but back no script, results table or claim;
  added two it omitted (DSB2018, SproutAngio) that are cited in the paper and
  backed by bundled results tables; renumbered the supplementary figures from the
  old ED 1-4 to S1-S4; pointed VesselExpress at the Zenodo record the paper cites;
  and aligned the Fig 6 data-availability wording with the manuscript.
- `CITATION.cff` now carries the concept DOI, which resolves to the latest release,
  rather than a version DOI that goes stale inside the archive.

No change to the measurement code: the only edit under `src/fluorostats/` is the
`__version__` string, so every number in the manuscript is unaffected.

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
