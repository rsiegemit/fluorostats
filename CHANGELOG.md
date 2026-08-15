# Changelog

Notable changes to fluorostats. Format follows [Keep a Changelog](https://keepachangelog.com);
versions follow [semantic versioning](https://semver.org).

## [0.8.0]

### Added
- Ships a `py.typed` marker (PEP 561) so the library's type hints are usable by
  downstream type checkers.
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
- Release-hardening: removed dead code, corrected all inaccurate docstrings/comments,
  de-duplicated the per-condition stats helper in `plots` and the shared quant2d/quant3d
  reporting tail in `cli`, and expanded the test suite to 354 tests at **100% line
  coverage** (vendor loaders exercised via mocks; a handful of unreachable version/lib
  fallbacks marked `# pragma: no cover`). No public API changed.
