# Changelog

Notable changes to fluorostats. Format follows [Keep a Changelog](https://keepachangelog.com);
versions follow [semantic versioning](https://semver.org).

## [0.8.0]

### Added
- **Keyence BZ-X support** (`fluorostats.keyence`): `load_keyence_stack` assembles a
  z-stack from a folder of per-slice `*_CHF` OME-TIFFs (all channels per file), and
  `parse_gci` reads the `.gci` acquisition metadata (pixel size, z-step, channel
  names, exposures). `io.load_volume` now auto-detects a Keyence folder. Worked
  example: `examples/keyence_tube_analysis.py`.
- Manifest-driven batch drivers: `fluorostats depth` (depth-penetration / permeability)
  and `fluorostats viability` (Live/Dead) — many stacks grouped by condition, matched
  blanks, multi-window AUC, tidy CSVs + publication figures.

### Fixed
- `plots`: strip-plot jitter was reseeded per group, so every group drew identical
  jitter — now one RNG per figure (independent points, still reproducible).
- `render3d`: voxel-cloud box aspect now reflects anisotropic voxel size and per-axis
  downsample rather than raw voxel counts; `__all__` now lists every public MIP helper.

### Changed
- Release-hardening pass: removed dead imports/variables, corrected a few misleading
  docstrings, and added depth-batch test coverage. No public API changed.
