# Engineering principles

Every change to fluorostats must satisfy these rules. Each rule has an ID and a check, so a review or an audit can cite
it (`violates SIN-1`). Sections 1-4 (modularity, generality, reuse, singularity) are the priorities: when two rules pull
against each other, those four win, then correctness of results (section 6), then everything else.

An intentional exception is written at the site as a comment that names the rule and the reason
(`# exception MOD-6: one click command per subcommand; options are the interface`). An exception without a reason is a
violation.

The goal: a small set of durable, general measurement tools built from swappable parts, so that a new assay is a
manifest, a new metric is one function in the module that owns its concern, and nothing is written to be thrown away.

## 1. Modularity (MOD)

Each piece owns all of its functionality behind a small interface, so it can be swapped, removed or tested alone.

| ID | Rule | Check |
| --- | --- | --- |
| MOD-1 | One concern per module. Reading (`io`, `io_tiff`, `io_vendors`, `io_canonical`, `keyence`), manifests (`manifest`), preprocessing, segmentation, each metric family (`metrics_2d`, `metrics_3d`, `skeleton`, `objects`, `morphometry`, `ring`, `texture`, `spatial`, `depth`, `viability`), statistics (`stats`, `agreement`, `power`), validation, QC, reporting (`report`, `depth_contrasts`) and figures (`plots`, `render3d`, `projections`, `style`, `pubfig`, `depth_figures`) each keep their logic, options and tests in their own module. | You can name the one concern a module serves; changing that concern touches one module. |
| MOD-2 | Swappable by name. Variants (threshold methods, count methods, reducers, file formats) are chosen by a named option resolved in one place (a dict/registry or one dispatch function), not by `if` chains repeated in each caller. Adding a variant is one implementation plus one registration. | Grep for a variant's name finds its implementation, its registration and its tests, nothing else. |
| MOD-3 | Dependencies point one way: measurement modules (`io` … `validate`) ← orchestration (`depth_batch`, `viability_batch`, `report`, `cli`, `cli_paths`, `cli_measure`) ← scripts (`benchmarks/`, `examples/`). Measurement modules never import plotting, `click` or orchestration; nothing in `src/` imports `benchmarks/` or `examples/`. No cycles; no function-level import used to dodge one. | The import graph has no upward edges and no cycles. |
| MOD-4 | Use only a module's public surface. No `module._private` access across module boundaries; if a caller needs it, the owner makes it public. A module's own test files (`test_<module>.py`, `test_<module>_<part>.py`) may exercise that module's private helpers. | No `_name` imported or accessed from outside its module, except by its own test files. |
| MOD-5 | Components meet at explicit data contracts: arrays are `(C, Z, Y, X)` / `(Z, Y, X)` / `(Y, X)` with a metadata dict carrying calibrated spacing in µm; metric functions return flat `dict`s of named scalars (functions measuring per object or per depth return arrays or tables and say so); batch drivers read a JSON manifest and write CSV tables plus a config record. A contract change is backward compatible or versioned. | A consumer needs only the artifact, not the producer's code path. |
| MOD-6 | Size limits: files ≤ 600 lines (1,000 hard maximum); function bodies ≤ 60 lines, docstrings excluded; nesting ≤ 4 levels. A file past the limit is split along its concerns (MOD-1), not by line count. | Line counts. |
| MOD-7 | Composition happens in one place per entry point (each CLI command, each batch driver's `run`). Metric functions never load files or build their own dependencies. | Swapping a component changes one composition site and one option. |

## 2. Generality (GEN)

Build for the class of problem, not for today's dataset.

| ID | Rule | Check |
| --- | --- | --- |
| GEN-1 | Nothing experiment-specific in library code: no paths, sample names, group labels, channel indices, dyes, assay titles or thresholds tied to one dataset. They are arguments or manifest fields with named defaults. | No literal path, condition name or dataset-specific number in `src/`. |
| GEN-2 | Derive facts from their source: voxel size from file metadata (or an explicit argument), never a constant; channel identity from metadata or an explicit override; group sizes from the data. A fallback that guesses is announced (ERR-2). | No calibration constant outside defaults that are documented as defaults. |
| GEN-3 | Work for every input that makes sense: 2D and 3D, any supported vendor format, any number of channels, isotropic or anisotropic spacing, one group or many. A function that cannot handle a case refuses it with a clear message. | Each public function states its accepted shapes and fails fast on others. |
| GEN-4 | Parameterize the part that varies; never fork. A new variant is a parameter or registry entry, not `metrics_v2.py` or a copied script with one line changed. | No file whose diff against another is mostly identical. |
| GEN-5 | New metrics, formats and methods plug in without editing their consumers: consumers iterate the registry or the returned dict's keys rather than enumerating names. | Adding a metric touches its module, its tests and (if it is a reported column) the one column list in `report`. |

## 3. Reuse (REU)

Make durable, integrated pieces; do not write one-time scripts.

| ID | Rule | Check |
| --- | --- | --- |
| REU-1 | Analyses are manifests, not scripts. A new dataset of an existing assay is a JSON manifest run by the existing driver (`fluorostats depth`, `fluorostats viability`, `quant2d`/`quant3d`). A new knob becomes a manifest field or CLI option on the shared tool. | No dataset-specific Python in the repository. |
| REU-2 | One tool per job, extended rather than duplicated. Before adding a script or module, name the existing tool it could not extend and why. | Each new file's commit names the tool it could not extend. |
| REU-3 | Scripts call the library; they do not reimplement it. `benchmarks/` and `examples/` import measurement, statistics and styling from `fluorostats`. A capability a script needs that the library lacks is added to the library (with tests) first. | No metric, test, matcher or style setting defined in a script that the library also defines. |
| REU-4 | Tools are durable: deterministic, idempotent, refuse to silently overwrite unrelated output, and record their own configuration and environment next to what they write. | Re-running a tool on the same input gives byte-identical tables. |
| REU-5 | Defaults are the measured best and cite their evidence (the benchmark table or validation that chose them). A default that can change results changes only with that evidence and a CHANGELOG entry. | Each non-obvious default names its evidence. |
| REU-6 | A throwaway probe becomes a function, an option or a test the second time it is needed, or is deleted. Exploratory code never lands on `main` as a one-off. | No file used by exactly one past run. |

## 4. Singularity (SIN)

Everything exists in exactly one place. Duplication is how two copies drift apart and one of them becomes wrong.

| ID | Rule | Check |
| --- | --- | --- |
| SIN-1 | One implementation per concern: one Gini, one CV, one Mann-Whitney/Cliff's δ wrapper, one BH-FDR, one bootstrap, one instance matcher, one tile grid, one figure style. Every caller imports it. Where callers deliberately differ (e.g. `ddof`), the difference is a parameter of the one implementation, not a second copy. | No two functions compute the same metric, test or transform. |
| SIN-2 | One definition per fact: a default, a column list, a label, a threshold name list, an error message for the same condition, is defined once and imported. Docs link to the definition instead of restating values that can change. | Grep for a value finds one definition. |
| SIN-3 | No copy-paste code. A block that appears twice becomes a function on the second use; a file that differs from another by a few lines becomes one file with a parameter. | No near-identical blocks across files. |
| SIN-4 | One entry point per job, one composition point per component, one registry per kind of plug-in (threshold methods, count methods, file formats). | Each job and plug-in kind has one place to look. |
| SIN-5 | One home per document fact (DOC-1); a superseded doc or script is deleted, not left beside its replacement. | No two live files describe the same thing. |

## 5. Minimal code (MIN)

| ID | Rule | Check |
| --- | --- | --- |
| MIN-1 | Every line serves a current requirement: a public capability, a benchmark, an example or a test. | You can name what a file or function serves. |
| MIN-2 | No speculative options. Generality (GEN) means not hard-coding today's case, not building for imagined ones. | Every option is used by a caller, manifest, example or test. |
| MIN-3 | Dead code leaves `main` (Git keeps it). | Nothing in `src/` is unreachable from the public API, the CLI or a test. |
| MIN-4 | Reuse existing dependencies first; a new dependency is justified in its commit. Vendor readers stay optional extras. | Every dependency is imported by live code. |
| MIN-5 | Shortest correct diff. Feature changes carry no drive-by refactors; cleanups are separate commits. | Every changed line traces to the commit's purpose. |

## 6. Results and statistics (EXP)

These rules exist because the published numbers depend on them.

| ID | Rule | Check |
| --- | --- | --- |
| EXP-1 | Everything is calibrated: lengths in µm, volumes in µm³, densities per mm² / mm³ from real spacing. Fractions are in [0, 1] and labelled as fractions, not percent. | Units are in the column name or the docstring. |
| EXP-2 | Small-sample inference is non-parametric by default: two-sided Mann-Whitney U with Cliff's δ, BH-FDR across any grid of tests, the biological replicate (stack, tube) as the unit of replication. A parametric test appears only when asked for and is labelled. | Reported p-values name the test, the n and the correction. |
| EXP-3 | Every metric is validated against a reference implementation or analytic ground truth, and that check is a regression test. | The metric has a test that would fail if it drifted. |
| EXP-4 | Deterministic: no random seed in the measurement path; any resampling (bootstrap, power) takes an explicit seed. | Two runs give identical bits. |
| EXP-5 | A change that can move a published number ships only with the evidence and a CHANGELOG entry, and every place that reports the number is checked. | The commit says which published numbers it can move, or "none". |

## 7. Data and artifacts (DAT)

| ID | Rule | Check |
| --- | --- | --- |
| DAT-1 | Raw microscopy never enters Git; collaborator or proprietary data never leaves `methods_paper/` (git-ignored). Only derived numeric tables are committed. | `git ls-files` has no image stacks; the tag tarball has no `methods_paper/`. |
| DAT-2 | Inputs are never modified in place; outputs go to a named output directory. | No writes next to input files. |
| DAT-3 | Tables written for figures carry the columns the figure needs and nothing derived ad hoc in the plotting script. A figure that must compute on the reader's machine (a live reproducibility run, a display overlay) says so in a `# exception DAT-3:` comment. | A figure script reads, filters and plots; it does not compute metrics outside a recorded exception. |

## 8. Configuration (CFG)

| ID | Rule | Check |
| --- | --- | --- |
| CFG-1 | Each CLI option has a default and a help line; options that change results say so. | `--help` documents every behaviour switch. |
| CFG-2 | A tool records what it resolved (config + environment) next to its output. | The output directory identifies how it was produced. |
| CFG-3 | Environment variables are read at entry points only (scripts, CLI), never inside measurement modules. | `os.environ` appears only in entry points. |
| CFG-4 | No secrets in code or Git. | Credential scan is clean. |

## 9. Code style (STY)

| ID | Rule | Check |
| --- | --- | --- |
| STY-1 | `ruff check src tests benchmarks examples` and `ruff format --check src tests` pass; `mypy` (configured in `pyproject.toml`) passes for `src/`. One statement per line. | CI is green. |
| STY-2 | Names say what a thing is or does; no `_v2`, `_new`, `_final`, `_gaps2` for live code or tests. | A reader can predict a module from its name. |
| STY-3 | Comments explain why. A comment that no longer matches the code is a bug. | Docstrings match behaviour. |
| STY-4 | Public functions have type hints and docstrings stating shapes and units. No function mutates its arguments unless its name says so. A renamed or retired keyword keeps working for one release with a `DeprecationWarning` (`fluorostats.deprecation`). | mypy; review. |

## 10. Errors (ERR)

| ID | Rule | Check |
| --- | --- | --- |
| ERR-1 | Fail fast on bad configuration or impossible inputs, with a message naming the fix. | Bad manifests exit before any image loads. |
| ERR-2 | No silent `except`. Catch specific errors; any fallback that changes what is measured (default spacing, dropped method, skipped stack) emits a warning or is recorded in the output. | Every broad catch is narrowed or surfaced. |
| ERR-3 | A partial result is never presented as complete: skipped files and failed fits are counted and flagged in the output tables. | Tables carry `fit_ok`-style flags and counts. |

## 11. Testing (TST)

| ID | Rule | Check |
| --- | --- | --- |
| TST-1 | 100% line coverage of `src/`. Genuinely unreachable defensive lines carry `# pragma: no cover` with the reason. | `pytest --cov` reports 100%. |
| TST-2 | A bug fix starts with a test that fails without the fix. | The fix commit includes it. |
| TST-3 | Tests are organized by the concern they test (`test_<module>.py`), not by when they were written. | No `*_gaps*.py`, `*2.py` test files. |
| TST-4 | No skipped, focused or placeholder tests on `main`, except environment-gated ones that say what they need. | Search for `skip`, `xfail`, `TODO`. |
| TST-5 | Refactors are behaviour-preserving and proven so: the full suite plus a golden comparison of outputs before and after. | The refactor commit names its proof. |

## 12. Documentation and commits (DOC)

| ID | Rule | Check |
| --- | --- | --- |
| DOC-1 | Each fact has one home. `README.md` is the user guide and index; `docs/` holds how things work; `CHANGELOG.md` the release record; `benchmarks/DATA_MANIFEST.md` the datasets. | No two docs state the same fact differently. |
| DOC-2 | A change updates the docs it affects in the same commit. | The commit includes them. |
| DOC-3 | Commits are authored by the project owner only: no `Co-Authored-By`, session links or tool footers in commits, PRs or release notes. | `git log` shows none. |
| DOC-4 | The deposited archive is public: anything shipped in a tagged release (manifest, README, results tables) must agree with the results it reports. | Release checklist. |

## Target layout

```text
src/fluorostats/   # the library. Measurement modules (no plotting, no click) <- orchestration (depth_batch,
                   # viability_batch, report, cli) ; figures in plots / render3d / projections / pubfig / depth_figures
tests/             # one test_<module>.py per module (TST-3)
benchmarks/        # figure scripts: read results/ and plot (DAT-3)
  producers/       # one producer per results table, sharing common.py (REU-3, REU-4)
  results/         # derived metric tables + <table>.run.json (committed); DATA_MANIFEST.md, VALIDATION.md
.github/workflows/ # CI: ruff, mypy, pytest at 100% coverage on Python 3.10-3.13
examples/          # worked examples of the public API, each importing everything it uses from fluorostats
docs/              # how things work; this file
methods_paper/     # git-ignored: local drafts and collaborator data (DAT-1)
```

## Current state

**Three audit rounds, 2026-10-14.** Each round ran five parallel read-only audits
(one per rule family) plus an independent review; fixes landed one per commit on
branch `engineering-audit`. Refactors were proven behaviour-identical with a golden
harness (every public function and CLI command on fixed inputs, plus every figure
and example script compared byte for byte); changes that alter outputs are listed
in `CHANGELOG.md` under Unreleased with the inputs they affect.

Now compliant (measured, not estimated):

- MOD-1/MOD-6: every file in `src/`, `tests/`, `benchmarks/` and `examples/` is at
  most 600 lines; every function body is at most 60 lines and nests at most four
  levels.
- MOD-2: one registry per variant family (file and folder formats, threshold, count
  and opening methods, reducers, report modes, condition modes, render themes,
  reported metrics).
- MOD-3/MOD-4: no import cycles or upward edges; `import fluorostats` loads no
  plotting backend; no private access across modules outside a module's own test
  file.
- MOD-5: one voxel-size argument, `voxel_size_um`; one key set per function
  (`live_dead_by_count` returns the same flat keys in every mode; `method="all"`
  also keeps the deprecated `by_method`); renamed or
  retired arguments keep working for one release with a DeprecationWarning.
- GEN-2/GEN-3: axis order and channel names from the file's labels (Olympus: from the
  plane file names, checked against the mainfile sizes; a mismatch raises), verified on
  45 public vendor sample files and an optional real-file regression suite; lengths
  converted from their stated unit; every defaulted, copied or assumed calibration
  recorded in `voxel_size_source`; results do not depend on the installed scipy or
  scikit-image version where those libraries changed behaviour.
- SIN: one implementation per concern, including the small-object rule, mean/SD/SEM,
  tiling, mask overlays, contrasts, figure style and the segmentation pipeline;
  benchmark producers share `benchmarks/producers/common.py`.
- ERR/CFG/EXP: values never measured are NaN; fallbacks warn or are recorded;
  manifests and options are validated before any image loads; every command and
  every producer records its configuration (producers also the git commit).
- TST/STY: 100% line coverage on Python 3.10-3.13; ruff (lint and format, pinned) and
  mypy clean; CI runs all of it on every push.
- DAT/REU: figure scripts read committed tables (recorded DAT-3 exceptions: live
  determinism runs and display overlays), and every table regenerates from HEAD
  except those `benchmarks/README.md` marks as not regenerated (cluster-only,
  wall-clock timing and collaborator tables).

Open, each needing an owner decision:

- [ ] `benchmarks/figS2_runtime.py`: panel (c) plots a typed value, not a measured
  one; the platform label and the deep-learning timings are typed (cluster-only,
  no run record).
- [ ] Typed figure annotations in `fig2_nuclei.py` (the deep-learning reference
  line and the panel (f) crossover/scope map), which the watershed line in panel (d)
  now contradicts.
- [ ] `benchmarks/results/ve_crops.npz` (a display panel) has no producer.
- [ ] GEN-1: `style.MATERIAL_COLORS` keeps its GelMA/Hybrid/Control default
  (existing manifests rely on it); `examples/keyence_tube_analysis.py` and
  `multi_condition_pipeline.py` are dataset-specific and could become manifest runs.
- [ ] MIN-2: deprecated names to remove in a release that announces it. These warn:
  `fluorostats.style` (use `pubfig`); the `render3d` MIP panels (use `projections`);
  the `depth_batch` window and figure helpers (use `depth`, `depth_figures`);
  `pubfig.panel(dx=, dy=)`; the depth manifest's `auc_window_um`; the renamed
  voxel-size keywords; `watershed_split`'s `footprint_size`; the `metrics_3d`
  skeleton re-exports; `plots.compute_pvalues`. These are plain dict keys or table
  columns, kept for compatibility and documented as deprecated, but they cannot warn:
  `live_dead_by_count`'s `by_method`; Keyence `px_um`/`z_step_um`/`vox_*` keys; the
  viability `*_cov_pct` columns.
- [ ] GEN-1: defaults whose values cite no evidence (each is named once and
  documented as uncited): depth `FIT_R2_THRESHOLD` (0.85) and default window;
  quant3d `--threshold-scale 0.9` and `--min-branch-length 5`; quant2d `--threshold
  li` and `--bg-radius 15`; the `segment` auto-rule constants; the `viability`
  SNR/crowding regime constants; `auto_crop`'s 5% border; `count_local_maxima`'s
  3 px / 0.15. Each needs a benchmark, a citation, or to stay a documented choice.

## Validating a change

1. Sections 1-4: it is a function in the module that owns its concern or an option on an existing tool, not a fork or a
   one-off; nothing experiment-specific is hard-coded; nothing it adds already exists elsewhere.
2. `ruff check` and the full test suite pass at 100% coverage.
3. If it can change a result, it says which published numbers it can move (EXP-5); if it is a refactor, it is proven
   behaviour-identical (TST-5).
4. The docs it affects change in the same commit (DOC-2).
