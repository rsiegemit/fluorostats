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
| MOD-1 | One concern per module. Reading (`io`, `keyence`), preprocessing, segmentation, each metric family (`metrics_2d`, `metrics_3d`, `skeleton`, `objects`, `morphometry`, `ring`, `texture`, `spatial`, `depth`, `viability`), statistics (`stats`, `agreement`, `power`), validation, QC, reporting and figures (`plots`, `render3d`, `style`) each keep their logic, options and tests in their own module. | You can name the one concern a module serves; changing that concern touches one module. |
| MOD-2 | Swappable by name. Variants (threshold methods, count methods, reducers, file formats) are chosen by a named option resolved in one place (a dict/registry or one dispatch function), not by `if` chains repeated in each caller. Adding a variant is one implementation plus one registration. | Grep for a variant's name finds its implementation, its registration and its tests, nothing else. |
| MOD-3 | Dependencies point one way: measurement modules (`io` … `validate`) ← orchestration (`depth_batch`, `viability_batch`, `report`, `cli`) ← scripts (`benchmarks/`, `examples/`). Measurement modules never import plotting, `click` or orchestration; nothing in `src/` imports `benchmarks/` or `examples/`. No cycles; no function-level import used to dodge one. | The import graph has no upward edges and no cycles. |
| MOD-4 | Use only a module's public surface. No `module._private` access across module boundaries; if a caller needs it, the owner makes it public. | No `_name` imported or accessed from outside its module. |
| MOD-5 | Components meet at explicit data contracts: arrays are `(C, Z, Y, X)` / `(Z, Y, X)` / `(Y, X)` with a metadata dict carrying calibrated spacing in µm; metric functions return flat `dict`s of named scalars; batch drivers read a JSON manifest and write CSV tables plus a config record. A contract change is backward compatible or versioned. | A consumer needs only the artifact, not the producer's code path. |
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

These rules exist because the paper's numbers depend on them.

| ID | Rule | Check |
| --- | --- | --- |
| EXP-1 | Everything is calibrated: lengths in µm, volumes in µm³, densities per mm² / mm³ from real spacing. Fractions are in [0, 1] and labelled as fractions, not percent. | Units are in the column name or the docstring. |
| EXP-2 | Small-sample inference is non-parametric by default: two-sided Mann-Whitney U with Cliff's δ, BH-FDR across any grid of tests, the biological replicate (stack, tube) as the unit of replication. A parametric test appears only when asked for and is labelled. | Reported p-values name the test, the n and the correction. |
| EXP-3 | Every metric is validated against a reference implementation or analytic ground truth, and that check is a regression test. | The metric has a test that would fail if it drifted. |
| EXP-4 | Deterministic: no random seed in the measurement path; any resampling (bootstrap, power) takes an explicit seed. | Two runs give identical bits. |
| EXP-5 | A change that can move a published number ships only with the evidence and a CHANGELOG entry, and the manuscript is checked for the number. | The commit says which published numbers it can move, or "none". |

## 7. Data and artifacts (DAT)

| ID | Rule | Check |
| --- | --- | --- |
| DAT-1 | Raw microscopy never enters Git; collaborator or proprietary data never leaves `methods_paper/` (git-ignored). Only derived numeric tables are committed. | `git ls-files` has no image stacks; the tag tarball has no `methods_paper/`. |
| DAT-2 | Inputs are never modified in place; outputs go to a named output directory. | No writes next to input files. |
| DAT-3 | Tables written for figures carry the columns the figure needs and nothing derived ad hoc in the plotting script. | A figure script reads, filters and plots; it does not compute metrics. |

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
| STY-1 | `ruff check src tests benchmarks examples` passes. One statement per line. | Lint is clean. |
| STY-2 | Names say what a thing is or does; no `_v2`, `_new`, `_final`, `_gaps2` for live code or tests. | A reader can predict a module from its name. |
| STY-3 | Comments explain why. A comment that no longer matches the code is a bug. | Docstrings match behaviour. |
| STY-4 | Public functions have type hints and docstrings stating shapes and units. No function mutates its arguments unless its name says so. | Review. |

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
| DOC-4 | The deposited archive is reviewer-facing: anything shipped in a tagged release (manifest, README, results tables) must agree with the manuscript. | Release checklist. |

## Target layout

```text
src/fluorostats/   # the library. Measurement modules (no plotting, no click) <- orchestration (depth_batch,
                   # viability_batch, report, cli) ; figures in plots / render3d / style
tests/             # one test_<module>.py per module (TST-3)
benchmarks/        # paper figure scripts: read results/, call the library, plot (REU-3, DAT-3)
  results/         # derived metric tables (committed); DATA_MANIFEST.md, VALIDATION.md
examples/          # worked examples of the public API, each importing everything it uses from fluorostats
docs/              # how things work; this file
methods_paper/     # git-ignored: manuscript work and collaborator data (DAT-1)
```

## Current state

Audit against these rules: pending (see the audit report for the findings and the proposed fixes).

## Validating a change

1. Sections 1-4: it is a function in the module that owns its concern or an option on an existing tool, not a fork or a
   one-off; nothing experiment-specific is hard-coded; nothing it adds already exists elsewhere.
2. `ruff check` and the full test suite pass at 100% coverage.
3. If it can change a result, it says which published numbers it can move (EXP-5); if it is a refactor, it is proven
   behaviour-identical (TST-5).
4. The docs it affects change in the same commit (DOC-2).
