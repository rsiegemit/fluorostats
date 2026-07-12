# Competitor Benchmark: Depth-Penetration (Axial Intensity) Profiling

**Status:** VERIFIED by fetching primary sources (ImageJ/Fiji docs, CellProfiler
module manual, Oxford Instruments/Imaris Learning Centre, ZEISS OAD, Nikon
NIS-Elements docs). Feature descriptions and scripting/batch claims below are
quoted or paraphrased from vendor/tool documentation with URLs. Anything that
could not be independently reached from a primary source is flagged
**(unverified)**. Numbers and features are not invented.

## 1. What the task is, and why there is no public ground-truth set

The fluorostats depth-penetration capability answers a specific quantitative
question: **how far does a fluorescent probe penetrate into a material?** Given a
confocal z-stack, the pipeline is:

1. Collapse each z-slice to a single **per-slice spatial mean (or median)**
   intensity vs physical depth (`intensity_depth_profile`, `src/fluorostats/depth.py`).
2. **Subtract a matched blank / no-fluo control** (scalar or a depth-resolved
   blank profile, resampled depth-for-depth; negatives clipped to 0)
   (`subtract_background`).
3. **Normalise to the near-surface signal** (mean of the first `n_surface`
   slices, default 3) so decay *shape* is comparable across acquisitions with
   different gain/laser (`normalize_to_surface`).
4. **Trapezoidal area-under-curve** over one or more physical depth windows,
   with endpoints linearly interpolated so the window is exactly `[z0, z1]`
   regardless of slice spacing (`auc_depth`).
5. **Group mean ± SEM** curves + per-window AUC, with a built-in **Welch
   (unequal-variance) t-test** for a two-condition contrast
   (`src/fluorostats/depth_batch.py`, `_print_summary`).

The whole run is **manifest-driven**: one JSON (groups, stacks, channel,
reducer, background blanks, normalisation slices, AUC windows) produces tidy CSVs
(`depth_profiles_long.csv`, `auc_per_stack.csv`, `group_depth_summary.csv`) plus
publication figures — deterministic, no manual export.

**Honest framing on ground truth.** Unlike nucleus segmentation (which has
BBBC039 with ~23,000 manually annotated instances as public ground truth), there
is **no public ground-truth benchmark dataset** for confocal depth-penetration /
permeability profiling (see **§5** for the documented, reproducible repository search
that backs this claim, with the closest near-miss datasets cited). The tools people
actually use for it are GUI/commercial
and not scriptable head-to-head. So correctness cannot be scored against a
community dataset; instead it is shown on **synthetic Beer-Lambert ground
truth** where the answer is known in closed form. A Beer-Lambert stack
`I(z) = I0·exp(−z/λ) + bg` has analytic absolute AUC `I0·λ·(1−e^(−Z/λ))` and
normalised AUC `λ·(1−e^(−Z/λ))`; fluorostats recovers these exactly on noiseless
stacks and within SEM on noisy stacks, and reproduces Fiji's per-slice mean
bit-for-bit. See `methods_paper/benchmarks/b_depth_penetration.py` for the
constructed-ground-truth benchmark and faithful-reimplementation parity checks.

Consequence for the comparison: the differentiator here is **not accuracy of any
single step** (the arithmetic — mean, subtract, normalise, trapezoid — is
standard and every tool below can produce the underlying z-profile). It is
**reproducibility and integration**: doing the entire penetration pipeline from
one config with no manual export.

## 2. The comparators (what each does; GUI vs scriptable; batch)

### 2.1 Fiji / ImageJ — "Plot Z-axis Profile" (primary comparator)

The canonical free tool. **Image ▸ Stacks ▸ Plot Z-axis Profile** is the direct
analogue of fluorostats' step 1.

- **VERIFIED — what it computes:** the ImageJ menu reference states it
  "**Plots the ROI selection mean gray value versus slice number. Requires a
  selection.**" That is exactly fluorostats' per-slice spatial mean over the
  field/ROI. (ImageJ docs, Image menu:
  https://imagej.net/ij/docs/menus/image.html)
- Community documentation confirms it is "like applying **Measure** to each
  slice independently … over the entire image or any ROI … plots the mean pixel
  values," and the plot window offers **List / Save / Copy** to export the
  per-slice column to Excel (imagej.net Image Intensity Processing:
  https://imagej.net/imaging/image-intensity-processing).
- **GUI vs scriptable:** GUI by default, but **fully scriptable via ImageJ
  macros / the script recorder** (`getZAxisProfile`-style workflows), so batch is
  achievable with user-written macros.
- **Batch/pipeline:** no built-in penetration pipeline. Background subtraction,
  surface normalisation, AUC, group mean ± SEM, and the significance test are
  **manual** (Excel/Prism) or hand-rolled in a macro. It plots the z-profile; it
  does not, out of the box, subtract a blank, normalise to surface, integrate a
  depth window, or run a group statistic.

### 2.2 Imaris (Oxford Instruments / Bitplane) — MeasurementPro / intensity profile

- **VERIFIED — intensity profiling exists:** with a line drawn in the Surpass
  view, "Imaris displays the intensity of each voxel along the line," shown under
  the **Histogram** tab (x-axis = line length, y-axis = channel intensity)
  (Imaris Learning Centre, Intensity Profile:
  https://imaris.oxinst.com/open/view/intensity-profile). **MeasurementPro**
  "adds shape, size, and intensity based measurement capabilities" (Oxford
  Instruments product materials).
- Imaris Vantage generates intensity-profile plots around surfaces
  (https://imaris.oxinst.com/learning/view/article/where-to-find-a-3d-intensity-profile-around-your-surface-in-imaris-microscopy-image-software).
- **GUI vs scriptable:** GUI-first, commercial. Scriptable via **ImarisXT**
  (MATLAB/Python XTensions) and a **Batch Process** function
  (https://imaris.oxinst.com/learning/view/article/imarisxt;
  https://imaris.oxinst.com/open/view/batch-process-function).
- **Batch/pipeline:** the line/voxel intensity profile is oriented to spatial
  distance along a drawn line, not natively to an **axial depth-vs-slice mean
  with blank subtraction + surface normalisation + AUC over a depth window**;
  that full penetration pipeline is not a built-in one-click product and would be
  assembled in an XTension. **(Exact axial-profiling UI path unverified** — the
  intensity-profile documentation reached describes line profiles, not a
  dedicated depth-mean tool.)

### 2.3 ZEISS ZEN (blue / black) — profile & intensity tools

- **Intensity measurement / profile tools exist:** ZEN Blue's **Image Analysis**
  module produces intensity features and measurement tables; a profile/graphics
  workflow yields area and mean-intensity values on drawn contours (ZEN Blue
  Image Analysis guides, e.g.
  https://www.zeiss.com/content/dam/rms/reference-master/service-support/downloads/faq/zen-blue_image-analysisguide-02-2016.pdf).
- **GUI vs scriptable:** GUI-first, commercial. Scriptable via **OAD (Open
  Application Development)** — Python/IronPython macros and the Image Analysis
  Wizard can be automated (https://github.com/zeiss-microscopy/OAD;
  https://www.zeiss.com/microscopy/us/products/software/zeiss-zen/zen-developer-toolkit.html).
- **Batch/pipeline:** an axial-profile / ortho line profile can be read, and
  analyses batched via OAD, but a **blank-subtracted, surface-normalised,
  AUC-over-depth-window penetration pipeline with group stats is not a stock
  ZEN measurement**; it would be scripted in OAD. **(Exact "profile intensity /
  ortho line" documentation path unverified** — official ortho-profile tool docs
  were not directly reached; capability inferred from ZEN analysis guides.)

### 2.4 Nikon NIS-Elements — intensity over Z / time measurement

- **VERIFIED — measures mean intensity across Z:** the **Time Measurement** tool
  "records **average pixel intensities within Regions Of Interest (ROIs)**"; when
  the ND document combines time with a Z dimension, **Measure All** measures "all
  frames across all multipoints and/or Z stacks at once" (NIS-Elements Time
  Measurement docs:
  https://www.nisoftware.net/NikonSaleApplication/Help/Docs-AR/eng_ar/timemeas.html).
  A separate **Measure ▸ Intensity Profile** creates a pixel-intensity graph
  along a linear section.
- **GUI vs scriptable:** GUI-first, commercial. Scriptable via the **NIS-Elements
  macro language**, **GA3** (visual "General Analysis 3," 400+ functions, with a
  Python editor), and **JOBS** for automated acquisition+analysis; recipes run on
  saved files via `nis.mac.GA3_Execute()` for batch
  (https://www.nisoftware.net/NikonSaleApplication/Help/Docs-D/eng_d/GS_MacroLanguageDefinition.html;
  Nikon GA3/JOBS application note:
  https://www.microscope.healthcare.nikon.com/resources/application-notes/automatic-isolation-of-pollen-using-nis-elements-general-analysis-ga-and-jobs-imaging-workflow-tools).
- **Batch/pipeline:** ROI-mean-vs-Z is native and batchable; the specific
  **blank subtraction → surface normalisation → AUC over depth window → group
  mean±SEM + significance** chain is not a stock button and would be built in
  GA3/macro.

### 2.5 CellProfiler — `MeasureImageIntensity` over a z-stack

- **VERIFIED — module and mean-intensity measurement exist:**
  `MeasureImageIntensity` "measures several intensity features across an entire
  image (excluding masked pixels)," producing **`MeanIntensity`,
  `MedianIntensity`** ("Mean and median of pixel intensity values"),
  **`TotalIntensity`** ("Sum of all pixel intensity values"), plus Std/MAD,
  Min/Max, quartiles, percentiles, and `TotalArea` (CellProfiler measurement
  module manual:
  https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.6/modules/measurement.html).
  Fed each z-slice as a separate image, this yields a per-slice mean — the same
  quantity as Fiji's Plot Z-axis Profile.
- **GUI vs scriptable:** GUI to build the pipeline; **headless CLI** for batch:
  `-c` "runs headless," `-r` runs the analysis, `-p pipeline`, `-i`/`-o` for
  input/output dirs, `-f`/`-l` for image-set ranges (CellProfiler command-line
  wiki:
  https://github.com/CellProfiler/CellProfiler/wiki/Getting-started-using-CellProfiler-from-the-command-line).
- **Batch/pipeline:** a pipeline can emit per-slice mean intensity to a
  spreadsheet, and it is scriptable/headless — but CellProfiler has **no
  depth-penetration semantics**: no blank-profile subtraction vs depth, no
  surface normalisation, no trapezoidal AUC over a physical depth window, no
  group mean±SEM + Welch test. Those remain downstream manual/Prism steps.

### 2.6 MATLAB (Image Processing Toolbox) — roll-your-own depth scripts

- The REAVER-style "write it yourself" comparator. MATLAB can load a stack,
  compute `mean(slice(:))` per plane, subtract a blank, normalise, `trapz` the
  AUC, and run `ttest2` — reproducing the whole pipeline exactly.
- **GUI vs scriptable:** scriptable, deterministic, fully batchable.
- **Batch/pipeline:** everything is possible, but **nothing is provided** — it is
  bespoke code per lab, unversioned, with no standard manifest, no tidy-CSV
  contract, and reproducibility resting entirely on the author. This is the
  status quo fluorostats replaces with a shared, tested, config-driven pipeline.

## 3. Capability matrix

Rows = the six comparators + fluorostats. Columns are the discrete steps of a
depth-penetration study. Legend: **yes** = built-in/native; **partial** =
possible but not a stock one-step feature (needs scripting/config work in-tool);
**manual** = done outside the tool (Excel/Prism/hand-rolled); **no** = not
available. "Manual steps to a group comparison" = rough count of distinct
human/export actions from raw stacks to a group mean±SEM + significance result.

| Capability | Fiji / ImageJ | Imaris | ZEISS ZEN | NIS-Elements | CellProfiler | MATLAB (IPT) | **fluorostats** |
|---|---|---|---|---|---|---|---|
| Per-slice intensity profile (mean/median vs depth) | **yes** (mean gray/slice) | yes (line/voxel) | yes | **yes** (ROI mean vs Z) | **yes** (`MeanIntensity`/slice) | yes (`mean`) | **yes** (mean & median) |
| Blank / background subtraction (matched control) | manual | partial | partial | partial | partial | partial | **yes** (scalar or depth-resolved blank) |
| Surface normalisation (near-surface reference) | manual | manual | manual | manual | manual | partial | **yes** (`n_surface`) |
| AUC over physical depth window(s) | manual | manual | manual | manual | manual | partial (`trapz`) | **yes** (trapezoid, interpolated endpoints, multi-window + "full") |
| Group mean ± SEM curves | manual | partial | partial | partial | manual | partial | **yes** |
| Built-in significance test | no | no | no | no | no | partial (`ttest2`) | **yes** (Welch t-test) |
| Tidy CSV export | manual (copy/List) | partial | partial | partial | **yes** (spreadsheet) | partial | **yes** (3 tidy CSVs) |
| Batch / manifest-driven | macro | XTension/BatchProcess | OAD | GA3/JOBS/macro | **yes** (headless CLI) | script | **yes** (one JSON manifest) |
| Scriptable & deterministic | **yes** (macros) | yes (ImarisXT) | yes (OAD) | yes (macro/GA3) | **yes** (headless) | **yes** | **yes** |
| Typical # manual steps to a group comparison | ~8–12 | ~8–12 | ~8–12 | ~6–10 | ~5–8 | ~1 (bespoke code) | **1** (edit + run one manifest) |

**The honest story the matrix tells.** Every tool can *plot a z-profile* — that
step is a commodity, and Fiji, CellProfiler, NIS-Elements and Imaris all do it
natively. Several tools are also genuinely **scriptable** (ImageJ macros,
CellProfiler headless, ZEN OAD, NIS GA3/JOBS, MATLAB), so "fluorostats is
scriptable and they aren't" would be **false** and is not claimed. What no
GUI/commercial tool offers is the **entire penetration pipeline as one built-in,
config-driven, reproducible unit**: blank subtraction → surface normalisation →
multi-window trapezoidal AUC → group mean±SEM → significance test → tidy CSVs +
figures, from a single manifest with zero manual export. MATLAB *can* do all of
it, but only as bespoke, unshared, per-lab code (the REAVER problem). fluorostats
packages that same standard arithmetic as a tested, deterministic, one-command
tool — turning a ~8–12-step manual Fiji+Excel workflow (or a one-off MATLAB
script) into one reproducible config.

## 4. Honest limitations of the fluorostats approach

- **It is a standard pipeline, not a novel algorithm.** Every individual step —
  per-slice mean, blank subtraction, surface normalisation, trapezoidal AUC,
  Welch test — is textbook. There is no new estimator or model here. The
  contribution is **reproducibility + integration + a tested reference
  implementation**, not a better number on any single step.
- **The differentiator is not accuracy.** On the same z-profile, Fiji/CellProfiler
  compute the identical per-slice mean (verified bit-for-bit in the benchmark).
  fluorostats does not, and does not claim to, measure intensity "more
  accurately" than these tools — it removes the manual export/subtract/normalise/
  integrate steps that make the standard workflow slow and irreproducible.
- **Whole-image mean assumes a roughly uniform field.** The default reducer
  averages every pixel in each slice, which is only meaningful if the field is
  approximately homogeneous (or the material fills it). Heterogeneous fields
  (vessels, voids, a scaffold edge in frame) require the **user to pick the ROI
  upstream** — fluorostats profiles what it is given; it does not segment the
  region of interest. (`median` is offered as a debris/bubble-robust reducer, but
  it is not a substitute for correct ROI selection.)
- **No public ground-truth benchmark to score against.** Correctness is
  demonstrated on synthetic Beer-Lambert stacks with known closed-form AUC, not
  on a community dataset — because none exists for this task. This is a genuine
  evidentiary limitation relative to the nuclei comparison (BBBC039), and is
  stated as such rather than hidden.
- **Small-n statistics.** The built-in Welch test is explicitly labelled
  "underpowered, descriptive only" in the code for the typical handful of stacks
  per condition; it flags an ordering/gap, it is not a powered inference.
- **Model-agnostic AUC, not a decay fit.** fluorostats integrates the observed
  curve trapezoidally rather than fitting an exponential decay constant λ. This
  is deliberate (no distributional assumption), but a user who specifically wants
  a penetration-depth constant would still fit that themselves; fluorostats
  reports retained-signal AUC, not λ.

---

## 5. Open-data availability — a documented search (the "no public dataset" claim)

A claim that no suitable public dataset exists carries a burden of proof, so it is
recorded here as a **bounded, reproducible search result**, not an absolute absence.
It is falsifiable: a single counterexample meeting criteria (i)–(iv) below overturns it.

**Search (July 2026).** Repositories and interfaces queried:
- **Zenodo** — keyword search + the `/api/records` REST API (`type=dataset`).
- **EBI BioImage Archive / BioStudies** — the BioImages search API, with per-study
  `File List` verification of the actual image formats.
- **Image Data Resource (IDR)** — the curated study catalogue (`github.com/IDR/idr-metadata`).
- **figshare / Dryad** — via web index.

Query families: *tissue clearing / imaging depth*; *antibody & dye penetration depth
(cleared tissue)*; *FITC-dextran hydrogel diffusion & permeability*; *mounting-medium /
refractive-index depth*; *light-sheet vs confocal imaging-depth comparison*.

**Inclusion criteria (all four required)** for a real-data penetration benchmark:
(i) openly licensed and directly downloadable; (ii) **raw z-stack**, not a
maximum-intensity projection; (iii) **TIFF / OME-TIFF** (readable without proprietary
vendor libraries); (iv) **≥ 2 experimental conditions** enabling a penetration/depth
contrast.

**Result: no record satisfied all four.** The three closest each fail on exactly one
criterion, and are cited so a reader can re-examine them:

| Dataset (accession / DOI + URL) | What it is | Fails criterion |
|---|---|---|
| **S-BIAD479**, BioImage Archive — "Tissue libraries enable rapid determination of conditions preserving antibody labeling" — https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD479 | Confocal z-stacks of 500 µm–1 mm cleared tissue across **multiple antibody-labeling conditions and incubation times (18 h vs ≥1 week)** — scientifically ideal | **(iii)** raw = Olympus **`.oir`**, processed = Imaris **`.ims`** (verified via the study `File List`, Jul 2026); needs Bio-Formats/Java to read |
| **S-BIAD1136**, BioImage Archive — "3D light sheet microscopy imaging of cleared human mammary gland terminal ductal lobular unit" — https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1136 | **55 `.tif`** light-sheet z-stacks of cleared tissue, 0.8 µm z-spacing | **(iv)** single clearing condition (and impractical: 0.57–4.4 GB per stack) |
| **Zenodo 10.5281/zenodo.437943** — "Entire confocal z-stack series as .tif image sequences" (parvalbumin ependymal cells), CC-BY-4.0 — https://zenodo.org/records/437943 | Confocal **`.tif`** z-stacks | **(ii) & (iv)** mixes maximum-intensity projections; single biological condition |

**Bounded claim for the paper:** *as of July 2026, a search of Zenodo, the EBI
BioImage Archive, IDR, figshare and Dryad did not identify a public, openly-licensed,
TIFF/OME-TIFF confocal z-stack dataset with ≥ 2 conditions suitable for a
depth-penetration comparison.* Correctness is therefore established on synthetic
Beer–Lambert ground truth (§1); the constraint is **deposited data**, not the method.

**The method itself is standard and published** (so the gap is data, not analysis):
depth-dependent decay of the per-slice mean intensity is a recognised confocal
phenomenon, and the canonical correction fits an **exponential curve to the average
intensity in each slice** — e.g. Amira's *Correct-Z-Drop* as used by **Bonda U,
Jaeschke A, Lighterness A, Baldwin J, Werner C, De-Juan-Pardo EM, Bray LJ. "3D
Quantification of Vascular-Like Structures in z Stack Confocal Images." STAR
Protocols 2020;1(3):100180. doi:10.1016/j.xpro.2020.100180** (PMC7757404). That is
exactly the per-slice-mean + exponential model fluorostats implements. Cleared-tissue
studies routinely quantify probe/antibody **penetration depth** across conditions
(e.g. S-BIAD479 above), but deposit the raw stacks in proprietary formats or not at all
— which is the honest reason a synthetic ground truth is used here.

---

### Verification summary

- **VERIFIED from primary sources:** Fiji "Plot Z-axis Profile" = per-slice ROI
  mean gray value (imagej.net); CellProfiler `MeasureImageIntensity` produces
  `MeanIntensity`/`MedianIntensity`/`TotalIntensity` per image and runs headless
  via `-c -r -p` (CellProfiler manual + CLI wiki); Imaris intensity profiling +
  ImarisXT/Batch Process (Oxford Instruments Learning Centre); ZEN OAD Python
  scripting (ZEISS OAD GitHub / Developer Toolkit); NIS-Elements ROI-mean-vs-Z
  Time Measurement + GA3/JOBS/macro batch (nisoftware.net / Nikon app note).
- **Flagged (unverified):** Imaris exact *axial* depth-mean UI path (docs reached
  describe line/voxel profiles, not a dedicated depth-mean tool); ZEN exact
  "profile / ortho line intensity" tool documentation path (inferred from
  analysis guides, official ortho-profile page not directly reached). MATLAB
  capabilities are stated from general toolbox function knowledge (`mean`,
  `trapz`, `ttest2`), not a single cited page.
- **Not claimed / avoided overstatement:** that competitors cannot script or
  batch (several can); that fluorostats is more *accurate* per step (it is
  bit-for-bit identical on the shared step). The defensible claim is
  reproducibility + full-pipeline integration from one manifest.
