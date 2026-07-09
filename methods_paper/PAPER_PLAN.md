# fluorostats methods paper — proposed plan (synthesized from 45+ exemplar papers)

*Draft plan for discussion. Built by triangulating structural conventions from 15
parallel literature studies across: DL segmentation tools, bioimage platforms,
scientific Python libraries, vascular/skeleton/viability/spatial-stats tool papers,
benchmark & data-challenge papers, Nature-Methods / Bioinformatics / JOSS venue
conventions, classical-vs-DL positioning, reproducibility/FAIR framing, the
bioprinting audience, figure design, and title/abstract craft. No single paper is
copied; every recommendation below is a convention seen across multiple exemplars.*

---

## 0. The one tension to resolve first (it shapes everything)

The exemplars split into two structural species, and fluorostats sits between them:

- **"Platform" papers** (Fiji, CellProfiler, QuPath, ilastik, Icy) deliberately have
  **no single headline metric**. They lead with breadth + integration + reproducibility,
  organize Results *by capability*, carry validation *inside* case studies, and spend a
  figure on an architecture schematic. Contribution = the union, not a win.
- **"Benchmark/method" papers** (Cellpose, StarDist, Omnipose, REAVER, scanpy) lead with
  a **quantified head-to-head** and live or die on comparison rigor.

fluorostats has *both* a breadth story (19 modules, integrated stats) **and** genuine
benchmark wins (≥ DL on separated nuclei; ties AngioTool; ties the Fiji viability macro).
The plan below **fuses the two**: a platform-style spine (Fig 1 schematic, Results-by-
capability, reproducibility as headline) carrying benchmark-paper rigor *inside each
capability section* (baseline validation, per-dataset tables, bootstrap CIs). This is the
structure that lets you claim breadth without looking unfocused and claim parity-or-better
without looking like an overreaching single-metric paper.

---

## 1. Positioning thesis (recommended framing)

**Primary claim (bounded, survivable):** fluorostats is *the reproducible, training-free,
CPU-only quantifier for the analyses that don't need a trained instance segmenter* —
delivering **reference-exact correctness** on every metric and **parity-or-better accuracy**
against established tools on well-separated targets, while **explicitly ceding the crowded/
overlapping-instance regime to deep learning.**

Two rhetorical rules the exemplars agree on:
- **Claim a regime, not a crown** (CellSeg3D, Cellpose, Omnipose all do this). "Matches or
  exceeds *on well-separated targets*; DL remains preferable when instances heavily overlap."
  A bounded claim survives review; "beats deep learning" does not.
- **Praise incumbents by name, then isolate one missing capability** (QuPath, scanpy). The
  gap is a *missing capability* (a validated, training-free, statistics-integrated quantifier),
  never "the competitors are bad."

**Two differentiators no comparator paper can claim** — foreground both:
1. **Reference-exact correctness** — numerical agreement with canonical implementations /
   analytic ground truth, metric by metric. DL papers benchmark accuracy but never prove
   bit-level correctness.
2. **Deterministic reproducibility** — no GPU, no training, no seed variance; bit-identical
   CPU reruns. A structural advantage, not a compromise.

---

## 2. Venue & format — the first decision for you (see Questions)

Two viable paths, from the venue studies:

- **Path A — Full method paper.** Nature Methods "Article" (3,000–5,000 words, Online-Methods-
  at-end, ≤6 display items, Extended Data), *or* eLife Tools & Resources, *or* PLoS Computational
  Biology (Software), *or* Bioinformatics Original Paper. These **require** head-to-head
  benchmarking against state-of-the-art on real data — which you already have. Highest prestige,
  highest bar, longest write.
- **Path B — Application note + JOSS.** A Bioinformatics Application Note (or SoftwareX / BMC
  Bioinformatics Software article) foregrounding utility + availability, **plus** a parallel JOSS
  submission that converts your 105 tests + CI + docs into a peer-reviewed quality stamp. Faster,
  lower ceiling, and the benchmark depth would be under-used.

**My recommendation:** Path A, targeting a **full method paper** (Nature Methods Article as the
stretch target; PLoS Comp Biol / eLife T&R as strong, more-attainable homes), because the
benchmark campaign is the paper's strongest asset and Path B would waste it. A JOSS note can still
be co-submitted later as a citable software artifact. **This is Question 1.**

The section plan below is written for Path A and scales down cleanly to Path B if you prefer.

---

## 3. Proposed section structure (Path A)

Canonical order adopted from Nature-Methods/scanpy/Cellpose, with the platform "Results-by-
capability" organization and benchmark-paper rigor folded in:

1. **Title** — see §9. Colon + differentiator ("training-free, CPU-only", "reproducible").
2. **Abstract** — ≤150–200 w, unstructured, unreferenced. Skeleton in §9.
3. **Introduction (no heading)** — 5 paragraphs, §9. Builds the gap from the field's *own*
   self-diagnosis (Spiller & Duarte Campos 2025; Pereira 2023: "image 3D, quantify 2D";
   under-reported parameters), praises incumbents, states the missing capability, previews the
   validation design + thesis, and ends with the honest DL-scope beat.
4. **Results** — organized by capability, each subsection self-contained with its own baseline
   validation + comparison + CIs:
   - 4.1 **Design & implementation overview** (brief; points to Fig 1 schematic + Methods).
     State 3–4 named design goals — *training-free, CPU-only, reference-exact, integrated stats* —
     then a one-paragraph module tour. (scikit-learn/scikit-image convention.)
   - 4.2 **Correctness: reference-exact validation** (the validity anchor; §5). Phantom + reference-
     implementation agreement table. Lead here — it licenses every later comparison.
   - 4.3 **Nuclei / instance measurement vs deep learning** (§6.1). BBBC039 head-to-head with
     bootstrap CIs; the DSB2018 "~91% of DL, zero training" result; the **baseline-validation**
     subsection; then the **scope boundary** (BBBC024 crossover) drawn as a curve.
   - 4.4 **Vascular networks (2D then 3D)** (§6.2). REAVER-protocol ranking (accuracy/precision
     split + zero-bias test); VesselExpress software-agreement with the honestly-named offset;
     synthetic-phantom exact-GT table.
   - 4.5 **Viability (Live/Dead), depth-resolved** (§6.3). The "2D biases viability +5–25%" paired
     figure; the tie-to-Fiji-macro (CCC 0.987); the regime decision-guide.
   - 4.6 **Spatial homogeneity + integrated statistics** (§6.4). The five-statistic validation
     (the Martin-gap wedge); the "correct-by-default" stats layer.
   - 4.7 **Performance / runtime** (§7). Separate section, scanpy-shaped: comparative timing table
     + a scaling note. Determinism as a results panel.
5. **Discussion** — no subheadings. What the tool is for; complementarity with DL (routes users to
   DL when out of scope); the reproducibility/auditability argument; impact for CPU-only labs and
   the bioprinting/TE audience.
6. **Scope & Limitations** — §8. The honesty ledger, reframed as characterized scope (mechanism +
   quantified boundary + mitigation per item), plus the "benchmark correctness safeguards" note.
7. **(Optional) Application vignette** — a worked GelMA-vs-hybrid bioprinting case (see Question 3).
8. **Online Methods** — exact metric definitions & formulae (with matching thresholds), dataset
   provenance, baseline configuration & tuning policy, statistical methods.
9. **Data availability / Code availability** — §10. Versioned Zenodo DOI, OSI license, accessions.
10. **Extended Data** — per-dataset breakdowns, full validation tables, extra comparators.
11. **Supplementary** — the software bundle, extended notes, the ~7 non-main figures, checklists.

---

## 4. The comparison / benchmark rigor playbook (applies to every §4.3–4.6)

Distilled from the benchmark-conventions, DL-tool, and vascular studies. These are the moves that
make "parity-or-better" survive a skeptical reviewer:

1. **Per-dataset tables, never a single grand winner.** Atomic unit = (dataset × method × metric).
   The famous challenge papers (CTC, DSB) explicitly refuse to collapse to one number; reviewers
   verify claims cell by cell.
2. **Define every metric formula with its matching threshold, and sweep the threshold.** Report
   F1/AP averaged over a range of IoU thresholds (DSB used 0.10–0.95), give closed forms in Methods.
   Single-threshold F1 reads as cherry-picking.
3. **A named "Baseline validation" subsection.** Reproduce each DL tool's *published* number before
   comparing (you already do: StarDist 0.871 ≈ published 0.864). Table of *our-reproduction vs
   original-paper*. This is the linchpin — the CellSeg3D peer reviews show that classical parity
   instantly triggers "were the DL baselines crippled?"; the reproduction table is the pre-emptive answer.
4. **One identical tuning/preprocessing policy for all methods, disclosed.** State it plainly
   (e.g. default parameters for all; no per-dataset training for fluorostats while DL used published
   weights — a fairness point *in your favor* when stated). Weber 2019 names self-assessment bias as
   the #1 threat when you wrote the method; disclosure defuses it.
5. **CIs + paired significance on every comparison** — this is where fluorostats *beats* the classic
   challenge papers (which have none). Present paired differences (fs − baseline) per image with
   bootstrap CIs: parity = CI overlaps 0; better = CI above 0. You already have this for BBBC039;
   extend the presentation.
6. **Draw scope boundaries as measurable curves with a mechanism** (Omnipose/Schulz style), not
   apologies. The BBBC024 clustering crossover becomes "parity below density X, DL above" with the
   mechanism named (CC labeling merges touching objects above threshold).
7. **A stratified generalization angle.** Because fluorostats is training-free, argue (and show
   across your 8 datasets/modalities) that performance doesn't collapse on unseen modalities the way
   trained DL can — the training-free generalization point.
8. **An explicit "limitations of this comparison" paragraph** (pipeline-generated GT, tools you
   couldn't include and why) — every challenge paper has one.

---

## 5. Validation / correctness section design (§4.2 — the anchor)

From the skeleton/topology + Python-library studies. This is fluorostats' strongest, most
distinctive section — none of BoneJ/skan/AnalyzeSkeleton/MitoGraph published the strongest version.

- **Lead with analytic phantoms with closed-form ground truth** (ball χ=1, torus χ=0, N balls χ=N,
  known branch/junction counts, known length) — agreement against *math*, not another program (which
  could share bugs). This is a strictly stronger claim than any comparator tool made.
- **A single expected-vs-measured error table** — a convention this field underuses: rows = phantoms/
  metrics, columns = expected | measured | abs error | % error. Concrete, auditable, communicates rigor
  better than overlaid histograms.
- **Separate exact (integer) from approximate (continuous) metrics explicitly.** Euler number, component
  counts, branch/junction counts are integers → bit-exact. Skeleton length is subject to rasterization →
  bounded % error with the discretization limit named. Pre-empts the "length isn't exact" objection.
- **State the shared-algorithm move, then prove it.** fluorostats implements the *same* Lee-1994 thinning
  as Fiji/skan → exact agreement is the *predicted, falsifiable* outcome → show the test that confirms it.
  Reframes "trivial agreement" as "a passed correctness contract."
- **Frame parity as a CI-tested invariant** (BoneJ2 model): the reference-parity checks run on every
  commit, so exactness is *maintained*, not a one-off.
- Structure a validation **table** (statsmodels-style) spanning all capabilities: metric × reference
  implementation × agreement (exact / max abs error / correlation) × dataset. You already have the data
  (stats 8/8, agreement 11/11, instance metrics 23/23, volume fraction 7/7, connectivity 6/6, etc.).

---

## 6. Per-capability section wedges (the specific angle each section should take)

### 6.1 Nuclei vs deep learning
- Report in *their* dialect first: AP/F1 swept across IoU thresholds on BBBC039/DSB2018, so a reviewer
  places fluorostats on the familiar axis before you introduce field metrics.
- Lead the framing mechanistically (Omnipose-style): name the axis where you win — *field/population
  metrics + separated-instance accuracy at zero training cost* — not "DL is heavy."
- Scope boundary (BBBC024) as the crossover curve; state it's a limit of the *whole non-DL class*, not
  fluorostats specifically.
- **Do not hide** the crowded-regime collapse — it's the paper's honest scope statement and, per the
  reproducibility study (Metrics Reloaded; "rankings should be interpreted with care"), enumerating
  failure modes reads as rigor in a top venue.

### 6.2 Vascular
- **Adopt REAVER's protocol verbatim**: one shared 36-image dataset, all tools through your unified
  quantification code, default parameters for all (declared as a limitation).
- **Split accuracy vs precision + a Bonferroni zero-bias test** — "ties AngioTool" becomes
  "statistically indistinguishable error distributions, and *unbiased* where a specialist is biased."
- 3D: the VesselExpress software-agreement as **Spearman + Dice + Bland-Altman with the offset named
  mechanistically** (Li-threshold more inclusive), plus the synthetic-phantom exact-GT table. Follow
  VesselExpress's four-pronged no-GT validation (phantom + literature values + inter-tool agreement +
  biological discrimination — the SproutAngio VEGF dose-response).
- Where a specialist wins (segmentation Dice, diameter), report your error *within the spread of manual
  inter-annotator disagreement* (VesSAP framing) — turns a loss into an adequacy signal. Concede diameter
  is the least reliable metric (every vascular paper does).

### 6.3 Viability
- **Prove the "2D biases viability +5–25%" headline the Theart way**: identical volumes through 3D vs
  2D/MIP/single-slice, paired per-sample deltas, report *direction + count of samples affected* (e.g.
  "MIP inflated live fraction in 27/30 stacks, median +11%"), not just a mean, with p-values.
- **Separate optical from biological with a known-composition phantom** (fixed true live fraction → any
  2D deviation is provably optical). This is the field's weak spot (Mali, Mountcastle assert biology
  without ruling out optics) and your strongest differentiator — standalone figure.
- **Tie-to-Fiji-macro led by CCC + Bland-Altman** (this literature underuses both; your CCC 0.987 is
  stronger than most exemplars report) + per-image overlays (Kerkhoff's signature).
- Narrative arc: gap (3D catches what 2D misses) → existing count tools live only in 2D → your maxima
  mode ties the macro *exactly* and extends to depth-resolved 3D. Reproducing the tool is the credibility
  down-payment that licenses the novel 3D claim.
- "No counting method is universal" → a **regime decision-guide table** (mode × density/depth/SNR),
  presented as actionable guidance, pre-empting "why three methods?"

### 6.4 Homogeneity + statistics
- **The Martin-gap wedge**: the closest state-of-the-art simple-index paper (Martin et al., dispersion
  indices, iScience 2026) *never validated against any rigorous point-pattern statistic* and shipped no
  Python / no significance layer. State this explicitly. fluorostats validates against **five** statistics
  (|ρ| 0.96–0.997) with **AUC 1.0** and integrated non-parametric stats in Python.
- Flagship figure = the spatstat/Amgad grammar: point-pattern panels (regular / Poisson / clustered)
  beside index values, with a CSR reference; then the five-statistic correlation table; then the ROC/AUC
  panel (which dispersion-index papers lack).
- Position the **integrated stats layer** with SuperPlots/Lazic "correct-by-default" rhetoric: open with
  a concrete failure cost (pseudoreplication false-positive inflation at small n), pitch fluorostats as
  making the correct choice (MWU + Cliff's δ + BH-FDR + bootstrap CIs, stratified/Scheirer-Ray-Hare) the
  default path. A Lazic-style remedy table (scenario → default test). This is a genuine differentiator —
  no general bioimage platform ships an integrated stats layer.

---

## 7. Performance section (§4.7)
Separate from correctness (scikit-learn split). Comparative timing table vs named tools (you have it:
14.5 ms/img; 15× StarDist, 380× Cellpose), named hardware (cores/RAM, scanpy-style), plus one scaling
note. Add a **determinism panel**: bit-identical reruns / zero seed variance — a reproducibility result
DL tools structurally cannot show.

---

## 8. Scope & Limitations design (turn the honesty ledger into a strength)
- **Rename "honesty ledger" → "Scope & Limitations," Scope subsection first** (Metrics-Reloaded "problem
  fingerprint" idiom). State the operational envelope (modality, instance density, SNR, threshold regime)
  so each limitation reads as a boundary of a well-scoped tool, not a defect.
- **Each item = mechanism + quantified boundary + mitigation/guidance.** E.g. crowded-instance collapse →
  "CC labeling merges touching objects above ~N/field; use DL there; fluorostats flags the regime." Add a
  threshold-sensitivity *curve* (turns "threshold matters" into a characterization).
- **Separate method limits from evaluation limits** (Maier-Hein 2018): pipeline-generated VesselExpress GT
  and small-pilot power optimism are *evaluation-side* caveats; frame small-pilot optimism as a statistical
  caveat for the reader's own experiments, citing bioimage-stats guidance.
- **The caught-and-fixed benchmark-script bugs — disclose, reframed.** Advice from the reproducibility
  study: put them under a **"Benchmark correctness safeguards"** methods note, not a confession. For each:
  the safeguard that caught it (regression test / invariant), impact direction+magnitude, the fixing commit,
  and that all reported numbers use the audited version. Cite Sandve/Miura to position bug-auditing as best
  practice. Short and factual. In a paper whose thesis is radical honesty this is on-brand and differentiating —
  but it is *optional* and unusual, so it's a judgment call (Question 4).

---

## 9. Title / Abstract / Introduction (drafts to react to)

**Candidate titles** (recommend #1):
1. *fluorostats: a training-free, CPU-only Python library for reproducible fluorescence-microscopy quantification*
2. *fluorostats: training-free fluorescence quantification with parity-or-better accuracy and integrated statistics*
3. *fluorostats: open-source, training-free quantification of volume fraction, vascular architecture, and Live/Dead viability*
4. *Reference-exact, training-free quantification of fluorescence microscopy on a laptop CPU*

**Abstract skeleton (~150–200 w):** need opener (what these metrics decide) → gap A (fragmented single-purpose
tools, most needing GPUs/training; field images 3D but quantifies 2D) → gap B (stats done by error-prone manual
export) → contribution named inline (open-source Python library + CLI, training-free on CPU, integrated
non-parametric stats/power/figures) → validation-design sentence (every metric reference-exact; benchmarked vs
StarDist/Cellpose/REAVER/AngioTool/VesselExpress/Fiji-macro on public data) → headline result (matches-or-exceeds
across N metrics on CPU; delineates the crowded-instance regime where DL remains preferable) → close (availability,
license).

**Introduction (5 paragraphs):** ¶1 why these measurements matter (biology first; 3D imaging now routine) → ¶2
pain part 1: fragmentation + 3D→2D collapse (name incumbents generously) → ¶3 pain part 2: the statistics leak +
reproducibility (the "absence of…" statement of need) → ¶4 the contribution (training-free + CPU-only + reference-
exact; the module families + integrated stats) → ¶5 preview validation design + thesis + the honest DL-scope beat +
one-line roadmap.

---

## 10. Figure plan (from the figure-design study)

**Build the one figure you don't have — Figure 1, a pipeline schematic**: microscopy input (multi-modality
thumbnails) → boxed modules (segmentation-free quantification engine → the task families) → integrated stats/
figure output, with an inset visually signalling *training-free / no GPU*. (Cellpose-Fig-1 / QuPath-Fig-1 analog.)

**Consolidate the 13 existing plots into ~6 task-grouped main figures** (not by plot type):
- **Fig 2 — Nuclei accuracy:** F1 ranking bars + bootstrap-CI forest plot (panels a/b).
- **Fig 3 — Qualitative gallery:** raw / GT / prediction triplets across modalities (incl. VesselExpress
  overlays), scale bars + inset zooms, magenta/green not red/green.
- **Fig 4 — Vascular & structural:** REAVER ranking + homogeneity five-statistic correlation.
- **Fig 5 — Agreement & validation:** viability agreement (scatter/Bland-Altman) + 3D phantom exact-GT.
- **Fig 6 — Robustness & cost:** clustering-degradation crossover curve + timing log-plot + scope-boundary bars.
- Plus **Table 1** — master validation/leaderboard table (best value bold).
Push per-dataset breakdowns, redundant metrics, extra sweeps to Extended Data/Supplementary (the remaining ~7 plots).

**Standardize now:** a fixed **Okabe-Ito** colorblind-safe palette (fluorostats always the same hue), viridis/
cividis for continuous, redundant encoding (shape + line style), self-contained captions (bold title + per-panel +
n + statistic + error definition), Nature-style bold lowercase panel labels. *(The current figures use a cream
background and default styling — a restyle pass to this system is worth doing.)*

---

## 11. Reproducibility & availability (FAIR — do this before submission)
- **Versioned Zenodo DOI for the exact release** (GitHub link alone gets flagged by Nature/JOSS/GigaScience);
  OSI license for code (BSD/MIT/Apache — community norm) + CC for data; version/commit cited for every result.
- **Data availability**: deposit benchmark images/GT with DOIs; list all public accessions (BBBC039/024, DSB2018,
  CTC, REAVER, SproutAngio, VesselExpress-Zenodo, Kerkhoff-Zenodo).
- **Environment lockfile / container + recorded seeds + one-command figure-regeneration scripts** (Sandve rules).
  Foreground **deterministic CPU** as a design guarantee.
- **Complete a community reporting checklist** (Montero-Llopis/MicCheck fluorescence reporting + QUAREP/Schmied
  analysis-workflow) as a supplement — pre-empts reviewer reproducibility objections; signals community-standard
  compliance (FAIR4RS).
- A standardized **"Availability and requirements"** block (project name, URL, OS, Python version, deps, license,
  restrictions) — mandated by several software venues, expected by all.

---

## 12. Open decisions → questions for Roy
1. **Venue/format** — full method paper (recommended) vs application note + JOSS. Drives length & structure.
2. **Positioning emphasis** — lead with "parity-or-better vs DL" (bolder, benchmark-forward) vs "the reproducible
   general quantifier + integrated stats" (breadth/platform-forward, DL comparison as one section)?
3. **Bioprinting application vignette** — include a worked GelMA-vs-hybrid Live/Dead + vascular + homogeneity case
   (makes it citable by the extrusion paper; the bioprinting audience expects it) — but this may require the
   deferred final Live/Dead rerun. Include / omit / include-as-supplement?
4. **Disclose the caught-and-fixed benchmark bugs** as a "Benchmark correctness safeguards" note (on-brand for the
   honesty thesis, but unusual) — yes / no?
5. **Figure restyle** — restyle the 13 figures to the Okabe-Ito system and build the Fig 1 schematic — now or later?
6. **Main-vs-supplementary split** — confirm the ~6-main-figure grouping above, or adjust.
