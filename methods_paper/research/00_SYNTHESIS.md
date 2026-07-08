# fluorostats methods paper — research synthesis & benchmark plan

Master synthesis of 13 parallel literature dossiers (~145 verified references)
positioning **fluorostats** against existing methods, per capability. Each
category file (`01`–`13`) holds the full citation table, per-reference notes,
and verification flags. This file is the executive layer: the positioning
thesis, a parity/differentiator/limit matrix, the consolidated head-to-head
benchmark plan, and a proof-stage citation-verification checklist.

---

## 1. Central thesis (the paper's spine)

> Existing bioimage tools **measure** well but **stop at the measurement table**;
> the bioprinting/tissue field **images in 3D but quantifies in 2D**
> (z-collapsed projections, manual counts) and leaves statistics to error-prone
> manual export. fluorostats closes both gaps: it delivers **automated,
> depth-resolved, statistically-grounded 3D quantification** — segmentation →
> volume/topology/skeleton/object metrics → FOV-normalized densities →
> non-parametric statistics + effect sizes + FDR + power — as one reproducible,
> open, scriptable pipeline over the confocal images the field already collects.

The claim is **NOT** "we beat deep learning at segmentation." It is:
**parity on the core measurements + three genuine differentiators, packaged for
reproducibility.**

### The three defensible differentiators
1. **Integrated non-parametric statistics layer** (Mann-Whitney + Cliff's δ +
   BH-FDR across strata + bootstrap fold-change CIs + Stouffer + Scheirer-Ray-Hare)
   — no general bioimage platform ships this; §10.
2. **FOV-normalized, voxel-size-invariant densities** (per mm³) — fixes the
   documented digital-zoom/magnification reproducibility drift (Riley 2023); §09, §03.
3. **Depth-resolved 3D quantification of Live/Dead & networks** — 2D/MIP area
   ratios systematically overestimate viability in thick constructs and miss
   depth-dependent core death; §06, §13. **This is the strongest single "we catch
   more" figure.**

Plus two supporting differentiators: **bootstrap power from pilot data with
joint BH-FDR power across metrics** (§11), and **matched-settings reproducible
3D renders** for fair comparison figures (§12).

---

## 2. Per-capability matrix: parity, differentiator, honest limit

| Capability | Field standard / comparator | fluorostats stance | Honest limit |
|---|---|---|---|
| **Platforms** (§01) | Fiji, CellProfiler, ilastik, QuPath, napari, scikit-image | Parity on measurement; unique stats+figures layer | Weaker ecosystem, no ML segmentation, lower throughput |
| **Segmentation** (§02) | Otsu/Li (classical); Cellpose, StarDist, U-Net (DL) | Parity on foreground Dice & volume fraction for separated/moderate density; training-free, deterministic, CPU | DL wins on touching/dense instance separation |
| **Vascular networks** (§03) | AngioTool, Angiogenesis Analyzer, REAVER (2D); VesSAP, VesselExpress (3D) | Matches 2D junction/branch/length; extends to true 3D + connectivity + FOV norm | REAVER reports diameter; 3D tools win at organ scale |
| **Skeleton** (§04) | AnalyzeSkeleton, skan (both Lee-1994 thinning) | **Provable numerical parity** (shared algorithm) + stats layer | No added skeleton rigor over AnalyzeSkeleton itself |
| **Topology** (§05) | BoneJ (Euler/Conn.D), MitoGraph PHI, persistent homology | Makes χ + component count + LCC fraction routine & comparative | No edge correction; scalar χ, not persistence diagrams |
| **Viability** (§06) | Manual/2D ImageJ Live/Dead area ratios | **Depth-resolved 3D live fraction catches core death 2D misses** | Calcein = metabolic viability, not architecture; needs per-z attenuation control |
| **Nuclei** (§07) | StarDist-3D, Cellpose, watershed, 3D Objects Counter | Parity when sparse; fast, training-free, stats-integrated | CC labeling under-counts touching nuclei in dense tissue |
| **Homogeneity** (§08) | Gini/CV on pixels (Martin 2026); Ripley's K, lacunarity | Adds spatial **tiling** (lateral uniformity) to segmentation-free Gini/CV | Single fixed tile scale; no CSR significance test |
| **Volume fraction** (§09) | Design-based stereology (gold standard); µCT BV/TV | Delesse-Glagolev-valid voxel counting + FOV normalization | Not unbiased design-based stereology; threshold-sensitive |
| **Statistics** (§10) | R/Prism/statsmodels (external, manual) | Correct-by-default small-n non-parametric stats in-pipeline | Not a full stats package (no mixed models, BCa) |
| **Power** (§11) | SIMR, FDRsamplesize2 (external, GLMM/parametric) | Pilot-driven nonparametric bootstrap power + joint FDR power, in-workflow | Small pilots → optimistic, wide-CI curves |
| **Visualization** (§12) | Imaris (gold, closed/paid), ClearVolume, napari, Vaa3D | Reproducible, matched-settings publication isosurfaces | No interactive/GPU volume rendering |
| **Application** (§13) | GelMA/hybrid bioprinting papers (qualitative imaging) | Fills the field's self-acknowledged quantification gap | — |

---

## 3. Consolidated benchmark plan (head-to-head, same images)

Every category proposed a benchmark; they collapse into **six experiments**.
Common design principle from §01: **agreement via Bland-Altman (bias + 95% LoA)
+ Lin's CCC / ICC + Spearman**, not just correlation. Use public ground-truth
datasets where they exist.

**B1 — Correctness against reference implementations (validity anchor).**
- Skeleton: fluorostats vs Fiji AnalyzeSkeleton on identical binary volumes +
  synthetic phantoms with known counts → expect **exact integer agreement**
  (shared Lee-1994 algorithm), <1–2% length deviation. (§04)
- Topology: fluorostats Euler number vs BoneJ + scikit-image on phantoms with
  analytically known χ (ball χ=1; k tunnels χ=1−k; N balls χ=N) → **zero-error
  pass criterion**; pin 6- vs 26-connectivity. (§05)
- LCC fraction vs MitoGraph PHI within numerical tolerance. (§05)

**B2 — Segmentation / signal-capture agreement (the "catch more" core).**
On Cell Tracking Challenge 3D fluorescence stacks (Fluo-C3DL-MDA231,
Fluo-C3DH-A549, dense Fluo-N3DL-TRIC) + DSB2018/BBBC: fluorostats vs
Otsu-in-Fiji vs Cellpose-3D vs StarDist-3D. Report (1) voxel Dice/IoU foreground,
(2) volume fraction Bland-Altman vs truth, (3) per-instance SEG (reported
honestly — fluorostats lags on dense), (4) CPU vs GPU runtime. Hypothesis stated
up front: **parity on foreground Dice + volume fraction for separated/moderate
density; lags only on per-instance SEG for dense tissue.** (§02, §07)

**B3 — 2D-vs-3D viability (the headline "we catch more" figure).**
Same Live/Dead confocal stacks spanning a death gradient: fluorostats 3D live
volume fraction vs mid-plane 2D area ratio vs MIP area ratio. Predicted: **2D/MIP
overestimate viability, and the overestimate grows with construct thickness**;
per-z live-fraction curve reveals monotonic depth-dependent death 2D misses.
Pair with attenuation control (per-z normalization) to prove the gradient is
biological, not optical. (§06, §13)

**B4 — Vascular network parity then extension.**
AngioTool + Angiogenesis Analyzer vs fluorostats on the SAME 2D network images
(tube-formation / Z-projections): junctions, branch/segment count, total length
via ICC + Bland-Altman → parity. Then show 3D recovers junctions/branches lost
to Z-projection overlap + connectivity with no 2D analog. Quantify "catches more"
as a **systematic offset**, not an assertion. (§03)

**B5 — FOV-normalization reproducibility (differentiator #2).**
Image one sample at two digital zooms / voxel sizes: raw counts diverge, but
FOV-normalized per-mm³ densities stay stable. Directly reproduces + solves the
Riley 2023 magnification-sensitivity finding. Pair with volume fraction vs
blinded Cavalieri/Delesse point counting (Fiji Stereology) → CCC within the
stereological CE. (§09)

**B6 — Homogeneity metric validation on synthetic controls.**
Parametric clustering sweep (jittered lattice → Poisson CSR → Thomas/Matérn
cluster), rendered as fluorescence: fluorostats 8×8 tile Gini/CV rises
monotonically with the clustering parameter and correlates (Spearman) with
Ripley's K deviation and (1−NN index); high AUC separating uniform vs clustered.
Include a tile-size sensitivity sweep (4×4→32×32) documenting the scale-dependence
limit. (§08)

**Validation-only (no comparator): power calibration.** Simulate pilots from
known non-normal effect sizes; confirm predicted power matches empirical
rejection rate over many full-size datasets, and joint FDR power matches realized
average power. (§11)

---

## 4. Closest prior art per capability (the citations that matter most)

| Capability | Single closest existing method to cite | Why it matters |
|---|---|---|
| Topology/LCC | **MitoGraph PHI** (Harwig 2018) | PHI ≈ fluorostats LCC fraction — proves the metric is established & biologically meaningful |
| Homogeneity | **Martin 2026 (iScience)** | Gini/CV on raw pixels, segmentation-free — direct methodological ancestor; fluorostats adds tiling |
| Skeleton | **skan (Nunez-Iglesias 2018) + AnalyzeSkeleton (Arganda-Carreras 2010)** | fluorostats builds on skan → provable parity |
| Volume fraction | **Delesse-Glagolev principle + Riley 2023** | Theoretical validity + the exact reproducibility gap FOV-norm solves |
| Viability gap | **Spiller & Duarte Campos 2025; Avnet 2024** | Peer-reviewed statements that the field NEEDS this tool |
| Euler in 3D fluorescence | **Chang 2021 (Theranostics, retinal light-sheet)** | Precedent for Euler number discriminating conditions in 3D fluorescence |
| Stats gap | **SuperPlots (Lord 2020); Lazic 2018** | Canonical pseudoreplication problem fluorostats' stratified design addresses |

---

## 5. Proof-stage citation verification checklist

All 13 agents were instructed to never fabricate DOIs and to flag anything not
fully fetched. Aggregated items to confirm before manuscript submission:

- **§01:** napari has no journal methods paper (cite Zenodo concept DOI / 2022
  conf. abstract); Bland-Altman 1986 Lancet DOI not fetched directly.
- **§02:** cite 3D StarDist as **Weigert 2020 (WACV)** not the 2018 2D MICCAI
  paper; Cellpose3 (2025) pagination unconfirmed; citation counts are
  order-of-magnitude.
- **§03:** 7 of 9 DOIs (Angiogenesis Analyzer, RAVE, Q-VAT, VesselExpress,
  VesSAP, VESNA, cleared-organ) from metadata — confirm DOIs + complete author
  lists.
- **§04:** Blum 1967 page range; MitoGraph pages (77–93); 3DVascNet author list;
  actin-tools Biology Open 2024 authors.
- **§05:** connEulor pages; Armstrong 2019 author order; Schlüter 2014 (not
  fetched); Ashworth co-authors; Robins 2016 volume/DOI; Chang 2021 full author
  list.
- **§06:** IOP review (Cadena 2024) author list provisional (URL 403); Roerdink &
  Bakker 2004; QuantICV 2020 author list/volume.
- **§07:** Malpica 1997 legacy DOI; NuMorph 2021 article number/author order;
  ClearMap (Renier 2016) unverified optional ref.
- **§08:** Martin 2026 iScience final volume/page at typeset; Liu & Xu ASME DOI
  confirmed via Crossref but 403 to scraper (not unverifiable).
- **§09:** Delesse 1848 / Glagolev 1933 primaries (cite via modern review);
  Peterson 2001 venue/DOI; Bouxsein 2010 µCT DOI/pages.
- **§10:** Cliff 1993 DOI (confirm vs APA/PsycNet); Stouffer 1949 is a book —
  cite Zaykin 2011 / Lipták 1958 for weighted-Z; minimal-reporting + rigor-guide
  author lists (#10–12).
- **§11:** Beasley et al. bootstrap-power (UNT handle, author/venue/year
  unconfirmed); Cohen 1988 is a book (ISBN not DOI).
- **§12:** napari DOI (Cambridge Core 500 on re-fetch — use Zenodo record);
  ParaView/VTK handbook chapters (ISBN, page numbers).
- **§13:** refs 2, 6, 12, 14, 15 author lists unconfirmed; ref 13 (microfluidic
  endothelial networks 2022) **ResearchGate-only — do not cite until
  publisher-verified**; refs 5, 9, 10 venue/DOI checks.

**No fabricated DOIs, authors, or titles were reported by any agent.** Every
uncertain field was flagged rather than filled in.

---

## 6. Recommended methods-paper structure

1. **Intro / gap** — lead with §13 (field images 3D, quantifies 2D) + §10
   (stats left to error-prone manual export). Cite Spiller 2025, Avnet 2024,
   SuperPlots, Lazic 2018.
2. **Design & implementation** — pipeline modules mapped to §01–§12 lineage
   (scikit-image, skan, Lee-1994, Otsu, BH-FDR, marching cubes).
3. **Validation** — B1 (correctness anchors: exact parity vs AnalyzeSkeleton /
   BoneJ / phantoms).
4. **Benchmarks** — B2 (segmentation agreement), B4 (vascular parity→extension),
   B5 (FOV reproducibility), B6 (homogeneity on synthetic controls).
5. **The "catch more" results** — B3 (2D-vs-3D viability, the headline figure).
6. **Statistics & power** — §10 differentiator + B-power calibration.
7. **Application** — the GelMA vs hybrid dataset already analyzed.
8. **Honest limitations** — dense-tissue instance separation (DL wins), no
   design-based stereology, single-scale homogeneity, small-pilot power.

---

*Source dossiers: `01`–`13` in this directory. ~145 references, DOI-tabled,
verification-flagged. Generated by 13 parallel research agents.*
