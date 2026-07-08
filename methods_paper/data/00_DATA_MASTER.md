# Master benchmark data table

Synthesis of 16 data + competitor-benchmark dossiers (all URLs fetched/verified
2026-07-08 by the agent fleet). This is the actionable index: what to download,
for which benchmark, against which competitor, and the number to beat.

**Recurring honesty constraint (all agents flagged it):** fluorostats is NOT an
instance segmenter. It reports **foreground Jaccard/Dice + object count +
volume/area fraction + skeleton/topology metrics** — NOT instance AP or CTC
SEG/DET/TRA. We cite competitor instance scores as *context*, never claim a
leaderboard rank, and compete on the semantic/foreground/network metrics we
legitimately compute.

---

## Tier A — download first (small, verified, highest value)

| Dataset | For | Competitor + number to beat | Size / license | URL |
|---|---|---|---|---|
| **REAVER Vascular Networks** ⭐ | **B4** vascular | REAVER (Corliss 2020) — beat its MAE vs manual on length density & branchpoints (REAVER cut AngioTool's error −76.5% length, −94.6% branchpoints) | 63 MB · CC-BY-4.0 · 36 imgs, manual GT (red=mask, green=skel) | `https://zenodo.org/records/3340165/files/REAVER_Vascular_Networks_Image_Dataset.zip` |
| **BBBC039** U2OS nuclei ⭐ | **B2** nuclei | StarDist / Cellpose / CellProfiler (CP F1≈0.82@IoU0.5) — score all vs GT | 78+3 MB · CC0 · 200 FOV, ~23k instance masks + counts | `https://data.broadinstitute.org/bbbc/BBBC039/images.zip` + `/masks.zip` |
| **VesselExpress test.tiff** | **B4** 3D vascular | VesselExpress (Spangenberg 2023) — length/junction density per mm³ | 48 MB · CC-BY-4.0 · 100×500×500, voxel 2.0×1.016×1.016 µm | `https://raw.githubusercontent.com/RUB-Bioinf/VesselExpress/master/VesselExpress/data/test.tiff` |
| **CTC Fluo-C3DH-A549** | **B2** 3D seg | CTC top SEG=0.9083 (context only) — report foreground Jaccard + volume fraction | 244 MB · confocal, silver GT | `http://data.celltrackingchallenge.net/training-datasets/Fluo-C3DH-A549.zip` |
| **CTC Fluo-N3DH-CHO** | **B2** 3D seg | CTC top SEG=0.9248 (context) | 98 MB | `http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CHO.zip` |
| **AngioTool test images** | **B4** vascular (home turf) | AngioTool (Zudaire 2011) — reproducibility only, no pixel GT | small · hindbrain/retina/allantois | `https://ccrod.cancer.gov/wiki-html/ROB2/Downloads_62196327.html` (resolve zip links) |

## Tier B — 3D synthetic / controlled (foreground + count, no instance seg needed)

| Dataset | For | Note | URL |
|---|---|---|---|
| **BBBC024** 3D synthetic HL60 | B2 seg + count | 120 imgs, exactly 20 nuclei/img, clustering 0–75%, CC BY 3.0 | `https://bbbc.broadinstitute.org/BBBC024` |
| **BBBC027** 3D synthetic colon | B2 seg | FG/BG + counts | `https://bbbc.broadinstitute.org/BBBC027` |
| **BBBC032** mouse blastocyst | B2 3D seg | real confocal, 3D nuclei masks, CC0, 1.14 GB | `https://data.broadinstitute.org/bbbc/BBBC032/BBBC032_v1_dataset.zip` |
| **VascuSynth** synthetic trees | B1 skeleton | 120 vols, bifurcation/branch GT | `https://vascusynth.cs.sfu.ca/Data.html` |

## Tier C — application-matched (B3 viability, larger)

| Dataset | For | Note | URL |
|---|---|---|---|
| **S-BIAD2130** ⭐ | **B3** viability | 3D bioprinted tumouroids, **Calcein-AM + EthD-1 Live/Dead z-stacks**, CC0, ~3 GB each | `https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2130` |
| **S-BIAD2215** | B3 viability | Calcein-AM viability, confocal z-series (17 planes @50 µm), 3D collagen, CC-BY | `https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2215` |
| **S-BIAD2920** | B4 application | endothelial self-organization / vessel formation, Leica confocal, 556 files, CC-BY | `https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2920` |
| **Zenodo 5089728** (Barbier 2017) | B3 attenuation | light-attenuated 3D confocal spheroid stacks — physical basis for 2D-vs-3D discrepancy, CC0, ~12 GB | `https://zenodo.org/records/5089728` |

## Tier D — topology / volume-fraction real data (mostly synthetic wins)

| Dataset | For | Note |
|---|---|---|
| **Synthetic phantoms** ⭐ | B1 topology + skeleton, B5 VF | **DONE — passing.** Exact zero-error ground truth; reproduces BoneJ's own validation |
| Digital Rocks DRP-372 (Santos 2022) | B1/B5 real | binary volumes with published porosity + Euler characteristic, CC-BY, >1 TB (subset) — resolve via DOI `10.17612/93pd-y471` |
| foam_ct_phantom (Pelt 2022) | B5 | analytic-porosity foam, conda-installable, Zenodo 3726909 |

---

## Competitor → dataset → metric → number-to-beat

| Competitor | Their benchmark data | Metric | Published number | fluorostats fair metric |
|---|---|---|---|---|
| **REAVER** | its own 36-img Zenodo set | MAE vs manual | −76.5% length, −94.6% branchpoints (vs AngioTool) | total length, junctions=branchpoints, area fraction — **beat MAE** |
| **AngioTool** | hindbrain/retina/allantois | reproducibility (no pixel GT) | none | junctions, length, area fraction (agreement) |
| **StarDist** | DSB2018 (BBBC038) | instance AP@0.5 | 0.864 | foreground Dice/IoU + count (NOT AP) |
| **Cellpose** | own dataset (gated, CC-BY-NC) | instance AP | ~0.8 (fig only, unverified) | foreground Dice + count |
| **CellProfiler** | BBBC039 / DSB2018 | F1@IoU0.5 | ~0.82 (unverified) | foreground + count + area fraction |
| **AnalyzeSkeleton** | Fiji "Bat Cochlea Volume" | branch/junction/length | exact (shared Lee-1994 algo) | **require exact match** |
| **BoneJ** | self-gen synthetic solids | Euler χ / Conn.D | "without fault" | **exact match on phantoms — DONE** |
| **MitoGraph** | budding-yeast + tube phantoms | length/volume, ~96% | ~95.9% reproducibility | LCC fraction = PHI, total length |
| **VesselExpress** | in-repo test.tiff | length/junction density | cortex 4.8 µm caliber, 1–2% vasc vol | length density, junction density per mm³ |
| **CTC methods** | CTC 3D fluorescence | SEG (Jaccard) | 0.71–0.92 per set | foreground Jaccard (context only) |

---

## Repository access (for deeper mining)

**Most fruitful: EMBL-EBI BioImage Archive** — only verified source with our exact
stains (Calcein-AM / EthD-1 z-stacks). BioStudies JSON API:
`https://www.ebi.ac.uk/biostudies/api/v1/BioImages/search?query=<term>`;
downloads at `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/<last3>/<ACC>/`.
IDR (OMERO JSON API) is the curated secondary. Zenodo/figshare/Dryad fill
nuclei/format needs.

**Confirmed gaps:** no deposited raw Live/Dead z-stacks of *GelMA* constructs;
no in-vitro Matrigel tube-formation assay sets. Closest: S-BIAD2130 (bioprinted
Live/Dead) and S-BIAD2920 (vessel formation).

---

## Execution order

1. **Tier A downloads** (running now) → B4 REAVER + B2 BBBC039 head-to-heads first.
2. **Fiji** (resolve current URL) → AnalyzeSkeleton exact-match + BoneJ (extends
   the passing B1 phantoms to "vs the field-standard tool").
3. **StarDist install** + DSB2018 → complete B2 nuclei tri-tool (fluorostats vs
   StarDist vs Cellpose vs manual, density-stratified).
4. **S-BIAD2130** (Tier C, larger) → B3 viability headline on public bioprinted
   Live/Dead data.
5. Each wired through `../benchmarks/agreement.py` (Bland-Altman + CCC + ICC).
