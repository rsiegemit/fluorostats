# Open Bioimage Repositories: A Practical Mining Guide for fluorostats Benchmarking

**Purpose.** Locate public, citable, downloadable confocal fluorescence z-stack data to benchmark **fluorostats**. Target modalities: Live/Dead viability (Calcein-AM / Ethidium homodimer / propidium iodide), endothelial network & tube-formation assays, bioprinted constructs (GelMA / hydrogel), and nuclei/DAPI staining.

**Constraint.** Public, citable, downloadable data only — no private data.

**Verification status.** All URLs and accessions below were fetched and confirmed to resolve on 2026-07-08. API query examples were executed and returned the counts/records quoted. Items we could not verify are explicitly flagged.

---

## TL;DR — Where to look first

| Modality | Best repository | Concrete first pull |
|---|---|---|
| Live/Dead (Calcein-AM + EthD) z-stack | **BioImage Archive** | S-BIAD2130 (CC0) |
| Calcein-AM viability, confocal z-series | **BioImage Archive** | S-BIAD2215 (CC BY 4.0) |
| Endothelial / vessel formation, confocal | **BioImage Archive** | S-BIAD2920 (CC BY 4.0) |
| Nuclei, 3D fluorescence z-stack | **BBBC** / **Cell Tracking Challenge** | BBBC050 / Fluo-N3DH-CHO |
| Nuclei/DAPI + bioprinted Live/Dead | **Dryad** | doi:10.5061/dryad.gf1vhhn0m (CC0) |
| Generic confocal z-stack (format fixture) | **figshare / Zenodo** | figshare 12387629 / Zenodo 437943 |

**The single most fruitful repository for our exact modalities is the EMBL-EBI BioImage Archive** — it is the only source with verified confocal fluorescence z-stacks carrying our precise stains (Calcein-AM / Ethidium homodimer live/dead), plus endothelial confocal and bioprinted-construct studies, all wget/curl/S3-downloadable under CC0 / CC BY. IDR is a strong secondary (rich OMERO JSON API, curated published studies). The general repos (Zenodo, figshare, Dryad) fill nuclei/DAPI and format-fixture needs but are sparse for Live/Dead + GelMA + tube-formation specifically.

---

## 1. EMBL-EBI BioImage Archive  ★ primary

- **URL:** https://www.ebi.ac.uk/bioimage-archive/
- **Holds:** Primary archive for biological light & electron microscopy image data tied to publications — fluorescence, confocal, high-content screening, light-sheet. Accession format `S-BIAD####` (also `S-JCBD-*` for Journal of Cell Biology deposits).
- **Scale:** Thousands of studies (accessions observed well above S-BIAD3600 in mid-2026), many TB. Shares BioStudies infrastructure.

### Search
- **Web UI:** faceted search; filter by collection, imaging method, organism.
- **API (BioStudies REST, returns JSON — verified):**
  - Search: `https://www.ebi.ac.uk/biostudies/api/v1/BioImages/search?query=Calcein` → `totalHits` + accession/title list. Add `&pageSize=N`.
  - Full study record: `https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD2130`
  - Download links: `https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD2130/info` → JSON with ftpLink / httpLink / globusLink.
  - Human-readable page (JS-rendered; use API for scripting): `https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2130`

### Download (all verified)
- **HTTPS (wget/curl-able):** `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/<last3digits>/<ACCESSION>/` — e.g. `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/130/S-BIAD2130/`. Files under `.../Files/`.
- **FTP (anonymous):** `ftp://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/130/S-BIAD2130/`
- **S3:** endpoint `https://uk1s3.embassy.ebi.ac.uk/`, bucket `bia-integrator-data` — `aws --endpoint-url https://uk1s3.embassy.ebi.ac.uk s3 ls s3://bia-integrator-data/<ACCESSION>/`
- **Globus:** collection origin_id `47772002-3e5b-4fd3-b97c-18cee38d6df2`
- **Aspera:** `fasp-public@fasp.ebi.ac.uk` port 33001

> Note: the path mode segment can be `fire` or `nfs` — read it from the `/info` endpoint rather than hardcoding.

### Licensing
Per-study, predominantly **CC0** or **CC BY 4.0** (license is a field in the study JSON).

### Best search terms
`Calcein`, `live/dead`, `Ethidium Homodimer`, `viability`, `confocal`, `bioprinted`, `hydrogel`, `endothelial`, `tube formation`, `organoid`, `DAPI`, `nuclei`.
**Flag:** `GelMA` returns **0 hits** — search `hydrogel` / `bioprinted` instead.

---

## 2. Image Data Resource (IDR)  ★ strong secondary

- **URL:** https://idr.openmicroscopy.org/
- **Holds:** Curated public repository of image datasets from published studies, built on OMERO. Screens (high-content) + projects/datasets (experiments). Study names follow `idrXXXX-author-topic` (e.g. `idr0026-weigelin-immunotherapy`).
- **Scale:** ~100+ curated studies; large per-study image counts (some projects >12,000 images).

### Search
- **Web UI:** search/browse at the landing page; each study has a landing page and thumbnail gallery.
- **API (OMERO REST/JSON — verified):**
  - Containers overview: `https://idr.openmicroscopy.org/webclient/api/containers/?group=-1` → `projects`, `screens`, `plates` arrays with `id`, `name`, `childCount`.
  - Projects list: `https://idr.openmicroscopy.org/api/v0/m/projects/?limit=200` → project names + IDs.
  - Also accessible via OMERO API clients in Python, R, Java, MATLAB, and REST/JSON.
  - Migrating to OME-Zarr; non-Zarr images remain downloadable for local viewing.

### Download
Per-image / per-dataset via the OMERO API and web client; OME-Zarr and original-file export. Use the project/dataset IDs from the API to drive downloads.

### Licensing
**CC BY 4.0** for content (OMERO software is GPL).

### Best search terms
`endothelial`, `angiogenesis`, `vascular`, `tumour`, `organoid`, `light-sheet`, `confocal`, `3D`, `viability`. Note web search engines index IDR study accessions poorly — enumerate via the API rather than Google.

---

## 3. Broad Bioimage Benchmark Collection (BBBC)

- **URL:** https://bbbc.broadinstitute.org/ — index: https://bbbc.broadinstitute.org/image_sets
- **Holds:** Curated catalog of ~50+ microscopy image sets (BBBC001–BBBC051+), each paired with ground truth (masks, counts, outlines, labels). Strong on fluorescence 2D nuclei; a few 3D / synthetic-3D sets.
- **Scale:** ~50 sets.

### Search
No API — static curated HTML list, grouped by ground-truth type. Each set at `https://bbbc.broadinstitute.org/BBBC###`.

### Download (verified)
Direct wget/curl-able zips: `https://data.broadinstitute.org/bbbc/BBBC###/<filename>.zip`. **Filenames are not uniformly predictable** — read each set's page (e.g. BBBC021 uses `BBBC021_v1_images_Week1_22123.zip`). Do not guess filenames.

### Licensing
Varies per set (CC BY 3.0; CC BY-NC-SA 3.0 for BBBC020; **BBBC021 is AstraZeneca copyright with no open license — flag before redistribution**). Always cite Ljosa et al., *Nature Methods* 9(7):637, 2012.

### Best search terms (browse index)
`nuclei`, `DAPI`, `U2OS`, `3D`, `fluorescence`. **No Live/Dead, tube-formation, or GelMA sets exist in BBBC.**

---

## 4. Cell Tracking Challenge (CTC)

- **URL:** https://celltrackingchallenge.net/ — 3D: https://celltrackingchallenge.net/3d-datasets/
- **Holds:** ~10 3D+time and ~10 2D+time time-lapse microscopy datasets (real + simulated), with train/test splits and reference annotations. Naming encodes modality: `Fluo-` = fluorescence, `N` = nuclei, `3D`.
- **Scale:** ~20 datasets.

### Search
Browse-only via the dataset pages; no query API. Registration only for leaderboard submission, not downloads.

### Download (verified)
Direct curl-able zips: `https://data.celltrackingchallenge.net/training-datasets/<CODE>.zip` and `.../test-datasets/<CODE>.zip`. Verified: `Fluo-N3DH-CHO.zip` (~108 MB), `Fluo-N3DH-CE.zip` (~3.4 GB).

### Licensing
Free to download for the challenge; **no explicit open license printed on the 3D page — flag: confirm reuse terms per dataset before redistribution.** Cite Maška et al., *Nature Methods* 2023.

### Best terms / match
`Fluo-N3DH-*` (fluorescent nuclei, 3D). Only nuclei/DAPI-adjacent modality matches our list.

---

## 5. Zenodo

- **URL:** https://zenodo.org/ — REST API: `https://zenodo.org/api/records`
- **Holds:** Millions of general research records (datasets, figures, software). Broad but noisy; format/quality varies per record.

### Search (REST API — verified)
- `GET https://zenodo.org/api/records?q=<query>&size=<n>&type=dataset&sort=bestmatch` (add `&communities=<slug>` to scope).
- Response: `hits.total` and `hits.hits[]`; each hit has `id`, `doi`, `metadata.title`, `metadata.license.id`, `links.self_html`, `files[]`.
- **Gotchas:** a literal `/` in `q` (e.g. `Live/Dead`) triggers a **500** (Lucene syntax) — quote it (`"live/dead"`) or escape. Loose multi-word queries inflate totals (161k+ hits); use quoted phrases + `AND` + `type=dataset`.

### Download (verified)
- API: `https://zenodo.org/api/records/{id}/files/{filename}/content`
- Human/curl: `https://zenodo.org/records/{id}/files/{filename}` (302 → content). Percent-encode spaces (`%20`).

### Licensing
Per-record (`metadata.license.id`); relevant hits mostly `cc-by-4.0`.

### Best search terms
`"live/dead" AND confocal AND (hydrogel OR bioprint*)`; `GelMA AND (confocal OR "z-stack" OR fluoresc*)`; `(angiogenesis OR "tube formation" OR HUVEC OR endothelial) AND confocal`; `DAPI AND nuclei AND "z-stack" AND confocal`.

---

## 6. figshare

- **URL:** https://figshare.com/ — API base: `https://api.figshare.com/v2/`
- **Holds:** General-purpose open repository; heavily used as journal "source data" host. Bioimage content is scattered — often supplementary z-stacks attached to papers, not purpose-built sets.

### Search (API)
- POST: `POST https://api.figshare.com/v2/articles/search` body `{"search_for": "confocal z-stack"}` (JSON; POST-only, GET returns 404).
- GET convenience (verified): `https://api.figshare.com/v2/articles?search_for=confocal&page_size=5`.
- Single article JSON (verified): `https://api.figshare.com/v2/articles/{id}` → `files[]` with `download_url` + `size`. No token needed for public reads.

### Download (verified)
`https://ndownloader.figshare.com/files/{file_id}` (302 → signed AWS S3 URL; no token for public files).

### Licensing
Predominantly **CC-BY 4.0** (default) and **CC0**; per-article in JSON.

### Best search terms
`confocal z-stack`, `Live/Dead calcein`, `endothelial confocal`, `GelMA bioprinting`, `DAPI nuclei confocal`; use `:title:` field-scoped queries to cut noise.

---

## 7. Dryad

- **URL:** https://datadryad.org/ — API base: `https://datadryad.org/api/v2/`
- **Holds:** Curated (human-reviewed) general research data; strong life sciences. Datasets packaged as versioned bundles (ZIP + README). Confocal content is supplementary but real.
- **Scale:** ~149 datasets match "confocal"; "bioprinting" = 5; "GelMA" = 0 (verified via API `total`).

### Search (API — verified)
- Search: `https://datadryad.org/api/v2/search?q=confocal` → `count`, `total`, `_embedded["stash:datasets"]`.
- Metadata: `https://datadryad.org/api/v2/datasets/{encoded-doi}` (DOI encoded as `doi%3A10.5061%2Fdryad.<suffix>`).
- Versions: `https://datadryad.org/api/v2/datasets/{encoded-doi}/versions` → `stash:download` href.

### Download (verified, with caveat)
- Dataset-level `/download` returned **HTTP 401** (needs a free Bearer token, ~10 h lifetime).
- Reliable programmatic path: **version-level** href `https://datadryad.org/api/v2/versions/{version_id}/download`.
- **Simplest token-free route:** the "Download dataset" button on the web UI page. **Flag:** budget for the token step if fully scripting Dryad.

### Licensing
**CC0 1.0** is the Dryad standard (confirmed on datasets below) — most permissive for redistribution.

### Best search terms
`bioprinting`, `confocal`, `calcein`, `Live/Dead`, `DAPI actin`, `endothelial angiogenesis`. `GelMA` = 0; search `bioprinting` and read READMEs.

---

## 8. EMPIAR — for completeness (poor fit)

- **URL:** https://www.ebi.ac.uk/empiar/
- **Holds:** Raw images underpinning cryo-EM maps/tomograms, volume EM, soft/hard X-ray tomography. **Not confocal fluorescence light microscopy.** Accession `EMPIAR-#####`.
- **Scale:** 3,038 entries, ~8.94 PB (2026-07-08).
- **Download (per docs):** FTP/HTTPS `ftp://ftp.ebi.ac.uk/empiar/world_availability/<ID>/`; Aspera; Globus GridFTP; HTTP tarball only < 1.5 GB.
- **Licensing:** **CC0**.
- **Fit:** Essentially none for our modalities. Exclude unless EM is specifically wanted.

---

## 9. SSBD (Systems Science of Biological Dynamics, RIKEN) — partial fit

- **URLs (verified):** https://ssbd.riken.jp/repository/ and http://ssbd.qbic.riken.jp/repository/
- **Structure:** SSBD:repository (open archive, all bioimaging types tied to publications) + SSBD:database (curated quantitative BD5/BDML data).
- **Scale:** repository ~385 projects, 54.5 TB (July 2026); quantitative `/data/` tier 687 objects.
- **Search:**
  - Web UI fields: Organism, Project ID, Title, Description, Person, Method, Paper. Unified browser: https://ssbd.riken.jp/unified/projects.
  - **API (Django Tastypie, JSON — verified):** base `http://ssbd.qbic.riken.jp/SSBD/api/v3/`; docs http://ssbd.qbic.riken.jp/restfulapi/. Example: `http://ssbd.qbic.riken.jp/SSBD/api/v3/data/?format=json&limit=1` → `total_count: 687`. Filter: `...?description__icontains=confocal` → 17 hits. Python `Py_SSBDapi` / Java `Java_SSBDapi` clients (openssbd GitHub, GPL v3).
- **Download:** per-project from repository pages; image viewing via integrated SSBD:OMERO. Dataset IDs numeric (e.g. `dataset-12515`).
- **Licensing:** per-dataset, author-declared (e.g. CC BY-SA seen). Check each `license` field.
- **Fit / flag:** strengths are quantitative dynamics, tracking, simulation (C. elegans, zebrafish, Ca²⁺/ERK). **Could not verify** a specific SSBD dataset matching Live/Dead-Calcein, GelMA, or endothelial tube-formation — the image tier is JS-rendered and not enumerable via the tools here. Treat as unverified for our modalities; manual browse recommended if needed.

---

## Specific promising accessions to pull (all verified to resolve)

### BioImage Archive (best matches)

| Accession | Title | Modality | License | URL |
|---|---|---|---|---|
| **S-BIAD2130** | High-throughput 3D engineered paediatric tumour models for precision medicine | **Best.** Live/Dead: "10 µM Ethidium Homodimer-1 and 5 µM Calcein AM"; z-stack fluorescence (green=live, red=dead); 3D bioprinted tumouroids; Zeiss CellDiscoverer 7 | **CC0** | https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2130 |
| **S-BIAD2215** | 3D collagen high-throughput screen identifies drugs that induce epithelial polarity… in colorectal cancer | Calcein-AM (3.75 µM) viability; **confocal** (ImageXpress confocal HT.ai); z-series 17 planes @ 50 µm; 3D collagen | **CC BY 4.0** | https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2215 |
| **S-BIAD2920** | Laminin and Fibronectin Cooperate to Guide Endothelial Self-Organization During Intersegmental Vessel Formation | **Endothelial** self-organization / vessel formation; confocal (Leica Stellaris 5 CLSM); 556 files | **CC BY 4.0** | https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2920 |
| **S-BIAD1284** | Visualization of cellular membrane damage and apoptosis in murine precision cut lung slices after vibratome slicing | Live/dead-style membrane-damage viability (surfaced by Calcein search) — verify stain details before use | check record | https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1284 |

**Download example (S-BIAD2130):** HTTPS `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/130/S-BIAD2130/` (24 files, HTTP 200); FTP `ftp://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/130/S-BIAD2130/`.

### BBBC / Cell Tracking Challenge (nuclei ground-truth)

| ID | Title | Modality | License | URL |
|---|---|---|---|---|
| **BBBC050** | 3D nuclei in mouse embryos (H2B-mCherry) | 3D nuclei fluorescence z-stack (histone-tagged, not DAPI) | CC BY 3.0 | https://bbbc.broadinstitute.org/BBBC050 |
| **BBBC020** | Murine bone-marrow macrophages, DAPI + CD11b | DAPI nuclei (2D) | CC BY-NC-SA 3.0 | https://bbbc.broadinstitute.org/BBBC020 |
| **Fluo-N3DH-CHO** | CHO cell nuclei, 3D+time confocal | 3D fluorescent nuclei | see §4 flag | https://celltrackingchallenge.net/3d-datasets/ |

### Dryad / figshare / Zenodo (general repos)

| ID / DOI | Title | Modality | License | URL |
|---|---|---|---|---|
| **10.5061/dryad.gf1vhhn0m** | A bioprinted model of pregnant human uterine myometrium | **Live/Dead + nuclei + bioprinted:** Calcein-AM (green, live) / dead + Hoechst nuclei in bioprinted tissue ring (.avi 34.56 MB) + proteomics | **CC0** | https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn0m |
| **10.5061/dryad.sbcc2frfv** | Fluorescent images of actin & DAPI-labelled MCF10A/MCF7/MDA-MB-231 | Nuclei/DAPI TIFF stacks (549 MB; widefield, not confocal — flag) | **CC0** | https://datadryad.org/dataset/doi:10.5061/dryad.sbcc2frfv |
| **zenodo 11353017** | 3D confocal of dorsal aorta, WT & Endoglin-deficient zebrafish | Endothelial 3D confocal (in-vivo, not tube assay) | CC BY 4.0 | https://zenodo.org/records/11353017 |
| **figshare 12387629** | Source data: raw confocal z-stacks (Leica TCS SPE, 40×, 2–3 µm slices) | Real confocal z-stack format fixture (~1.2 GB; biology differs) | CC-BY 4.0 | https://api.figshare.com/v2/articles/12387629 |
| **zenodo 437943** | Entire confocal z-stack series as .tif image sequences | Generic confocal z-stack TIFFs (format reference) | CC BY 4.0 | https://zenodo.org/records/437943 |

---

## Honest gaps (no verified clean match)

- **GelMA-specific fluorescence confocal z-stacks:** genuine gap. `GelMA` = 0 hits in BioImage Archive and Dryad; Zenodo/figshare GelMA hits were posters/spreadsheets/articles, not image stacks. Bioprinted studies exist (S-BIAD2130 is Matrigel-based; S-BIAD616 / S-BIAD3391 are bioprinted organoids but **label-free interferometry, not fluorescence**). Use `hydrogel` / `bioprinted` and inspect per-study.
- **Endothelial *tube-formation* assay (the specific in-vitro Matrigel tube assay):** not verified anywhere. Closest are in-vivo / self-organization / junctional endothelial confocal sets (S-BIAD2920; Zenodo 11353017, 7229061). Dig further with the BioImage Archive `tube formation endothelial` search (417 hits, many pollen-tube false positives).
- **Live/Dead in general repos (Zenodo/figshare/Dryad):** sparse — Dryad gf1vhhn0m is the one clean CC0 hit, and it's a video not a z-stack. The BioImage Archive (S-BIAD2130 / S-BIAD2215) is the reliable home.

---

## Sources
- https://www.ebi.ac.uk/bioimage-archive/ · https://www.ebi.ac.uk/bioimage-archive/help-download/
- https://idr.openmicroscopy.org/
- https://bbbc.broadinstitute.org/
- https://celltrackingchallenge.net/3d-datasets/
- https://zenodo.org/ · https://developers.zenodo.org/
- https://figshare.com/ · https://docs.figshare.com/
- https://datadryad.org/
- https://www.ebi.ac.uk/empiar/
- https://ssbd.riken.jp/repository/ · http://ssbd.qbic.riken.jp/restfulapi/
