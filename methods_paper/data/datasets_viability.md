# Viability (Live/Dead) Datasets — Benchmark B3

Public, citable datasets for benchmarking **fluorostats** depth-resolved 3D live-fraction
vs. 2D/MIP area-ratio methods. Target: Calcein-AM (green/live) + PI/Ethidium/EthD-1
(red/dead) confocal z-stacks in 3D constructs (spheroids, organoids, hydrogels).

**Verdict (honest):** This is genuinely the hardest category. The overwhelming majority
of "3D Live/Dead" published work only shares protocols, figures, or 2D projections — not
raw z-stacks. However, **at least one excellent raw z-stack Live/Dead dataset exists**
(S-BIAD2130, below) plus a strong light-attenuation z-stack dataset (Zenodo 5089728).
Together they fully support B3. All URLs below were verified by fetching (dates 2026-07-08).

Status legend: ✅ verified & downloadable · ⚠️ verified but caveat · ❌ not usable as-is

---

## ⭐ PRIMARY — S-BIAD2130 (BioImage Archive) — raw Live/Dead z-stacks in hydrogel constructs ✅

- **Name:** High-throughput 3D engineered paediatric tumour models for precision medicine —
  "LiveDead Imaging" collection.
- **Citation:** Jung M, Poltavets V, Skhinas JN, Tax G, Kamili A, et al. (2025).
  Deposited BioImage Archive **S-BIAD2130**. Released 2025-09-26.
  Associated peer-reviewed paper DOI not yet linked in metadata (verify at publication).
- **Accession/landing:** https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2130
- **Direct download (HTTP mirror of FTP):**
  https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/130/S-BIAD2130/Files/
  - `Files/Figure 2/Fig 2A/zccs154 - 1.1kPa FN CN LN.tif` — **3.0 GB** (fluorescence z-stack)
  - `Files/Figure 2/Fig 2A/zccs373 ....tif` — 3.3 GB + `..._Brightfield.tif` 3.3 GB
  - `Files/Figure 2/Fig 2B/zccs59 ....tif` — 3.7 GB (+ 1.2 GB brightfield), zccs207 3.4 GB, zccs227 3.2 GB
  - Also `Supp Figure 2/`, `Supp Figure 6/`, and `LiveDead Imaging.tsv` / `.json` (channel metadata).
  - FTP: `ftp://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/130/S-BIAD2130/` · Globus transfer also offered.
- **License:** **CC0** (public domain).
- **Format / dimensionality:** Multi-GB `.tif` **3D z-stacks** (the `z`-prefixed filenames are
  z-stacks; paired fluorescence + brightfield). Genuine 3D volumes, not projections.
- **Channels/stains:** **Calcein AM (live, green) + Ethidium Homodimer-1 (dead, red)** — the exact
  Live/Dead pairing B3 needs. Imaged on Zeiss CellDiscoverer 7. Constructs are engineered
  hydrogel tumour models (1.1 kPa FN/CN/LN matrix).
- **Feeds:** **B3** — the headline dataset. Depth-resolved 3D live-fraction from these z-stacks
  vs. MIP/2D area-ratio directly demonstrates the "we catch more in 3D" figure.
- **Caveats:** Large (each stack ~3 GB; plan bandwidth/storage). No published paper DOI wired into
  the record yet — cite the S-BIAD accession. Confirm exact z-step / voxel size from the
  `LiveDead Imaging.json` before quantifying absolute depths.

---

## ⭐ Zenodo 5089728 — light-attenuated 3D confocal spheroid z-stacks ✅ (depth-attenuation demo)

- **Name:** Data from: *Ellipsoid segmentation model for analyzing light-attenuated 3D confocal
  image stacks of fluorescent multi-cellular spheroids.*
- **Citation:** Barbier M, Jaensch S, Cornelissen F, Vidic S, Gjerde K, De Hoogt R, Graeser R,
  Gustin E, Chong YT (2017). *PLoS ONE* 11(6):e0156942. Paper DOI **10.1371/journal.pone.0156942**;
  data DOI **10.5061/dryad.0m9n7** (Zenodo mirror record 5089728).
- **Landing:** https://zenodo.org/records/5089728
- **Direct downloads:** per-file from the record, e.g.
  `field_001.tif` … `field_004.tif` (~965 MB each, PC346c spheroids),
  `stack_333.tif` … `stack_340.tif` (~965 MB each, LNCaP spheroids),
  `S2_File.tif` (366 MB example stack), `S3_File.zip` (11.6 MB ground-truth masks).
  Total ~12.2 GB current version.
- **License:** **CC0**.
- **Format / dimensionality:** Raw **3D confocal `.tif` z-stacks** of spheroids >100 µm diameter.
- **Channels/stains:** EdU (proliferation) + nuclear counterstain — **not** a Calcein/PI Live/Dead
  pair. But the paper's entire point is modelling **signal attenuation with depth** in spheroids.
- **Feeds:** **B3 (supporting)** — the cleanest public demonstration that fluorescence signal
  attenuates with z-depth in a 3D construct, which is *the* physical basis for the 2D-vs-3D
  discrepancy fluorostats corrects. Use to validate the depth-attenuation correction even where
  a true Live/Dead pair isn't present.
- **Caveats:** Not Live/Dead channels; frames the depth-attenuation argument rather than the
  live-fraction argument directly.

---

## Zenodo 8278594 — Calcein-AM + PI spheroid images (CNN dataset) ⚠️

- **Name:** Prediction of Spheroid Cell Death using Fluorescence Staining and CNNs.
- **Citation:** Srisongkram T, Syahid NF, Piyasawetkul T, et al. (Khon Kaen University, 2023).
  Data DOI **10.5281/zenodo.8278594** (paper "submitted" at deposit; verify final DOI).
- **Landing:** https://zenodo.org/records/8278594
- **Direct download:** https://zenodo.org/records/8278594/files/image_pickles.zip?download=1
  (584 MB compressed; ~21.6 GB uncompressed).
- **License:** **CC-BY-4.0**.
- **Channels/stains:** **Calcein AM (live) + Propidium Iodide (dead)** — correct Live/Dead pair.
- **Format / dimensionality:** ⚠️ **Python pickle arrays**, not raw microscopy files; appears to be
  **2D images** (or single planes) prepared for CNN training — depth/z information not preserved
  or not documented.
- **Feeds:** **B3 (2D reference only)** — usable as the *2D area-ratio* side of the comparison, or
  as a Calcein/PI ground-truth reference, but cannot supply 3D depth resolution.
- **Caveats:** Pickled non-standard format; no confirmed z-stacks. Good as a 2D counterpoint,
  not as the 3D star.

---

## Zenodo 5220610 — 3D spheroid confocal stack (single, StarDist demo) ⚠️

- **Name:** Segmenting cells in a spheroid in 3D using 2D StarDist within TrackMate.
- **Citation:** Zenodo record, DOI **10.5281/zenodo.5220610**. CC-BY-4.0.
- **Landing:** https://zenodo.org/records/5220610
- **Files:** `Spheroid-3D.tif` (33.6 MB) + label image + a day-6 stack (50.3 MB) + preview PNG.
- **Format:** single **3D confocal spheroid z-stack**, small and easy to fetch.
- **Channels:** nuclear/cell-body fluorescence — **not** Live/Dead.
- **Feeds:** **B3 (smoke-test only)** — a tiny, fast 3D spheroid stack good for pipeline sanity
  checks and figure prototyping before committing to the multi-GB S-BIAD2130 download.
- **Caveats:** Not viability channels; single sample; demo-scale.

---

## Also surfaced (context, not primary for B3)

- **BioImage Archive S-BIAD1284** — murine precision-cut lung slices, Calcein-AM + EthD-1 live/dead
  imaging. https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1284 — 3D tissue Live/Dead but
  not a spheroid/construct; worth checking file formats if more Live/Dead z-stacks are needed.
- **BioImage Archive S-BIAD448** — patient cancer organoids, NucRed Dead 647 / TO-PRO-3 viability.
  https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD448 — organoid viability but different
  dead-marker chemistry (not PI/EthD-1).

---

## What does NOT exist publicly (searched, honestly reported)

- **Raw Live/Dead (Calcein/PI) confocal z-stacks of bioprinted hydrogel constructs** with deposited
  data: extensive search of Zenodo, figshare, and paper supplements found **only protocols and
  figures**, no deposited raw z-stacks. The bioprinting field near-universally publishes 2D
  projections / Fiji-quantified numbers, not raw volumes.
- No matching dataset found in **IDR** or **EMPIAR** for 3D Live/Dead viability z-stacks
  (EMPIAR is EM-focused; IDR search returned no Calcein/PI spheroid z-stack study).

**Bottom line:** A citable public Live/Dead z-stack dataset **does exist** — use **S-BIAD2130**
(CC0, raw Calcein-AM/EthD-1 3D `.tif` z-stacks in hydrogel tumour models) as the B3 headline, with
**Zenodo 5089728** (CC0, light-attenuated 3D spheroid stacks) to substantiate the depth-attenuation
mechanism and **Zenodo 8278594** (Calcein/PI, 2D) as the 2D-method reference.
