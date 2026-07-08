# Bioprinting / Hydrogel Construct Confocal Datasets for fluorostats Benchmarking

Public, citable datasets of **cells in 3D-printed / hydrogel tissue constructs imaged by fluorescence (confocal / light-sheet) microscopy** — application-matched to fluorostats' GelMA-vs-hybrid quantification use case.

All entries below were verified by fetching the record/API/FTP listing on 2026-07-08. URLs marked VERIFIED returned live metadata and file listings. No links or DOIs were invented.

---

## Availability verdict (read first)

Raw, downloadable bioprinting confocal data is **genuinely scarce** but it **does exist**. The dominant pattern in bioprinting papers is figure-only publication (representative live/dead panels, no raw stacks). General web search surfaces almost none of it. The two richest sources of *raw* fluorescence stacks of printed/hydrogel constructs are:

1. **Zenodo** — a handful of paper-linked deposits carry raw `.czi` / `.tif` z-stacks (multi-GB).
2. **BioImage Archive (EMBL-EBI)** — S-BIAD accessions hold raw multi-GB TIFF fluorescence stacks of bioprinted constructs, but must be browsed via the FTP tree (the HTML study pages are JS-rendered and search APIs undercount).

**IDR (Image Data Resource)** returned no matching bioprinting/hydrogel study via fetchable endpoints (JS app; needs manual review at idr.openmicroscopy.org). **Dryad** returned zero results for bioprinting+hydrogel+fluorescence. **figshare** search API endpoint used was invalid (needs POST-based search or manual browse).

**Best candidate:** Zenodo 6198612 — 3D-bioprinted **alginate-gelatin** cardiac patches, ~15.9 GB of raw `.czi` confocal (Hoechst/vimentin/cTNT/CD31), CC-BY-4.0. Closest single match to fluorostats' extrusion-printed soft-hydrogel construct type with multi-channel confocal.
Direct: `https://zenodo.org/records/6198612`

---

## 1. 3D-bioprinted alginate-gelatin cardiac patches with spheroids — VERIFIED (BEST CANDIDATE)

- **Citation:** Roche CD, Lin H, de Bock CE, Beck D, Xue M, Gentile C (2023). "3D bioprinted alginate-gelatin hydrogel patches containing cardiac spheroids recover heart function in a mouse model of myocardial infarction." *Bioprinting.* Data DOI **10.5281/zenodo.6198612**.
- **URL:** https://zenodo.org/records/6198612
- **Download pattern:** `https://zenodo.org/api/records/6198612/files/<FILENAME>/content`
- **Size / format:** ~15.9 GB across 24 files. Raw **`.czi`** confocal stacks (+ `.mp4`, `.xlsx`, `.pzfx`, `.zip`).
- **License:** CC-BY-4.0.
- **Modality / dims:** Multi-channel **confocal fluorescence**, 3D bioprinted constructs. Channels: Hoechst (nuclei), vimentin, cTNT (cardiomyocyte), CD31 (endothelial).
- **Material / cells:** **Alginate-gelatin** extrusion bioink; cardiac spheroids (cardiomyocytes, endothelial, fibroblast).
- **Relevance:** Strong. Soft extrusion-printed hydrogel + multi-channel confocal of cells within the construct — directly exercises fluorostats' per-channel intensity/coverage quantification on the same construct class. Caveat: alginate-gelatin, not GelMA; cardiac (not the driving paper's tissue). Native `.czi` requires Bio-Formats to read.

## 2. Light-sheet-bioprinted skin construct — raw IF stacks — VERIFIED

- **Citation:** Pampaloni F et al. (BRIGHTER / B-BRIGHTER, Goethe Univ. Frankfurt), 2025. Supporting data for Hafa et al., *Advanced Materials* 36(8):2306258, DOI **10.1002/adma.202306258**. Data DOI **10.5281/zenodo.17060664**.
- **URL:** https://zenodo.org/records/17060664
- **Download pattern:** `https://zenodo.org/api/records/17060664/files/<FILENAME>/content`
- **Size / format:** ~2.5 GB. Raw **`.czi`** (788 MB) + per-channel **`.tif`** stacks (C1-C4, ~197 MB each) + 3D projection `.tif`, timelapse `.tif`, `.stl`/`.gcode` print files, movies, day-30 JPGs.
- **License:** CC-BY-4.0.
- **Modality / dims:** **Light-sheet fluorescence** (with confocal-format CZI), 3D bioprinted. Channels: Hoechst 3342, Collagen IV-488, Vimentin-568, E-cadherin-633.
- **Material / cells:** **Dextran-CD-link** hydrogel bioink; Hs27 fibroblast + HaCaT keratinocyte co-culture (skin).
- **Relevance:** Strong for 3D multi-channel quantification and channel-split TIFF workflows. Caveat: dextran-based bioink (not GelMA); light-sheet rather than confocal (comparable intensity-quantification target).

## 3. High-throughput 3D-engineered paediatric tumour models (bioprinted hydrogel) — VERIFIED

- **Citation:** Jung M, Poltavets V, Skhinas JN, ... Kavallaris M (2025). "High-throughput 3D engineered paediatric tumour models for precision medicine." BioImage Archive **S-BIAD2130**, DOI **10.6019/S-BIAD2130**. (Linked to a primary publication; imaged on Zeiss CellDiscoverer 7.)
- **URL:** https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2130
- **FTP tree:** `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/130/S-BIAD2130/Files/`
- **Size / format:** Multi-GB raw **`.tif`** stacks (individual files 3.0-3.3 GB, e.g. `Figure 2/Fig 2A/zccs154 - 1.1kPa FN CN LN.tif`) plus brightfield TIFFs and a `LiveDead Imaging.json`/`.tsv` manifest.
- **License:** **CC0** (public domain).
- **Modality / dims:** Live/dead + multi-channel **fluorescence** (green=live, red=dead) + brightfield, 3D bioprinted constructs.
- **Material / cells:** **Engineered ECM-mimic peptide hydrogels** (tunable stiffness, e.g. 1.1 kPa); human paediatric tumour cells.
- **Relevance:** Strong for the live/dead viability-quantification use case fluorostats targets, with real raw stacks and CC0 licensing. Caveat: peptide/ECM hydrogel rather than GelMA; large files (browse the FTP tree — the JS study page and search API undercount the imaging files).

## 4. Bioprinted osteochondral construct (GelMA-nHA + THA) — POSTER ONLY — VERIFIED (caveat)

- **Citation:** Jahangir S, Vecstaudža J, Canciani E, Locs J, Alini M, Serra T (2022). "Development of bioprinted osteochondral tissue: an in-vitro model for drug discovery." TERMIS poster. DOI **10.5281/zenodo.6587069**.
- **URL:** https://zenodo.org/records/6587069
- **Size / format:** **PDF only** (`abstract TERMIS.pdf`, 105.5 KB). No raw images.
- **License:** CC-BY.
- **Material / cells:** **GelMA-nHA** (bone) + tyramine-HA (cartilage); osteoblasts, endothelial cells, chondrocyte micropellets. Live/dead (Calcein-AM / EthD-1) described but **not deposited**.
- **Relevance:** Material-perfect (actual GelMA + live/dead confocal) but **no downloadable image data** — cite only as evidence of the application; not usable as a benchmark. Listed to document the figure-only reality of GelMA bioprinting deposits.

## 5. Multi-scale engineered vasculature via volumetric bioprinting — PDF ONLY — VERIFIED (caveat)

- **Citation:** *Advanced Materials*, DOI **10.1002/adma.202521171**. Zenodo **10.5281/zenodo.17964998**. Gelatin-norbornene hydrogel, capillary-scale endothelial networks, volumetric bioprinting.
- **URL:** https://zenodo.org/records/17964998
- **Size / format:** **PDF (supporting-info document) only** at time of check — no raw stacks.
- **License:** CC-BY-4.0.
- **Relevance:** Endothelial-network-in-printed-hydrogel match on paper, but no raw imaging deposited. Cite as context; not a benchmark.

---

## Repositories worth manual review (endpoints not fully fetchable here)

- **IDR — https://idr.openmicroscopy.org/about/studies.html** — JS-rendered; search for "bioprint"/"hydrogel"/"organoid" manually. Likely holds relevant 3D fluorescence studies not indexed by fetchable API.
- **figshare — https://figshare.com/search?q=GelMA+bioprinting+confocal** — API search needs POST/query auth; browse manually for paper-linked GelMA deposits.
- **BioImage Archive full-text search** undercounts imaging assets; when a study looks relevant, inspect its FTP tree at `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/<last3digits>/S-BIAD<n>/Files/` to see the real `.tif`/`.czi` stacks.
- Related BioImage Archive bioprinting studies (not confocal): **S-BIAD616**, **S-BIAD3391** (HSLCI interferometry of bioprinted organoids — different modality); **S-BIAD705** (cells in agarose hydrogel droplets, InCell fluorescence — droplet, not printed construct).

## Honest caveats summary

- No public deposit found using **GelMA + confocal + raw stacks** together — GelMA bioprinting data is overwhelmingly figure-only (records 4-5). The application-matched *raw* data that exists uses adjacent bioinks: alginate-gelatin (#1), dextran (#2), peptide-ECM hydrogel (#3).
- The GelMA-vs-hybrid comparison fluorostats drives toward has no single ready-made raw benchmark; #1 (alginate-gelatin) is the closest soft-hydrogel + multi-channel confocal analog.
- All raw formats here are `.czi`/proprietary `.tif` needing Bio-Formats; confirm fluorostats' loader handles these before committing to a benchmark.
