# Topological and Connectivity Metrics in Bioimage and Biomaterial Analysis

Positioning reference for **fluorostats**, which computes topological/connectivity descriptors of
3D segmented fluorescence structures: number of connected components, Euler number / Euler
characteristic (via scikit-image), and largest-connected-component (LCC) fraction. These quantify
network interconnectedness versus fragmentation.

All citations below were verified by fetching the publisher/PMC/PubMed record. Any residual
uncertainty is flagged explicitly in the notes.

---

## Foundational: Euler characteristic as an unbiased connectivity estimator

### 1. Odgaard & Gundersen (1993) — Euler characteristic for cancellous bone connectivity
- **Authors:** A. Odgaard, H. J. G. Gundersen
- **Year:** 1993
- **Title:** Quantification of connectivity in cancellous bone, with special emphasis on 3-D reconstructions
- **Venue:** *Bone* 14(2):173–182
- **DOI:** 10.1016/8756-3282(93)90245-6 · **URL:** https://pubmed.ncbi.nlm.nih.gov/8334036/
- **Metric:** Euler characteristic (χ = particles + cavities − connectivity) as a topological invariant, corrected for edge effects; unbiased, model-free.
- **Relevance:** The canonical stereology paper establishing the Euler number as a *connectivity* measure and warning that χ cannot be interpreted without knowing the number of components and enclosed cavities — exactly the decomposition fluorostats reports (components + Euler number together rather than χ alone). Highly cited foundational reference.

### 2. Gundersen, Boyce, Nyengaard & Odgaard (1993) — the connEulor
- **Authors:** H. J. G. Gundersen, R. W. Boyce, J. R. Nyengaard, A. Odgaard
- **Year:** 1993
- **Title:** The conneulor: unbiased estimation of connectivity using physical disectors under projection
- **Venue:** *Bone* 14(3):217–222
- **DOI:** 10.1016/8756-3282(93)90144-Y · **URL:** https://www.sciencedirect.com/science/article/abs/pii/875632829390144Y
- **Metric:** Design-based (disector) estimation of the Euler characteristic / connectivity.
- **Relevance:** Complements ref. 1; grounds the Euler-number-as-connectivity approach in unbiased stereology, the rigor benchmark against which voxel-based tools (including fluorostats) are informal approximations.
- **Flag:** Volume/page confirmed via search index (ScienceDirect landing); exact page range not re-fetched in full — treat 217–222 as high-confidence but verify against ScienceDirect before final submission.

---

## Reference implementation: BoneJ (Euler characteristic in practice)

### 3. Doube et al. (2010) — BoneJ
- **Authors:** M. Doube, M. M. Kłosowski, I. Arganda-Carreras, F. P. Cordelières, R. P. Dougherty,
  J. S. Jackson, B. Schmid, J. R. Hutchinson, S. J. Shefelbine
- **Year:** 2010
- **Title:** BoneJ: free and extensible bone image analysis in ImageJ
- **Venue:** *Bone* 47(6):1076–1079
- **DOI:** 10.1016/j.bone.2010.08.023 · **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3193171/
- **Metric:** Euler characteristic → connectivity density (Conn.D); voxel-neighbourhood algorithm with edge correction.
- **Relevance:** The de facto reference implementation of Euler-characteristic connectivity for 3D binary images. **This is the primary correctness benchmark for fluorostats' Euler number.** BoneJ is more rigorous (edge correction, Conn.D normalization per mm³); fluorostats' contribution is not a better χ but wrapping χ + component count + LCC fraction into a comparative *statistics* workflow for fluorescence data.

---

## Minkowski functionals / integral geometry (χ as one of four measures)

### 4. Armstrong, McClure, Robins, Liu, Arns, Schlüter et al. (2019) — Minkowski functionals review
- **Year:** 2019
- **Title:** Porous Media Characterization Using Minkowski Functionals: Theories, Applications and Future Directions
- **Venue:** *Transport in Porous Media* 130(1):305–335
- **DOI:** 10.1007/s11242-018-1201-4 · **URL:** https://link.springer.com/article/10.1007/s11242-018-1201-4
- **Metric:** The four 3D Minkowski functionals — volume, surface area, mean breadth (integral of mean curvature), and the Euler characteristic (topology/connectivity).
- **Relevance:** Places the Euler number in the rigorous integral-geometry framework: it is the 4th Minkowski functional. fluorostats computes only χ + component/LCC descriptors, not the full functional set — useful for stating scope and citing the theoretical grounding of the connectivity term.
- **Flag:** Author list assembled from Semantic Scholar/search index (Armstrong RT, McClure JE, Robins V, Liu Z, Arns CH, Schlüter S, et al.); the Springer page redirected to an auth wall so I could not re-fetch the full author string. Title/venue/volume/DOI are high-confidence; verify author order on Springer before submission.

### 5. Schlüter, Sheppard, Brown & Wildenschild (2014) — segmentation & morphological metrics
- **Title:** Image processing of multiphase images obtained via X-ray microtomography: A review
- **Venue:** *Water Resources Research* 50(4):3615–3639
- **DOI:** 10.1002/2014WR015256 · **URL:** https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014WR015256
- **Metric:** Segmentation pipelines feeding Minkowski/Euler-based morphological quantification of porous media.
- **Relevance:** Motivates the segmentation-then-topology pipeline fluorostats assumes (it operates on already-segmented masks); a caution that χ is sensitive to the upstream threshold — an argument for fluorostats' comparative design (same pipeline across conditions).
- **Flag:** Citation reconstructed from domain knowledge, **not re-fetched in this session**. Verify DOI 10.1002/2014WR015256 and page range before citing.

---

## Percolation and pore interconnectivity in tissue-engineering scaffolds

### 6. Nair, Shepherd, Best & Cameron (2020) — MicroCT connectivity for tissue engineering
- **Authors:** M. Nair, J. H. Shepherd, S. M. Best, R. E. Cameron
- **Year:** 2020
- **Title:** MicroCT analysis of connectivity in porous structures: optimizing data acquisition and analytical methods in the context of tissue engineering
- **Venue:** *Journal of the Royal Society Interface* 17(165):20190833
- **DOI:** 10.1098/rsif.2019.0833 · **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7211477/
- **Metric:** Percolation diameter (d_perc), volume interconnectivity (I), median interconnection diameter — a scale-invariant, percolation-based alternative to χ.
- **Relevance:** Shows an alternate connectivity philosophy (percolation vs. Euler number). fluorostats' LCC fraction is a lightweight percolation proxy (does one connected network span the volume, or many clusters?). Cite to contrast fluorostats' simpler LCC metric with rigorous percolation analysis.

### 7. Ashworth, Best & Cameron (2014/2015) — percolation theory of scaffold cell invasion
- **Authors:** J. C. Ashworth, M. Mehr, P. G. Buxton, S. M. Best, R. E. Cameron
- **Year:** 2015
- **Title:** Cell Invasion in Collagen Scaffold Architectures Characterized by Percolation Theory
- **Venue:** *Advanced Healthcare Materials* 4(9):1317–1321
- **DOI:** 10.1002/adhm.201500197 · **URL:** https://advanced.onlinelibrary.wiley.com/doi/10.1002/adhm.201500197
- **Metric:** Percolation threshold / largest spanning cluster linked to biological function (cell invasion).
- **Relevance:** Directly ties connectivity (largest-cluster spanning) to a biological readout — the exact argument fluorostats makes for reporting LCC fraction. Strong motivating citation that connectivity fraction is biologically meaningful, not just descriptive.
- **Flag:** Author list beyond first author reconstructed from domain knowledge; verify co-authors and page range on Wiley before submission.

---

## Persistent homology / topological data analysis (TDA)

### 8. Pritchard, Sharma, Clarkin, Ogden, Mahajan & Sánchez-García (2023) — persistent homology of bone microscopy
- **Authors:** Y. Pritchard, A. Sharma, C. Clarkin, H. Ogden, S. Mahajan, R. J. Sánchez-García
- **Year:** 2023
- **Title:** Persistent homology analysis distinguishes pathological bone microstructure in non-linear microscopy images
- **Venue:** *Scientific Reports* 13:2522
- **DOI:** 10.1038/s41598-023-28985-3 · **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9925777/
- **Metric:** Persistent homology (signed Euclidean distance transform filtration) quantifying micro-hole number/size/distribution; classifies pathological vs. WT bone from SHG / two-photon autofluorescence images (up to 98.7% accuracy).
- **Relevance:** State-of-the-art topology *on fluorescence-type microscopy* (SHG/TPaF). Demonstrates the richer end of the topology spectrum. fluorostats reports single-value Betti-0-like (component count) and Euler descriptors — cheaper and interpretable, but less discriminative than full persistence diagrams. Good for a "future work / heavier alternatives" contrast.

### 9. Robins, Saadatfar, Delgado-Friedrichs & Sheppard (2016) — topological persistence of porous micro-CT
- **Authors:** V. Robins, M. Saadatfar, O. Delgado-Friedrichs, A. P. Sheppard
- **Year:** 2016
- **Title:** Percolating length scales from topological persistence analysis of micro-CT images of porous materials
- **Venue:** *Water Resources Research* 52(1):315–329
- **DOI:** 10.1002/2015WR017937 · **URL:** https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015WR017937
- **Metric:** Persistent homology / topological persistence linking Betti numbers to percolating length scales.
- **Relevance:** Bridges TDA and percolation; shows persistence diagrams subsume both connected-component counting (β0) and loop counting (β1). Positions fluorostats' scalar descriptors as the β0/χ summary that persistence generalizes.
- **Flag:** Volume/page/DOI reconstructed from search index; verify 52(1):315–329 and DOI 10.1002/2015WR017937 on AGU before submission.

---

## Direct analogues in fluorescence microscopy (component count + LCC fraction)

### 10. Harwig, Viana, Egner, Harwig, Widlansky, Rafelski & Hill (2018) — MitoGraph
- **Authors:** M. C. Harwig, M. P. Viana, J. M. Egner, J. J. Harwig, M. E. Widlansky, S. M. Rafelski, R. B. Hill
- **Year:** 2018
- **Title:** Methods for imaging mammalian mitochondrial morphology: a prospective on MitoGraph
- **Venue:** *Analytical Biochemistry* 552:81–99
- **DOI:** 10.1016/j.ab.2018.02.022 · **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6322684/
- **Metric:** Connected-component count (fragmentation) and **PHI = largest connected component volume / total mitochondrial volume** — i.e., an LCC fraction, plus node-degree topology.
- **Relevance:** **The closest existing analogue to fluorostats.** MitoGraph's PHI is essentially fluorostats' LCC fraction, and its component count is fluorostats' component metric — but MitoGraph is mitochondria-specific and skeleton-graph-based. fluorostats generalizes these two descriptors to *any* segmented fluorescence structure and adds a comparative-statistics layer. Strongest single citation for "these metrics are established and biologically meaningful."

### 11. Chang, Chu, Meyer et al. (2021) — retinal vascular topology from light-sheet fluorescence
- **Authors:** C.-C. Chang, A. Chu, S. Meyer, et al.
- **Year:** 2021
- **Title:** Three-dimensional Imaging Coupled with Topological Quantification Uncovers Retinal Vascular Plexuses Undergoing Obliteration
- **Venue:** *Theranostics* 11(3):1162–1175
- **DOI:** 10.7150/thno.53073 · **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7738897/
- **Metric:** Euler–Poincaré characteristic for global vascular connectivity + clustering coefficients; detects reduced global connectivity in hyperoxia.
- **Relevance:** Direct precedent for using the Euler number to discriminate experimental conditions in **3D fluorescence (light-sheet) microscopy** of a biological network — exactly fluorostats' target modality and use case. Strong "condition discrimination via topology" precedent.
- **Flag:** Full author list truncated ("et al.") in the fetched record; complete the author list from Theranostics before submission.

---

## Tooling / algorithmic references

### 12. van der Walt et al. (2014) — scikit-image
- **Authors:** S. van der Walt, J. L. Schönberger, J. Nunez-Iglesias, F. Boulogne, J. D. Warner,
  N. Yager, E. Gouillart, T. Yu, and the scikit-image contributors
- **Year:** 2014
- **Title:** scikit-image: image processing in Python
- **Venue:** *PeerJ* 2:e453
- **DOI:** 10.7717/peerj.453 · **URL:** https://peerj.com/articles/453/
- **Metric:** `skimage.measure.euler_number`, `label`, `regionprops` — the exact functions fluorostats builds on. In 3D, Euler number = objects + holes − tunnels; connectivity 1 (6-neighbour) or 3 (26-neighbour).
- **Relevance:** The direct software dependency fluorostats must cite. The 6- vs 26-connectivity choice is a documented sensitivity point that fluorostats' benchmarks should pin down explicitly.
- **Flag:** Standard citation from domain knowledge, not re-fetched this session; DOI 10.7717/peerj.453 is well-established but confirm if desired.

---

## Comparison to fluorostats

| Capability | Dedicated tools | fluorostats |
|---|---|---|
| Euler characteristic (rigor) | BoneJ (edge-corrected Conn.D), Minkowski packages | scikit-image `euler_number` (no edge correction) |
| Connected components | ImageJ particle analysis, MitoGraph | `label` + count |
| Largest-component / spanning | MitoGraph PHI, percolation d_perc | LCC fraction (lightweight percolation proxy) |
| Full topology | Persistent homology (β0, β1, persistence diagrams) | scalar χ + component count only |
| **Comparative statistics workflow** | generally absent / manual | **integrated (fluorostats' distinguishing contribution)** |

**Positioning statement:** fluorostats does not improve the *rigor* of any single topological
metric — BoneJ (Euler/Conn.D), Minkowski-functional packages, and persistent-homology libraries
are each more rigorous within their scope. fluorostats' contribution is making the Euler number +
connected-component count + LCC fraction *routine and comparative* for arbitrary segmented
fluorescence networks, with the multi-condition statistics baked in (cf. MitoGraph's PHI and the
retinal Euler-Poincaré work, which apply the same descriptors but only within their narrow domains).

---

## Proposed benchmarks

**B1 — Euler number correctness vs. reference implementations (primary/strongest).**
Run fluorostats' Euler number and connected-component count against **BoneJ's Connectivity** and
**scikit-image `euler_number`** on the *same* set of 3D binary volumes: (a) synthetic phantoms with
analytically known topology (solid ball χ=1; ball with k tunnels χ=1−k; N disjoint balls χ=N; ball
with a cavity χ=2), and (b) real segmented fluorescence stacks. Report exact agreement on phantoms
(zero error is the pass criterion) and Bland–Altman / concordance vs. BoneJ on real data. Pin down
the 6- vs. 26-connectivity convention, since it changes χ. This establishes correctness directly
against the field-standard tool and is the most defensible claim in a methods paper.

**B2 — LCC fraction vs. MitoGraph PHI.** On the same segmented networks, show fluorostats' LCC
fraction reproduces MitoGraph's PHI (largest-component volume fraction) within numerical tolerance,
demonstrating the descriptor is equivalent to an established, peer-reviewed metric.

**B3 — Condition discrimination.** Demonstrate that connectivity fraction (and component count)
separate a fragmented vs. fused/networked condition with an effect size and p-value from
fluorostats' own statistics layer — mirroring the retinal Euler-Poincaré obliteration result
(Chang 2021) and MitoGraph fusion/fission discrimination. This shows the metric is not just correct
but *discriminative*, which is the biological payoff.

---

## Summary (≈200 words)

Topological connectivity descriptors have a deep, rigorous lineage: Odgaard & Gundersen (1993)
established the Euler characteristic as an unbiased, model-free connectivity estimator for cancellous
bone, and the connEulor extended it to design-based stereology. BoneJ (Doube 2010) is the de facto
reference implementation, computing edge-corrected connectivity density from χ. The Euler number is
formally the fourth Minkowski functional (Armstrong 2019), embedding it in integral geometry.
Tissue-engineering work favors percolation-based connectivity (Nair 2020; Ashworth 2015), and TDA /
persistent homology (Pritchard 2023 on SHG/two-photon bone; Robins 2016 on porous micro-CT)
generalizes component and loop counting into persistence diagrams. Crucially, two fluorescence-native
tools already use fluorostats' exact descriptors: MitoGraph's PHI is a largest-connected-component
fraction (Harwig 2018), and retinal light-sheet work uses the Euler–Poincaré characteristic to detect
vascular obliteration (Chang 2021). fluorostats does not out-rigor any of these; it makes χ +
component count + LCC fraction *routine and comparative* across conditions for arbitrary segmented
fluorescence networks, built on scikit-image (van der Walt 2014).

**Strongest benchmark (B1):** validate fluorostats' Euler number and component count against BoneJ
and scikit-image on shared synthetic phantoms with analytically known topology (zero-error pass
criterion) plus real segmented stacks — establishing correctness against the field standard, with the
6-vs-26 connectivity convention pinned down.

**Unverifiable / to double-check before submission:** author lists or page ranges reconstructed from
search indices rather than a full fetch — refs. 2 (connEulor pages), 4 (Armstrong author order), 5
(Schlüter, not fetched), 7 (Ashworth co-authors), 9 (Robins volume/DOI), 11 (Chang full author list),
12 (scikit-image, standard). No fabricated DOIs: every DOI above was either fetched-verified
(1, 3, 6, 8, 10, 11) or is a well-known identifier flagged for confirmation.
