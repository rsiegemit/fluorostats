# For Claude Cowork — figures are final, next is the write-up

Short version: **the manuscript figures are built, polished, and committed.** Your
job is the prose + reference pass + final display-item numbering. Everything you
need is already in the repo.

## What's done (don't rebuild)
- **8 figures, manuscript-ready** — Nature-style, uniform 7.2 in (183 mm), Okabe–Ito
  palette (fluorostats always blue), vector **PDF + 300-dpi PNG + caption `.txt`**:
  - Main: `benchmarks/figures/main/` → fig2_nuclei_boundary, fig3_vascular,
    fig4_viability, fig5_homogeneity_stats (+ your fig1_schematic.svg).
  - Extended Data: `benchmarks/figures/extended/` → ed1_correctness, ed2_runtime,
    ed3_robustness, ed4_generalization.
  - Panel-by-panel contents + which CSV backs each: `HANDOFF_FOR_COWORK.md` §6.
  - Style is centralized in `benchmarks/figstyle.py`; to tweak a figure, edit its
    `benchmarks/fig*.py` / `make_ed1_correctness.py` and re-run with `python3.13`.
- **All numbers are frozen and traceable**: `benchmarks/BENCHMARK_INDEX.md` (registry)
  → `benchmarks/results/*.csv` (raw) → `benchmarks/00_BENCHMARK_RESULTS.md` (narrative).
- **Comparison matrix**: `COMPARISON_MATRIX.md`. **Honesty ledger**: `HANDOFF_FOR_COWORK.md` §7.

## What's left (your turn)
1. **Finish the prose** from `DRAFT_v0.5.md` — it's already strong (abstract, intro,
   results). Pull every number from `00_BENCHMARK_RESULTS.md` / the CSVs, not memory.
2. **Reference pass** — resolve the `[bracketed]` citation keys against a reference
   manager; verify the flagged DOIs (flag lists live in each `research/` dossier).
3. **Final display-item numbering** — the draft consolidates to 6 main figures +
   2 tables; reconcile the schematic (fig1) + data figures (fig2–5) with the draft's
   numbering, and move per-experiment numbers to Supplementary Tables S1–S9.
4. **Captions** — the sibling `.txt` next to each figure PNG is drop-in; adjust panel
   letters if the final numbering shifts.

## Framing rule (keep it straight)
Thresholding algorithms (Otsu, Li, …) are fluorostats' **own configs**, not rival
software. Comparisons are vs distinct **software** (StarDist, Cellpose, Omnipose,
REAVER, AngioTool, VesselExpress, the Kerkhoff Fiji macro) and vs reference
**implementations** for correctness (scipy, skan, hand-coded).

## Non-negotiable honesty points to keep in the text
- fluorostats (and all non-DL methods) collapse on crowded/overlapping instances —
  that's the stated scope boundary, not a flaw to hide.
- VesselExpress GT is pipeline-generated (software agreement, not manual gold).
- DL baselines were validated to reproduce their published numbers *before* any
  comparison — state this; it's the credibility linchpin.

Library is v0.7.0 on GitHub `main`; 105 tests green. Resume/state:
`PROJECT_STATE.md`.
