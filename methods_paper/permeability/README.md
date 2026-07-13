# FITC-dextran 2000 kDa penetration: GelMA vs volumetric GelMA–CMCMA hybrid

Quantitative depth-penetration analysis for the volumetric GelMA–CMCMA
manuscript. Regenerate everything with the library CLI:

```bash
fluorostats depth methods_paper/tools/permeability_fd2000.json
```

The reusable engine lives in the installed package (`fluorostats.depth` +
`fluorostats.depth_batch`); only this manifest and these outputs are
paper-specific.

## Method (methods-section ready)

Confocal z-stacks (Olympus `.oib`, single FITC/Alexa-488 channel, axial
step 4.8 µm, slice 0 = gel surface) were read with **fluorostats**
(`fluorostats.io.load_volume`). For every stack the mean FITC intensity of
each z-slice was computed over the full field of view
(`fluorostats.depth.intensity_depth_profile`), giving mean intensity versus
depth. The matched no-fluorescence control (`GelMA no fluo`,
`Hybrid no fluo`) was profiled identically and subtracted depth-for-depth
(interpolated onto the signal's depth axis; values floored at 0). Each
background-subtracted profile was normalised to its near-surface signal
(mean of the first 3 slices, 0–9.6 µm). Penetration was summarised as the
area under the profile over 0–100 µm (trapezoidal, endpoints interpolated;
`fluorostats.depth.auc_depth`). Group curves are mean ± SEM over the
primary matched stacks (GelMA n = 2, Hybrid n = 3, 512², 100 slices);
the shorter 1024²/30-slice stacks are reported as `aux` only.

## Files

| File | Contents |
|------|----------|
| `depth_profiles_long.csv` | Tidy per-slice data — one row per (stack, depth). Columns: `group, stack, role, depth_um, raw_mean, bg_subtracted, normalized`. **Import this into Prism.** |
| `auc_per_stack.csv` | One row per stack: voxel size, slice count, surface reference, and `auc_absolute`/`auc_normalized`/`window_covered_um` for **each** window (`_0_100um`, `_0_200um`, `_full`). |
| `group_depth_summary.csv` | Group mean ± SEM vs depth (primary stacks), for both bg-subtracted and normalized curves. |
| `fig_depth_absolute.{png,pdf}` | Mean FITC intensity (blank-subtracted) vs depth. |
| `fig_depth_normalized.{png,pdf}` | Normalized (surface = 1) intensity vs depth — the headline figure. |
| `fig_auc_normalized_{0_100um,0_200um,full}.{png,pdf}` | AUC dot/bar plot, one per window (report the normalized ones). |
| `fig_auc_absolute_{...}.{png,pdf}` | AUC on absolute intensity — confounded by per-stack gain; supplementary only. |
| `fig_auc_retention_multiwindow.{png,pdf}` | Single-panel summary: mean retained fraction (AUC_norm ÷ window width) for all three windows — shows the GelMA/hybrid gap widening with depth. |

## AUC over multiple depth windows (normalized, primary stacks)

| Window | GelMA (n=2) | Hybrid (n=3) | Hybrid / GelMA | Welch p |
|-------:|:-----------:|:------------:|:--------------:|:-------:|
| 0–100 µm | 86.7 | 98.1 | 1.1× | 0.23 |
| 0–200 µm | 97.5 | 188.6 | 1.9× | 0.0009 |
| full (0–475 µm) | 100.8 | 357.4 | 3.5× | 0.037 |

GelMA's AUC saturates near 100 (signal is gone past ~150 µm) while the hybrid
keeps accumulating — the deeper the window, the larger the difference. (n = 2 vs 3;
p-values are descriptive.) Windows are set by `auc_windows_um` in the manifest
(`[z0, z1]` pairs or `"full"`); add/remove windows and re-run — one line, no code.

## Result

- **GelMA** holds near-surface signal to ~48 µm then decays rapidly: 47 % at
  96 µm, 3 % at 192 µm.
- **Hybrid** decays gradually: 96 % at 96 µm, 86 % at 192 µm, 54 % at 384 µm.
- AUC 0–100 µm (normalized): GelMA 86.7 (n = 2) vs Hybrid 98.1 (n = 3).

The hybrid retains FITC-dextran signal markedly deeper, consistent with
better hydration of the volumetric GelMA–CMCMA bioink.

> **Report `auc_normalized`, not `auc_absolute`.** GelMA was imaged at higher
> gain (surface ≈ 2400 a.u. vs ≈ 1300 for hybrid), so absolute AUC would
> spuriously favour GelMA. Normalisation removes the acquisition gain and
> isolates decay shape.

> **AUC window note.** The GelMA/hybrid divergence is largest *beyond* 100 µm,
> so AUC is reported over 0–100 µm (as requested), 0–200 µm, and full depth (see
> the table above and `fig_auc_retention_multiwindow`). Windows are configured via
> `"auc_windows_um"` in the manifest — add or change windows and re-run, no code changes.
