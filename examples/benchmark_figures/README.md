# Example: reproducible benchmark figures

Five multi-panel figures from the FluoroStats methods-paper benchmarks that
reproduce **from bundled precomputed metrics** — no large raw datasets required.
Each script also doubles as a worked example of the library's analysis modules.

| script | figure | library modules exercised |
|---|---|---|
| `fig3_vascular.py` | vessel-network accuracy (REAVER 6-tool, VesselExpress, 3D phantom, Bland–Altman) | `metrics_3d`, `skeleton`, `agreement` |
| `fig5_homogeneity_stats.py` | spatial homogeneity + end-to-end statistics | `objects.centroid_homogeneity`, `stats` (Mann–Whitney, Cliff's δ, BH-FDR, bootstrap CI), `power.power_curve` |
| `make_ed1_correctness.py` | topology / skeleton / density-normalisation correctness | `objects`, `skeleton`, density normalisation |
| `fig_ed3_robustness.py` | segmentation robustness to noise + denoising | `segment`, `preprocess` |
| `fig_ed4_generalization.py` | cross-dataset generalisation (DSB2018, CTC 3D) | `segment`, `validate` |

`figstyle.py` is the shared publication-styling helper (Okabe–Ito palette, panel
labels, scale bars). `results/` holds the small precomputed metric tables the
figures plot (and `ve_crops.npz`, small light-sheet image crops for one panel).
`previews/` shows the expected output.

Run any of them:

```bash
pip install -e ".[all]"          # from the repo root
cd examples/benchmark_figures
python fig5_homogeneity_stats.py     # writes figures/ (git-ignored)
```

## Not included here
Three further benchmark figures (nucleus segmentation vs deep learning,
depth-resolved viability, and the combined applications panel) depend on large
public raw microscopy datasets (BBBC039/BBBC024, BioImage Archive, Olympus
`.oib` stacks) rather than small tables, so they are not bundled. The underlying
analysis all runs through the same `fluorostats` modules shown above.
