# Example: publication figure from synthetic ground truth

`depth_penetration_figure.py` is a self-contained demonstration of the
`fluorostats.depth` module — it needs **no external data**. It builds synthetic
Beer–Lambert z-stacks with a known penetration constant λ, runs them through the
depth-penetration pipeline (per-slice mean → background subtraction → surface
normalisation → AUC + single-exponential λ fit), and renders a three-panel
figure showing (a) ground-truth recovery, (b) λ recovery on the identity line,
and (c) two-condition discrimination.

`figstyle.py` is the shared publication-styling helper (Okabe–Ito palette,
panel labels, scale bars, consistent type sizes) used to build the figure.

Run it:

```bash
pip install -e ".[all]"          # from the repo root
cd examples/publication_figure
python depth_penetration_figure.py
```

It writes the figure to `figures/main/` and summary CSVs to `results/`
(both are git-ignored). `depth_penetration.png` in this folder is a committed
preview of the expected output.
