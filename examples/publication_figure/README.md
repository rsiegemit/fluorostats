# Example: publication figure from synthetic ground truth

`depth_penetration_figure.py` is a self-contained demonstration of the
`fluorostats.depth` module — it needs **no external data**. It builds synthetic
Beer–Lambert z-stacks with a known penetration constant λ, runs them through the
depth-penetration pipeline (per-slice mean → background subtraction → surface
normalisation → AUC + single-exponential λ fit), and renders a three-panel
figure showing (a) ground-truth recovery, (b) λ recovery on the identity line,
and (c) two-condition discrimination.

The figure is built with `fluorostats.pubfig`, the library's publication-figure
kit (Okabe–Ito palette, panel labels, scale bars, consistent type sizes) — the
same kit the benchmark figures use.

Run it:

```bash
pip install -e ".[all]"          # from the repo root
cd examples/publication_figure
python depth_penetration_figure.py
```

It writes `figures/main/depth_penetration.{pdf,png}` with its caption (`.txt`,
every number read from the run) and `results/depth_penetration_*.csv` (both
folders are git-ignored). `depth_penetration.png` in this folder is a committed
preview of the expected output.
