"""B-cluster-2D — fluorostats 2D cluster/coverage metrics vs GT (BBBC039).

Exercises fluorostats.metrics_2d (area_fraction, coverage_metrics) directly:
for each image we binarize with 5 skimage threshold methods, feed the binary
mask into fluorostats, and compare the resulting cluster/coverage metrics to
the ground-truth instance masks.

GT definitions (BBBC039 RGBA instance masks, channel 0):
    foreground = channel0 > 0            (interior + boundary)
    nucleus count = CC of channel0 == 1  (interior only, splits touching cells)
    cluster count = CC of channel0 > 0   (merges touching cells; apples-to-apples
                                          with fluorostats n_components)

Agreement (Lin's CCC, Spearman rho) is reported per threshold method for
area fraction, cluster count, and nucleus count.
"""
import glob, sys
import numpy as np, pandas as pd, tifffile
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import agreement_report

import fluorostats.metrics_2d as fs

DL = Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/BBBC039")
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(parents=True, exist_ok=True)

imgs = {Path(p).stem: p for p in glob.glob(str(DL / "images" / "images" / "*.tif"))}
masks = {Path(p).stem: p for p in glob.glob(str(DL / "masks" / "masks" / "*.png"))}
common = sorted(set(imgs) & set(masks))

METHODS = {
    "Otsu_1979": filters.threshold_otsu,
    "Li_1993": filters.threshold_li,
    "Isodata_1978": filters.threshold_isodata,
    "Triangle_1977": filters.threshold_triangle,
    "Yen_1995": filters.threshold_yen,
}

# Ground truth ----------------------------------------------------------------
gt_af, gt_nuclei, gt_clusters = [], [], []
# fluorostats predictions -----------------------------------------------------
pred_af = {m: [] for m in METHODS}
pred_ncomp = {m: [] for m in METHODS}
pred_largest = {m: [] for m in METHODS}
pred_meanarea = {m: [] for m in METHODS}

for stem in common:
    im = tifffile.imread(imgs[stem]).astype(np.float32)
    mk = np.array(Image.open(masks[stem]))
    ch0 = mk[..., 0] if mk.ndim == 3 else mk

    gt_fg = ch0 > 0
    gt_af.append(float(gt_fg.mean()))
    gt_nuclei.append(int(ndi.label(ch0 == 1)[1]))
    gt_clusters.append(int(ndi.label(gt_fg)[1]))

    for name, fn in METHODS.items():
        try:
            binary = im > fn(im)
            # Drive fluorostats' 2D metrics on the binary mask.
            cm = fs.coverage_metrics(binary)
            pred_af[name].append(float(fs.area_fraction(binary)))
            pred_ncomp[name].append(int(cm["n_components"]))
            pred_largest[name].append(float(cm["largest_component_fraction"]))
            pred_meanarea[name].append(float(cm["mean_cluster_area_px"]))
        except Exception:
            for d in (pred_af, pred_ncomp, pred_largest, pred_meanarea):
                d[name].append(np.nan)

gt_af = np.array(gt_af, float)
gt_nuclei = np.array(gt_nuclei, float)
gt_clusters = np.array(gt_clusters, float)

# Sanity: fluorostats area_fraction must equal binary.mean() exactly ----------
# (verifies the library metric is the plain foreground fraction, not a variant)
consistency_ok = True
for name in METHODS:
    p = np.array(pred_af[name], float)
    # recompute independently for one method as an internal check
    if name == "Otsu_1979":
        indep = []
        for stem in common:
            im = tifffile.imread(imgs[stem]).astype(np.float32)
            indep.append(float((im > filters.threshold_otsu(im)).mean()))
        consistency_ok = bool(np.allclose(p, np.array(indep), atol=1e-9, equal_nan=True))

# Agreement tables ------------------------------------------------------------
def agree(pred_dict, gt, label):
    rows = []
    for name in METHODS:
        p = np.array(pred_dict[name], float)
        m = np.isfinite(p) & np.isfinite(gt)
        rep = agreement_report(p[m], gt[m], name, "GT")
        rows.append({
            "metric": label, "method": name,
            "CCC": round(rep["ccc"], 3), "spearman": round(rep["spearman"], 3),
            "pearson": round(rep["pearson"], 3), "bias": round(rep["bias"], 4),
            "MAE": round(float(np.abs(p[m] - gt[m]).mean()), 4), "n": int(m.sum()),
        })
    return rows

rows = []
rows += agree(pred_af, gt_af, "area_fraction")
rows += agree(pred_ncomp, gt_clusters, "cluster_count_vs_gtclusters")
rows += agree(pred_ncomp, gt_nuclei, "cluster_count_vs_gtnuclei")

df = pd.DataFrame(rows)
df.to_csv(RES / "b_cluster_2d.csv", index=False)

# Descriptive stats for the other cluster metrics (no direct GT counterpart) --
desc_rows = []
for name in METHODS:
    desc_rows.append({
        "method": name,
        "mean_largest_frac": round(float(np.nanmean(pred_largest[name])), 3),
        "mean_cluster_area_px": round(float(np.nanmean(pred_meanarea[name])), 1),
        "mean_n_components": round(float(np.nanmean(pred_ncomp[name])), 1),
    })
desc = pd.DataFrame(desc_rows)
desc.to_csv(RES / "b_cluster_2d_descriptive.csv", index=False)

# Markdown report -------------------------------------------------------------
def md_table(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

def block(label):
    sub = df[df.metric == label].sort_values("CCC", ascending=False)
    return md_table(sub[["method", "CCC", "spearman", "pearson", "bias", "MAE", "n"]])

md = f"""# B-cluster-2D — fluorostats 2D cluster/coverage metrics vs GT (BBBC039)

**Dataset:** BBBC039 2D nuclei, n = {len(common)} images.
**Library:** fluorostats {getattr(__import__('fluorostats'), '__version__', '?')} — `metrics_2d.area_fraction`, `metrics_2d.coverage_metrics`.
**Threshold methods:** Otsu, Li, Isodata, Triangle, Yen (skimage.filters).

Ground truth from BBBC039 RGBA instance masks (channel 0):
`foreground = ch0>0`; `nucleus_count = CC(ch0==1)`; `cluster_count = CC(ch0>0)`.

GT means: area_fraction = {gt_af.mean():.4f}, nuclei/img = {gt_nuclei.mean():.1f}, clusters/img = {gt_clusters.mean():.1f}.

fluorostats `area_fraction` == independent `(img>thr).mean()`: **{consistency_ok}** (Otsu spot-check, atol 1e-9).

## Area fraction agreement (fluorostats area_fraction vs GT)
{block("area_fraction")}

## Cluster count agreement (fluorostats n_components vs GT foreground clusters)
{block("cluster_count_vs_gtclusters")}

## Cluster count vs GT nucleus count (expected weaker — clusters merge touching nuclei)
{block("cluster_count_vs_gtnuclei")}

## Descriptive cluster metrics (no direct GT counterpart)
{md_table(desc)}

## Notes
- fluorostats `n_components` counts connected foreground clusters, so it is
  compared against GT *cluster* count (touching-nuclei-merged), not the raw
  nucleus count; the nucleus-count row is included to quantify that gap.
- `largest_component_fraction`, `mean_cluster_area_px`, and `median_cluster_area_px`
  have no per-image GT counterpart in BBBC039 and are reported descriptively.
"""
(RES / "b_cluster_2d.md").write_text(md)

print("=== B-cluster-2D (n=%d) ===" % len(common))
print("area_fraction consistency (fluorostats == manual):", consistency_ok)
for lbl in ("area_fraction", "cluster_count_vs_gtclusters", "cluster_count_vs_gtnuclei"):
    print("\n---", lbl, "---")
    print(df[df.metric == lbl].sort_values("CCC", ascending=False)
          [["method", "CCC", "spearman", "bias", "MAE"]].to_string(index=False))
print("\nWrote:", RES / "b_cluster_2d.csv", RES / "b_cluster_2d.md")
