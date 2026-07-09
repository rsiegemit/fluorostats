"""Generate publication comparison figures for the methods-paper handoff."""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fluorostats.style import apply_style, PALETTE
apply_style()
R=Path(__file__).resolve().parent/"results"; F=Path(__file__).resolve().parent/"figures"/"handoff"; F.mkdir(parents=True,exist_ok=True)
P=PALETTE; FS="fluorostats"
def hl(names): return [P["accent"] if FS in str(n).lower() else P["primary"] for n in names]
def save(fig,name): fig.tight_layout(); fig.savefig(F/name,dpi=200,bbox_inches="tight"); plt.close(fig); print("saved",name)

# 1. nuclei 12-method ranking
d=pd.read_csv(R/"b2_nuclei_methods.csv").sort_values("mean_F1")
fig,ax=plt.subplots(figsize=(8,5)); ax.barh(d.method,d.mean_F1,color=hl(d.method),edgecolor="black",lw=.5)
ax.set_xlabel("instance F1 @ IoU 0.5"); ax.set_title("Nuclei segmentation — 12-method comparison (BBBC039)")
ax.axvline(d[d.method.str.contains("fluorostats")].mean_F1.iloc[0],ls="--",c=P["muted"],lw=1); save(fig,"fig1_nuclei_ranking.png")

# 2. fluorostats vs DL with bootstrap CIs
d=pd.read_csv(R/"b_dl_ci.csv"); dd=d[d.method.isin(["fluorostats","StarDist","Cellpose"])]
fig,ax=plt.subplots(figsize=(6.5,4.2))
y=np.arange(len(dd)); cols=hl(dd.method)
for i,(_,r) in enumerate(dd.iterrows()):
    ax.errorbar(r.mean_F1,i,xerr=[[r.mean_F1-r.CI_low],[r.CI_high-r.mean_F1]],fmt="o",
        color=P["ink"],capsize=5,ms=9,mfc=cols[i],mec="black")
ax.set_yticks(y); ax.set_yticklabels(dd.method); ax.set_xlabel("mean F1 (95% bootstrap CI, n=200)")
ax.set_title("fluorostats vs validated DL — BBBC039\n(paired diff significant: +0.025 & +0.034, CIs exclude 0)"); save(fig,"fig2_dl_ci.png")

# 3. clustering degradation curve
d=pd.read_csv(R/"b_clustering_curve.csv"); x=[0,25,50,75]
fig,ax=plt.subplots(figsize=(7,5))
for _,r in d.iterrows():
    c=P["accent"] if "fluorostats" in r["method"] else P["muted"]
    lw=2.6 if "fluorostats" in r["method"] else 1.4
    ax.plot(x,[r["c00"],r["c25"],r["c50"],r["c75"]],marker="o",label=r["method"],color=c,lw=lw)
ax.axhline(0.96,ls=":",c=P["primary"],label="DL (c75)")
ax.set_xlabel("nuclear clustering / overlap (%)"); ax.set_ylabel("instance F1")
ax.set_title("Clustering degradation — the whole non-DL class collapses"); ax.legend(fontsize=7,frameon=False); save(fig,"fig3_clustering_curve.png")

# 4. master timing (log scale)
d=pd.read_csv(R/"b_timing_all_metrics.csv").dropna(subset=["ms"]).sort_values("ms")
fig,ax=plt.subplots(figsize=(8,11))
cols=[P["accent"] if g=="fluorostats" else P["primary"] for g in d.group]
ax.barh(d.metric,d.ms,color=cols,edgecolor="black",lw=.3); ax.set_xscale("log")
ax.set_xlabel("ms per call (log)"); ax.set_title("Per-metric runtime — fluorostats (red) vs comparators (blue)")
ax.tick_params(labelsize=6); save(fig,"fig4_timing_all.png")

# 5. viability external vs Kerkhoff macro
d=pd.read_csv(R/"b_viability_external_summary.csv").sort_values("MAE")
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
a1.barh(d.method,d.MAE,color=hl(d.method),edgecolor="black",lw=.5); a1.set_xlabel("MAE vs ground-truth viability"); a1.set_title("Error (lower better)")
a2.barh(d.method,d.CCC,color=hl(d.method),edgecolor="black",lw=.5); a2.set_xlabel("Lin's CCC"); a2.set_title("Agreement (higher better)")
fig.suptitle("Viability vs published Kerkhoff Fiji macro (Zenodo synthetic GT)",fontweight="semibold"); save(fig,"fig5_viability_external.png")

# 6. noise robustness
d=pd.read_csv(R/"b_noise_robustness.csv")
fig,ax=plt.subplots(figsize=(7,4.5))
for c in [x for x in d.columns if x!="noise_sd"]:
    col=P["accent"] if "fluorostats" in c else P["muted"]; lw=2.6 if "fluorostats" in c else 1.3
    ax.plot(d.noise_sd,d[c],marker="o",label=c,color=col,lw=lw)
ax.set_xlabel("added Gaussian noise (σ)"); ax.set_ylabel("foreground Dice"); ax.set_title("Noise robustness (BBBC024)"); ax.legend(fontsize=7,frameon=False); save(fig,"fig6_noise.png")

# 7. scope boundary
sep=pd.read_csv(R/"b2_nuclei_methods.csv"); cr=pd.read_csv(R/"b2_crowded_c75_comparison.csv")
fs_sep=sep[sep.method.str.contains("fluorostats")].mean_F1.iloc[0]
dl_sep=sep[sep.method.str.contains("StarDist")].mean_F1.iloc[0]
fig,ax=plt.subplots(figsize=(6.5,4.5)); x=np.arange(2); w=.35
fsv=[fs_sep, cr[cr.method.str.contains("fluorostats",case=False)].mean_F1.iloc[0]]
dlv=[dl_sep, cr[cr.method.str.contains("StarDist",case=False)].mean_F1.iloc[0]]
ax.bar(x-w/2,fsv,w,label="fluorostats",color=P["accent"],edgecolor="black")
ax.bar(x+w/2,dlv,w,label="DL (StarDist)",color=P["primary"],edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels(["well-separated","crowded (c75)"]); ax.set_ylabel("instance F1")
ax.set_title("Scope boundary: fluorostats ≥ DL separated; DL wins crowded"); ax.legend(frameon=False); save(fig,"fig7_scope_boundary.png")

# 8. vascular tool ranking
d=pd.read_csv(R/"b4_reaver_ranking.csv").sort_values("MAE",ascending=False)
fig,ax=plt.subplots(figsize=(7,4.2)); ax.barh(d.tool,d.MAE,color=hl(d.tool),edgecolor="black",lw=.5)
ax.set_xlabel("area-fraction MAE vs manual GT (lower better)"); ax.set_title("Vascular tools on REAVER's own benchmark (n=36)"); save(fig,"fig8_vascular_ranking.png")

# 9. homogeneity vs spatial stats
d=pd.read_csv(R/"b_homogeneity_multi_corr.csv"); d["abs"]=d["spearman_gini_vs_ref"].abs()
d=d.sort_values("abs")
fig,ax=plt.subplots(figsize=(7,4)); ax.barh(d.reference,d["abs"],color=P["primary"],edgecolor="black",lw=.5)
ax.set_xlim(0,1.05); ax.set_xlabel("|Spearman| vs fluorostats Gini"); ax.set_title("Homogeneity tracks 5 established spatial statistics"); save(fig,"fig9_homogeneity.png")

# 10. 3D vascular phantom accuracy
d=pd.read_csv(R/"b_vascular_phantom_3d.csv")
fig,ax=plt.subplots(figsize=(6.5,4)); x=np.arange(len(d)); w=.35
ax.bar(x-w/2,d.len_err_pct,w,label="skeleton length err %",color=P["accent"],edgecolor="black")
ax.bar(x+w/2,d.vf_err_pct,w,label="volume fraction err %",color=P["primary"],edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels(d.phantom); ax.set_ylabel("error vs exact GT (%)")
ax.set_title("3D vascular phantom — fluorostats vessel metrics vs exact GT"); ax.legend(frameon=False); save(fig,"fig10_vascular_phantom.png")
print("ALL FIGURES DONE ->", F)
