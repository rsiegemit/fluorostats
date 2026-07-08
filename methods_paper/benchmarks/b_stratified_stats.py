"""Validate stats.stratified_mann_whitney vs independent scipy per-stratum + BH refs."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from fluorostats.stats import stratified_mann_whitney
RES=Path(__file__).resolve().parent/"results"
rng=np.random.default_rng(0)
# synthetic: 2 groups x 3 strata (e.g. donors), 2 value cols, with a real effect
rows=[]
for st in ["s1","s2","s3"]:
    for g,shift in [("A",0),("B",1.2)]:
        for _ in range(12):
            rows.append({"grp":g,"stratum":st,"vol":rng.normal(shift,1),"len":rng.normal(shift*0.5,1)})
df=pd.DataFrame(rows)
out=stratified_mann_whitney(df,["vol","len"],"grp","A","B",strata=["stratum"])
# independent reference: scipy MWU per (value_col, stratum) + hand BH across the grid
refp=[]
for vc in ["vol","len"]:
    for st in ["s1","s2","s3"]:
        a=df[(df.grp=="A")&(df.stratum==st)][vc]; b=df[(df.grp=="B")&(df.stratum==st)][vc]
        refp.append(mannwhitneyu(a,b,alternative="two-sided").pvalue)
refp=np.array(refp); order=np.argsort(refp); m=len(refp)
bh=np.empty(m); bh[order]=np.minimum.accumulate((refp[order]*m/np.arange(1,m+1))[::-1])[::-1]
# align: match fluorostats output rows to refs by (value col, stratum)
checks=[]
pcol=[c for c in out.columns if c.lower() in ("p","pvalue","p_value")][0]
raw_ok=np.allclose(sorted(out[pcol].values), sorted(refp), atol=1e-9)
checks.append(("raw p-values match scipy per-stratum", raw_ok))
qcol=[c for c in out.columns if "q" in c.lower() or "fdr" in c.lower() or "adj" in c.lower()]
if qcol:
    q_ok=np.allclose(sorted(out[qcol[0]].values), sorted(bh), atol=1e-9)
    checks.append(("BH-FDR q-values match hand-coded BH", q_ok))
print("=== stratified_mann_whitney validation ===")
print(out.to_string(index=False)); print()
res=[]
for name,ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"); res.append({"check":name,"result":"PASS" if ok else "FAIL"})
pd.DataFrame(res).to_csv(RES/"b_stratified_stats.csv",index=False)
