"""Does live_dead_by_count(method='auto') track the best fixed method per regime?"""
import re, glob, numpy as np, pandas as pd, tifffile
from pathlib import Path
from skimage.draw import disk
from fluorostats.viability import live_dead_by_count
RES=Path(__file__).resolve().parent/"results"; rng=np.random.default_rng(0)
def field2(nl,nd,radius,noise,size=400):
    live=np.zeros((size,size),np.float32); dead=np.zeros((size,size),np.float32)
    def place(img,n):
        g=int(np.ceil(np.sqrt(n))); xs=np.linspace(radius+2,size-radius-2,g)
        for (y,x) in [(y,x) for y in xs for x in xs][:n]:
            rr,cc=disk((y,x),radius,shape=img.shape); img[rr,cc]=255.0
    place(live,nl); place(dead,nd)
    if noise>0: live=np.clip(live+rng.normal(0,noise,live.shape),0,None); dead=np.clip(dead+rng.normal(0,noise,dead.shape),0,None)
    return live,dead
def v(d): return d["viability"]
rows=[]
# regimes with known true viability = nl/(nl+nd)
cases=[("separated_small",90,10,3,0),("separated_large",90,10,16,0),
       ("noisy",90,10,3,80),("sparse_clean",45,5,5,0)]
for name,nl,nd,r,nz in cases:
    live,dead=field2(nl,nd,r,nz); true=nl/(nl+nd)
    res={"regime":name,"true":round(true,3)}
    for m in ["cc","maxima","watershed","auto"]:
        try: res[m]=round(abs(v(live_dead_by_count(live,dead,method=m))-true),3)
        except Exception: res[m]=np.nan
    a=live_dead_by_count(live,dead,method="auto"); res["auto_chose"]=a.get("method_live")
    rows.append(res)
# crowded regime: Kerkhoff synthetic (real overlapping)
D=Path("/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/data/downloads/kerkhoff_lda/synthetic")
kf=[f for f in glob.glob(str(D/"**/*.tif"),recursive=True) if "700 Live; 700 Dead-1" in f]
if kf:
    im=tifffile.imread(kf[0]).astype(np.float32); live,dead=im[1],im[0]; true=0.5
    res={"regime":"crowded_kerkhoff","true":true}
    for m in ["cc","maxima","watershed","auto"]:
        res[m]=round(abs(v(live_dead_by_count(live,dead,method=m))-true),3)
    res["auto_chose"]=live_dead_by_count(live,dead,method="auto").get("method_live")
    rows.append(res)
df=pd.DataFrame(rows); df.to_csv(RES/"b_viability_auto.csv",index=False)
print("=== viability |error| by method per regime (lower=better) ===")
print(df.to_string(index=False))
# did auto match the best fixed method?
def best(r): 
    fixed={k:r[k] for k in ["cc","maxima","watershed"] if pd.notna(r[k])}; return min(fixed.values())
df["best_fixed"]=df.apply(best,axis=1)
df["auto_optimal"]=(df["auto"]<=df["best_fixed"]+0.02)
print(f"\nauto within 0.02 of best fixed method in {df.auto_optimal.sum()}/{len(df)} regimes")
