"""objects.centroid_homogeneity vs 4 established spatial stats on regular->clustered sweep."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy import stats as sps
from fluorostats.objects import centroid_homogeneity
RES=Path(__file__).resolve().parent/"results"
rng=np.random.default_rng(0); H=W=512; N=300
def pattern(clust):  # clust in [0,1]: 0=regular grid, 1=tight clusters
    if clust<=0:
        g=int(np.sqrt(N)); xs=np.linspace(20,W-20,g); ys=np.linspace(20,H-20,g)
        pts=np.array([(y,x) for y in ys for x in xs])
    else:
        k=max(1,int((1-clust)*40)+3); cen=rng.uniform(40,W-40,(k,2)); sd=8+(1-clust)*60
        pts=np.array([cen[i%k]+rng.normal(0,sd,2) for i in range(N)])
    return np.clip(pts,1,W-2)
def clark_evans(p):
    d,_=cKDTree(p).query(p,k=2); nn=d[:,1].mean(); exp=0.5/np.sqrt(len(p)/(H*W)); return nn/exp
def morisita(p,g=8):
    hx=np.histogram2d(p[:,0],p[:,1],bins=g,range=[[0,H],[0,W]])[0].ravel()
    n=hx.sum(); return len(hx)*(np.sum(hx*(hx-1)))/(n*(n-1)) if n>1 else np.nan
def quadrat_var(p,g=8):
    hx=np.histogram2d(p[:,0],p[:,1],bins=g,range=[[0,H],[0,W]])[0].ravel(); return hx.var()/hx.mean()
rows=[]
for clust in np.linspace(0,1,11):
    for seed in range(4):
        p=pattern(clust)
        p3=np.column_stack([np.zeros(len(p)), p[:,0], p[:,1]])
        ch=centroid_homogeneity(p3, (1,H,W))
        homog=ch.get("gini", ch.get("homogeneity", list(ch.values())[0]))
        rows.append({"clust":round(float(clust),2),"centroid_homogeneity":float(homog),
                     "clark_evans":clark_evans(p),"morisita":morisita(p),"quadrat_var":quadrat_var(p)})
df=pd.DataFrame(rows); df.to_csv(RES/"b_centroid_homogeneity.csv",index=False)
print("=== centroid_homogeneity vs established spatial stats (clustering sweep) ===")
print("Spearman rho vs clustering level:")
for c in ["centroid_homogeneity","clark_evans","morisita","quadrat_var"]:
    print(f"  {c:24s} {sps.spearmanr(df.clust,df[c]).statistic:+.3f}")
print("\nSpearman of centroid_homogeneity vs each reference stat:")
for c in ["clark_evans","morisita","quadrat_var"]:
    print(f"  vs {c:20s} {abs(sps.spearmanr(df.centroid_homogeneity,df[c]).statistic):.3f}")
