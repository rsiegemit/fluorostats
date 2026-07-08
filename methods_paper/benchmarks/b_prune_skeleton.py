"""skeleton.prune_skeleton — spur removal vs 4 alternatives on trees with known spurs."""
import numpy as np, pandas as pd
from pathlib import Path
from skimage.morphology import skeletonize
import skan
from fluorostats.skeleton import prune_skeleton
RES=Path(__file__).resolve().parent/"results"
rng=np.random.default_rng(0)
def draw(v,p0,p1):
    p0=np.array(p0,float);p1=np.array(p1,float);n=int(np.abs(p1-p0).max())+1
    for t in np.linspace(0,1,n*2):
        y,x=np.round(p0+t*(p1-p0)).astype(int); v[y,x]=True
def tree_with_spurs(n_spurs, spur_len=4):
    N=256; v=np.zeros((N,N),bool); tb=[0]
    def rec(p,ang,ln,d):
        if d>3 or ln<6: return
        e=(p[0]+ln*np.cos(ang),p[1]+ln*np.sin(ang)); draw(v,(int(p[1]),int(p[0])),(int(e[1]),int(e[0]))); tb[0]+=1
        rec(e,ang-0.5,ln*0.62,d+1); rec(e,ang+0.5,ln*0.62,d+1)
    rec((N//2,N-20),-np.pi/2,42,0)
    sk=skeletonize(v); true_branches=len(skan.summarize(skan.Skeleton(sk),separator="_"))
    ys,xs=np.where(sk)
    for _ in range(n_spurs):  # add short spurs off random skeleton pixels
        i=rng.integers(len(ys)); a=rng.uniform(0,2*np.pi)
        e=(xs[i]+spur_len*np.cos(a), ys[i]+spur_len*np.sin(a))
        draw(sk,(ys[i],xs[i]),(int(np.clip(e[1],1,N-2)),int(np.clip(e[0],1,N-2))))
    return skeletonize(sk), true_branches
def nbranch(sk):
    try: return len(skan.summarize(skan.Skeleton(sk),separator="_"))
    except Exception: return 0
def skan_prune(sk,L):  # independent ref: drop leaf branches shorter than L
    try:
        s=skan.Skeleton(sk); df=skan.summarize(s,separator="_")
        keep=~((df["branch_type"]==1)&(df["branch_distance"]<L))
        return int(keep.sum())
    except Exception: return nbranch(sk)
rows=[]
for ns in [5,10,20]:
    sk,true_b=tree_with_spurs(ns)
    meas={
     "no_prune": nbranch(sk),
     "fluorostats_prune_L6": nbranch(prune_skeleton(sk,min_branch_length_px=6)),
     "fluorostats_prune_L10": nbranch(prune_skeleton(sk,min_branch_length_px=10)),
     "skan_leafprune_L6(ref)": skan_prune(sk,6),
     "skan_leafprune_L10(ref)": skan_prune(sk,10),
    }
    for m,b in meas.items():
        rows.append({"n_spurs":ns,"true_branches":true_b,"method":m,"measured":b,"err":abs(b-true_b)})
df=pd.DataFrame(rows); df.to_csv(RES/"b_prune_skeleton.csv",index=False)
print("=== prune_skeleton: branch recovery on spurred trees (5 methods) ===")
print(df.pivot(index="method",columns="n_spurs",values="err").to_string())
print("\n(err = |measured - true branches|; lower=better spur removal. true≈15)")
print(df.groupby("method")["err"].mean().round(2).sort_values().to_string())
