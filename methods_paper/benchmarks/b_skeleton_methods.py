"""Skeletonization algorithm comparison — fluorostats (Lee-1994) vs 4 alternatives."""
import numpy as np, pandas as pd
from pathlib import Path
from skimage.morphology import skeletonize, medial_axis, thin
import skan
RES=Path(__file__).resolve().parent/"results"

def draw(vol,p0,p1):
    p0=np.array(p0,float);p1=np.array(p1,float);n=int(np.abs(p1-p0).max())+1
    for t in np.linspace(0,1,n*2):
        z,y,x=np.round(p0+t*(p1-p0)).astype(int); vol[z,y,x]=True

def tree(depth,length=40,shrink=0.62):
    N=260;v=np.zeros((60,N,N),bool);z=30;br=[0];bi=[0]
    def rec(p,ang,ln,d):
        if d>depth or ln<5:return
        e=(p[0]+ln*np.cos(ang),p[1]+ln*np.sin(ang))
        draw(v,(z,int(p[1]),int(p[0])),(z,int(e[1]),int(e[0])));br[0]+=1
        if d<depth: bi[0]+=1; rec(e,ang-0.5,ln*shrink,d+1); rec(e,ang+0.5,ln*shrink,d+1)
    rec((N//2,N-20),-np.pi/2,length,0); return v,br[0],bi[0]

def metrics(skel):
    if skel.sum()<3: return 0,0
    try:
        s=skan.Skeleton(skel); df=skan.summarize(s,separator="_")
        return float(df["branch_distance"].sum()), len(df)
    except Exception: return 0,0

# ground-truth total length: sum of drawn segment lengths (approx by branch phantom)
methods={"Lee_1994(fluorostats)":lambda m:skeletonize(m),
         "medial_axis":lambda m:medial_axis(m) if m.ndim==2 else skeletonize(m),
         "thin":lambda m:thin(m) if m.ndim==2 else skeletonize(m),
         "skeletonize_2d_zhang":lambda m:skeletonize(m,method='zhang') if m.ndim==2 else skeletonize(m)}
rows=[]
for depth in [2,3]:
    vol,nb,nbif=tree(depth)
    # collapse to 2D (tree is planar at z=30) for 2D skeleton methods
    m2d=vol[30]
    for name,fn in methods.items():
        try: L,B=metrics(np.asarray(fn(m2d)))
        except Exception: L,B=0,0
        rows.append({"phantom":f"tree_d{depth}","method":name,"true_branches":nb,
                     "measured_branches":B,"total_length":round(L,1),"branch_err":abs(B-nb)})
df=pd.DataFrame(rows); df.to_csv(RES/"b_skeleton_methods.csv",index=False)
print("=== Skeletonization algorithms — branch recovery on known trees ===")
print(df.to_string(index=False))
print("\nfluorostats uses Lee-1994 (skimage skeletonize); compared vs medial_axis, thin, Zhang.")
