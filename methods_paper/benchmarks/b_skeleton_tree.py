"""Validate fluorostats skeleton metrics on synthetic trees with known structure."""
import numpy as np, pandas as pd
from pathlib import Path
from fluorostats.skeleton import skeleton_metrics, n_junction_nodes
from skimage.morphology import skeletonize
RES=Path(__file__).resolve().parent/"results"

def draw_line(vol,p0,p1):
    p0=np.array(p0,float); p1=np.array(p1,float)
    n=int(np.abs(p1-p0).max())+1
    for t in np.linspace(0,1,n*2):
        z,y,x=np.round(p0+t*(p1-p0)).astype(int)
        vol[z,y,x]=True

def build_tree(depth, length=40, shrink=0.62):
    """Binary bifurcating tree in 3D. Returns volume + true (#branches,#bifurcations)."""
    N=260; vol=np.zeros((60,N,N),bool); z=30
    branches=[0]; bifs=[0]
    def rec(p, ang, ln, d):
        if d>depth or ln<5: return
        p2=(z, p[1]+ln*np.sin(ang), p[0]+ln*np.cos(ang)*0+ln*np.cos(ang))
        # 2D tree embedded at fixed z: use (x,y)
        end=(p[0]+ln*np.cos(ang), p[1]+ln*np.sin(ang))
        draw_line(vol,(z,int(p[1]),int(p[0])),(z,int(end[1]),int(end[0])))
        branches[0]+=1
        if d<depth:
            bifs[0]+=1
            rec(end, ang-0.5, ln*shrink, d+1)
            rec(end, ang+0.5, ln*shrink, d+1)
    rec((N//2, N-20), -np.pi/2, length, 0)
    return vol, branches[0], bifs[0]

rows=[]
for depth in [2,3,4]:
    vol,nb,nbif=build_tree(depth)
    m=skeleton_metrics(vol, prune=True, min_branch_length_um=4)
    rows.append({"tree_depth":depth,"true_branches":nb,"fs_n_branches":m["n_branches"],
                 "true_bifurcations":nbif,"fs_n_junction_nodes":m["n_junction_nodes"],
                 "branch_match":abs(m["n_branches"]-nb)<=1,
                 "bifurcation_match":abs(m["n_junction_nodes"]-nbif)<=1})
df=pd.DataFrame(rows); df.to_csv(RES/"b_skeleton_tree.csv",index=False)
print("=== fluorostats skeleton vs known synthetic tree structure ===")
print(df.to_string(index=False))
npass=int((df.branch_match & df.bifurcation_match).sum())
print(f"\n{npass}/{len(df)} trees: branch & bifurcation counts match ground truth (+-1)")
