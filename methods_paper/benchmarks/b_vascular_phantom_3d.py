"""3D vascular phantom with EXACT ground truth — fluorostats vessel metrics + skeletonization methods."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import skan
from fluorostats.skeleton import skeleton_metrics
from fluorostats.metrics_3d import volume_fraction
RES=Path(__file__).resolve().parent/"results"
def draw_cyl(vol, p0, p1, radius):
    p0=np.array(p0,float); p1=np.array(p1,float); L=np.linalg.norm(p1-p0)
    n=int(L*2)+2; true_len=L
    for t in np.linspace(0,1,n):
        c=np.round(p0+t*(p1-p0)).astype(int)
        zz,yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1,-radius:radius+1]
        ball=(zz**2+yy**2+xx**2)<=radius**2
        z0,y0,x0=c-radius
        sl=tuple(slice(max(0,a),min(s,a+2*radius+1)) for a,s in zip(c-radius,vol.shape))
        bsl=tuple(slice(max(0,-a),max(0,-a)+(s.stop-s.start)) for a,s in zip(c-radius,sl))
        vol[sl]|=ball[bsl]
    return true_len
def phantom(depth, size=140, radius=3):
    vol=np.zeros((size,size,size),bool); segs=[0]; tl=[0.0]
    def rec(p,vec,ln,d):
        if d>depth or ln<8: return
        e=p+vec*ln; tl[0]+=draw_cyl(vol,p,e,radius); segs[0]+=1
        if d<depth:
            for dth in (-0.5,0.5):
                nv=vec.copy(); nv[1]=vec[1]*np.cos(dth)-vec[2]*np.sin(dth); nv[2]=vec[1]*np.sin(dth)+vec[2]*np.cos(dth)
                rec(e,nv/np.linalg.norm(nv),ln*0.7,d+1)
    rec(np.array([size//2,size//2,20.]),np.array([0,0,1.]),34,0)
    return vol, tl[0], segs[0]
rows=[]
for depth in [1,2]:
    vol,true_len,true_seg=phantom(depth)
    true_vf=vol.mean()
    m=skeleton_metrics(vol, voxel_size_um=(1.0,1.0,1.0))
    meas_len=m.get("total_length_um",m.get("total_length"))
    meas_br=m.get("n_branches")
    vf=volume_fraction(vol)["volume_fraction"] if isinstance(volume_fraction(vol),dict) else volume_fraction(vol)
    rows.append({"phantom":f"tree_d{depth}","true_centerline_len":round(true_len,1),
                 "fs_skeleton_len":round(float(meas_len),1),
                 "len_err_pct":round(abs(meas_len-true_len)/true_len*100,1),
                 "true_segments":true_seg,"fs_branches":meas_br,
                 "true_vf":round(true_vf,4),"fs_vf":round(float(vf),4),
                 "vf_err_pct":round(abs(vf-true_vf)/true_vf*100,1)})
df=pd.DataFrame(rows); df.to_csv(RES/"b_vascular_phantom_3d.csv",index=False)
print("=== 3D vascular phantom — fluorostats vessel metrics vs EXACT GT ===")
print(df.to_string(index=False))
