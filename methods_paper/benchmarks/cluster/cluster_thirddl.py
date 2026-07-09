"""Third DL baseline (Omnipose, fallback Cellpose-cyto3) on BBBC039, F1@IoU0.5.
Reuses the exact GT loader + matcher from cluster_infer.py for parity."""
import sys, glob, os, csv
import numpy as np, tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed
IMG=os.path.expanduser('~/fluorostats_bench/data/BBBC039/images')
MSK=os.path.expanduser('~/fluorostats_bench/data/BBBC039_masks/masks')
def gt_instances(stem):
    m=np.array(Image.open(os.path.join(MSK,stem+'.png'))); r=m[...,0] if m.ndim==3 else m
    interior=(r==1); boundary=(r>=2); seeds,n=ndi.label(interior)
    if n==0: return np.zeros(r.shape,int),0
    return watershed(boundary.astype(np.uint8),seeds,mask=(r>0)),n
def f1_at(pred,gt,thr=0.5):
    pids=[p for p in np.unique(pred) if p!=0]; gids=[g for g in np.unique(gt) if g!=0]
    if not gids: return (1.0 if not pids else 0.0)
    matched=set(); tp=0; gm={g:(gt==g) for g in gids}
    for p in pids:
        pm=pred==p; ys,xs=np.where(pm); cand=np.unique(gt[ys,xs]); cand=cand[cand!=0]; best=0; bg=None
        for g in cand:
            inter=(pm&gm[g]).sum(); union=(pm|gm[g]).sum(); iou=inter/union if union else 0
            if iou>best: best=iou; bg=g
        if best>=thr and bg not in matched: matched.add(bg); tp+=1
    fp=len(pids)-tp; fn=len(gids)-tp
    return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
# model: prefer omnipose, else cellpose cyto3
tool='omnipose'
try:
    from cellpose_omni import models as om
    m=om.CellposeModel(gpu=False, model_type='cyto2_omni')
    def run(img): return m.eval(img, channels=[0,0], omni=True)[0]
except Exception as e:
    print('omnipose unavailable:',repr(e)[:100],flush=True)
    from cellpose import models
    try:
        m=models.CellposeModel(gpu=False, model_type='cyto3'); tool='cellpose_cyto3'
        def run(img): return m.eval(img, channels=[0,0])[0]
    except Exception:
        m=models.Cellpose(gpu=False, model_type='cyto2'); tool='cellpose_cyto2'
        def run(img): return m.eval(img, channels=[0,0], diameter=None)[0]
files=sorted(glob.glob(os.path.join(IMG,'*.tif')))
rows=[]
for i,f in enumerate(files):
    stem=os.path.basename(f).replace('.tif','')
    img=tifffile.imread(f).astype(np.float32)
    try:
        pred=run(img); gt,_=gt_instances(stem); fv=f1_at(pred,gt); n=int(len(np.unique(pred))-1)
    except Exception as e:
        n=-1; fv=-1; print('ERR',stem,repr(e)[:100],flush=True)
    rows.append((stem,n,round(fv,4) if fv>=0 else -1))
    if i%40==0: print(f'[{i}/{len(files)}] f1={fv}',flush=True)
with open(os.path.expanduser('~/fluorostats_bench/thirddl_eval.csv'),'w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['image','thirddl_count','thirddl_f1']); w.writerows(rows)
valid=[r[2] for r in rows if r[2]>=0]
print(f'DONE tool={tool} n={len(valid)} MEAN_F1={np.mean(valid):.4f}' if valid else 'DONE no valid',flush=True)
