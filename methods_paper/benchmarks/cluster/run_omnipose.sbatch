#!/bin/bash
#SBATCH -p mi2101x
#SBATCH -t 00:40:00
#SBATCH -N 1
#SBATCH -J fl_omni
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
module purge; module load pytorch/2.7.1
export PYTHONPATH=/share/sw/ai/pytorch/2.7.1:$HOME/.local/lib/python3.9/site-packages
# try omnipose; if unavailable, fall back to cellpose cyto3 as a distinct baseline
pip install --user -q omnipose 2>/dev/null
python3 - <<'PY'
import glob, numpy as np, tifffile, csv, os
from pathlib import Path
imgs=sorted(glob.glob(os.path.expanduser('~/fluorostats_bench/data/BBBC039/images/images/*.tif')))[:50]
def gt_count(name):
    import re
    return None
# instance F1 helper (IoU 0.5)
def f1(pred, gt):
    from scipy import ndimage as ndi
    pl=pred; gl=gt
    pi=[i for i in np.unique(pl) if i]; gi=[i for i in np.unique(gl) if i]
    if not pi and not gi: return 1.0
    if not pi or not gi: return 0.0
    tp=0; used=set()
    for g in gi:
        gm=gl==g; best=0; bj=None
        idx=np.unique(pl[gm])
        for p in idx:
            if p==0 or p in used: continue
            pm=pl==p; iou=(gm&pm).sum()/((gm|pm).sum()+1e-9)
            if iou>best: best=iou; bj=p
        if best>=0.5 and bj is not None: tp+=1; used.add(bj)
    fp=len(pi)-len(used); fn=len(gi)-tp
    return 2*tp/(2*tp+fp+fn+1e-9)
# GT masks
def load_gt(name):
    import png
    mp=os.path.expanduser('~/fluorostats_bench/data/BBBC039_masks/masks/'+Path(name).stem+'.png')
    if not os.path.exists(mp): return None
    a=tifffile.imread(mp) if mp.endswith('.tif') else __import__('imageio').v2.imread(mp)
    a=np.asarray(a)
    ch0=a[...,0] if a.ndim==3 else a
    from scipy import ndimage as ndi
    lab,_=ndi.label(ch0==1) if (ch0==1).any() else ndi.label(ch0>0)
    return lab
tool='omnipose'
try:
    from cellpose import models
    try:
        m=models.CellposeModel(gpu=True, model_type='cyto3'); tool='cellpose_cyto3'
    except Exception:
        m=models.Cellpose(gpu=True, model_type='cyto2'); tool='cellpose_cyto2'
    # prefer omnipose if importable
    try:
        import omnipose; from cellpose_omni import models as omodels
        m=omodels.CellposeModel(gpu=True, model_type='bact_fluor_omni'); tool='omnipose'
    except Exception: pass
except Exception as e:
    print('MODEL_LOAD_FAIL', e); raise
rows=[]
for f in imgs:
    im=tifffile.imread(f).astype('float32')
    try:
        out=m.eval(im, channels=[0,0])
        mask=out[0]
    except Exception as e:
        mask=np.zeros(im.shape,int)
    gt=load_gt(f)
    if gt is None: continue
    rows.append((Path(f).name, int(mask.max()), round(f1(mask,gt),4), int(gt.max())))
import csv
with open(os.path.expanduser('~/fluorostats_bench/thirddl_eval.csv'),'w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['image','count','f1','gt_count']); w.writerows(rows)
import numpy as np
print(f'TOOL={tool} N={len(rows)} MEAN_F1={np.mean([r[2] for r in rows]):.4f}')
PY
echo DONE
