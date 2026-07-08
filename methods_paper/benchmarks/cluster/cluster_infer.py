"""Run a DL nuclei segmenter on BBBC039; output per-image count + F1@IoU0.5.

Usage: python3 cluster_infer.py <cellpose|stardist>

GT instances are reconstructed from BBBC039's 3-class masks (R channel:
1=interior, >=2=boundary) via watershed of the interior seeds. F1 matches
predicted instances to GT instances at IoU>=0.5 — the metric used in the
published nuclei benchmarks (DSB2018), so mean F1 can be checked against the
published StarDist/Cellpose numbers to confirm the baseline is faithful.
"""
import sys, glob, os, csv
import numpy as np, tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed

TOOL = sys.argv[1]
IMG = os.path.expanduser('~/fluorostats_bench/data/BBBC039/images')
MSK = os.path.expanduser('~/fluorostats_bench/data/BBBC039_masks/masks')

def gt_instances(stem):
    m = np.array(Image.open(os.path.join(MSK, stem + '.png')))
    r = m[...,0] if m.ndim==3 else m
    interior = (r==1); boundary = (r>=2)
    seeds, n = ndi.label(interior)
    if n==0: return np.zeros(r.shape, int), 0
    filled = watershed(boundary.astype(np.uint8), seeds, mask=(r>0))
    return filled, n

def f1_at(pred, gt, thr=0.5):
    pids = [p for p in np.unique(pred) if p!=0]
    gids = [g for g in np.unique(gt) if g!=0]
    if not gids: return (1.0 if not pids else 0.0), 0, len(pids), 0
    matched_g=set(); tp=0
    gt_masks={g:(gt==g) for g in gids}
    for p in pids:
        pm = pred==p; best=0; bg=None
        ys,xs=np.where(pm)
        cand=np.unique(gt[ys,xs]); cand=cand[cand!=0]
        for g in cand:
            gm=gt_masks[g]; inter=(pm&gm).sum(); union=(pm|gm).sum()
            iou=inter/union if union else 0
            if iou>best: best=iou; bg=g
        if best>=thr and bg not in matched_g:
            matched_g.add(bg); tp+=1
    fp=len(pids)-tp; fn=len(gids)-tp
    f1= 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
    return f1, tp, len(pids), len(gids)

def get_model():
    if TOOL=='cellpose':
        from cellpose import models
        gpu = os.environ.get('FL_GPU','0')=='1'
        m = models.Cellpose(gpu=gpu, model_type='nuclei')
        def run(img): return m.eval(img, channels=[0,0], diameter=None)[0]
        return run
    else:
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
        m = StarDist2D.from_pretrained('2D_versatile_fluo')
        def run(img): return m.predict_instances(normalize(img))[0]
        return run

run = get_model()
files = sorted(glob.glob(os.path.join(IMG,'*.tif')))
rows=[]
for i,f in enumerate(files):
    stem=os.path.basename(f).replace('.tif','')
    img=tifffile.imread(f).astype(np.float32)
    try:
        pred=run(img); n=int(len(np.unique(pred))-1)
        gt,ngt=gt_instances(stem)
        f1,tp,npred,ngt2=f1_at(pred,gt)
    except Exception as e:
        n=-1; f1=-1; tp=-1; ngt=-1; print('ERR',stem,repr(e)[:120],flush=True)
    rows.append((stem,n,round(f1,4) if f1>=0 else -1, ngt))
    if i%40==0: print(f'[{i}/{len(files)}] {stem[:16]} count={n} f1={f1}',flush=True)
out=os.path.expanduser(f'~/fluorostats_bench/{TOOL}_eval.csv')
with open(out,'w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['image',f'{TOOL}_count',f'{TOOL}_f1','gt_count']); w.writerows(rows)
import numpy as _np
valid=[r[2] for r in rows if r[2]>=0]
print(f'DONE {TOOL}: {len(valid)} valid, mean F1={_np.mean(valid):.3f}' if valid else f'DONE {TOOL}: no valid',flush=True)
