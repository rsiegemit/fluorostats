"""Run a DL segmenter on a directory of 2D image slices; F1 vs label slices."""
import sys, glob, os, csv
import numpy as np, tifffile
TOOL=sys.argv[1]; IMG=os.path.expanduser(sys.argv[2]); LAB=os.path.expanduser(sys.argv[3]); OUT=os.path.expanduser(sys.argv[4])
def f1(pred,gt,thr=0.5):
    pids=[p for p in np.unique(pred) if p]; gids=[g for g in np.unique(gt) if g]
    if not gids: return 0.0
    gm={g:(gt==g) for g in gids}; matched=set(); tp=0
    for p in pids:
        pm=pred==p; cand=np.unique(gt[pm]); cand=cand[cand!=0]; best=0; bg=None
        for g in cand:
            i=(pm&gm[g]).sum(); u=(pm|gm[g]).sum(); iou=i/u if u else 0
            if iou>best: best=iou; bg=g
        if best>=thr and bg not in matched: matched.add(bg); tp+=1
    fp=len(pids)-tp; fn=len(gids)-tp
    return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
if TOOL=='cellpose':
    from cellpose import models; m=models.Cellpose(gpu=False, model_type='nuclei')
    run=lambda im: m.eval(im, channels=[0,0], diameter=None)[0]
else:
    from stardist.models import StarDist2D; from csbdeep.utils import normalize
    m=StarDist2D.from_pretrained('2D_versatile_fluo'); run=lambda im: m.predict_instances(normalize(im))[0]
rows=[]
for f in sorted(glob.glob(IMG+'/*.tif')):
    stem=os.path.basename(f); im=tifffile.imread(f).astype('float32'); gt=tifffile.imread(LAB+'/'+stem)
    try: pred=run(im); n=int(len(np.unique(pred))-1); score=f1(pred,gt)
    except Exception as e: n=-1; score=-1; print('ERR',stem,repr(e)[:100],flush=True)
    rows.append((stem,n,round(score,4)))
with open(OUT,'w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['image',f'{TOOL}_count',f'{TOOL}_f1']); w.writerows(rows)
v=[r[2] for r in rows if r[2]>=0]
print(f'DONE {TOOL} mean_F1={np.mean(v):.3f}' if v else 'no valid',flush=True)
