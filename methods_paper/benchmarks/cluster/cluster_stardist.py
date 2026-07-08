"""Run StarDist (2D_versatile_fluo) on BBBC039, write per-image counts."""
import glob, os, csv
import numpy as np, tifffile
from stardist.models import StarDist2D
from csbdeep.utils import normalize
model = StarDist2D.from_pretrained('2D_versatile_fluo')
files = sorted(glob.glob(os.path.expanduser('~/fluorostats_bench/data/BBBC039/images/*.tif')))
rows = []
for i, f in enumerate(files):
    img = tifffile.imread(f).astype(np.float32)
    try:
        labels, _ = model.predict_instances(normalize(img))
        n = int(labels.max())
    except Exception as e:
        n = -1; print('ERR', f, e, flush=True)
    rows.append((os.path.basename(f).replace('.tif',''), n))
    if i % 40 == 0:
        print(f'[{i}/{len(files)}] -> {n}', flush=True)
with open(os.path.expanduser('~/fluorostats_bench/stardist_counts.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['image','stardist_count']); w.writerows(rows)
print('DONE stardist', len(rows), flush=True)
