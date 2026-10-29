"""Phat hien chu ky (ACF tren tap train, luat v2 nested) cho cac dataset cua --grid full, so voi chu ky tac gia.

Chay:  python utils/detect_all_full.py [--only Weather,Traffic]  -> in bang + ghi results_dacd/full/cycles_detected.json
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.detect_cycles import detect
from utils.run_dacd_experiments import FULL, ROOT

ap = argparse.ArgumentParser()
ap.add_argument('--only', default='')
ap.add_argument('--refine', type=int, default=0)
a = ap.parse_args()
out = []
for ds, d in FULL.items():
    if a.only and ds not in a.only.split(','):
        continue
    path = os.path.join(ROOT, 'dataset', d['path'])
    if d['data'] == 'PEMS':
        v = np.load(path, allow_pickle=True)['data'][:, :, 0].astype(float); n = int(len(v) * 0.6)
    elif d['data'] == 'Solar':
        v = np.loadtxt(path, delimiter=','); n = int(len(v) * 0.7)
    else:
        v = pd.read_csv(path).drop(columns=['date']).values.astype(float)
        n = {'ETTh1': 8640, 'ETTh2': 8640, 'ETTm1': 34560, 'ETTm2': 34560}.get(d['data'], int(len(v) * 0.7))
    res, _, _ = detect(v[:n], nested=True, refine=bool(a.refine))
    r = dict(dataset=ds, author_L=d['L'], cycles=res['cycles'], acf=[round(x, 3) for x in res['acf_at_cycles']],
             match=d['L'] in res['cycles'])
    out.append(r)
    print('%-12s tac gia L=%-4d  DACD tu tim: %-12s ACF=%s  %s' % (ds, d['L'], r['cycles'], r['acf'], 'KHOP' if r['match'] else 'KHAC'), flush=True)
os.makedirs(os.path.join(ROOT, 'results_dacd', 'full'), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, 'results_dacd', 'full', 'cycles_detected.json'), 'w'), indent=1)
