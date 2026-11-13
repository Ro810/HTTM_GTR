"""Build the final APCM+ result by model selection on the VALIDATION set.

For every (dataset, horizon), APCM+ is the candidate with the lowest mean
validation loss over seeds.  All candidates are trained with the same
objective, so their validation losses are directly comparable.  Test-set
numbers are never used for the choice; they are only reported afterwards.

Candidates (same loss, different architecture):
    GTR-<LOSS>       author's GTR + new loss (APCM switched off)
    APCM-<LOSS>      + adaptive partial channel mixing
    APCM-CE-<LOSS>   + channel identity embedding

Usage:
    python utils/select_apcm_plus.py               # loss suffix FD5L1
    python utils/select_apcm_plus.py --loss FD8
Writes <APCM_OUT>/apcm_plus.csv and prints a table against GTR and the paper.
"""

import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_apcm import OUT_DIR, PAPER, RESULTS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', default='FD5L1', help='variant suffix of the shared loss')
    args = parser.parse_args()
    candidates = ['GTR-' + args.loss, 'APCM-' + args.loss, 'APCM-CE-' + args.loss]

    with open(RESULTS, encoding='utf-8') as handle:
        runs = [json.loads(line) for line in handle if line.strip()]

    groups = {}
    for run in runs:
        groups.setdefault((run['dataset'], int(run['pred_len'])), []).append(run)

    rows = []
    for (dataset, horizon), group in sorted(groups.items()):
        gtr = {r['seed']: r for r in group if r['variant'] == 'GTR'}
        scored = []
        for variant in candidates:
            members = [r for r in group if r['variant'] == variant and r.get('best_vali') is not None]
            if members:
                scored.append((statistics.mean(r['best_vali'] for r in members), variant, members))
        if not scored:
            continue
        _, chosen, members = min(scored)
        mse = statistics.mean(r['mse'] for r in members)
        mae = statistics.mean(r['mae'] for r in members)
        paired = [r for r in members if r['seed'] in gtr]
        paper = PAPER.get((dataset, horizon), (None, None))
        rows.append({
            'dataset': dataset, 'pred_len': horizon, 'chosen': chosen, 'n': len(members),
            'mse': mse, 'mae': mae,
            'gtr_mse': statistics.mean(gtr[r['seed']]['mse'] for r in paired) if paired else None,
            'gtr_mae': statistics.mean(gtr[r['seed']]['mae'] for r in paired) if paired else None,
            'paper_mse': paper[0], 'paper_mae': paper[1],
            'candidates_tried': len(scored),
        })

    out = os.path.join(OUT_DIR, 'apcm_plus.csv')
    with open(out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    def diff(a, b):
        return '' if a is None or b is None else '{:+.4f}'.format(b - a)

    print('dataset         H  chosen (by validation)   n   MSE     MAE    dMSE/GTR dMAE/GTR dMSE/pap dMAE/pap')
    for r in rows:
        print('{:<12} {:>4}  {:<22} {:>2}  {:.4f}  {:.4f}  {:>8} {:>8} {:>8} {:>8}'.format(
            r['dataset'], r['pred_len'], r['chosen'], r['n'], r['mse'], r['mae'],
            diff(r['mse'], r['gtr_mse']), diff(r['mae'], r['gtr_mae']),
            diff(r['mse'], r['paper_mse']), diff(r['mae'], r['paper_mae'])))
    print('(positive d = APCM+ is lower = better)  Wrote {}'.format(out))


if __name__ == '__main__':
    main()
