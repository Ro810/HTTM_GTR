"""Summarize APCM experiments and compute paired gains against GTR."""

import csv
import json
import os
import statistics


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, os.environ.get('APCM_OUT', 'results_apcm'))
RESULTS = os.path.join(OUT_DIR, 'runs.jsonl')
OUTPUT = os.path.join(OUT_DIR, 'summary.csv')

# GTR (Ours) column of the author's Table 7 (look-back 96), as (MSE, MAE).
_H_LONG, _H_SHORT = (96, 192, 336, 720), (12, 24, 48, 96)
_PAPER_ROWS = {
    'ETTh1': [(0.367, 0.391), (0.420, 0.421), (0.476, 0.447), (0.493, 0.476)],
    'ETTh2': [(0.290, 0.342), (0.362, 0.389), (0.414, 0.429), (0.423, 0.442)],
    'ETTm1': [(0.305, 0.349), (0.349, 0.375), (0.380, 0.398), (0.435, 0.436)],
    'ETTm2': [(0.168, 0.249), (0.232, 0.292), (0.287, 0.330), (0.386, 0.390)],
    'Electricity': [(0.134, 0.229), (0.152, 0.245), (0.171, 0.264), (0.208, 0.300)],
    'Solar': [(0.176, 0.235), (0.193, 0.244), (0.201, 0.250), (0.205, 0.251)],
    'Traffic': [(0.440, 0.263), (0.454, 0.274), (0.472, 0.282), (0.514, 0.301)],
    'Weather': [(0.154, 0.200), (0.202, 0.244), (0.259, 0.287), (0.341, 0.342)],
    'PEMS03': [(0.057, 0.156), (0.070, 0.172), (0.094, 0.198), (0.127, 0.228)],
    'PEMS04': [(0.065, 0.164), (0.075, 0.177), (0.093, 0.197), (0.114, 0.219)],
    'PEMS07': [(0.051, 0.142), (0.063, 0.156), (0.082, 0.178), (0.110, 0.202)],
    'PEMS08': [(0.071, 0.168), (0.095, 0.192), (0.148, 0.234), (0.256, 0.293)],
}
PAPER = {(ds, h): v for ds, rows in _PAPER_ROWS.items()
         for h, v in zip(_H_SHORT if ds.startswith('PEMS') else _H_LONG, rows)}


def mean(values):
    return statistics.mean(values) if values else None


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    if not os.path.exists(RESULTS):
        raise SystemExit('No results found at {}'.format(RESULTS))

    with open(RESULTS, encoding='utf-8') as handle:
        runs = [json.loads(line) for line in handle if line.strip()]

    by_group = {}
    for run in runs:
        key = (run['dataset'], int(run['pred_len']), run['variant'])
        by_group.setdefault(key, []).append(run)

    baseline = {}
    for run in runs:
        if run['variant'] == 'GTR':
            baseline[(run['dataset'], int(run['pred_len']), int(run['seed']))] = run

    rows = []
    for (dataset, horizon, variant), group in sorted(by_group.items()):
        mses = [float(run['mse']) for run in group]
        maes = [float(run['mae']) for run in group]
        paired = []
        d_mse, d_mae = [], []
        wins = 0
        for run in group:
            base = baseline.get((dataset, horizon, int(run['seed'])))
            if base is None or variant == 'GTR':
                continue
            gain = (float(base['mse']) - float(run['mse'])) / float(base['mse']) * 100.0
            paired.append(gain)
            d_mse.append(float(base['mse']) - float(run['mse']))
            d_mae.append(float(base['mae']) - float(run['mae']))
            wins += int(float(run['mse']) < float(base['mse']))

        rows.append({
            'dataset': dataset,
            'pred_len': horizon,
            'variant': variant,
            'n': len(group),
            'mse_mean': mean(mses),
            'mse_std': std(mses),
            'mae_mean': mean(maes),
            'mae_std': std(maes),
            'paired_mse_gain_pct': mean(paired),
            'wins_vs_gtr': '{} / {}'.format(wins, len(paired)) if paired else '',
            # Absolute paired reductions vs our GTR run (positive = better).
            'd_mse_vs_gtr': mean(d_mse),
            'd_mae_vs_gtr': mean(d_mae),
            # Reductions vs the number printed in the paper (Table 7).
            'd_mse_vs_paper': (PAPER[(dataset, horizon)][0] - mean(mses)) if (dataset, horizon) in PAPER else None,
            'd_mae_vs_paper': (PAPER[(dataset, horizon)][1] - mean(maes)) if (dataset, horizon) in PAPER else None,
        })

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, spec='{:+.4f}'):
        return '' if value is None else spec.format(value)

    print('dataset       H    variant          n   MSE     MAE     gain%    dMSE    dMAE  '
          'dMSEpap dMAEpap  wins')
    for row in rows:
        print('{:<12} {:>4} {:<16} {:>2}   {:.4f}  {:.4f}  {:>7} {:>7} {:>7} {:>7} {:>7}  {}'.format(
            row['dataset'], row['pred_len'], row['variant'], row['n'],
            row['mse_mean'], row['mae_mean'], fmt(row['paired_mse_gain_pct'], '{:+.2f}'),
            fmt(row['d_mse_vs_gtr']), fmt(row['d_mae_vs_gtr']),
            fmt(row['d_mse_vs_paper']), fmt(row['d_mae_vs_paper']), row['wins_vs_gtr']))
    print('Wrote {}'.format(OUTPUT))


if __name__ == '__main__':
    main()
