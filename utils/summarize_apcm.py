"""Summarize APCM experiments and compute paired gains against GTR."""

import csv
import json
import os
import statistics


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results_apcm', 'runs.jsonl')
OUTPUT = os.path.join(ROOT, 'results_apcm', 'summary.csv')


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
        wins = 0
        for run in group:
            base = baseline.get((dataset, horizon, int(run['seed'])))
            if base is None or variant == 'GTR':
                continue
            gain = (float(base['mse']) - float(run['mse'])) / float(base['mse']) * 100.0
            paired.append(gain)
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
        })

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    print('dataset       H    variant      n   MSE mean+/-std     gain vs GTR   wins')
    for row in rows:
        gain = '' if row['paired_mse_gain_pct'] is None else '{:+.2f}%'.format(row['paired_mse_gain_pct'])
        print('{:<12} {:>4} {:<12} {:>2}   {:.4f}+/-{:.4f}   {:>11}   {}'.format(
            row['dataset'], row['pred_len'], row['variant'], row['n'],
            row['mse_mean'], row['mse_std'], gain, row['wins_vs_gtr']))
    print('Wrote {}'.format(OUTPUT))


if __name__ == '__main__':
    main()
