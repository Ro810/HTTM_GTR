"""Select APCM+ variants using validation MSE only.

Every candidate must have results for the same selection seeds. Test MSE/MAE
are read only after the variant has been selected; they never affect the
choice. This prevents test leakage and unfair comparisons caused by one
variant having more completed seeds than another.
"""

import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_apcm import OUT_DIR, PAPER, RESULTS  # noqa: E402


def build_selection_rows(runs, loss_suffix='FD5L1', selection_seeds=None):
    """Return selected result rows and human-readable skip warnings."""
    candidates = ['GTR-' + loss_suffix, 'APCM-' + loss_suffix,
                  'APCM-CE-' + loss_suffix]
    requested_seeds = set(selection_seeds or [])
    groups = {}
    for run in runs:
        groups.setdefault((run['dataset'], int(run['pred_len'])), []).append(run)

    rows = []
    warnings = []
    for (dataset, horizon), group in sorted(groups.items()):
        by_variant = {}
        for variant in candidates:
            by_variant[variant] = {
                int(run['seed']): run for run in group
                if run['variant'] == variant and run.get('best_vali_mse') is not None
            }

        missing = [variant for variant in candidates if not by_variant[variant]]
        if missing:
            # Old result groups containing none of the corrected candidates are
            # irrelevant. Partly completed corrected groups get a warning.
            if len(missing) != len(candidates):
                warnings.append('{} H{}: missing {}'.format(
                    dataset, horizon, ', '.join(missing)))
            continue

        common_seeds = set.intersection(*(set(by_variant[v]) for v in candidates))
        chosen_seeds = requested_seeds or common_seeds
        incomplete = [v for v in candidates if not chosen_seeds.issubset(by_variant[v])]
        if not chosen_seeds or incomplete:
            warnings.append('{} H{}: candidates do not share requested seeds {}'.format(
                dataset, horizon, ','.join(map(str, sorted(chosen_seeds)))))
            continue

        scores = {
            variant: statistics.mean(
                float(by_variant[variant][seed]['best_vali_mse'])
                for seed in chosen_seeds
            )
            for variant in candidates
        }
        chosen = min(candidates, key=lambda variant: (scores[variant], variant))

        # After the variant is fixed, report every completed seed for it. The
        # two seed columns in the CSV make selection and reporting transparent.
        members = [by_variant[chosen][seed] for seed in sorted(by_variant[chosen])]
        mse = statistics.mean(float(run['mse']) for run in members)
        mae = statistics.mean(float(run['mae']) for run in members)
        gtr = {int(run['seed']): run for run in group if run['variant'] == 'GTR'}
        report_seed_set = {int(run['seed']) for run in members}
        has_paired_gtr = report_seed_set.issubset(gtr)
        paper = PAPER.get((dataset, horizon), (None, None))
        rows.append({
            'dataset': dataset,
            'pred_len': horizon,
            'chosen': chosen,
            'selection_metric': 'vali_mse',
            'selection_seeds': ','.join(map(str, sorted(chosen_seeds))),
            'report_seeds': ','.join(str(run['seed']) for run in members),
            'mean_vali_mse': scores[chosen],
            'n': len(members),
            'mse': mse,
            'mae': mae,
            'gtr_mse': (statistics.mean(float(gtr[seed]['mse']) for seed in report_seed_set)
                        if has_paired_gtr else None),
            'gtr_mae': (statistics.mean(float(gtr[seed]['mae']) for seed in report_seed_set)
                        if has_paired_gtr else None),
            'paper_mse': paper[0],
            'paper_mae': paper[1],
            'candidates_tried': len(candidates),
        })
    return rows, warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', default='FD5L1', help='variant suffix of the shared loss')
    parser.add_argument(
        '--selection_seeds', default='',
        help='comma-separated seeds required for every candidate; default: shared seeds',
    )
    args = parser.parse_args()
    selection_seeds = [int(x) for x in args.selection_seeds.split(',') if x]

    if not os.path.exists(RESULTS):
        raise SystemExit('No corrected results found at {}'.format(RESULTS))
    with open(RESULTS, encoding='utf-8') as handle:
        runs = [json.loads(line) for line in handle if line.strip()]

    rows, warnings = build_selection_rows(runs, args.loss, selection_seeds or None)
    for warning in warnings:
        print('SKIP: ' + warning, file=sys.stderr)
    if not rows:
        raise SystemExit('No complete candidate groups with validation MSE were found')

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'apcm_plus.csv')
    with open(out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    def diff(a, b):
        return '' if a is None or b is None else '{:+.4f}'.format(b - a)

    print('dataset         H  chosen (by val MSE)       n   MSE     MAE    dMSE/GTR dMAE/GTR dMSE/pap dMAE/pap')
    for row in rows:
        print('{:<12} {:>4}  {:<24} {:>2}  {:.4f}  {:.4f}  {:>8} {:>8} {:>8} {:>8}'.format(
            row['dataset'], row['pred_len'], row['chosen'], row['n'], row['mse'], row['mae'],
            diff(row['mse'], row['gtr_mse']), diff(row['mae'], row['gtr_mae']),
            diff(row['mse'], row['paper_mse']), diff(row['mae'], row['paper_mae'])))
    print('(positive d = selected model is lower = better) Wrote {}'.format(out))


if __name__ == '__main__':
    main()
