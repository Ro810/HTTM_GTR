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


def default_candidates(loss_suffix='FD5L1'):
    return ['GTR-' + loss_suffix, 'APCM-' + loss_suffix, 'APCM-CE-' + loss_suffix]


def build_selection_rows(runs, loss_suffix='FD5L1', selection_seeds=None, candidates=None):
    """Return selected result rows and human-readable skip warnings."""
    candidates = list(candidates or default_candidates(loss_suffix))
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


def build_dataset_rows(runs, candidates, selection_seeds=None, threshold=0.01):
    """Pick ONE candidate per dataset for all horizons, by validation MSE only.

    Each candidate is scored by the mean over horizons of its validation MSE
    divided by the first candidate's validation MSE at the same horizon, so
    every horizon weighs the same whatever its error scale. A dataset is
    skipped when any candidate lacks a horizon or a selection seed. Returns
    (horizon rows, dataset summaries, warnings).
    """
    candidates = list(candidates)
    reference = candidates[0]
    requested_seeds = set(selection_seeds or [])
    by_dataset = {}
    for run in runs:
        if run.get('best_vali_mse') is None:
            continue
        cell = by_dataset.setdefault(run['dataset'], {}).setdefault(int(run['pred_len']), {})
        cell.setdefault(run['variant'], {})[int(run['seed'])] = run

    rows, summaries, warnings = [], [], []
    for dataset, cells in sorted(by_dataset.items()):
        horizons = sorted(h for h, cell in cells.items() if any(v in cell for v in candidates))
        if not horizons:
            continue
        missing = sorted({'{} H{}'.format(v, h) for h in horizons for v in candidates
                          if v not in cells[h]})
        if missing:
            warnings.append('{}: missing {}'.format(dataset, ', '.join(missing)))
            continue
        common = set.intersection(*(set(cells[h][v]) for h in horizons for v in candidates))
        seeds = requested_seeds or common
        if not seeds or not seeds.issubset(common):
            warnings.append('{}: candidates do not share seeds {}'.format(
                dataset, ','.join(map(str, sorted(seeds)))))
            continue

        def vali(variant, horizon):
            return statistics.mean(float(cells[horizon][variant][s]['best_vali_mse'])
                                   for s in seeds)

        scores = {v: statistics.mean(vali(v, h) / vali(reference, h) for h in horizons)
                  for v in candidates}
        chosen = min(candidates, key=lambda v: (scores[v], v))

        dataset_rows = []
        for horizon in horizons:
            members = cells[horizon][chosen]
            report_seeds = sorted(members)
            gtr = cells[horizon].get('GTR', {})
            paired = set(report_seeds).issubset(gtr)
            paper = PAPER.get((dataset, horizon), (None, None))
            dataset_rows.append({
                'dataset': dataset,
                'pred_len': horizon,
                'chosen': chosen,
                'relative_vali_mse': scores[chosen],
                'selection_seeds': ','.join(map(str, sorted(seeds))),
                'report_seeds': ','.join(map(str, report_seeds)),
                'n': len(report_seeds),
                'mse': statistics.mean(float(members[s]['mse']) for s in report_seeds),
                'mae': statistics.mean(float(members[s]['mae']) for s in report_seeds),
                'gtr_mse': (statistics.mean(float(gtr[s]['mse']) for s in report_seeds)
                            if paired else None),
                'gtr_mae': (statistics.mean(float(gtr[s]['mae']) for s in report_seeds)
                            if paired else None),
                'paper_mse': paper[0],
                'paper_mae': paper[1],
                'candidates_tried': len(candidates),
            })
        rows.extend(dataset_rows)

        def avg(key):
            values = [row[key] for row in dataset_rows]
            return None if any(v is None for v in values) else statistics.mean(values)

        mse, gtr_mse, paper_mse = avg('mse'), avg('gtr_mse'), avg('paper_mse')
        gain = None if gtr_mse is None else gtr_mse - mse
        summaries.append({
            'dataset': dataset,
            'chosen': chosen,
            'horizons': len(horizons),
            'mse': mse,
            'gtr_mse': gtr_mse,
            'd_mse_vs_gtr': gain,
            'd_mse_vs_paper': None if paper_mse is None else paper_mse - mse,
            'meets_threshold': None if gain is None else gain >= threshold,
            'scores': scores,
        })
    return rows, summaries, warnings


def load_runs():
    if not os.path.exists(RESULTS):
        raise SystemExit('No corrected results found at {}'.format(RESULTS))
    with open(RESULTS, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(rows, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    with open(out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return out


def fmt(value, spec='{:+.4f}'):
    return '' if value is None else spec.format(value)


def main_per_dataset(runs, candidates, selection_seeds, threshold, only, tag):
    if only:
        runs = [run for run in runs if run['dataset'] in only]
    rows, summaries, warnings = build_dataset_rows(runs, candidates, selection_seeds, threshold)
    for warning in warnings:
        print('SKIP: ' + warning, file=sys.stderr)
    if not rows:
        raise SystemExit('No dataset has every candidate on every horizon and seed')
    out = write_csv(rows, 'per_dataset{}.csv'.format('_' + tag if tag else ''))

    print('Candidates ({} tried, first = score reference): {}'.format(
        len(candidates), ', '.join(candidates)))
    print('dataset       chosen (by val MSE)          H   MSE     GTR     dMSE/GTR dMSE/pap  >= {}'
          .format(threshold))
    for s in summaries:
        print('{:<12}  {:<28} {:>2}  {:.4f}  {:>6}  {:>8} {:>8}  {}'.format(
            s['dataset'], s['chosen'], s['horizons'], s['mse'], fmt(s['gtr_mse'], '{:.4f}'),
            fmt(s['d_mse_vs_gtr']), fmt(s['d_mse_vs_paper']),
            {True: 'YES', False: 'no', None: '(no paired GTR)'}[s['meets_threshold']]))
    print('(d = reference - selected; positive = selected model is better) Wrote {}'.format(out))
    print('\nRelative validation MSE per candidate (lower is better):')
    for s in summaries:
        ranked = sorted(s['scores'].items(), key=lambda item: item[1])
        print('{:<12}  '.format(s['dataset']) +
              '  '.join('{} {:.4f}'.format(v, score) for v, score in ranked))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', default='FD5L1', help='variant suffix of the shared loss')
    parser.add_argument(
        '--selection_seeds', default='',
        help='comma-separated seeds required for every candidate; default: shared seeds',
    )
    parser.add_argument('--candidates', default='',
                        help='comma-separated variants; default: GTR/APCM/APCM-CE with --loss')
    parser.add_argument('--per_dataset', action='store_true',
                        help='choose one candidate per dataset for all horizons')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='mean MSE reduction vs paired GTR counted as a clear gain')
    parser.add_argument('--only', default='',
                        help='per_dataset: comma-separated datasets to include')
    parser.add_argument('--tag', default='',
                        help='per_dataset: output is per_dataset_<tag>.csv')
    args = parser.parse_args()
    selection_seeds = [int(x) for x in args.selection_seeds.split(',') if x]
    candidates = [x for x in args.candidates.split(',') if x] or default_candidates(args.loss)

    runs = load_runs()
    if args.per_dataset:
        only = {x for x in args.only.split(',') if x}
        main_per_dataset(runs, candidates, selection_seeds or None, args.threshold,
                         only, args.tag)
        return

    rows, warnings = build_selection_rows(runs, args.loss, selection_seeds or None, candidates)
    for warning in warnings:
        print('SKIP: ' + warning, file=sys.stderr)
    if not rows:
        raise SystemExit('No complete candidate groups with validation MSE were found')

    out = write_csv(rows, 'apcm_plus.csv')

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
