"""Run reproducible GTR vs GTA vs APCM experiments.

Fast screening (one seed, two representative datasets):
    python utils/run_apcm_experiments.py --stage screen

Confirm selected datasets with three seeds and all horizons:
    python utils/run_apcm_experiments.py --stage full --only Electricity,Traffic

Runs are resumable. Results and logs are stored under results_apcm/.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results_apcm')
LOG = os.path.join(OUT, 'logs')

LONG = [96, 192, 336, 720]
SHORT = [12, 24, 48, 96]
SEEDS = [2024, 2025, 2026]

DATA = {
    'ETTh1': dict(data='ETTh1', path='ETTh1.csv', enc=7, cycle=24, horizons=LONG,
                  extra='--batch_size 256 --learning_rate 0.001 --train_epochs 30 --patience 5 --dropout 0.5'),
    'ETTh2': dict(data='ETTh2', path='ETTh2.csv', enc=7, cycle=24, horizons=LONG,
                  extra='--batch_size 256 --learning_rate 0.001 --train_epochs 30 --patience 5 --dropout 0.5'),
    'ETTm1': dict(data='ETTm1', path='ETTm1.csv', enc=7, cycle=96, horizons=LONG,
                  extra='--batch_size 256 --learning_rate 0.001 --train_epochs 30 --patience 5 --dropout 0.5'),
    'ETTm2': dict(data='ETTm2', path='ETTm2.csv', enc=7, cycle=96, horizons=LONG,
                  extra='--batch_size 256 --learning_rate 0.001 --train_epochs 30 --patience 5 --dropout 0.5'),
    'Electricity': dict(data='custom', path='electricity.csv', enc=321, cycle=168, horizons=LONG,
                        extra='--batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'Traffic': dict(data='custom', path='traffic.csv', enc=862, cycle=168, horizons=LONG,
                    extra='--individual 1 --batch_size 16 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'Solar': dict(data='Solar', path='solar_AL.txt', enc=137, cycle=144, horizons=LONG,
                  extra='--use_revin 0 --batch_size 64 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'Weather': dict(data='custom', path='weather.csv', enc=21, cycle=144, horizons=LONG,
                    extra='--batch_size 64 --learning_rate 0.001 --train_epochs 30 --patience 5 --dropout 0.5'),
    'PEMS03': dict(data='PEMS', path='PEMS03.npz', enc=358, cycle=288, horizons=SHORT,
                   extra='--use_revin 0 --batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'PEMS04': dict(data='PEMS', path='PEMS04.npz', enc=307, cycle=288, horizons=SHORT,
                   extra='--use_revin 0 --batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'PEMS07': dict(data='PEMS', path='PEMS07.npz', enc=883, cycle=288, horizons=SHORT,
                   extra='--use_revin 0 --batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'PEMS08': dict(data='PEMS', path='PEMS08.npz', enc=170, cycle=288, horizons=SHORT,
                   extra='--use_revin 1 --batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
}

VARIANTS = {
    'GTR': '--model GTR',
    # Author's single-token inter-channel ablation. This is a required control,
    # not the proposed method.
    'GTA': '--model GTR --gtr_agg 1',
    'APCM': ('--model GTRAPCM --apcm_patch_len 16 --apcm_dim 32 '
             '--apcm_latents 8 --apcm_dropout 0.1 --apcm_temperature 0.2 --apcm_spectral 1'),
    # Ablation used to show whether frequency-aware routing contributes.
    'APCM-noFFT': ('--model GTRAPCM --apcm_patch_len 16 --apcm_dim 32 '
                   '--apcm_latents 8 --apcm_dropout 0.1 --apcm_temperature 0.2 --apcm_spectral 0'),
    # APCM plus a learnable per-channel identity in the routing (starts at APCM).
    'APCM-NE': ('--model GTRAPCM --apcm_patch_len 16 --apcm_dim 32 '
                '--apcm_latents 8 --apcm_dropout 0.1 --apcm_temperature 0.2 --apcm_spectral 1 '
                '--apcm_node_emb 1'),
    # RevIN off, following the author's own ablation (paper Table 8, page 18).
    # Placed after the dataset flags, so this --use_revin 0 overrides them.
    'GTR-noRevIN': '--model GTR --use_revin 0',
    'APCM-noRevIN': ('--model GTRAPCM --apcm_patch_len 16 --apcm_dim 32 '
                     '--apcm_latents 8 --apcm_dropout 0.1 --apcm_temperature 0.2 --apcm_spectral 1 '
                     '--use_revin 0'),
}


def job_key(dataset, horizon, variant, seed):
    return '{}_{}_{}_s{}'.format(dataset, horizon, variant, seed)


def completed_keys():
    path = os.path.join(OUT, 'runs.jsonl')
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as handle:
        return {json.loads(line)['key'] for line in handle if line.strip()}


def build_jobs(stage, only, variants, seeds=None, pred_lens=None):
    seeds = seeds or ([2024] if stage == 'screen' else SEEDS)
    pred_lens = set(pred_lens or [])
    selected = list(DATA) if not only else [x for x in only.split(',') if x]
    unknown = sorted(set(selected) - set(DATA))
    if unknown:
        raise ValueError('Unknown datasets: {}'.format(', '.join(unknown)))
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError('Unknown variants: {}'.format(', '.join(unknown)))

    jobs = []
    if stage == 'screen':
        screen = [('Electricity', 96), ('PEMS08', 12)]
        if only:
            screen = [(ds, h) for ds, h in screen if ds in selected]
        for ds, horizon in screen:
            if pred_lens and horizon not in pred_lens:
                continue
            for variant in variants:
                for seed in seeds:
                    jobs.append((ds, horizon, variant, seed))
    else:
        for ds in selected:
            for horizon in DATA[ds]['horizons']:
                if pred_lens and horizon not in pred_lens:
                    continue
                for variant in variants:
                    for seed in seeds:
                        jobs.append((ds, horizon, variant, seed))
    return jobs


def run_one(dataset, horizon, variant, seed, threads):
    cfg = DATA[dataset]
    key = job_key(dataset, horizon, variant, seed)
    args = (
        '-u run.py --is_training 1 --root_path ./dataset/ --data_path {path} '
        '--model_id APCM_{key} --data {data} --features M --seq_len 96 '
        '--pred_len {horizon} --enc_in {enc} --cycle {cycle} {extra} {model_args} '
        '--itr 1 --random_seed {seed} --num_workers 0'
    ).format(path=cfg['path'], key=key, data=cfg['data'], horizon=horizon,
             enc=cfg['enc'], cycle=cfg['cycle'], extra=cfg['extra'],
             model_args=VARIANTS[variant], seed=seed)

    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), PYTHONIOENCODING='utf-8')
    started = time.time()
    process = subprocess.run([sys.executable] + args.split(), cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding='utf-8')
    output = process.stdout + process.stderr
    with open(os.path.join(LOG, key + '.log'), 'w', encoding='utf-8') as handle:
        handle.write(args + '\n' + output)

    metrics = re.findall(r'mse:([\d.eE+-]+), mae:([\d.eE+-]+)', output)
    if process.returncode != 0 or not metrics:
        return key, None
    validation = [float(v) for v in re.findall(r'Vali Loss: ([\d.eE+-]+)', output)]
    return key, {
        'key': key,
        'dataset': dataset,
        'pred_len': horizon,
        'variant': variant,
        'seed': seed,
        'mse': float(metrics[-1][0]),
        'mae': float(metrics[-1][1]),
        'best_vali': min(validation) if validation else None,
        'epochs': len(re.findall(r'Epoch: \d+ cost time', output)),
        'seconds': round(time.time() - started, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['screen', 'full'], default='screen')
    parser.add_argument('--only', default='')
    parser.add_argument('--variants', default='GTR,GTA,APCM')
    parser.add_argument('--seeds', default='',
                        help='comma-separated seeds; default: 2024 for screen, 2024,2025,2026 for full')
    parser.add_argument('--pred_lens', default='',
                        help='optional comma-separated forecast horizons')
    parser.add_argument('--threads', type=int, default=5)
    args = parser.parse_args()

    variants = [x for x in args.variants.split(',') if x]
    seeds = [int(x) for x in args.seeds.split(',') if x]
    pred_lens = [int(x) for x in args.pred_lens.split(',') if x]
    jobs = build_jobs(args.stage, args.only, variants, seeds or None, pred_lens or None)
    os.makedirs(LOG, exist_ok=True)
    done = completed_keys()
    jobs = [job for job in jobs if job_key(*job) not in done]
    print('{} jobs to run (results are resumable)'.format(len(jobs)), flush=True)

    for index, job in enumerate(jobs, 1):
        key, result = run_one(*job, args.threads)
        if result is None:
            print('[{}/{}] FAILED {} (see log)'.format(index, len(jobs), key), flush=True)
            continue
        with open(os.path.join(OUT, 'runs.jsonl'), 'a', encoding='utf-8', newline='') as handle:
            handle.write(json.dumps(result) + '\n')
        print('[{}/{}] {} mse={:.4f} mae={:.4f}'.format(
            index, len(jobs), key, result['mse'], result['mae']), flush=True)


if __name__ == '__main__':
    main()
