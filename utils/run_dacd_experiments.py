"""Chay luoi thuc nghiem DACD vs GTR, song song tren CPU, ghi ket qua vao results_dacd/runs.jsonl.

Moi dong jsonl = 1 lan chay (dataset, pred_len, variant, seed, mse, mae, ...).
Chay lai se bo qua cac lan da co ket qua (co the dung giua chung).

Chay:  python utils/run_dacd_experiments.py [--workers 4] [--threads 5] [--only ETTh1,ETTm2]
       python utils/run_dacd_experiments.py --grid full [--only Solar,Weather] [--variants GTR,DACD-F1]
         (--grid full: 12 dataset cua bai bao, sieu tham so goc cua tac gia, ket qua o results_dacd/full/;
          danh cho may co GPU, xem docs/DACD_RUN_GUIDE.md)
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results_dacd')
LOG = os.path.join(OUT, 'logs')
SEEDS = [2024, 2025, 2026]

# Sieu tham so lay tu scripts/GTR/*.sh (Electricity: 6 epoch / patience 2 vi chay CPU, ap dung chung moi bien the)
DATA = {
    'ETTh1': dict(data='ETTh1', path='ETTh1.csv', enc_in=7, L=24, extra='--dropout 0.5 --batch_size 256 --learning_rate 0.001 --train_epochs 30 --patience 5'),
    'ETTh2': dict(data='ETTh2', path='ETTh2.csv', enc_in=7, L=24, extra='--dropout 0.5 --batch_size 256 --learning_rate 0.001 --train_epochs 30 --patience 5'),
    'ETTm1': dict(data='ETTm1', path='ETTm1.csv', enc_in=7, L=96, extra='--dropout 0.5 --batch_size 256 --learning_rate 0.001 --train_epochs 30 --patience 5'),
    'ETTm2': dict(data='ETTm2', path='ETTm2.csv', enc_in=7, L=96, extra='--dropout 0.5 --batch_size 256 --learning_rate 0.001 --train_epochs 30 --patience 5'),
    'Electricity': dict(data='custom', path='electricity.csv', enc_in=321, L=168, extra='--batch_size 32 --learning_rate 0.003 --train_epochs 6 --patience 2'),
}

# --grid full: dung NGUYEN sieu tham so trong scripts/GTR/*.sh (12 dataset, 4 horizon)
ETT_ARGS = '--dropout 0.5 --batch_size 256 --learning_rate 0.001 --train_epochs 30 --patience 5'
LONG, SHORT = [96, 192, 336, 720], [12, 24, 48, 96]
FULL = {
    'ETTh1': dict(data='ETTh1', path='ETTh1.csv', enc_in=7, L=24, extra=ETT_ARGS, pls=LONG),
    'ETTh2': dict(data='ETTh2', path='ETTh2.csv', enc_in=7, L=24, extra=ETT_ARGS, pls=LONG),
    'ETTm1': dict(data='ETTm1', path='ETTm1.csv', enc_in=7, L=96, extra=ETT_ARGS, pls=LONG),
    'ETTm2': dict(data='ETTm2', path='ETTm2.csv', enc_in=7, L=96, extra=ETT_ARGS, pls=LONG),
    'Weather': dict(data='custom', path='weather.csv', enc_in=21, L=144, pls=LONG,
                    extra='--dropout 0.5 --batch_size 64 --learning_rate 0.001 --train_epochs 30 --patience 5'),
    'Solar': dict(data='Solar', path='solar_AL.txt', enc_in=137, L=144, pls=LONG,
                  extra='--use_revin 0 --batch_size 64 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'Electricity': dict(data='custom', path='electricity.csv', enc_in=321, L=168, pls=LONG,
                        extra='--batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'Traffic': dict(data='custom', path='traffic.csv', enc_in=862, L=168, pls=LONG,
                    extra='--individual 1 --batch_size 16 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'PEMS03': dict(data='PEMS', path='PEMS03.npz', enc_in=358, L=288, pls=SHORT,
                   extra='--use_revin 0 --batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'PEMS04': dict(data='PEMS', path='PEMS04.npz', enc_in=307, L=288, pls=SHORT,
                   extra='--use_revin 0 --batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'PEMS07': dict(data='PEMS', path='PEMS07.npz', enc_in=883, L=288, pls=SHORT,
                   extra='--use_revin 0 --batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
    'PEMS08': dict(data='PEMS', path='PEMS08.npz', enc_in=170, L=288, pls=SHORT,
                   extra='--use_revin 1 --batch_size 32 --learning_rate 0.003 --train_epochs 30 --patience 5'),
}


def gtr(L, seeds=None):
    return dict(variant='GTR-L%d' % L, args='--model GTR --cycle %d' % L, seeds=seeds)


def dacd(cycles='auto', mix='gate', name=None, seeds=None, future=0):
    return dict(variant=name or 'DACD-%s' % mix, seeds=seeds,
                args='--model GTRDACD --cycles %s --dacd_mix %s --dacd_future %d' % (cycles, mix, future))


def grid_v3():
    """v3: bang mau chu ky dat dung pha tren chan troi du bao (F1) / them du bao phan du (F2)."""
    jobs = []
    add = lambda ds, pls, cfgs: jobs.extend((ds, pl, c) for pl in pls for c in cfgs)
    for ds in ['ETTh1', 'ETTh2']:
        add(ds, [96, 336], [gtr(DATA[ds]['L']), dacd(name='DACD-F1', future=1), dacd(name='DACD-F2', future=2)])
    return [(ds, pl, c, s) for ds, pl, c in jobs for s in (c.get('seeds') or SEEDS)]


def grid_full(variants, seeds):
    """12 dataset x 4 horizon. GTR = chu ky tac gia; DACD* = tu phat hien (luat v2: chu ky phu la boi so nguyen)."""
    cfgs = {'GTR': lambda ds: gtr(FULL[ds]['L']),
            'DACD': lambda ds: dacd('auto --dacd_nested 1', name='DACD'),
            'DACD-F1': lambda ds: dacd('auto --dacd_nested 1', name='DACD-F1', future=1),
            'DACD-F2': lambda ds: dacd('auto --dacd_nested 1', name='DACD-F2', future=2)}
    return [(ds, pl, cfgs[v](ds), s) for ds in FULL for pl in FULL[ds]['pls'] for v in variants for s in seeds]


def grid(extra=False):
    """(dataset, pred_len, cfg, seed). Thu tu = thu tu uu tien."""
    jobs = []
    add = lambda ds, pls, cfgs: jobs.extend((ds, pl, c) for pl in pls for c in cfgs)
    # 1) Bang chinh, H=96: GTR tac gia / khong chu ky (L=1) / DACD tu dong
    #    (ETTh1, ETTh2, ETTm1: DACD tu dong ra 1 chu ky trung tac gia -> trung GTR, da kiem chung bit-by-bit)
    add('ETTm2', [96], [gtr(96), gtr(336), gtr(1), dacd(mix='uniform'), dacd()])
    # Electricity (CPU): dot 1 = du 4 cau hinh seed 2024, dot 2 = them seed cho GTR-L168 va DACD-gate
    add('Electricity', [96], [gtr(168, [2024]), dacd(seeds=[2024]), gtr(24, [2024]), dacd(mix='uniform', seeds=[2024]),
                              gtr(168, [2025, 2026]), dacd(seeds=[2025, 2026])])
    for ds in ['ETTh1', 'ETTh2', 'ETTm1']:
        add(ds, [96], [gtr(DATA[ds]['L']), gtr(1), dacd()])
    # 2) Do nhay theo L tren ETTh1 (dong luc: dat sai L thi sao)
    add('ETTh1', [96], [gtr(L) for L in [12, 23, 25, 48, 168]])
    # 3) Ep 2 ung vien {24,168} tren ETTh: bo chon co tu bo chu ky yeu khong?
    for ds in ['ETTh1', 'ETTh2']:
        add(ds, [96], [dacd('24,168', 'uniform', 'DACD-24+168-uniform'), dacd('24,168', 'gate', 'DACD-24+168-gate')])
    # 4) Cac horizon dai tren ETTm2 (chi chay khi --extra, khong kip tren CPU)
    if extra:
        add('ETTm2', [192, 336, 720], [gtr(96), dacd()])
    return [(ds, pl, c, s) for ds, pl, c in jobs for s in (c.get('seeds') or SEEDS)]


def key(ds, pl, variant, seed):
    return '%s_%d_%s_s%d' % (ds, pl, variant, seed)


def done_keys():
    p = os.path.join(OUT, 'runs.jsonl')
    if not os.path.exists(p):
        return set()
    return {json.loads(l)['key'] for l in open(p, encoding='utf-8') if l.strip()}


def run_one(ds, pl, cfg, seed, threads):
    d = DATA[ds]
    k = key(ds, pl, cfg['variant'], seed)
    cmd = ('-u run.py --is_training 1 --root_path ./dataset/ --data_path {path} --model_id {k} '
           '--data {data} --features M --seq_len 96 --pred_len {pl} --enc_in {enc} {extra} {margs} '
           '--itr 1 --random_seed {seed} --num_workers 0').format(
        path=d['path'], k=k, data=d['data'], pl=pl, enc=d['enc_in'],
        extra=d['extra'], margs=cfg['args'], seed=seed)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONIOENCODING='utf-8')
    t0 = time.time()
    p = subprocess.run([sys.executable] + cmd.split(), cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
    out = p.stdout + p.stderr
    with open(os.path.join(LOG, k + '.log'), 'w', encoding='utf-8') as f:
        f.write(cmd + '\n' + out)
    m = re.findall(r'mse:([\d.eE+-]+), mae:([\d.eE+-]+)', out)
    if p.returncode != 0 or not m:
        return k, None
    cyc = re.search(r'DACD cycles: ([\d,]+)', out)
    vali = [float(v) for v in re.findall(r'Vali Loss: ([\d.eE+-]+)', out)]
    return k, dict(key=k, vali=min(vali) if vali else None, dataset=ds, pred_len=pl, variant=cfg['variant'], seed=seed,
                   cycles=cyc.group(1) if cyc else cfg['args'].split('--cycle ')[-1],
                   mse=float(m[-1][0]), mae=float(m[-1][1]),
                   epochs=len(re.findall(r'Epoch: \d+ cost time', out)), seconds=round(time.time() - t0, 1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=5)
    ap.add_argument('--only', default='')
    ap.add_argument('--skip', default='')
    ap.add_argument('--extra', action='store_true')
    ap.add_argument('--grid', default='v2', help='v2 | v3 (CPU, bao cao 21/10) | full (12 dataset, GPU)')
    ap.add_argument('--variants', default='GTR,DACD,DACD-F1,DACD-F2', help='--grid full: bien the can chay')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='--grid full: danh sach seed')
    a = ap.parse_args()
    if a.grid == 'full':
        DATA, OUT = FULL, os.path.join(OUT, 'full')
        LOG = os.path.join(OUT, 'logs')
        jobs = grid_full(a.variants.split(','), [int(x) for x in a.seeds.split(',')])
    else:
        jobs = grid_v3() if a.grid == 'v3' else grid(a.extra)
    os.makedirs(LOG, exist_ok=True)
    done = done_keys()
    todo = [j for j in jobs if key(j[0], j[1], j[2]['variant'], j[3]) not in done
            and (not a.only or j[0] in a.only.split(',')) and j[0] not in a.skip.split(',')]
    print('%d jobs to run (%d already done)' % (len(todo), len(done)), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, *j, a.threads) for j in todo]
        for i, f in enumerate(as_completed(futs), 1):
            k, r = f.result()
            if r is None:
                print('[%d/%d] FAILED %s (xem logs)' % (i, len(todo), k), flush=True)
                continue
            with open(os.path.join(OUT, 'runs.jsonl'), 'a', encoding='utf-8') as fo:
                fo.write(json.dumps(r) + '\n')
            print('[%d/%d] %s mse=%.4f mae=%.4f (%d ep, %.0fs)' % (i, len(todo), k, r['mse'], r['mae'], r['epochs'], r['seconds']), flush=True)
