"""Tong hop ket qua DACD: bang mean +- std qua 3 seed, hinh so sanh, va phan tich trong so bo chon.

Chay:  python utils/analyze_dacd.py            (bang + hinh tu results_dacd/runs.jsonl)
       python utils/analyze_dacd.py --gate     (them phan tich trong so bo chon, can checkpoint)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results_dacd')
sys.stdout.reconfigure(encoding='utf-8')

# So lieu GTR trong bai bao (Table 7, T=96) de doi chieu muc tai lap
PAPER = {('ETTh1', 96): (0.367, 0.391), ('ETTh2', 96): (0.290, 0.342), ('ETTm1', 96): (0.305, 0.349),
         ('ETTm2', 96): (0.168, 0.249), ('ETTm2', 192): (0.232, 0.292), ('ETTm2', 336): (0.287, 0.330),
         ('ETTm2', 720): (0.386, 0.390), ('Electricity', 96): (0.134, 0.229)}


def load():
    rows = [json.loads(l) for l in open(os.path.join(OUT, 'runs.jsonl'), encoding='utf-8') if l.strip()]
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(['dataset', 'pred_len', 'variant'])
    s = g.agg(cycles=('cycles', 'first'), n=('seed', 'count'),
              mse=('mse', 'mean'), mse_std=('mse', 'std'), mae=('mae', 'mean'), mae_std=('mae', 'std'),
              epochs=('epochs', 'mean'), seconds=('seconds', 'mean')).reset_index()
    s['paper_mse'] = [PAPER.get((d, p), (np.nan,))[0] if v.startswith('GTR-L') and str(c) == str(author(d)) else np.nan
                      for d, p, v, c in zip(s.dataset, s.pred_len, s.variant, s.cycles)]
    s.to_csv(os.path.join(OUT, 'summary.csv'), index=False, float_format='%.4f')
    return s


def author(ds):
    return {'ETTh1': 24, 'ETTh2': 24, 'ETTm1': 96, 'ETTm2': 96, 'Electricity': 168}[ds]


def paired(df, ds, pl, a, b):
    """Chenh lech MSE cung seed (b - a): am = b tot hon a."""
    x = df[(df.dataset == ds) & (df.pred_len == pl)].pivot(index='seed', columns='variant', values='mse')
    if a not in x or b not in x:
        return None
    d = (x[b] - x[a]).dropna()
    return d.mean(), d.std(), int((d < 0).sum()), len(d)


def fig_sweep(df):
    x = df[(df.dataset == 'ETTh1') & (df.pred_len == 96) & df.variant.str.startswith('GTR-L')].copy()
    if x.empty:
        return
    x['L'] = x.variant.str[5:].astype(int)
    s = x.groupby('L').mse.agg(['mean', 'std']).reset_index().sort_values('L')
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    pos = np.arange(len(s))
    colors = ['#c0392b' if L == 24 else '#7f8c8d' for L in s.L]
    ax.bar(pos, s['mean'], yerr=s['std'], color=colors, capsize=3)
    ax.set_xticks(pos, ['L=%d' % L + (' (tác giả)' if L == 24 else '') for L in s.L], fontsize=8)
    lo, hi = s['mean'].min(), s['mean'].max()
    ax.set_ylim(lo - 0.6 * (hi - lo) - 0.002, hi + 0.4 * (hi - lo) + 0.002)
    ax.set_ylabel('MSE (test, H=96)')
    ax.set_title('ETTh1: GTR nhạy với độ dài chu kỳ L (3 seed)', fontsize=10)
    for p, m in zip(pos, s['mean']):
        ax.text(p, m, '%.4f' % m, ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_etth1_L_sweep.png'), dpi=160)
    plt.close(fig)


def fig_main(s):
    cases = [('ETTm2', 96, ['GTR-L1', 'GTR-L96', 'GTR-L336', 'DACD-uniform', 'DACD-gate']),
             ('Electricity', 96, ['GTR-L24', 'GTR-L168', 'DACD-uniform', 'DACD-gate'])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (ds, pl, vs) in zip(axes, cases):
        x = s[(s.dataset == ds) & (s.pred_len == pl)].set_index('variant').reindex(vs).dropna(subset=['mse'])
        if x.empty:
            ax.set_visible(False)
            continue
        colors = ['#2e86c1' if v.startswith('DACD') else '#7f8c8d' for v in x.index]
        pos = np.arange(len(x))
        ax.bar(pos, x.mse, yerr=x.mse_std.fillna(0), color=colors, capsize=3)
        ax.set_xticks(pos, x.index, fontsize=8, rotation=15)
        lo, hi = x.mse.min(), x.mse.max()
        ax.set_ylim(lo - 0.6 * (hi - lo) - 0.001, hi + 0.4 * (hi - lo) + 0.001)
        for p, m in zip(pos, x.mse):
            ax.text(p, m, '%.4f' % m, ha='center', va='bottom', fontsize=7)
        ax.set_title('%s, H=%d (MSE, 3 seed)' % (ds, pl), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_main_compare.png'), dpi=160)
    plt.close(fig)


def load_model(ds, pl, variant, seed=2024):
    """Nap lai mo hinh (GTR hoac GTRDACD) da huan luyen tu log + checkpoint. Tra ve (model, args)."""
    import torch
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import run as runmod
    from models import GTR, GTRDACD
    path = os.path.join(OUT, 'logs', '%s_%d_%s_s%d.log' % (ds, pl, variant, seed))
    toks = open(path, encoding='utf-8').readline().split()
    args = runmod.parser.parse_args(toks[toks.index('run.py') + 1:])
    if args.model == 'GTRDACD':
        for line in open(path, encoding='utf-8'):
            if line.startswith('DACD cycles:'):
                args.cycles = line.split()[2]
        args.cycle = int(np.lcm.reduce([int(c) for c in args.cycles.split(',')]))
    args.use_gpu = False
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(args.model_id, args.model, args.data, args.features,
                                                             args.seq_len, args.pred_len, args.cycle, seed)
    model = {'GTR': GTR, 'GTRDACD': GTRDACD}[args.model].Model(args).float()
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints', setting, 'checkpoint.pth'), map_location='cpu'))
    model.eval()
    return model, args


def gate_weights(ds, pl, variant, seed=2024):
    """Nap checkpoint GTRDACD, chay tren tap test, tra ve (phase, w) voi w: (n_mau, C, K)."""
    import torch
    model, args = load_model(ds, pl, variant, seed)
    from data_provider.data_factory import data_provider
    cyc = args.cycles
    _, loader = data_provider(args, 'test')
    phases, ws = [], []
    with torch.no_grad():
        for bx, _, _, _, bc in loader:
            model(bx.float(), bc.int())
            ws.append(model.last_weights.numpy())
            phases.append(bc.numpy())
    return [int(c) for c in cyc.split(',')], np.concatenate(phases), np.concatenate(ws)


def fig_gate(cases):
    fig, axes = plt.subplots(len(cases), 2, figsize=(10, 3.2 * len(cases)), squeeze=False)
    report, curves = {}, {}
    for row, (ds, pl, variant) in enumerate(cases):
        try:
            cyc, ph, w = gate_weights(ds, pl, variant)
        except (FileNotFoundError, AttributeError) as e:
            print('bo qua', ds, variant, e)
            continue
        wl = w[..., -1]  # trong so cua chu ky dai nhat
        Lb = cyc[-1]
        p = ph % Lb
        per_phase = pd.Series(wl.mean(1)).groupby(p).mean()
        ax = axes[row, 0]
        ax.plot(per_phase.index, per_phase.values, color='#2e86c1')
        ax.set_xlabel('pha trong chu kỳ %d (vị trí bắt đầu dự báo)' % Lb)
        ax.set_ylabel('w(Q_%d) trung bình' % Lb)
        ax.set_title('%s: trọng số thay đổi theo từng mẫu' % ds, fontsize=10)
        ax = axes[row, 1]
        per_ch = wl.mean(0)
        ax.hist(per_ch, bins=30, color='#7f8c8d')
        ax.set_xlabel('w(Q_%d) trung bình theo kênh' % Lb)
        ax.set_ylabel('số kênh')
        ax.set_title('%s: mỗi kênh chọn khác nhau (%d kênh)' % (ds, w.shape[1]), fontsize=10)
        report[ds + '_' + variant] = dict(cycles=cyc, w_long_mean=float(wl.mean()), w_long_std_samples=float(wl.mean(1).std()),
                                          w_long_min_ch=float(per_ch.min()), w_long_max_ch=float(per_ch.max()),
                                          phase_range=[float(per_phase.min()), float(per_phase.max())])
        curves[ds + '_' + variant] = dict(phase=[int(i) for i in per_phase.index], w_long=[round(float(v), 4) for v in per_phase.values],
                                          per_channel=[round(float(v), 4) for v in per_ch])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_gate_weights.png'), dpi=160)
    plt.close(fig)
    json.dump(report, open(os.path.join(OUT, 'gate_weights.json'), 'w'), indent=1)
    json.dump(curves, open(os.path.join(OUT, 'gate_curves.json'), 'w'))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', action='store_true')
    a = ap.parse_args()
    df = load()
    s = summarize(df)
    pd.set_option('display.width', 200)
    print(s.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print('\nCung seed, chenh MSE (b - a), so seed b tot hon:')
    for ds, pl, a_, b_ in [('ETTm2', 96, 'GTR-L96', 'DACD-gate'), ('ETTm2', 96, 'GTR-L96', 'DACD-uniform'),
                           ('ETTm2', 96, 'DACD-uniform', 'DACD-gate'),
                           ('Electricity', 96, 'GTR-L168', 'DACD-gate'), ('Electricity', 96, 'GTR-L168', 'DACD-uniform'),
                           ('Electricity', 96, 'DACD-uniform', 'DACD-gate'),
                           ('ETTh1', 96, 'GTR-L24', 'DACD-24+168-gate'), ('ETTh2', 96, 'GTR-L24', 'DACD-24+168-gate')] + \
                          [('ETTm2', h, 'GTR-L96', 'DACD-gate') for h in (192, 336, 720)]:
        r = paired(df, ds, pl, a_, b_)
        if r:
            print('  %-11s H=%-3d %-14s -> %-20s %+.4f +- %.4f  (%d/%d seed)' % (ds, pl, a_, b_, *r))
    fig_sweep(df)
    fig_main(s)
    if a.gate:
        fig_gate([('Electricity', 96, 'DACD-gate'), ('ETTm2', 96, 'DACD-gate'), ('ETTh1', 96, 'DACD-24+168-gate')])
