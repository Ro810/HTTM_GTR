"""Tong hop ket qua --grid full: bang MSE/MAE trung binh qua seed, chenh lech cung seed so voi GTR.

Cot 'best_by_vali' = bien the DACD co Vali Loss trung binh thap nhat (chon KHONG nhin tap test).
Chay:  python utils/summarize_full.py [--dir results_dacd/full]   -> in bang + ghi summary.csv
"""
import argparse, json, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(ROOT, 'results_dacd', 'full'))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.dir, 'runs.jsonl'), encoding='utf-8') if l.strip()]
    df = pd.DataFrame(rows)
    df['variant'] = df.variant.where(~df.variant.str.startswith('GTR-L'), 'GTR')

    s = df.groupby(['dataset', 'pred_len', 'variant']).agg(
        cycles=('cycles', 'first'), n=('seed', 'count'), mse=('mse', 'mean'), mae=('mae', 'mean'),
        vali=('vali', 'mean')).reset_index()

    # chenh lech cung seed so voi GTR (am = DACD tot hon) va so seed thang
    base = df[df.variant == 'GTR'].set_index(['dataset', 'pred_len', 'seed'])[['mse', 'mae']]
    d = df[df.variant != 'GTR'].join(base, on=['dataset', 'pred_len', 'seed'], rsuffix='_gtr').dropna(subset=['mse_gtr'])
    d['d_mse'], d['d_mae'] = d.mse - d.mse_gtr, d.mae - d.mae_gtr
    w = d.groupby(['dataset', 'pred_len', 'variant']).agg(
        d_mse=('d_mse', 'mean'), d_mae=('d_mae', 'mean'),
        wins=('d_mse', lambda x: '%d/%d' % ((x < 0).sum(), len(x)))).reset_index()
    s = s.merge(w, on=['dataset', 'pred_len', 'variant'], how='left')

    dacd = s[s.variant != 'GTR'].dropna(subset=['vali'])
    pick = dacd.loc[dacd.groupby(['dataset', 'pred_len']).vali.idxmin(), ['dataset', 'pred_len', 'variant']]
    s = s.merge(pick.rename(columns={'variant': 'best_by_vali'}), on=['dataset', 'pred_len'], how='left')
    s['best_by_vali'] = s.best_by_vali == s.variant

    s.to_csv(os.path.join(a.dir, 'summary.csv'), index=False, float_format='%.4f')
    pd.set_option('display.width', 200)
    print(s.to_string(index=False, float_format=lambda v: '%.4f' % v))

    # trung binh 4 horizon (giong cot Avg cua bai bao), chi khi du 4 horizon
    avg = s.groupby(['dataset', 'variant']).agg(H=('pred_len', 'count'), mse=('mse', 'mean'), mae=('mae', 'mean'))
    avg = avg[avg.H == 4]
    print('\nTrung binh 4 horizon:\n' + (avg.to_string(float_format=lambda v: '%.4f' % v) if len(avg) else '(chua co dataset nao du 4 horizon)'))
