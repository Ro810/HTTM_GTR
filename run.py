import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import math
import numpy as np
import pandas as pd
from utils.detect_cycles import detect

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2026, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='GTR',
                    help='model name, options: [Informer, Autoformer, ...]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# TQNet & CycleNet
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

# DACD
parser.add_argument('--cycles', type=str, default='auto',
                    help='GTRDACD: "auto" = detect by ACF on train split, or list like "24,168"')
parser.add_argument('--dacd_mix', type=str, default='gate', help='GTRDACD: gate or uniform')
parser.add_argument('--dacd_delta', type=float, default=0.02, help='GTRDACD: harmonic threshold for auto detection')
parser.add_argument('--dacd_future', type=int, default=0,
                    help='GTRDACD v3: 0 off, 1 add phase-aligned cycle template R_k on the forecast horizon, '
                         '2 also remove R_k from the input window (residual)')
parser.add_argument('--dacd_nested', type=int, default=0, help='GTRDACD v2: secondary cycles must be integer multiples of the base cycle')
parser.add_argument('--dacd_refine', type=int, default=0, help='GTRDACD v4: refine the base cycle by mean ACF over its multiples')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')
parser.add_argument('--seg_len', type=int, default=48, help='segment length')
parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

def detect_train_cycles(args):
    """DACD buoc 1: ACF tren dung phan TRAIN theo cach chia cua data_loader.py."""
    path = os.path.join(args.root_path, args.data_path)
    if args.data == 'PEMS':                      # Dataset_PEMS: .npz, train = 60% dau
        values = np.load(path, allow_pickle=True)['data'][:, :, 0].astype(float)
        n_train = int(len(values) * 0.6)
    elif args.data == 'Solar':                   # Dataset_Solar: .txt khong header, train = 70% dau
        values = np.loadtxt(path, delimiter=',')
        n_train = int(len(values) * 0.7)
    else:
        values = pd.read_csv(path).drop(columns=['date']).values.astype(float)
        if args.data in ('ETTh1', 'ETTh2'):
            n_train = 12 * 30 * 24
        elif args.data in ('ETTm1', 'ETTm2'):
            n_train = 12 * 30 * 24 * 4
        else:                                    # Dataset_Custom (Electricity, Traffic, Weather)
            n_train = int(len(values) * 0.7)
    res, _, _ = detect(values[:n_train], delta=args.dacd_delta, nested=bool(args.dacd_nested), refine=bool(args.dacd_refine))
    if not res['cycles']:
        raise ValueError('DACD: no cycle detected on train split')
    return res['cycles']


if __name__ == "__main__":
    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.model == 'GTRDACD':
        if args.cycles == 'auto':
            args.cycles = ','.join(str(c) for c in detect_train_cycles(args))
        # data loader tra ve t mod args.cycle; dung lcm de moi L_k deu suy ra dung pha
        args.cycle = math.lcm(*[int(c) for c in args.cycles.split(',')])
        print('DACD cycles: {} (loader cycle = lcm = {})'.format(args.cycles, args.cycle))

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main


    if args.is_training:
        for ii in range(args.itr):

            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.cycle,
                fix_seed)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.cycle,
            fix_seed)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
