"""Training objectives for GTR / GTR-APCM.

``--loss`` picks the point-wise loss in the time domain:
    mse      plain MSE (the author's setting, default)
    mae      L1
    huber    smooth L1 with delta 1 (data are z-scored, so 1 = one std)
    mse_mae  0.5 * MSE + 0.5 * MAE

``--loss_freq a`` (0 <= a <= 1) mixes in a frequency-domain term in the style
of FreDF (Wang et al., ICLR 2025):
    loss = a * mean |rfft(pred) - rfft(true)| + (1 - a) * time_loss
The FFT runs along the forecast horizon, so errors in the shape of the whole
future curve (phase, period amplitude) are penalized directly instead of
step by step.  a = 0 gives back the time-domain loss exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


TIME_LOSSES = ('mse', 'mae', 'huber', 'mse_mae')


class ForecastLoss(nn.Module):
    def __init__(self, kind='mse', freq_weight=0.0):
        super().__init__()
        if kind not in TIME_LOSSES:
            raise ValueError('--loss must be one of {}, got {}'.format(TIME_LOSSES, kind))
        if not 0.0 <= freq_weight <= 1.0:
            raise ValueError('--loss_freq must be in [0, 1], got {}'.format(freq_weight))
        self.kind = kind
        self.freq_weight = freq_weight

    def time_loss(self, pred, true):
        if self.kind == 'mse':
            return F.mse_loss(pred, true)
        if self.kind == 'mae':
            return F.l1_loss(pred, true)
        if self.kind == 'huber':
            return F.smooth_l1_loss(pred, true, beta=1.0)
        return 0.5 * F.mse_loss(pred, true) + 0.5 * F.l1_loss(pred, true)

    def forward(self, pred, true):
        loss = self.time_loss(pred, true)
        if self.freq_weight > 0:
            # (B, S, C): transform along the horizon S.
            freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(true, dim=1)).abs().mean()
            loss = self.freq_weight * freq + (1 - self.freq_weight) * loss
        return loss
