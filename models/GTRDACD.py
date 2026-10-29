import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GTR import GTR


class Model(nn.Module):
    """GTR + DACD (Dynamic Adaptive Cycle Detection).

    Khac GTR goc:
      - K chu ky L_1..L_K (tu dong phat hien bang ACF tren train, xem run.py --cycles auto),
        moi chu ky mot bang Q_k (L_k x N) va mot khoi GTR rieng.
      - Bo chon (gate) nhin cua so dau vao, cho trong so w (B, C, K) rieng tung mau, tung kenh:
        g = sum_k w_k * GTR_k(x, Q_k).
      - (v3, --dacd_future) bang mau chu ky R_k (L_k x N), khoi tao 0, dat DUNG PHA len chan troi du bao:
        y += sum_k w_k * R_k[(t + S + j) mod L_k]; future=2 con tru R_k khoi cua so dau vao (du bao phan du).
        GTR goc chi truy xuat Q tren pha cua cua so dau vao, khong dung pha cua tuong lai.
        (Loader tra ve pha tai s_end chu khong phai s_begin -> moi chi so lech cung mot hang so,
        R_k la bang tu do nen chi la hoan vi hang; khoang cach dau vao -> tuong lai van dung bang S.)
    Voi K = 1, future = 0 mo hinh trung khop GTR goc (cung thu tu khoi tao tham so).
    cycle_index do data loader tra ve la t mod lcm(L_1..L_K) (run.py dat --cycle = lcm),
    nen (cycle_index + j) mod L_k = (t + j) mod L_k.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycles = [int(c) for c in str(configs.cycles).split(',')]
        self.mix = configs.dacd_mix

        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.individual = configs.individual

        K = len(self.cycles)
        self.Q = nn.ParameterList([nn.Parameter(torch.zeros(L, self.enc_in), requires_grad=True)
                                   for L in self.cycles])
        self.GTR = nn.ModuleList([GTR(d_series=self.seq_len, c=self.enc_in, CI=self.individual)
                                  for _ in self.cycles])
        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

        # Bo chon: khoi tao 0 -> bat dau bang tron deu, hoc dan cach lech trong so
        self.use_gate = K > 1 and self.mix == 'gate'
        if self.use_gate:
            self.gate = nn.Linear(self.seq_len, K)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        # v3: bang mau chu ky cho chan troi du bao (tao sau cung de khong doi thu tu khoi tao)
        self.future = getattr(configs, 'dacd_future', 0)
        if self.future:
            self.R = nn.ParameterList([nn.Parameter(torch.zeros(L, self.enc_in)) for L in self.cycles])
        self.last_weights = None  # (B, C, K), de phan tich sau khi test

    def forward(self, x, cycle_index):
        # RevIN normalize
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Trong so bo chon (B, C, K) nhin cua so goc
        K = len(self.cycles)
        if self.use_gate:
            w = F.softmax(self.gate(x.permute(0, 2, 1)), dim=-1)
            self.last_weights = w.detach()
        else:
            w = torch.full((x.shape[0], self.enc_in, K), 1.0 / K, device=x.device)

        if self.future == 2:
            x = x - self.template(cycle_index, 0, self.seq_len, w)

        # (B, S, C) -> (B, C, S)
        x_input = x.permute(0, 2, 1)

        # DACD: truy xuat tung bang Q_k nhu GTR goc
        steps = torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)
        g_all = []
        for L, Q, gtr in zip(self.cycles, self.Q, self.GTR):
            gather_index = (cycle_index.view(-1, 1) + steps) % L
            query_input = Q[gather_index].permute(0, 2, 1)
            g_all.append(gtr(x_input, query_input))
        g_all = torch.stack(g_all, dim=-1)  # (B, C, S, K)

        # Tron theo bo chon
        if K == 1:
            global_information = g_all[..., 0]
        else:
            global_information = (g_all * w.unsqueeze(2)).sum(-1)

        # Projection + MLP
        input_proj = self.input_proj(x_input + global_information)
        hidden = self.model(input_proj)
        output = self.output_proj(hidden + input_proj).permute(0, 2, 1)

        if self.future:
            output = output + self.template(cycle_index, self.seq_len, self.pred_len, w)

        # RevIN de-normalize
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output

    def template(self, cycle_index, start, length, w):
        """sum_k w_k * R_k[(t + start + j) mod L_k], j < length -> (B, length, C)."""
        steps = torch.arange(start, start + length, device=cycle_index.device).view(1, -1)
        out = 0
        for k, (L, R) in enumerate(zip(self.cycles, self.R)):
            out = out + R[(cycle_index.view(-1, 1) + steps) % L] * w[:, :, k].unsqueeze(1)
        return out
