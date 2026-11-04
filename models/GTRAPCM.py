"""GTR with Adaptive Partial Channel Mixing (APCM).

APCM keeps the original GTR path unchanged and adds a lightweight, patch-wise
channel correction.  Unlike the single-token GTA ablation from the GTR paper,
APCM uses multiple latent channel groups and assigns channels to groups for each
sample and temporal patch.  Its cost is O(B * P * C * K * D), linear in the
number of channels C, rather than the O(C^2) cost of full channel attention.

The final correction projection is zero-initialized.  Therefore a newly created
GTRAPCM model starts with exactly the same function as GTR (given the same base
parameters), and the extra branch only departs from the baseline when training
finds a useful cross-channel signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GTR import GTR


class AdaptivePartialChannelMixer(nn.Module):
    """Patch-wise, permutation-equivariant channel mixer with latent groups."""

    def __init__(self, patch_len=16, hidden_dim=32, num_latents=8,
                 dropout=0.1, temperature=0.2, spectral=True):
        super().__init__()
        if patch_len <= 0:
            raise ValueError('apcm_patch_len must be positive')
        if hidden_dim <= 0:
            raise ValueError('apcm_dim must be positive')
        if num_latents <= 0:
            raise ValueError('apcm_latents must be positive')
        if temperature <= 0:
            raise ValueError('apcm_temperature must be positive')

        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.temperature = temperature
        self.spectral = spectral

        spectral_dim = patch_len // 2 + 1 if spectral else 0
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_len + spectral_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.latent_queries = nn.Parameter(torch.empty(num_latents, hidden_dim))
        nn.init.normal_(self.latent_queries, std=hidden_dim ** -0.5)

        # A sample/patch/channel-specific gate prevents forced channel mixing.
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self.dropout = nn.Dropout(dropout)
        self.patch_decoder = nn.Linear(hidden_dim, patch_len)
        # Baseline-preserving initialization: APCM initially contributes zero.
        nn.init.zeros_(self.patch_decoder.weight)
        nn.init.zeros_(self.patch_decoder.bias)

        # Kept for post-hoc analysis; populated with detached tensors in forward.
        self.last_channel_to_latent = None
        self.last_gate = None

    def forward(self, x):
        """Return a cross-channel correction for x shaped (B, C, T)."""
        batch, channels, length = x.shape
        pad = (-length) % self.patch_len
        if pad:
            # Replication avoids injecting artificial zeros at the final patch.
            x = F.pad(x, (0, pad), mode='replicate')

        patches = x.unfold(-1, self.patch_len, self.patch_len)
        # (B, C, P, L) -> (B, P, C, L)
        patches = patches.permute(0, 2, 1, 3).contiguous()

        if self.spectral:
            spectrum = torch.log1p(torch.abs(torch.fft.rfft(patches, dim=-1)))
            encoded_input = torch.cat([patches, spectrum], dim=-1)
        else:
            encoded_input = patches
        h = self.patch_encoder(encoded_input)  # (B, P, C, D)

        # Soft clustering: every latent selects relevant channels in each patch.
        h_norm = F.normalize(h, dim=-1)
        q_norm = F.normalize(self.latent_queries, dim=-1)
        scores = torch.einsum('bpcd,kd->bpck', h_norm, q_norm)
        # h and q are unit-normalized already; temperature alone controls the
        # softness.  A second sqrt(D) scale would make large-C assignments
        # almost uniform and collapse the latent groups.
        scores = scores / self.temperature
        channel_to_latent = F.softmax(scores, dim=2)  # normalize over channels
        latent_tokens = torch.einsum('bpck,bpcd->bpkd', channel_to_latent, h)

        # Each channel reads from the latent groups it most closely matches.
        latent_for_channel = F.softmax(scores, dim=-1)  # normalize over groups
        mixed = torch.einsum('bpck,bpkd->bpcd', latent_for_channel, latent_tokens)

        gate = torch.sigmoid(self.gate(h))
        correction = self.patch_decoder(self.dropout((mixed - h) * gate))
        # (B, P, C, L) -> (B, C, P*L) -> original length
        correction = correction.permute(0, 2, 1, 3).contiguous()
        correction = correction.view(batch, channels, -1)[..., :length]

        self.last_channel_to_latent = channel_to_latent.detach()
        self.last_gate = gate.detach()
        return correction


class Model(nn.Module):
    """Original GTR backbone plus a baseline-preserving APCM correction."""

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.individual = configs.individual

        # Keep this construction order identical to models/GTR.py so the base
        # parameters have identical initialization for the same random seed.
        self.Q = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)
        self.GTR = GTR(d_series=self.seq_len, c=self.enc_in, CI=self.individual)
        self.GTR.agg = bool(getattr(configs, 'gtr_agg', 0))
        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

        # Construct APCM last so it cannot perturb the seeded initialization of
        # the original GTR parameters above.
        self.APCM = AdaptivePartialChannelMixer(
            patch_len=getattr(configs, 'apcm_patch_len', 16),
            hidden_dim=getattr(configs, 'apcm_dim', 32),
            num_latents=getattr(configs, 'apcm_latents', 8),
            dropout=getattr(configs, 'apcm_dropout', 0.1),
            temperature=getattr(configs, 'apcm_temperature', 0.2),
            spectral=bool(getattr(configs, 'apcm_spectral', 1)),
        )

    def forward(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)  # (B, C, T)

        gather_index = (cycle_index.view(-1, 1) +
                        torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        query_input = self.Q[gather_index].permute(0, 2, 1)
        global_information = self.GTR(x_input, query_input)

        gtr_features = x_input + global_information
        enhanced_features = gtr_features + self.APCM(gtr_features)

        input_proj = self.input_proj(enhanced_features)
        hidden = self.model(input_proj)
        output = self.output_proj(hidden + input_proj).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean
        return output
