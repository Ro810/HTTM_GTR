from types import SimpleNamespace

import torch

from models import GTR, GTRAPCM


def configs(**overrides):
    values = dict(
        seq_len=24,
        pred_len=12,
        enc_in=7,
        cycle=24,
        d_model=32,
        dropout=0.0,
        use_revin=1,
        individual=0,
        gtr_agg=0,
        apcm_patch_len=8,
        apcm_dim=16,
        apcm_latents=4,
        apcm_dropout=0.0,
        apcm_temperature=0.2,
        apcm_spectral=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_apcm_starts_exactly_at_gtr_baseline():
    cfg = configs()
    torch.manual_seed(123)
    baseline = GTR.Model(cfg).eval()
    torch.manual_seed(123)
    proposed = GTRAPCM.Model(cfg).eval()

    x = torch.randn(3, cfg.seq_len, cfg.enc_in)
    cycle_index = torch.tensor([0, 3, 9])
    with torch.no_grad():
        expected = baseline(x, cycle_index)
        actual = proposed(x, cycle_index)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_apcm_shape_and_correction_gradient():
    cfg = configs(seq_len=25, apcm_patch_len=8)
    model = GTRAPCM.Model(cfg).train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    cycle_index = torch.tensor([0, 4])

    output = model(x, cycle_index)
    assert output.shape == (2, cfg.pred_len, cfg.enc_in)
    output.square().mean().backward()
    gradient = model.APCM.patch_decoder.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
