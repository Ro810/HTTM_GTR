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


def test_apcm_node_emb_starts_exactly_at_apcm_and_gtr():
    cfg = configs()
    torch.manual_seed(7)
    baseline = GTR.Model(cfg).eval()
    torch.manual_seed(7)
    plain = GTRAPCM.Model(cfg).eval()
    torch.manual_seed(7)
    with_ne = GTRAPCM.Model(configs(apcm_node_emb=1)).eval()

    # The shared parameters get the same seeded values with and without NE.
    plain_state = plain.state_dict()
    for name, value in with_ne.state_dict().items():
        if name in plain_state:
            torch.testing.assert_close(value, plain_state[name], rtol=0, atol=0)

    x = torch.randn(3, cfg.seq_len, cfg.enc_in)
    cycle_index = torch.tensor([1, 5, 11])
    with torch.no_grad():
        expected = baseline(x, cycle_index)
        torch.testing.assert_close(plain(x, cycle_index), expected, rtol=0, atol=0)
        torch.testing.assert_close(with_ne(x, cycle_index), expected, rtol=0, atol=0)


def test_apcm_node_emb_receives_gradient_once_mixing_is_active():
    cfg = configs(apcm_node_emb=1)
    model = GTRAPCM.Model(cfg).train()
    mixer = model.APCM
    with torch.no_grad():
        # Simulate a few optimizer steps: decoder and static weight move off 0.
        mixer.patch_decoder.weight.normal_(std=0.1)
        mixer.node_weight.fill_(0.5)
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    model(x, torch.tensor([0, 4])).square().mean().backward()
    for grad in (mixer.node_emb.grad, mixer.node_weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_apcm_node_emb_rejects_wrong_channel_count():
    model = GTRAPCM.Model(configs(apcm_node_emb=1))
    try:
        model.APCM(torch.randn(2, 5, 24))
    except ValueError:
        return
    raise AssertionError('expected a ValueError for a channel-count mismatch')


def test_default_loss_matches_author_mse():
    import torch.nn as nn
    from utils.losses import ForecastLoss
    pred, true = torch.randn(4, 12, 7), torch.randn(4, 12, 7)
    assert torch.allclose(ForecastLoss('mse', 0.0)(pred, true), nn.MSELoss()(pred, true))
    # Pure frequency loss and mixed losses are finite and differentiable.
    pred.requires_grad_(True)
    for kind in ('mae', 'huber', 'mse_mae'):
        loss = ForecastLoss(kind, 0.5)(pred, true)
        loss.backward()
        assert torch.isfinite(loss)


def test_channel_embedding_starts_at_apcm():
    cfg = configs()
    torch.manual_seed(7)
    plain = GTRAPCM.Model(cfg).eval()
    torch.manual_seed(7)
    with_ce = GTRAPCM.Model(configs(chan_emb=1)).eval()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    idx = torch.tensor([1, 5])
    with torch.no_grad():
        assert torch.allclose(plain(x, idx), with_ce(x, idx))
    assert with_ce.channel_emb.shape == (cfg.enc_in, cfg.d_model)
