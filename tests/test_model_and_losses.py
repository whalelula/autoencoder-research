from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from ae_research.losses.same import MultiResolutionSTFTLoss, dual_axis_kl
from ae_research.models.decoder import MERTMirrorDecoder, mert_feature_lengths


def test_mirror_decoder_exact_output_length():
    target_length = 24_000
    frames = mert_feature_lengths(target_length)[-1]
    decoder = MERTMirrorDecoder(
        semantic_dim=16,
        conv_dims=[8, 8, 8, 8, 8, 8, 8],
        kernels=[10, 3, 3, 3, 3, 2, 2],
        strides=[5, 2, 2, 2, 2, 2, 2],
        audio_channels=2,
    )
    features = torch.randn(2, frames, 16)
    waveform, latent = decoder(features, target_length)
    assert waveform.shape == (2, 2, target_length)
    assert latent.shape == (2, 8, frames)


def test_mirror_decoder_uses_each_encoder_layer_width():
    target_length = 128
    kernels = [5, 3, 2]
    strides = [2, 2, 2]
    conv_dims = [4, 8, 12]
    frames = mert_feature_lengths(target_length, kernels, strides)[-1]
    decoder = MERTMirrorDecoder(
        semantic_dim=16,
        conv_dims=conv_dims,
        kernels=kernels,
        strides=strides,
        audio_channels=2,
    )
    assert decoder.projection.out_features == 12
    assert [(layer.in_channels, layer.out_channels) for layer in decoder.layers] == [
        (12, 8),
        (8, 4),
        (4, 2),
    ]
    waveform, _ = decoder(torch.randn(2, frames, 16), target_length)
    assert waveform.shape == (2, 2, target_length)


def test_dual_axis_kl_has_gradient():
    latent = torch.randn(2, 8, 16, requires_grad=True)
    loss = dual_axis_kl(latent)
    assert torch.isfinite(loss)
    loss.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_same_mrstft_is_zero_for_identical_audio():
    waveform = torch.randn(1, 1, 4096) * 0.05
    loss_module = MultiResolutionSTFTLoss(
        [32, 64, 128, 256, 512, 1024, 2048],
        sample_rate=24_000,
        use_k_weighting=False,
    )
    loss, components = loss_module(waveform, waveform)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert all(value.item() == pytest.approx(0.0, abs=1e-6) for value in components.values())
