from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from ae_research.losses.same import MultiResolutionSTFTLoss, dual_axis_kl  # noqa: E402
from ae_research.models.decoder import (  # noqa: E402
    MERTMirrorDecoder,
    mert_feature_lengths,
)
from ae_research.training.trainer import _nonfinite_details  # noqa: E402


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


def test_same_mrstft_identical_audio_is_finite():
    waveform = torch.randn(1, 1, 4096) * 0.05
    loss_module = MultiResolutionSTFTLoss(
        [32, 64, 128, 256, 512, 1024, 2048],
        sample_rate=24_000,
        use_k_weighting=False,
    )
    loss, components = loss_module(waveform, waveform)
    assert torch.isfinite(loss)
    assert components["sc"].item() == pytest.approx(0.0, abs=1e-6)
    assert components["lm"].item() == pytest.approx(0.0, abs=1e-6)
    assert components["complex"].item() == pytest.approx(0.0, abs=1e-6)
    assert components["if"].item() >= 0.0
    assert components["gd"].item() >= 0.0


@pytest.mark.parametrize("amplitude", [1e-3, 1e-5, 1e-7, 1e-12])
def test_same_mrstft_low_energy_gradient_is_finite(amplitude):
    torch.manual_seed(1)
    prediction = (torch.randn(1, 1, 4096) * amplitude).requires_grad_()
    torch.manual_seed(2)
    reference = torch.randn(1, 1, 4096) * amplitude
    loss_module = MultiResolutionSTFTLoss(
        [32, 64, 128, 256, 512, 1024, 2048],
        sample_rate=24_000,
        use_k_weighting=False,
    )
    loss, components = loss_module(prediction, reference)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert torch.isfinite(prediction.grad).all()


def test_nonfinite_details_identifies_nan_and_inf():
    details = _nonfinite_details(
        [
            ("finite", torch.ones(2)),
            ("broken", torch.tensor([float("nan"), float("inf")])),
        ]
    )
    assert details == ["broken: shape=(2,), nan=1, inf=1"]
