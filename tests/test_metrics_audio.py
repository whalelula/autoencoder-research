from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from ae_research.metrics import BandwiseSpectralErrors  # noqa: E402


def test_bandwise_spectral_errors_identical_audio_is_zero_or_unavailable():
    metric = BandwiseSpectralErrors(
        sample_rate=24_000,
        n_fft=512,
        hop_length=128,
        n_mels=64,
    )
    waveform = torch.randn(2, 1, 4096)
    values = metric(waveform, waveform)

    assert values["STFT/low"].item() == pytest.approx(0.0, abs=1e-6)
    assert values["STFT/mid"].item() == pytest.approx(0.0, abs=1e-6)
    assert values["STFT/high"].item() == pytest.approx(0.0, abs=1e-6)
    assert values["STFT/air"] is None
    assert values["MEL/low"].item() == pytest.approx(0.0, abs=1e-6)
    assert values["MEL/mid"].item() == pytest.approx(0.0, abs=1e-6)
    assert values["MEL/high"].item() == pytest.approx(0.0, abs=1e-6)
    assert values["MEL/air"] is None


def test_bandwise_spectral_errors_air_band_exists_above_40khz_sample_rate():
    metric = BandwiseSpectralErrors(
        sample_rate=44_100,
        n_fft=512,
        hop_length=128,
        n_mels=64,
    )
    estimate = torch.zeros(1, 1, 4096)
    target = torch.randn(1, 1, 4096) * 0.01
    values = metric(estimate, target)

    assert values["STFT/air"] is not None
    assert values["MEL/air"] is not None
    assert torch.isfinite(values["STFT/air"])
    assert torch.isfinite(values["MEL/air"])
