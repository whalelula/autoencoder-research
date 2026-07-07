from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from ae_research.evaluation.sa3_same import _match_reference_format  # noqa: E402


def test_match_reference_format_downmixes_resamples_and_trims():
    waveform = torch.stack(
        (
            torch.linspace(-1.0, 1.0, 44_100),
            torch.linspace(1.0, -1.0, 44_100),
        )
    ).unsqueeze(0)
    converted = _match_reference_format(
        waveform,
        source_rate=44_100,
        target_rate=24_000,
        target_channels=1,
        target_samples=12_000,
    )
    assert converted.shape == (1, 1, 12_000)
    assert torch.isfinite(converted).all()


def test_match_reference_format_repeats_mono_and_pads():
    waveform = torch.ones(2, 1, 100)
    converted = _match_reference_format(
        waveform,
        source_rate=24_000,
        target_rate=24_000,
        target_channels=2,
        target_samples=120,
    )
    assert converted.shape == (2, 2, 120)
    assert converted[:, :, :100].eq(1).all()
    assert converted[:, :, 100:].eq(0).all()


def test_match_reference_format_rejects_unsupported_channel_conversion():
    with pytest.raises(ValueError, match="Cannot convert"):
        _match_reference_format(
            torch.randn(1, 3, 100),
            source_rate=24_000,
            target_rate=24_000,
            target_channels=2,
            target_samples=100,
        )
