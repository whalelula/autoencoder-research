from __future__ import annotations

import torch
import torchaudio
from torch import nn

FREQUENCY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 500.0),
    ("mid", 500.0, 4_000.0),
    ("high", 4_000.0, 12_000.0),
    ("air", 12_000.0, 20_000.0),
)
BANDWISE_SPECTRAL_METRIC_NAMES: tuple[str, ...] = tuple(
    f"{metric}/{name}"
    for metric in ("STFT", "MEL")
    for name, _, _ in FREQUENCY_BANDS
)


def si_sdr(
    estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Scale-invariant SDR in dB, averaged over batch and channels."""
    if estimate.shape != target.shape:
        raise ValueError("SI-SDR inputs must have identical shapes")
    estimate = estimate.float() - estimate.float().mean(dim=-1, keepdim=True)
    target = target.float() - target.float().mean(dim=-1, keepdim=True)
    target_energy = target.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    projection = (
        (estimate * target).sum(dim=-1, keepdim=True) * target / target_energy
    )
    noise = estimate - projection
    ratio = projection.square().sum(dim=-1) / noise.square().sum(dim=-1).clamp_min(eps)
    return (10.0 * torch.log10(ratio.clamp_min(eps))).mean()


class LogMelL1(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
        )
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate_mel = self.mel(estimate.float())
        target_mel = self.mel(target.float())
        return (
            torch.log(estimate_mel + self.eps)
            - torch.log(target_mel + self.eps)
        ).abs().mean()


class BandwiseSpectralErrors(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        eps: float = 1e-5,
        bands: tuple[tuple[str, float, float], ...] = FREQUENCY_BANDS,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.eps = float(eps)
        self.bands = tuple((name, float(low), float(high)) for name, low, high in bands)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=int(n_mels),
            power=1.0,
        )
        self.register_buffer(
            "window", torch.hann_window(self.n_fft), persistent=False
        )
        stft_freqs = torch.linspace(0.0, self.sample_rate / 2.0, self.n_fft // 2 + 1)
        mel_fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=0.0,
            f_max=self.sample_rate / 2.0,
            n_mels=int(n_mels),
            sample_rate=self.sample_rate,
            norm=None,
            mel_scale="htk",
        )
        mel_weights = mel_fbanks.sum(dim=0).clamp_min(self.eps)
        mel_centers = (mel_fbanks * stft_freqs[:, None]).sum(dim=0) / mel_weights
        self.register_buffer(
            "stft_band_masks",
            self._make_masks(stft_freqs),
            persistent=False,
        )
        self.register_buffer(
            "mel_band_masks",
            self._make_masks(mel_centers),
            persistent=False,
        )

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(
            f"{metric}/{name}"
            for metric in ("STFT", "MEL")
            for name, _, _ in self.bands
        )

    def _make_masks(self, freqs: torch.Tensor) -> torch.Tensor:
        masks = []
        nyquist = self.sample_rate / 2.0
        for _, low, high in self.bands:
            clipped_high = min(high, nyquist)
            masks.append((freqs >= low) & (freqs < clipped_high))
        return torch.stack(masks)

    def _stft_magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = waveform.shape
        flattened = waveform.float().reshape(batch * channels, samples)
        value = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()
        return value.reshape(batch, channels, value.shape[-2], value.shape[-1])

    def _band_l1(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor,
        prefix: str,
    ) -> dict[str, torch.Tensor | None]:
        values: dict[str, torch.Tensor | None] = {}
        estimate_log = torch.log(estimate + self.eps)
        target_log = torch.log(target + self.eps)
        for band_index, (name, _, _) in enumerate(self.bands):
            mask = masks[band_index]
            key = f"{prefix}/{name}"
            if not bool(mask.any()):
                values[key] = None
                continue
            values[key] = (estimate_log[..., mask, :] - target_log[..., mask, :]).abs().mean()
        return values

    def forward(
        self, estimate: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        if estimate.shape != target.shape:
            raise ValueError(
                f"Bandwise spectral inputs must match, got {estimate.shape} and {target.shape}"
            )
        estimate_stft = self._stft_magnitude(estimate)
        target_stft = self._stft_magnitude(target)
        estimate_mel = self.mel(estimate.float())
        target_mel = self.mel(target.float())
        return {
            **self._band_l1(estimate_stft, target_stft, self.stft_band_masks, "STFT"),
            **self._band_l1(estimate_mel, target_mel, self.mel_band_masks, "MEL"),
        }
