from __future__ import annotations

import torch
import torchaudio
from torch import nn


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

