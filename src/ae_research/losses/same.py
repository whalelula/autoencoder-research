from __future__ import annotations

import math
from typing import Any

import torch
import torchaudio
from torch import nn


def dual_axis_kl(latent: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """SAME's KL-like moment penalty over time and channels.

    This is not a stochastic-VAE KL. ``latent`` has shape [B, C, T].
    """
    if latent.ndim != 3:
        raise ValueError("latent must have shape [batch, channels, time]")
    mean_time = latent.mean(dim=-1)
    var_time = latent.var(dim=-1, unbiased=False)
    time_term = (
        mean_time.square() + var_time - torch.log(var_time + eps) - 1.0
    ).mean()

    mean_channel = latent.mean(dim=1)
    var_channel = latent.var(dim=1, unbiased=False)
    channel_term = (
        mean_channel.square() + var_channel - torch.log(var_channel + eps) - 1.0
    ).mean()
    return time_term + 0.4 * channel_term


class MultiResolutionSTFTLoss(nn.Module):
    """Phase-aware SAME MR-STFT loss (paper equations 2--7)."""

    def __init__(
        self,
        fft_sizes: list[int] | tuple[int, ...],
        *,
        sample_rate: int,
        hop_ratio: float = 0.25,
        use_k_weighting: bool = True,
        stereo_representations: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.fft_sizes = tuple(int(value) for value in fft_sizes)
        self.sample_rate = int(sample_rate)
        self.hop_ratio = float(hop_ratio)
        self.use_k_weighting = bool(use_k_weighting)
        self.stereo_representations = bool(stereo_representations)
        self.eps = float(eps)
        for size in self.fft_sizes:
            self.register_buffer(
                f"window_{size}", torch.hann_window(size), persistent=False
            )

    def _perceptual_preemphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.use_k_weighting:
            return waveform
        # Differentiable BS.1770-style cascade with clamp=False. The convenience
        # biquad functions clamp to [-1, 1], which is undesirable inside a loss.
        device, dtype = waveform.device, waveform.dtype

        def coefficients(values: list[float]) -> torch.Tensor:
            return torch.tensor(values, device=device, dtype=dtype)

        shelf_frequency = min(1681.974, self.sample_rate * 0.45)
        amplitude = 10.0 ** (4.0 / 40.0)
        omega = 2.0 * math.pi * shelf_frequency / self.sample_rate
        cosine, sine = math.cos(omega), math.sin(omega)
        alpha = sine / 2.0 * math.sqrt(2.0)
        beta = 2.0 * math.sqrt(amplitude) * alpha
        b0 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine + beta)
        b1 = -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cosine)
        b2 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine - beta)
        a0 = (amplitude + 1) - (amplitude - 1) * cosine + beta
        a1 = 2 * ((amplitude - 1) - (amplitude + 1) * cosine)
        a2 = (amplitude + 1) - (amplitude - 1) * cosine - beta
        waveform = torchaudio.functional.lfilter(
            waveform,
            coefficients([a0 / a0, a1 / a0, a2 / a0]),
            coefficients([b0 / a0, b1 / a0, b2 / a0]),
            clamp=False,
        )

        omega = 2.0 * math.pi * 38.135 / self.sample_rate
        cosine, sine = math.cos(omega), math.sin(omega)
        alpha = sine / (2.0 * 0.5003)
        b0, b1, b2 = (1 + cosine) / 2, -(1 + cosine), (1 + cosine) / 2
        a0, a1, a2 = 1 + alpha, -2 * cosine, 1 - alpha
        return torchaudio.functional.lfilter(
            waveform,
            coefficients([a0 / a0, a1 / a0, a2 / a0]),
            coefficients([b0 / a0, b1 / a0, b2 / a0]),
            clamp=False,
        )

    def _representations(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.stereo_representations or waveform.shape[1] != 2:
            return waveform
        left, right = waveform[:, :1], waveform[:, 1:2]
        mid = (left + right) / math.sqrt(2.0)
        side = (left - right) / math.sqrt(2.0)
        return torch.cat((left, right, mid, side), dim=1)

    def _stft(self, waveform: torch.Tensor, fft_size: int) -> torch.Tensor:
        batch, channels, samples = waveform.shape
        flattened = waveform.reshape(batch * channels, samples)
        value = torch.stft(
            flattened,
            n_fft=fft_size,
            hop_length=max(1, round(fft_size * self.hop_ratio)),
            win_length=fft_size,
            window=getattr(self, f"window_{fft_size}"),
            center=True,
            return_complex=True,
        )
        return value.reshape(batch, channels, value.shape[-2], value.shape[-1])

    def _resolution(
        self, predicted: torch.Tensor, reference: torch.Tensor, fft_size: int
    ) -> dict[str, torch.Tensor]:
        x_complex = self._stft(predicted, fft_size)
        y_complex = self._stft(reference, fft_size)
        x = x_complex.abs()
        y = y_complex.abs()
        reduce_dims = (-2, -1)

        numerator = torch.linalg.vector_norm(x - y, dim=reduce_dims)
        denominator = torch.linalg.vector_norm(x + y, dim=reduce_dims)
        spectral_contrast = (numerator / (denominator + self.eps)).mean()

        sigma = torch.sqrt(
            x.std(dim=reduce_dims, unbiased=False, keepdim=True).square()
            + y.std(dim=reduce_dims, unbiased=False, keepdim=True).square()
        ).detach()
        log_magnitude = (
            torch.log1p(x / (sigma + self.eps))
            - torch.log1p(y / (sigma + self.eps))
        ).abs().mean()

        def phasor_loss(
            x_product: torch.Tensor,
            y_product: torch.Tensor,
            x_magnitude_product: torch.Tensor,
            y_magnitude_product: torch.Tensor,
        ) -> torch.Tensor:
            weight = torch.sqrt(
                (x_magnitude_product * y_magnitude_product).clamp_min(0.0)
            )
            weight = (weight / (weight.mean(dim=reduce_dims, keepdim=True) + self.eps)).detach()
            # The relative phasor angle gives the same cosine distance as
            # unit-normalizing both complex products, while preserving an exact
            # zero for identical inputs (including zero-energy bins).
            cosine_distance = 1.0 - torch.cos(
                torch.angle(x_product * y_product.conj())
            )
            return (weight * cosine_distance).mean()

        x_time_mag = x[..., 1:] * x[..., :-1]
        y_time_mag = y[..., 1:] * y[..., :-1]
        instantaneous_frequency = phasor_loss(
            x_complex[..., 1:] * x_complex[..., :-1].conj(),
            y_complex[..., 1:] * y_complex[..., :-1].conj(),
            x_time_mag,
            y_time_mag,
        )

        x_freq_mag = x[..., 1:, :] * x[..., :-1, :]
        y_freq_mag = y[..., 1:, :] * y[..., :-1, :]
        group_delay = phasor_loss(
            x_complex[..., 1:, :] * x_complex[..., :-1, :].conj(),
            y_complex[..., 1:, :] * y_complex[..., :-1, :].conj(),
            x_freq_mag,
            y_freq_mag,
        )

        complex_delta_squared = (x_complex - y_complex).abs().square()
        complex_scale = complex_delta_squared.std(
            dim=reduce_dims, unbiased=False, keepdim=True
        ).detach()
        complex_distance = torch.log1p(
            complex_delta_squared / (complex_scale + self.eps)
        ).mean()
        return {
            "sc": spectral_contrast,
            "lm": log_magnitude,
            "if": instantaneous_frequency,
            "gd": group_delay,
            "complex": complex_distance,
        }

    def forward(
        self, predicted: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if predicted.shape != reference.shape:
            raise ValueError(
                f"MR-STFT inputs must match, got {predicted.shape} and {reference.shape}"
            )
        predicted = self._representations(
            self._perceptual_preemphasis(predicted.float())
        )
        reference = self._representations(
            self._perceptual_preemphasis(reference.float())
        )
        components = {
            key: predicted.new_zeros(()) for key in ("sc", "lm", "if", "gd", "complex")
        }
        for fft_size in self.fft_sizes:
            resolution = self._resolution(predicted, reference, fft_size)
            for key, value in resolution.items():
                components[key] = components[key] + value
        total = sum(components.values(), predicted.new_zeros(()))
        return total, components


class SameObjective(nn.Module):
    def __init__(self, config: dict[str, Any], *, sample_rate: int) -> None:
        super().__init__()
        self.mrstft_weight = float(config["mrstft_weight"])
        self.kl_weight = float(config["kl_weight"])
        self.mrstft = MultiResolutionSTFTLoss(
            config["fft_sizes"],
            sample_rate=sample_rate,
            hop_ratio=float(config["hop_ratio"]),
            use_k_weighting=bool(config["use_k_weighting"]),
            stereo_representations=bool(config["stereo_representations"]),
            eps=float(config["eps"]),
        )
        self.eps = float(config["eps"])

    def forward(
        self,
        reconstruction: torch.Tensor,
        reference: torch.Tensor,
        latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mrstft, components = self.mrstft(reconstruction, reference)
        kl = dual_axis_kl(latent.float(), self.eps)
        total = self.mrstft_weight * mrstft + self.kl_weight * kl
        return {
            "total": total,
            "mrstft": mrstft,
            "kl": kl,
            **{f"mrstft_{key}": value for key, value in components.items()},
        }
