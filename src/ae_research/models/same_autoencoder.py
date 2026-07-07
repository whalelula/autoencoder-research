from __future__ import annotations

import math
from typing import Any

import torch
from einops import rearrange
from torch import nn

from ae_research.vendor.stable_audio_tools.same import SAMEDecoder, SAMEEncoder


class PatchedPretransform(nn.Module):
    """Parameter-free SAME patch pretransform."""

    def __init__(self, *, channels: int, patch_size: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        self.downsampling_ratio = self.patch_size
        self.encoded_channels = self.channels * self.patch_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"PatchedPretransform expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )
        pad_len = (self.patch_size - (x.shape[-1] % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)
        return rearrange(x, "b c (l h) -> b (c h) l", h=self.patch_size)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b (c h) l -> b c (l h)", h=self.patch_size)


class SoftNormBottleneck(nn.Module):
    """SoftNorm bottleneck from Stable Audio Tools."""

    def __init__(
        self,
        *,
        dim: int = 32,
        noise_augment_dim: int = 0,
        noise_regularize: bool = False,
        auto_scale: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.noise_augment_dim = int(noise_augment_dim)
        self.scaling_factor = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))
        self.noise_scaling_factor = nn.Parameter(
            torch.ones(1, self.noise_augment_dim, 1)
        )
        self.noise_regularize = bool(noise_regularize)
        self.freeze = bool(freeze)
        if self.freeze:
            self.scaling_factor.requires_grad = False
            self.bias.requires_grad = False
            self.noise_scaling_factor.requires_grad = False
        if auto_scale:
            self.register_parameter(
                "running_std",
                nn.Parameter(torch.ones(1), requires_grad=False),
            )

    def encode(
        self, x: torch.Tensor, *, return_info: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        info = {}
        x = x * self.scaling_factor + self.bias
        if self.training and hasattr(self, "running_std") and not self.freeze:
            self.running_std.data = (
                self.running_std.data * 0.999 + x.std().detach() * 0.001
            ).clamp(min=1e-4)
        if hasattr(self, "running_std"):
            x = x / self.running_std
        if self.training and return_info:
            var = (x.std(dim=-1) ** 2).clip(min=1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-1)
            loss = (mean * mean + var - logvar - 1).mean()
            var = (x.std(dim=-2) ** 2).clip(min=1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-2)
            loss = loss + 0.4 * (mean * mean + var - logvar - 1).mean()
            info["softnorm_loss"] = loss
        if return_info:
            return x, info
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "running_std"):
            x = x * self.running_std
        if self.noise_regularize:
            scaling = self.running_std if hasattr(self, "running_std") else x.std(
                dim=-1
            ).unsqueeze(-1)
            scale = 5e-2 if self.training else 1e-3
            x = x + torch.randn_like(x) * scaling * scale
        if self.noise_augment_dim > 0:
            noise = self.noise_scaling_factor * torch.randn(
                x.shape[0], self.noise_augment_dim, x.shape[-1], device=x.device
            ).type_as(x)
            x = torch.cat([x, noise], dim=1)
        return x


class SameAutoencoder(nn.Module):
    """Stable Audio Tools SAME-style autoencoder adapted to this training loop."""

    def __init__(
        self,
        model_config: dict[str, Any],
        *,
        audio_channels: int,
        data_sample_rate: int,
    ) -> None:
        super().__init__()
        self.data_sample_rate = int(data_sample_rate)
        self.audio_channels = int(audio_channels)
        self.latent_dim = int(model_config["latent_dim"])

        pretransform_config = model_config["pretransform"]
        if str(pretransform_config["type"]) != "patched":
            raise ValueError("SAME-S pretransform must be patched")
        self.pretransform = PatchedPretransform(**pretransform_config["config"])
        if self.pretransform.channels != self.audio_channels:
            raise ValueError(
                "data.channels must match model.pretransform.config.channels"
            )

        encoder_config = model_config["encoder"]
        decoder_config = model_config["decoder"]
        if str(encoder_config["type"]) != "same" or str(decoder_config["type"]) != "same":
            raise ValueError("SAME-S encoder and decoder types must be same")
        self.encoder = SAMEEncoder(**encoder_config["config"])
        self.decoder = SAMEDecoder(**decoder_config["config"])

        self.bottleneck: SoftNormBottleneck | None = None
        bottleneck_config = model_config.get("bottleneck")
        if bottleneck_config is not None:
            if str(bottleneck_config["type"]) != "softnorm":
                raise ValueError("SAME-S bottleneck must be softnorm")
            self.bottleneck = SoftNormBottleneck(**bottleneck_config["config"])

        self.strides = [int(value) for value in encoder_config["config"]["strides"]]
        self.downsampling_ratio = self.pretransform.downsampling_ratio * math.prod(
            self.strides
        )
        self.output_activation = str(model_config.get("output_activation", "none"))
        if self.output_activation not in {"none", "tanh"}:
            raise ValueError("SAME output_activation must be 'none' or 'tanh'")

    def _match_length(
        self, waveform: torch.Tensor, target_num_samples: int
    ) -> torch.Tensor:
        if waveform.shape[-1] > target_num_samples:
            waveform = waveform[..., :target_num_samples]
        elif waveform.shape[-1] < target_num_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_num_samples - waveform.shape[-1])
            )
        return waveform

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        target_num_samples = waveform.shape[-1]
        patched = self.pretransform.encode(waveform)
        latent = self.encoder(patched)
        info: dict[str, torch.Tensor] = {}
        if self.bottleneck is not None:
            latent, info = self.bottleneck.encode(latent, return_info=True)
            decoder_latent = self.bottleneck.decode(latent)
        else:
            decoder_latent = latent
        reconstruction = self.pretransform.decode(self.decoder(decoder_latent))
        reconstruction = self._match_length(reconstruction, target_num_samples)
        if self.output_activation == "tanh":
            reconstruction = torch.tanh(reconstruction)
        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "patched": patched,
            **info,
        }


def default_same_s_config() -> dict[str, Any]:
    """A practical SAME-S-style open configuration for this repository."""
    return {
        "type": "same",
        "variant": "same_s",
        "pretransform": {
            "type": "patched",
            "config": {
                "patch_size": 256,
                "channels": 2,
            },
        },
        "encoder": {
            "type": "same",
            "config": {
                "in_channels": 512,
                "channels": 128,
                "c_mults": [6],
                "strides": [16],
                "latent_dim": 256,
                "transformer_depths": [6],
                "checkpointing": False,
                "differential": True,
                "dyt": True,
                "dim_heads": 64,
                "variable_stride": True,
                "chunk_size": 32,
                "chunk_midpoint_shift": True,
                "mask_noise": 0.0,
            },
        },
        "decoder": {
            "type": "same",
            "config": {
                "out_channels": 512,
                "channels": 128,
                "c_mults": [6],
                "strides": [16],
                "latent_dim": 256,
                "transformer_depths": [6],
                "sinusoidal_blocks": [0],
                "checkpointing": False,
                "differential": True,
                "dyt": True,
                "dim_heads": 64,
                "variable_stride": True,
                "chunk_size": 32,
                "chunk_midpoint_shift": True,
                "conv_mapping": True,
                "mask_noise": 0.01,
            },
        },
        "bottleneck": {
            "type": "softnorm",
            "config": {
                "dim": 256,
                "noise_augment_dim": 0,
                "noise_regularize": True,
                "auto_scale": True,
                "freeze": True,
            },
        },
        "latent_dim": 256,
        "downsampling_ratio": 4096,
        "io_channels": 2,
        "output_activation": "none",
    }
