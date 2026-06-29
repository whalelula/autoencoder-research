from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

MERT_KERNELS = (10, 3, 3, 3, 3, 2, 2)
MERT_STRIDES = (5, 2, 2, 2, 2, 2, 2)


def mert_feature_lengths(
    input_length: int,
    kernels: Sequence[int] = MERT_KERNELS,
    strides: Sequence[int] = MERT_STRIDES,
) -> list[int]:
    lengths = [int(input_length)]
    current = int(input_length)
    for kernel, stride in zip(kernels, strides):
        current = (current - kernel) // stride + 1
        if current <= 0:
            raise ValueError(
                f"Audio length {input_length} is too short for MERT's convolution stack"
            )
        lengths.append(current)
    return lengths


class MERTMirrorDecoder(nn.Module):
    """Linear projection plus transposed convolutions derived from a MERT config."""

    def __init__(
        self,
        semantic_dim: int,
        conv_dims: Sequence[int],
        kernels: Sequence[int],
        strides: Sequence[int],
        audio_channels: int = 1,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.conv_dims = tuple(int(value) for value in conv_dims)
        self.kernels = tuple(int(value) for value in kernels)
        self.strides = tuple(int(value) for value in strides)
        if not self.conv_dims:
            raise ValueError("MERT convolution config must contain at least one layer")
        if not (
            len(self.conv_dims) == len(self.kernels) == len(self.strides)
        ):
            raise ValueError("conv_dims, kernels, and strides must have equal lengths")
        if any(value <= 0 for value in (*self.conv_dims, *self.kernels, *self.strides)):
            raise ValueError("MERT convolution dimensions, kernels, and strides must be positive")

        self.audio_channels = int(audio_channels)
        self.projection = nn.Linear(semantic_dim, self.conv_dims[-1])
        layers = []
        for encoder_index in reversed(range(len(self.conv_dims))):
            output_channels = (
                self.conv_dims[encoder_index - 1]
                if encoder_index > 0
                else self.audio_channels
            )
            layers.append(
                nn.ConvTranspose1d(
                    self.conv_dims[encoder_index],
                    output_channels,
                    kernel_size=self.kernels[encoder_index],
                    stride=self.strides[encoder_index],
                )
            )
        self.layers = nn.ModuleList(layers)
        self.activation = nn.GELU()
        if output_activation not in {"tanh", "identity"}:
            raise ValueError("output_activation must be 'tanh' or 'identity'")
        self.output_activation = output_activation

    def project(self, semantic_features: torch.Tensor) -> torch.Tensor:
        if semantic_features.ndim != 3:
            raise ValueError("semantic_features must have shape [batch, frames, hidden]")
        return self.projection(semantic_features).transpose(1, 2)

    def decode(self, latent: torch.Tensor, target_num_samples: int) -> torch.Tensor:
        lengths = mert_feature_lengths(
            target_num_samples,
            kernels=self.kernels,
            strides=self.strides,
        )
        expected_frames = lengths[-1]
        if latent.shape[-1] != expected_frames:
            raise ValueError(
                f"Latent has {latent.shape[-1]} frames, but target length "
                f"{target_num_samples} implies {expected_frames}. Ensure the waveform passed "
                "to MERT and decoder has the same 24 kHz length."
            )
        desired_lengths = list(reversed(lengths[:-1]))
        value = latent
        for index, (layer, desired) in enumerate(zip(self.layers, desired_lengths)):
            value = layer(
                value,
                output_size=(value.shape[0], layer.out_channels, desired),
            )
            if index != len(self.layers) - 1:
                value = self.activation(value)
        return torch.tanh(value) if self.output_activation == "tanh" else value

    def forward(
        self, semantic_features: torch.Tensor, target_num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.project(semantic_features)
        return self.decode(latent, target_num_samples), latent
