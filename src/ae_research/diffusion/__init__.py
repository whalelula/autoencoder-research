"""Interfaces and shared utilities for downstream latent diffusion."""

from typing import Protocol

import torch

from .codec import (
    FrozenAutoencoderCodec,
    FrozenCodecAdapter,
    LatentNormalizer,
    load_frozen_codec,
)
from .config import (
    DEFAULT_DIT_DEPTH,
    DEFAULT_TEXT_ENCODER,
    load_dit_config,
    validate_dit_config,
)


class LatentCodec(Protocol):
    def encode(self, waveform: torch.Tensor) -> torch.Tensor: ...

    def decode(self, latent: torch.Tensor, target_num_samples: int) -> torch.Tensor: ...


class LatentGenerator(Protocol):
    def sample(self, batch_size: int, num_frames: int, **conditions) -> torch.Tensor: ...


__all__ = [
    "DEFAULT_DIT_DEPTH",
    "DEFAULT_TEXT_ENCODER",
    "FrozenAutoencoderCodec",
    "FrozenCodecAdapter",
    "LatentCodec",
    "LatentGenerator",
    "LatentNormalizer",
    "load_dit_config",
    "load_frozen_codec",
    "validate_dit_config",
]
