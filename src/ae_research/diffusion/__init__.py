"""Reserved interfaces for the downstream latent diffusion stage.

No denoiser, conditioner, or scheduler is selected yet. Keeping this package
empty avoids silently baking an arbitrary diffusion design into AE experiments.
"""

from typing import Protocol

import torch


class LatentCodec(Protocol):
    def encode(self, waveform: torch.Tensor) -> torch.Tensor: ...

    def decode(self, latent: torch.Tensor, target_num_samples: int) -> torch.Tensor: ...


class LatentGenerator(Protocol):
    def sample(self, batch_size: int, num_frames: int, **conditions) -> torch.Tensor: ...


__all__ = ["LatentCodec", "LatentGenerator"]

