from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class FrozenMERTEncoder(nn.Module):
    """Frozen MERT wrapper that preserves gradients only outside the encoder."""

    def __init__(
        self,
        model_name: str,
        *,
        layer: int = -1,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.layer = int(layer)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        config = self.model.config
        self.hidden_size = int(config.hidden_size)
        self.num_attention_heads = int(config.num_attention_heads)
        self.sample_rate = int(config.sample_rate)
        self.conv_dims = tuple(int(value) for value in config.conv_dim)
        self.conv_kernels = tuple(int(value) for value in config.conv_kernel)
        self.conv_strides = tuple(int(value) for value in config.conv_stride)
        if not (
            len(self.conv_dims)
            == len(self.conv_kernels)
            == len(self.conv_strides)
            == 7
        ):
            raise ValueError(
                f"{model_name} does not expose the expected seven-layer MERT "
                "convolution stack"
            )

    def train(self, mode: bool = True) -> "FrozenMERTEncoder":
        # A parent model's train() must not re-enable MERT dropout.
        super().train(False)
        self.model.eval()
        return self

    @staticmethod
    def _normalize(waveform: torch.Tensor) -> torch.Tensor:
        """Match Wav2Vec2FeatureExtractor's per-example zero-mean/unit-var norm."""
        mean = waveform.mean(dim=-1, keepdim=True)
        variance = waveform.var(dim=-1, unbiased=False, keepdim=True)
        return (waveform - mean) / torch.sqrt(variance + 1e-7)

    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"Expected [batch, channels, samples], got {waveform.shape}")
        mono = waveform.mean(dim=1)
        return self._normalize(mono)

    @property
    def feature_extractor(self) -> nn.Module:
        return self.model.feature_extractor

    @property
    def feature_projection(self) -> nn.Module:
        return self.model.feature_projection

    def encode_normalized(self, normalized: torch.Tensor) -> torch.Tensor:
        if normalized.ndim != 2:
            raise ValueError(
                f"Expected normalized [batch, samples], got {normalized.shape}"
            )
        request_hidden = self.layer != -1
        with torch.no_grad():
            outputs: Any = self.model(
                input_values=normalized,
                output_hidden_states=request_hidden,
                return_dict=True,
            )
        if self.layer == -1:
            return outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        if not -len(hidden_states) <= self.layer < len(hidden_states):
            raise IndexError(
                f"MERT layer {self.layer} out of range for {len(hidden_states)} hidden states"
            )
        return hidden_states[self.layer]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encode_normalized(self.preprocess(waveform))
