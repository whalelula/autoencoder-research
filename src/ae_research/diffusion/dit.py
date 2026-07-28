from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def _sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Embed a scalar with transformer-style sinusoidal features."""
    if values.ndim != 1:
        raise ValueError("Scalar conditions must have shape [batch]")
    half = dim // 2
    if half == 0:
        raise ValueError("Embedding dimension must be at least 2")
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = values.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def _modulate(
    value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return value * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ResidualPointwiseConv(nn.Module):
    """A zero-initialised 1x1 convolution on a residual branch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.conv(value)


class FeedForward(nn.Module):
    def __init__(self, dim: int, *, multiplier: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(dim, round(dim * multiplier))
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, hidden = self.in_proj(value).chunk(2, dim=-1)
        value = self.activation(gate) * hidden
        return self.out_proj(self.dropout(value))


def _apply_partial_rope(
    value: torch.Tensor, *, rotary_dim: int
) -> torch.Tensor:
    """Apply RoPE to the first ``rotary_dim`` channels of [B,H,T,D]."""
    if rotary_dim <= 0:
        return value
    rotary_dim = min(rotary_dim, value.shape[-1])
    rotary_dim -= rotary_dim % 2
    if rotary_dim == 0:
        return value
    positions = torch.arange(value.shape[-2], device=value.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, rotary_dim, 2, device=value.device, dtype=torch.float32)
        / rotary_dim
    )
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos().to(value.dtype)[None, None, :, :]
    sin = angles.sin().to(value.dtype)[None, None, :, :]
    rotated, remainder = value[..., :rotary_dim], value[..., rotary_dim:]
    even, odd = rotated[..., 0::2], rotated[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).flatten(-2)
    return torch.cat((rotated, remainder), dim=-1)


class SelfAttention(nn.Module):
    def __init__(
        self, dim: int, *, num_heads: int, dropout: float, rotary_dim: int = 32
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.rotary_dim = min(int(rotary_dim), self.head_dim)
        self.qkv_projection = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.output_projection = nn.Linear(dim, dim, bias=False)
        self.dropout = float(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, dim = value.shape
        qkv = self.qkv_projection(value).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, values = qkv.unbind(dim=2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        values = values.transpose(1, 2)
        query = _apply_partial_rope(query, rotary_dim=self.rotary_dim)
        key = _apply_partial_rope(key, rotary_dim=self.rotary_dim)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            values,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return self.output_projection(attended)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_value_projection = nn.Linear(dim, dim * 2, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.output_projection = nn.Linear(dim, dim, bias=False)
        self.dropout = float(dropout)

    def forward(
        self, value: torch.Tensor, context: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, dim = value.shape
        context_length = context.shape[1]
        query = self.query_projection(value).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        key, values = self.key_value_projection(context).reshape(
            batch, context_length, 2, self.num_heads, self.head_dim
        ).unbind(dim=2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        values = values.transpose(1, 2)
        attention_mask = mask[:, None, None, :].bool()
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            values,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return self.output_projection(attended)


class AdaLNTransformerBlock(nn.Module):
    """Vanilla self/cross-attention block with AdaLN on self-attention and FFN."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mlp_multiplier: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(
            dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, dropout=dropout
        )
        self.ff_norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim, multiplier=mlp_multiplier, dropout=dropout
        )
        self.adaLN_bias = nn.Parameter(torch.zeros(dim * 6))
        self.dropout = nn.Dropout(dropout)

        # Match the stable-audio transformer convention of initially inert branches.
        nn.init.zeros_(self.self_attn.output_projection.weight)
        nn.init.zeros_(self.cross_attn.output_projection.weight)
        nn.init.zeros_(self.ff.out_proj.weight)
        nn.init.zeros_(self.ff.out_proj.bias)

    def forward(
        self,
        value: torch.Tensor,
        *,
        global_modulation: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        (
            self_shift,
            self_scale,
            self_gate,
            ff_shift,
            ff_scale,
            ff_gate,
        ) = (global_modulation + self.adaLN_bias.unsqueeze(0)).chunk(6, dim=-1)

        normalized = _modulate(
            self.self_norm(value), self_shift, self_scale
        )
        attended = self.self_attn(normalized)
        self_gate = torch.sigmoid(1.0 - self_gate).unsqueeze(1)
        value = value + self_gate * self.dropout(attended)

        cross_input = self.cross_norm(value)
        attended = self.cross_attn(cross_input, text, text_mask)
        value = value + self.dropout(attended)

        feed_forward = self.ff(
            _modulate(self.ff_norm(value), ff_shift, ff_scale)
        )
        ff_gate = torch.sigmoid(1.0 - ff_gate).unsqueeze(1)
        return value + ff_gate * self.dropout(feed_forward)


class AudioDiffusionTransformer(nn.Module):
    """A compact Stable Audio 3-style 1-D DiT with x-prediction output.

    Inputs and outputs use ``[batch, autoencoder_channels, frames]``. Text is
    supplied as frozen encoder tokens in ``[batch, text_tokens, text_dim]``.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        model_dim: int,
        text_dim: int,
        depth: int = 11,
        num_heads: int = 8,
        mlp_multiplier: float = 4.0,
        dropout: float = 0.0,
        repa_layer: int | None = None,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or model_dim <= 0 or text_dim <= 0:
            raise ValueError("latent_dim, model_dim, and text_dim must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if repa_layer is not None and not 0 <= repa_layer < depth:
            raise ValueError("repa_layer must be a valid zero-based block index")

        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)
        self.text_dim = int(text_dim)
        self.depth = int(depth)
        self.repa_layer = repa_layer

        self.preprocess_conv = ResidualPointwiseConv(self.latent_dim)
        self.input_projection = nn.Linear(
            self.latent_dim, self.model_dim, bias=False
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, self.model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim, bias=False),
        )
        self.timestep_embedding = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.duration_embedding = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.global_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim * 6),
        )
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    self.model_dim,
                    num_heads=num_heads,
                    mlp_multiplier=mlp_multiplier,
                    dropout=dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.output_norm = nn.RMSNorm(self.model_dim)
        self.output_projection = nn.Linear(
            self.model_dim, self.latent_dim, bias=False
        )
        self.postprocess_conv = ResidualPointwiseConv(self.latent_dim)

        self.repa_head: nn.Module | None = None
        if self.repa_layer is not None:
            self.repa_head = nn.Sequential(
                nn.RMSNorm(self.model_dim),
                nn.Linear(self.model_dim, self.latent_dim),
            )
            nn.init.zeros_(self.repa_head[-1].weight)
            nn.init.zeros_(self.repa_head[-1].bias)

        nn.init.zeros_(self.output_projection.weight)

    def _global_condition(
        self, timestep: torch.Tensor, duration: torch.Tensor
    ) -> torch.Tensor:
        timestep_features = _sinusoidal_embedding(timestep.float(), self.model_dim)
        duration_features = _sinusoidal_embedding(duration.float(), self.model_dim)
        return self.timestep_embedding(timestep_features.to(timestep.dtype)) + self.duration_embedding(
            duration_features.to(duration.dtype)
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        *,
        duration: torch.Tensor,
        text_embedding: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if noisy_latent.ndim != 3:
            raise ValueError("noisy_latent must have shape [batch, channels, frames]")
        batch, channels, _ = noisy_latent.shape
        if channels != self.latent_dim:
            raise ValueError(
                f"Expected {self.latent_dim} latent channels, got {channels}"
            )
        if timestep.shape != (batch,) or duration.shape != (batch,):
            raise ValueError("timestep and duration must each have shape [batch]")
        if text_embedding.ndim != 3 or text_embedding.shape[0] != batch:
            raise ValueError(
                "text_embedding must have shape [batch, tokens, text_dim]"
            )
        if text_embedding.shape[-1] != self.text_dim:
            raise ValueError(
                f"Expected text_dim={self.text_dim}, got {text_embedding.shape[-1]}"
            )
        if text_mask.shape != text_embedding.shape[:2]:
            raise ValueError("text_mask must have shape [batch, text_tokens]")
        text_mask = text_mask.bool()
        if not text_mask.any(dim=1).all():
            raise ValueError("Every text condition must contain at least one valid token")

        value = self.preprocess_conv(noisy_latent).transpose(1, 2)
        value = self.input_projection(value)
        text = self.text_projection(text_embedding.to(dtype=value.dtype))
        global_condition = self._global_condition(
            timestep.to(dtype=value.dtype), duration.to(dtype=value.dtype)
        )
        global_modulation = self.global_modulation(global_condition)

        base_prediction: torch.Tensor | None = None
        for index, block in enumerate(self.blocks):
            value = block(
                value,
                global_modulation=global_modulation,
                text=text,
                text_mask=text_mask,
            )
            if index == self.repa_layer:
                assert self.repa_head is not None
                base_prediction = self.repa_head(value).transpose(1, 2)

        x_prediction = self.output_projection(self.output_norm(value)).transpose(1, 2)
        x_prediction = self.postprocess_conv(x_prediction)
        output = {"x_pred": x_prediction}
        if base_prediction is not None:
            output["base_x_pred"] = base_prediction
        return output

    def muon_parameters(self) -> Iterable[nn.Parameter]:
        """Yield only attention QKV and feed-forward projection matrices."""
        for block in self.blocks:
            yield block.self_attn.qkv_projection.weight
            yield block.cross_attn.query_projection.weight
            yield block.cross_attn.key_value_projection.weight
            yield block.ff.in_proj.weight
            yield block.ff.out_proj.weight

    def high_lr_parameters(self) -> Iterable[nn.Parameter]:
        """Yield zero-initialized branches that need to leave zero promptly."""
        yield self.preprocess_conv.conv.weight
        for block in self.blocks:
            yield block.self_attn.output_projection.weight
            yield block.cross_attn.output_projection.weight
            yield block.ff.out_proj.weight
            yield block.ff.out_proj.bias
        yield self.output_projection.weight
        yield self.postprocess_conv.conv.weight
        if self.repa_head is not None:
            yield self.repa_head[-1].weight
            yield self.repa_head[-1].bias

    @classmethod
    def from_config(
        cls, config: dict[str, Any], *, latent_dim: int
    ) -> "AudioDiffusionTransformer":
        repa = config.get("repa", {})
        repa_layer = None
        if bool(repa.get("enabled", False)):
            # RAEv2 uses a deliberately shallow early head (after block 8/28).
            # The proportional zero-based index for the default audio depth is 2.
            depth = int(config.get("depth", 11))
            repa_layer = int(repa.get("layer", min(2, depth - 1)))
        return cls(
            latent_dim=latent_dim,
            model_dim=int(config["model_dim"]),
            text_dim=int(config["text_dim"]),
            depth=int(config.get("depth", 11)),
            num_heads=int(config.get("num_heads", 8)),
            mlp_multiplier=float(config.get("mlp_multiplier", 4.0)),
            dropout=float(config.get("dropout", 0.0)),
            repa_layer=repa_layer,
        )


__all__ = [
    "AdaLNTransformerBlock",
    "AudioDiffusionTransformer",
    "ResidualPointwiseConv",
]
