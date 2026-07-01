from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _drop_path(
    value: torch.Tensor,
    probability: float,
    training: bool,
) -> torch.Tensor:
    if probability == 0.0 or not training:
        return value
    keep_probability = 1.0 - probability
    shape = (value.shape[0],) + (1,) * (value.ndim - 1)
    keep = value.new_empty(shape).bernoulli_(keep_probability)
    return value * keep / keep_probability


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _drop_path(value, self.probability, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.attention_dropout = float(attention_dropout)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, dim = value.shape
        qkv = self.qkv(value).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, attention_value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        output = F.scaled_dot_product_attention(
            query,
            key,
            attention_value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, dim)
        return self.projection_dropout(self.projection(output))


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.attention_dropout = float(attention_dropout)
        self.query_projection = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_projection = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_projection = nn.Linear(dim, dim, bias=qkv_bias)
        self.output_projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, dim = value.shape
        context_length = context.shape[1]
        query = self.query_projection(value).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key = self.key_projection(context).reshape(
            batch_size, context_length, self.num_heads, self.head_dim
        )
        attention_value = self.value_projection(context).reshape(
            batch_size, context_length, self.num_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        attention_value = attention_value.transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query,
            key,
            attention_value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, dim)
        return self.projection_dropout(self.output_projection(output))


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(projection_dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class DeltaEncoderLayer(nn.Module):
    """PAE-style self-attention, cross-attention, and MLP delta block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        projection_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: Optional[float] = None,
        cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_cross_attention = bool(cross_attention)

        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = SelfAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        self.self_scale = (
            LayerScale(dim, layer_scale_init)
            if layer_scale_init is not None
            else nn.Identity()
        )
        self.self_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.use_cross_attention:
            self.cross_query_norm = nn.LayerNorm(dim)
            self.cross_context_norm = nn.LayerNorm(dim)
            self.cross_attention = CrossAttention(
                dim,
                num_heads,
                qkv_bias=qkv_bias,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )
            self.cross_scale = (
                LayerScale(dim, layer_scale_init)
                if layer_scale_init is not None
                else nn.Identity()
            )
            self.cross_drop_path = (
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio, projection_dropout)
        self.mlp_scale = (
            LayerScale(dim, layer_scale_init)
            if layer_scale_init is not None
            else nn.Identity()
        )
        self.mlp_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        value: torch.Tensor,
        semantic_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value = value + self.self_drop_path(
            self.self_scale(self.self_attention(self.self_norm(value)))
        )
        if self.use_cross_attention:
            if semantic_features is None:
                raise ValueError("semantic_features are required for cross-attention")
            value = value + self.cross_drop_path(
                self.cross_scale(
                    self.cross_attention(
                        self.cross_query_norm(value),
                        self.cross_context_norm(semantic_features),
                    )
                )
            )
        return value + self.mlp_drop_path(
            self.mlp_scale(self.mlp(self.mlp_norm(value)))
        )


def sinusoidal_position_embedding(
    sequence_length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a variable-length 1D sine/cosine position embedding."""
    if dim % 2 != 0:
        raise ValueError("Position embedding dimension must be even")
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    frequencies = torch.arange(dim // 2, device=device, dtype=torch.float32)
    frequencies = torch.pow(10000.0, -frequencies / (dim / 2))
    angles = positions[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return embedding.to(dtype=dtype).unsqueeze(0)


class AudioDetailAwareModule(nn.Module):
    """Inject waveform detail into frozen MERT features before decoding."""

    def __init__(
        self,
        *,
        feature_extractor: nn.Module,
        feature_projection: nn.Module,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        projection_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_scale_init: Optional[float] = None,
        cross_attention: bool = True,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        self.dim = int(dim)

        # Start from MERT's pretrained patchification weights while keeping this
        # trainable copy independent from the frozen semantic encoder.
        self.feature_extractor = copy.deepcopy(feature_extractor)
        self.feature_projection = copy.deepcopy(feature_projection)
        self.feature_extractor.requires_grad_(True)
        self.feature_projection.requires_grad_(True)
        if hasattr(self.feature_extractor, "_requires_grad"):
            self.feature_extractor._requires_grad = True

        drop_path_rates = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.layers = nn.ModuleList(
            [
                DeltaEncoderLayer(
                    dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    projection_dropout=projection_dropout,
                    attention_dropout=attention_dropout,
                    drop_path=drop_path_rates[index],
                    layer_scale_init=layer_scale_init,
                    cross_attention=cross_attention,
                )
                for index in range(depth)
            ]
        )
        self.fusion_projection = nn.Linear(dim, dim * 2)
        self.semantic_norm = nn.LayerNorm(dim)
        nn.init.zeros_(self.fusion_projection.weight)
        nn.init.zeros_(self.fusion_projection.bias)

    def patchify(self, normalized_waveform: torch.Tensor) -> torch.Tensor:
        if normalized_waveform.ndim != 2:
            raise ValueError(
                "normalized_waveform must have shape [batch, samples]"
            )
        detail_features = self.feature_extractor(normalized_waveform)
        if detail_features.ndim != 3:
            raise ValueError("MERT feature extractor must return [batch, channels, frames]")
        detail_features = detail_features.transpose(1, 2)
        detail_tokens = self.feature_projection(detail_features)
        if isinstance(detail_tokens, tuple):
            detail_tokens = detail_tokens[0]
        if detail_tokens.ndim != 3 or detail_tokens.shape[-1] != self.dim:
            raise ValueError(
                "MERT feature projection must return "
                f"[batch, frames, {self.dim}], got {tuple(detail_tokens.shape)}"
            )
        position = sinusoidal_position_embedding(
            detail_tokens.shape[1],
            self.dim,
            device=detail_tokens.device,
            dtype=detail_tokens.dtype,
        )
        return detail_tokens + position

    def forward(
        self,
        normalized_waveform: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_features.ndim != 3:
            raise ValueError("semantic_features must have shape [batch, frames, hidden]")
        if semantic_features.shape[-1] != self.dim:
            raise ValueError(
                f"Expected semantic dimension {self.dim}, got {semantic_features.shape[-1]}"
            )
        detail_tokens = self.patchify(normalized_waveform)
        if detail_tokens.shape[:2] != semantic_features.shape[:2]:
            raise ValueError(
                "Detail and MERT token grids must align exactly, got "
                f"{tuple(detail_tokens.shape[:2])} and "
                f"{tuple(semantic_features.shape[:2])}"
            )

        for layer in self.layers:
            detail_tokens = layer(detail_tokens, semantic_features)

        gamma, beta = self.fusion_projection(detail_tokens).chunk(2, dim=-1)
        # PAE's VFM tokens are already normalized, whereas intermediate MERT
        # layers are not. Express SFT as a normalized residual so W=0 gives an
        # exact identity while gamma/beta still modulate a stable feature scale.
        normalized_semantic = self.semantic_norm(semantic_features)
        return semantic_features + normalized_semantic * gamma + beta
