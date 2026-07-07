from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from torch.nn.utils import weight_norm

from .transformer import TransformerBlock


# Adapted from Stability-AI/stable-audio-tools
# commit 3241adba4fc2a85cf5b29d9eb68d42f40a28e820:
# stable_audio_tools/models/autoencoders.py.


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


class Transpose(nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return rearrange(x, "... a b -> ... b a")


def _zero_pad_modulo_sequence(
    x: torch.Tensor, size: int, dim: int = -2
) -> torch.Tensor:
    input_len = x.shape[dim]
    pad_len = (size - input_len % size) % size
    if pad_len > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        x = torch.cat(
            [x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim
        )
    return x


class TransformerResamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        sliding_window: list[int] | None = None,
        chunk_size: int = 128,
        chunk_midpoint_shift: bool = False,
        type: str = "encoder",
        transformer_depth: int = 3,
        checkpointing: bool = False,
        conformer: bool = False,
        layer_scale: bool = False,
        dim_heads: int = 128,
        differential: bool = True,
        variable_stride: bool = False,
        feat_scale: bool = False,
        sinusoidal_blocks: int = 0,
        mask_noise: float = 0.0,
        ff_mult: int = 3,
        mapping_bias: bool = True,
        cross_attn: bool = False,
        dyt: bool = True,
        conv_mapping: bool = False,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if type not in ["encoder", "decoder"]:
            raise ValueError(f"Unknown type {type}. Must be 'encoder' or 'decoder'")

        self.checkpointing = checkpointing
        transformer_dim = out_channels if type == "encoder" else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable_stride = variable_stride
        self.stride = stride
        self.mapping = (
            WNConv1d(
                in_channels,
                out_channels,
                3 if conv_mapping else 1,
                padding="same",
                bias=mapping_bias,
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.chunk_size = chunk_size
        self.chunk_midpoint_shift = chunk_midpoint_shift
        self.type = type
        self.mask_noise = mask_noise
        self.sliding_window_latents = sliding_window

        self.sliding_window_seq = self._get_sliding_window_size(sliding_window, stride)
        self.input_seg_size, self.output_seg_size, self.sub_chunk_size = (
            self._get_seg_sizes(stride)
        )
        self.transformer_depth = transformer_depth
        transformers = []
        for i in range(transformer_depth):
            sinusoidal = (transformer_depth - i) < sinusoidal_blocks
            transformers.append(
                TransformerBlock(
                    transformer_dim,
                    dim_heads=dim_heads,
                    causal=False,
                    zero_init_branch_outputs=not layer_scale,
                    norm_type="dyt" if dyt else "rms_norm",
                    conformer=conformer,
                    layer_scale=layer_scale,
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": differential,
                        "feat_scale": feat_scale,
                    },
                    ff_kwargs={
                        "mult": ff_mult,
                        "no_bias": False,
                        "sinusoidal": sinusoidal,
                    },
                    norm_kwargs={"eps": 1e-3},
                    cross_attend=cross_attn,
                )
            )

        token_channels = out_channels if type == "encoder" else in_channels
        token_length = self.output_seg_size if not self.variable_stride else 1
        self.new_tokens = nn.Parameter(
            1e-5 * torch.randn(1, token_length, token_channels)
        )
        self.transformers = nn.ModuleList(transformers)

        if freeze_backbone:
            for param in self.transformers.parameters():
                param.requires_grad = False
            self.new_tokens.requires_grad = False

    def _get_sliding_window_size(
        self,
        window: list[int] | None,
        stride: int,
        prepend_cond_length: int = 0,
    ) -> list[int] | None:
        if window is None:
            return None
        return [win * (stride + 1 + prepend_cond_length) for win in window]

    def _get_seg_sizes(
        self, stride: int, prepend_cond_length: int = 0
    ) -> tuple[int, int, int]:
        sub_chunk_size = stride + 1 + prepend_cond_length
        if self.sliding_window_latents is None:
            if self.chunk_size % stride != 0:
                raise ValueError(
                    f"Stride must fit evenly into chunk size: {self.chunk_size}"
                )
        input_seg_size = stride if self.type == "encoder" else 1
        output_seg_size = 1 if self.type == "encoder" else stride
        return input_seg_size, output_seg_size, sub_chunk_size

    def forward(
        self,
        x: torch.Tensor,
        stride: int | None = None,
        return_features: bool = False,
        override_new_tokens: torch.Tensor | None = None,
        prepend_cond: torch.Tensor | None = None,
        cross_attn_cond: torch.Tensor | None = None,
    ):
        batch_size = x.shape[0]
        if return_features:
            features = []

        if stride is None:
            input_seg_size = self.input_seg_size
            output_seg_size = self.output_seg_size
            sub_chunk_size = self.sub_chunk_size
            sliding_window = self.sliding_window_seq
        else:
            if not self.variable_stride:
                raise ValueError("Cannot override stride unless variable_stride is set")
            prepend_len = prepend_cond.shape[-2] if prepend_cond is not None else 0
            input_seg_size, output_seg_size, sub_chunk_size = self._get_seg_sizes(
                stride, prepend_len
            )
            sliding_window = self._get_sliding_window_size(
                self.sliding_window_latents, stride, prepend_len
            )

        if self.type == "encoder":
            if self.transformer_depth > 0:
                pad_modulo = self.chunk_size if sliding_window is None else input_seg_size
                x = _zero_pad_modulo_sequence(x, pad_modulo, dim=-1)
            x = self.mapping(x)

        if self.transformer_depth > 0:
            x = rearrange(x, "... a b -> ... b a")
            if return_features:
                features.append(x)
            if self.type != "encoder":
                if sliding_window is None:
                    active_stride = stride if stride is not None else self.stride
                    x = _zero_pad_modulo_sequence(x, self.chunk_size // active_stride)
                else:
                    x = _zero_pad_modulo_sequence(x, input_seg_size)

            x = rearrange(x, "b (n c) d -> (b n) c d", c=input_seg_size)
            new_token_seq_dim = -1 if not self.variable_stride else output_seg_size
            new_tokens = self.new_tokens.expand([x.shape[0], new_token_seq_dim, -1])
            if override_new_tokens is not None:
                override_new_tokens = rearrange(
                    override_new_tokens, "b (n c) d -> (b n) c d", c=output_seg_size
                )
                new_tokens = new_tokens + override_new_tokens
            elif self.mask_noise > 0:
                new_tokens = new_tokens + torch.randn_like(new_tokens) * self.mask_noise
            x = torch.cat([x, new_tokens], dim=-2)

            if prepend_cond is not None:
                n = x.shape[0] // batch_size
                cond_folded = prepend_cond.unsqueeze(1).expand(
                    batch_size, n, prepend_cond.shape[-2], x.shape[-1]
                )
                cond_folded = cond_folded.reshape(
                    n * batch_size, prepend_cond.shape[-2], x.shape[-1]
                )
                x = torch.cat([cond_folded, x], dim=-2)
            x = rearrange(x, "(b n) c d -> b (n c) d", b=batch_size)

            if sliding_window is None:
                prepend_len = prepend_cond.shape[-2] if prepend_cond is not None else 0
                active_stride = stride if stride is not None else self.stride
                effective_chunk_size = (
                    self.chunk_size + self.chunk_size * (1 + prepend_len) // active_stride
                )

            if sliding_window is None and self.chunk_midpoint_shift:
                split = self.transformer_depth // 2
                shift = effective_chunk_size // 2
                nc = x.shape[1] // effective_chunk_size
                x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)
                cross_attn_first = (
                    None
                    if cross_attn_cond is None
                    else cross_attn_cond.repeat_interleave(nc, dim=0)
                )
                for layer in self.transformers[:split]:
                    x = (
                        checkpoint(
                            layer,
                            x,
                            context=cross_attn_first,
                            self_attention_flash_sliding_window=None,
                        )
                        if self.checkpointing
                        else layer(x, context=cross_attn_first)
                    )
                    if return_features:
                        features.append(
                            rearrange(
                                x, "(b nc) cc d -> b (nc cc) d", b=batch_size
                            )
                        )
                x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)

                x = torch.cat([x[:, :shift, :], x, x[:, -shift:, :]], dim=1)
                nc_shifted = x.shape[1] // effective_chunk_size
                x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)
                cross_attn_second = (
                    None
                    if cross_attn_cond is None
                    else cross_attn_cond.repeat_interleave(nc_shifted, dim=0)
                )
                for layer in self.transformers[split:]:
                    x = (
                        checkpoint(
                            layer,
                            x,
                            context=cross_attn_second,
                            self_attention_flash_sliding_window=None,
                        )
                        if self.checkpointing
                        else layer(x, context=cross_attn_second)
                    )
                    if return_features:
                        feat = rearrange(
                            x, "(b nc) cc d -> b (nc cc) d", b=batch_size
                        )
                        features.append(feat[:, shift:-shift, :])
                x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)
                x = x[:, shift:-shift, :]
            else:
                if sliding_window is None:
                    x = rearrange(
                        x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size
                    )
                for layer in self.transformers:
                    x = (
                        checkpoint(
                            layer,
                            x,
                            context=cross_attn_cond,
                            self_attention_flash_sliding_window=sliding_window,
                        )
                        if self.checkpointing
                        else layer(
                            x,
                            context=cross_attn_cond,
                            self_attention_flash_sliding_window=sliding_window,
                        )
                    )
                    if return_features:
                        features.append(x)
                if sliding_window is None:
                    x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)

            x = rearrange(x, "b (n c) d -> (b n) c d", c=sub_chunk_size)
            x = x[:, -output_seg_size:, :]
            x = rearrange(x, "(b n) c d -> b d (n c)", b=batch_size)

        if self.type == "decoder":
            x = self.mapping(x)
        if return_features:
            return x, features
        return x


class SAMEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        transformer_depths: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        c_mults = c_mults or [1, 2, 4, 8]
        strides = strides or [2, 4, 8, 8]
        transformer_depths = transformer_depths or [3, 3, 3, 3]
        self.in_channels = in_channels
        self.strides = strides
        channel_dims = [in_channels] + [c * channels for c in c_mults]
        self.depth = len(c_mults)
        layers: list[nn.Module] = []
        for i in range(self.depth):
            layers.append(
                TransformerResamplingBlock(
                    in_channels=channel_dims[i],
                    out_channels=channel_dims[i + 1],
                    stride=strides[i],
                    transformer_depth=transformer_depths[i],
                    **kwargs,
                )
            )
        layers += [Transpose(), nn.Linear(channel_dims[-1], latent_dim), Transpose()]
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        override_stride: list[int] | None = None,
        return_features: bool = False,
        **kwargs,
    ):
        if override_stride is not None:
            if len(override_stride) != self.depth:
                raise ValueError("override_stride must contain one stride per layer")
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerResamplingBlock):
                stride = override_stride[i] if override_stride is not None else None
                if return_features:
                    x, features = layer(x, stride=stride, return_features=True)
                else:
                    x = layer(x, stride=stride)
            else:
                x = layer(x)
        if return_features:
            return x, features
        return x


class SAMEDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        transformer_depths: list[int] | None = None,
        sinusoidal_blocks: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        c_mults = c_mults or [1, 2, 4, 8]
        strides = strides or [2, 4, 8, 8]
        transformer_depths = transformer_depths or [3, 3, 3, 3]
        sinusoidal_blocks = sinusoidal_blocks or [0 for _ in c_mults]
        channel_dims = [out_channels] + [c * channels for c in c_mults]
        self.depth = len(c_mults)
        layers: list[nn.Module] = [
            Transpose(),
            nn.Linear(latent_dim, channel_dims[-1]),
            Transpose(),
        ]
        for i in range(self.depth, 0, -1):
            layers.append(
                TransformerResamplingBlock(
                    in_channels=channel_dims[i],
                    out_channels=channel_dims[i - 1],
                    stride=strides[i - 1],
                    type="decoder",
                    transformer_depth=transformer_depths[i - 1],
                    sinusoidal_blocks=sinusoidal_blocks[i - 1],
                    **kwargs,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, override_stride: list[int] | None = None, **kwargs
    ) -> torch.Tensor:
        if override_stride is not None:
            if len(override_stride) != self.depth:
                raise ValueError("override_stride must contain one stride per layer")
        transformer_layer_index = 0
        for layer in self.layers:
            if isinstance(layer, TransformerResamplingBlock):
                stride = (
                    override_stride[transformer_layer_index]
                    if override_stride is not None
                    else None
                )
                x = layer(x, stride=stride)
                transformer_layer_index += 1
            else:
                x = layer(x)
        return x

