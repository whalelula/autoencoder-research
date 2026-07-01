from __future__ import annotations

import torch
import torchaudio
from torch import nn

from .decoder import MERTMirrorDecoder
from .detail_aware import AudioDetailAwareModule
from .mert import FrozenMERTEncoder


class SemanticAudioAutoencoder(nn.Module):
    """Frozen semantic encoder with a trainable waveform decoder."""

    def __init__(
        self,
        model_config: dict,
        *,
        audio_channels: int,
        data_sample_rate: int,
    ) -> None:
        super().__init__()
        self.data_sample_rate = int(data_sample_rate)
        self.encoder = FrozenMERTEncoder(
            str(model_config["mert_name"]),
            layer=int(model_config["mert_layer"]),
            trust_remote_code=bool(model_config["trust_remote_code"]),
        )
        self.native_sample_rate = self.encoder.sample_rate
        detail_config = model_config.get("detail_aware", {})
        self.detail_aware: AudioDetailAwareModule | None = None
        if bool(detail_config.get("enabled", False)):
            self.detail_aware = AudioDetailAwareModule(
                feature_extractor=self.encoder.feature_extractor,
                feature_projection=self.encoder.feature_projection,
                dim=self.encoder.hidden_size,
                depth=int(detail_config.get("depth", 6)),
                num_heads=self.encoder.num_attention_heads,
                mlp_ratio=float(detail_config.get("mlp_ratio", 4.0)),
                qkv_bias=bool(detail_config.get("qkv_bias", True)),
                projection_dropout=float(
                    detail_config.get("projection_dropout", 0.0)
                ),
                attention_dropout=float(
                    detail_config.get("attention_dropout", 0.0)
                ),
                drop_path_rate=float(detail_config.get("drop_path_rate", 0.0)),
                layer_scale_init=detail_config.get("layer_scale_init"),
                cross_attention=bool(
                    detail_config.get("cross_attention", True)
                ),
            )
        self.decoder = MERTMirrorDecoder(
            semantic_dim=self.encoder.hidden_size,
            conv_dims=self.encoder.conv_dims,
            kernels=self.encoder.conv_kernels,
            strides=self.encoder.conv_strides,
            audio_channels=audio_channels,
            output_activation=str(model_config["output_activation"]),
        )

    def train(self, mode: bool = True) -> "SemanticAudioAutoencoder":
        super().train(mode)
        self.encoder.eval()
        return self

    def _to_native_rate(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.data_sample_rate == self.native_sample_rate:
            return waveform
        return torchaudio.functional.resample(
            waveform, self.data_sample_rate, self.native_sample_rate
        )

    def _from_native_rate(
        self, waveform: torch.Tensor, target_num_samples: int
    ) -> torch.Tensor:
        if self.data_sample_rate != self.native_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, self.native_sample_rate, self.data_sample_rate
            )
        if waveform.shape[-1] > target_num_samples:
            waveform = waveform[..., :target_num_samples]
        elif waveform.shape[-1] < target_num_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_num_samples - waveform.shape[-1])
            )
        return waveform

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        target_num_samples = waveform.shape[-1]
        native = self._to_native_rate(waveform)
        normalized = self.encoder.preprocess(native)
        semantic_features = self.encoder.encode_normalized(normalized)
        modulated_features = semantic_features
        if self.detail_aware is not None:
            modulated_features = self.detail_aware(normalized, semantic_features)
        reconstruction_native, latent = self.decoder(
            modulated_features, native.shape[-1]
        )
        reconstruction = self._from_native_rate(
            reconstruction_native, target_num_samples
        )
        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "semantic_features": semantic_features,
            "modulated_features": modulated_features,
        }
