"""Frozen autoencoder adapters for latent diffusion training.

Both supported project autoencoders expose a common ``[batch, channels, time]``
latent interface here.  Semantic/MERT models use their semantic representation
directly (before the waveform decoder's projection); SAME models use their
SoftNorm bottleneck representation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch import nn


CodecKind = Literal["semantic", "same"]
ModelFactory = Callable[..., nn.Module]


def _channel_vector(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().clone()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    elif tensor.ndim == 2 and 1 in tensor.shape:
        tensor = tensor.reshape(-1)
    elif tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[-1] == 1:
        tensor = tensor.reshape(-1)
    elif tensor.ndim != 1:
        raise ValueError(f"{name} must be scalar or contain one value per latent channel")
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")
    return tensor


class LatentNormalizer(nn.Module):
    """Fixed, exactly reversible channel-wise latent normalization."""

    def __init__(
        self,
        mean: torch.Tensor | Sequence[float] | float,
        std: torch.Tensor | Sequence[float] | float,
        *,
        latent_dim: int | None = None,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        channel_mean = _channel_vector(mean, "mean")
        channel_std = _channel_vector(std, "std")
        if latent_dim is not None:
            latent_dim = int(latent_dim)
            if latent_dim <= 0:
                raise ValueError("latent_dim must be positive")
            if channel_mean.numel() == 1 and latent_dim > 1:
                channel_mean = channel_mean.expand(latent_dim).clone()
            if channel_std.numel() == 1 and latent_dim > 1:
                channel_std = channel_std.expand(latent_dim).clone()
        if channel_mean.numel() != channel_std.numel():
            raise ValueError("mean and std must contain the same number of channels")
        eps = float(eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        if torch.any(channel_std < eps):
            raise ValueError("std must be at least eps in every channel")
        self.eps = eps
        self.register_buffer("mean", channel_mean.reshape(1, -1, 1), persistent=True)
        self.register_buffer("std", channel_std.reshape(1, -1, 1), persistent=True)

    @property
    def latent_dim(self) -> int:
        return int(self.mean.shape[1])

    @classmethod
    def identity(cls, latent_dim: int) -> "LatentNormalizer":
        latent_dim = int(latent_dim)
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        return cls(torch.zeros(latent_dim), torch.ones(latent_dim))

    @classmethod
    def from_stats(
        cls,
        path: str | Path,
        *,
        latent_dim: int | None = None,
        map_location: str | torch.device = "cpu",
        eps: float = 1.0e-8,
    ) -> "LatentNormalizer":
        """Load ``mean``/``std`` (or channel-prefixed aliases) from a ``.pt`` file."""

        stats_path = Path(path)
        try:
            stats = torch.load(stats_path, map_location=map_location, weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch
            stats = torch.load(stats_path, map_location=map_location)
        if not isinstance(stats, Mapping):
            raise ValueError(f"Latent stats must be a mapping: {stats_path}")
        mean = stats.get("mean", stats.get("channel_mean"))
        std = stats.get("std", stats.get("channel_std"))
        if std is None:
            variance = stats.get("var", stats.get("channel_var"))
            if variance is not None:
                std = torch.as_tensor(variance, dtype=torch.float32).clamp_min(0).sqrt()
        if mean is None or std is None:
            raise ValueError("Latent stats must contain mean and std/var tensors")
        return cls(mean, std, latent_dim=latent_dim, eps=eps)

    def _validate(self, latent: torch.Tensor) -> None:
        if latent.ndim != 3:
            raise ValueError(
                f"Latent must have shape [batch, channels, frames], got {tuple(latent.shape)}"
            )
        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Latent has {latent.shape[1]} channels, expected {self.latent_dim}"
            )

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        self._validate(latent)
        mean = self.mean.to(device=latent.device, dtype=latent.dtype)
        std = self.std.to(device=latent.device, dtype=latent.dtype)
        return (latent - mean) / std

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        self._validate(latent)
        mean = self.mean.to(device=latent.device, dtype=latent.dtype)
        std = self.std.to(device=latent.device, dtype=latent.dtype)
        return latent * std + mean

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.normalize(latent)


class ChannelStatsAccumulator:
    """Numerically stable population statistics over batch and time."""

    def __init__(self, channels: int) -> None:
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.count = 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros(channels, dtype=torch.float64)

    def update(self, latent: torch.Tensor) -> None:
        if latent.ndim != 3 or latent.shape[1] != self.mean.numel():
            raise ValueError("latent must have shape [batch, channels, frames]")
        values = latent.detach().to(device="cpu", dtype=torch.float64)
        batch_count = int(values.shape[0] * values.shape[2])
        if batch_count == 0:
            return
        batch_mean = values.mean(dim=(0, 2))
        centered = values - batch_mean.reshape(1, -1, 1)
        batch_m2 = centered.square().sum(dim=(0, 2))
        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2.add_(batch_m2).add_(
            delta.square(), alpha=self.count * batch_count / total
        )
        self.mean.add_(delta, alpha=batch_count / total)
        self.count = total

    def finalize(self, *, eps: float) -> dict[str, torch.Tensor | int]:
        if self.count == 0:
            raise RuntimeError("Cannot compute latent statistics from zero values")
        variance = (self.m2 / self.count).clamp_min(0.0)
        std = variance.sqrt().clamp_min(float(eps))
        return {
            "mean": self.mean.float(),
            "std": std.float(),
            "var": variance.float(),
            "count": self.count,
        }


def _load_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Autoencoder config must contain a mapping: {config_path}")
    return value


def _config_parts(config: Mapping[str, Any]) -> tuple[dict[str, Any], int, int]:
    data = config.get("data")
    if isinstance(data, Mapping):
        sample_rate = data.get("sample_rate")
        channels = data.get("channels")
    else:
        sample_rate = config.get("sample_rate")
        channels = config.get("audio_channels", config.get("channels"))
    if sample_rate is None or int(sample_rate) <= 0:
        raise ValueError("Autoencoder config must define a positive sample rate")
    if channels is None or int(channels) not in (1, 2):
        raise ValueError("Autoencoder config channels must be 1 or 2")
    model_config = config.get("model", config)
    if not isinstance(model_config, Mapping):
        raise ValueError("Autoencoder model config must be a mapping")
    return dict(model_config), int(sample_rate), int(channels)


def _canonical_kind(value: Any, model_config: Mapping[str, Any]) -> CodecKind:
    requested = str(value or "auto").lower().replace("-", "_")
    configured = str(model_config.get("type", "semantic_mert_autoencoder")).lower()
    if requested in {"same", "same_s"}:
        kind: CodecKind = "same"
    elif requested in {"semantic", "semantic_mert_autoencoder"}:
        kind = "semantic"
    elif requested == "auto":
        kind = "same" if configured == "same" else "semantic"
    else:
        raise ValueError("Codec kind must be auto, semantic, or same")
    if configured == "same" and kind != "same":
        raise ValueError("Requested semantic codec for a SAME autoencoder config")
    if configured in {"semantic", "semantic_mert_autoencoder"} and kind != "semantic":
        raise ValueError("Requested SAME codec for a semantic autoencoder config")
    return kind


def _default_model_factory(
    model_config: dict[str, Any],
    *,
    audio_channels: int,
    data_sample_rate: int,
    kind: CodecKind,
) -> nn.Module:
    # Imports are intentionally deferred so importing the codec utilities never
    # constructs or downloads a text/audio foundation model.
    if kind == "same":
        from ae_research.models.same_autoencoder import SameAutoencoder

        return SameAutoencoder(
            model_config,
            audio_channels=audio_channels,
            data_sample_rate=data_sample_rate,
        )
    from ae_research.models.autoencoder import SemanticAudioAutoencoder

    return SemanticAudioAutoencoder(
        model_config,
        audio_channels=audio_channels,
        data_sample_rate=data_sample_rate,
    )


def _call_model_factory(
    factory: ModelFactory,
    model_config: dict[str, Any],
    *,
    channels: int,
    sample_rate: int,
    kind: CodecKind,
) -> nn.Module:
    try:
        model = factory(
            model_config,
            audio_channels=channels,
            data_sample_rate=sample_rate,
            kind=kind,
        )
    except TypeError as error:
        # Project model constructors predate the codec and do not accept kind.
        if "kind" not in str(error):
            raise
        model = factory(
            model_config,
            audio_channels=channels,
            data_sample_rate=sample_rate,
        )
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    return model


def _state_dict(value: Any) -> Mapping[str, torch.Tensor] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    if not all(torch.is_tensor(item) for item in value.values()):
        return None
    return value  # type: ignore[return-value]


def _load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    kind: CodecKind,
    map_location: str | torch.device,
    strict: bool,
) -> Mapping[str, Any]:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Autoencoder checkpoint must be a mapping: {path}")

    if kind == "semantic" and "decoder" in checkpoint:
        decoder_state = _state_dict(checkpoint["decoder"])
        if decoder_state is None or not hasattr(model, "decoder"):
            raise ValueError("Semantic checkpoint has an invalid decoder state")
        model.decoder.load_state_dict(decoder_state, strict=strict)
        detail_aware = getattr(model, "detail_aware", None)
        if detail_aware is not None:
            detail_state = _state_dict(checkpoint.get("detail_aware"))
            if detail_state is None:
                if strict:
                    raise KeyError("Semantic checkpoint is missing detail_aware state")
            else:
                detail_aware.load_state_dict(detail_state, strict=strict)
        return checkpoint

    candidate = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = _state_dict(candidate)
    if state is None:
        raise ValueError(f"Checkpoint does not contain a model state dict: {path}")
    model.load_state_dict(state, strict=strict)
    return checkpoint


def _checkpoint_audio_format(checkpoint: Mapping[str, Any]) -> tuple[int, int] | None:
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        return None
    try:
        _, sample_rate, channels = _config_parts(config)
    except ValueError:
        return None
    return sample_rate, channels


def _infer_latent_dim(model: nn.Module, kind: CodecKind) -> int:
    if kind == "same":
        candidates = (
            getattr(model, "latent_dim", None),
            getattr(getattr(model, "encoder", None), "latent_dim", None),
        )
    else:
        encoder = getattr(model, "encoder", None)
        candidates = (
            getattr(encoder, "hidden_size", None),
            getattr(model, "latent_dim", None),
        )
    for candidate in candidates:
        if candidate is not None and int(candidate) > 0:
            return int(candidate)
    raise ValueError("Could not infer autoencoder latent_dim from the loaded model")


class FrozenAutoencoderCodec(nn.Module):
    """Load and expose a project autoencoder as a permanently frozen codec."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        kind: str = "auto",
        expected_sample_rate: int | None = None,
        expected_channels: int | None = None,
        latent_stats_path: str | Path | None = None,
        normalizer: LatentNormalizer | None = None,
        normalizer_eps: float = 1.0e-8,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        model_factory: ModelFactory | None = None,
    ) -> None:
        super().__init__()
        config = _load_mapping(config_path)
        model_config, sample_rate, channels = _config_parts(config)
        codec_kind = _canonical_kind(kind, model_config)
        if expected_sample_rate is not None and int(expected_sample_rate) != sample_rate:
            raise ValueError(
                f"Diffusion sample rate {expected_sample_rate} does not match "
                f"autoencoder sample rate {sample_rate}"
            )
        if expected_channels is not None and int(expected_channels) != channels:
            raise ValueError(
                f"Diffusion channels {expected_channels} do not match "
                f"autoencoder channels {channels}"
            )

        factory = model_factory or _default_model_factory
        model = _call_model_factory(
            factory,
            model_config,
            channels=channels,
            sample_rate=sample_rate,
            kind=codec_kind,
        )
        checkpoint = _load_checkpoint(
            model,
            checkpoint_path,
            kind=codec_kind,
            map_location=map_location,
            strict=strict,
        )
        checkpoint_format = _checkpoint_audio_format(checkpoint)
        if checkpoint_format is not None and checkpoint_format != (sample_rate, channels):
            raise ValueError(
                "Autoencoder checkpoint audio format does not match its config: "
                f"checkpoint={checkpoint_format}, config={(sample_rate, channels)}"
            )

        latent_dim = _infer_latent_dim(model, codec_kind)
        if normalizer is not None and latent_stats_path is not None:
            raise ValueError("Pass normalizer or latent_stats_path, not both")
        if normalizer is None:
            if latent_stats_path is None:
                normalizer = LatentNormalizer.identity(latent_dim)
            else:
                normalizer = LatentNormalizer.from_stats(
                    latent_stats_path,
                    latent_dim=latent_dim,
                    map_location=map_location,
                    eps=normalizer_eps,
                )
        if normalizer.latent_dim != latent_dim:
            raise ValueError(
                f"Latent normalizer has {normalizer.latent_dim} channels, "
                f"but codec has {latent_dim}"
            )

        self.autoencoder = model
        self.normalizer = normalizer
        self.kind: CodecKind = codec_kind
        self.sample_rate = sample_rate
        self.channels = channels
        self.latent_dim = latent_dim
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self._freeze()

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        kind: CodecKind,
        sample_rate: int,
        channels: int,
        normalizer: LatentNormalizer | None = None,
    ) -> "FrozenAutoencoderCodec":
        """Build an adapter around an in-memory model (primarily for tests/tools)."""

        if int(sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if int(channels) not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        latent_dim = _infer_latent_dim(model, kind)
        if normalizer is None:
            normalizer = LatentNormalizer.identity(latent_dim)
        if normalizer.latent_dim != latent_dim:
            raise ValueError("Normalizer channels must match model latent_dim")
        instance.autoencoder = model
        instance.normalizer = normalizer
        instance.kind = kind
        instance.sample_rate = int(sample_rate)
        instance.channels = int(channels)
        instance.latent_dim = latent_dim
        instance.config_path = None
        instance.checkpoint_path = None
        instance._freeze()
        return instance

    @property
    def model(self) -> nn.Module:
        """Compatibility alias for callers that refer to the wrapped model."""

        return self.autoencoder

    def _freeze(self) -> None:
        super().train(False)
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()
        self.normalizer.requires_grad_(False)
        self.normalizer.eval()

    def train(self, mode: bool = True) -> "FrozenAutoencoderCodec":
        # ``parent.train()`` must never re-enable dropout, running statistics,
        # or SAME's training-only bottleneck behaviour.
        super().train(False)
        self._freeze()
        return self

    def _validate_waveform(self, waveform: torch.Tensor) -> None:
        if waveform.ndim != 3:
            raise ValueError(
                f"Waveform must have shape [batch, channels, samples], got {tuple(waveform.shape)}"
            )
        if waveform.shape[1] != self.channels:
            raise ValueError(
                f"Waveform has {waveform.shape[1]} channels, expected {self.channels}"
            )

    def _validate_raw_latent(self, latent: torch.Tensor) -> None:
        if latent.ndim != 3:
            raise ValueError("Codec latent must have shape [batch, channels, frames]")
        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Codec produced {latent.shape[1]} latent channels, expected {self.latent_dim}"
            )

    def _encode_semantic(self, waveform: torch.Tensor) -> torch.Tensor:
        model = self.autoencoder
        encoder = getattr(model, "encoder", None)
        if (
            encoder is not None
            and hasattr(encoder, "preprocess")
            and hasattr(encoder, "encode_normalized")
        ):
            native = model._to_native_rate(waveform) if hasattr(model, "_to_native_rate") else waveform
            normalized = encoder.preprocess(native)
            semantic_features = encoder.encode_normalized(normalized)
            detail_aware = getattr(model, "detail_aware", None)
            modulated_features = (
                detail_aware(normalized, semantic_features)
                if detail_aware is not None
                else semantic_features
            )
        else:
            outputs = model(waveform)
            if not isinstance(outputs, Mapping) or "modulated_features" not in outputs:
                raise TypeError("Semantic autoencoder must return modulated_features")
            modulated_features = outputs["modulated_features"]
        if modulated_features.ndim != 3:
            raise ValueError("modulated_features must have shape [batch, frames, hidden]")
        # Semantic RAE latents are the modulated representation itself, not the
        # narrower convolutional projection produced inside the audio decoder.
        return modulated_features.transpose(1, 2).contiguous()

    def _encode_same(self, waveform: torch.Tensor) -> torch.Tensor:
        model = self.autoencoder
        try:
            patched = model.pretransform.encode(waveform)
            latent = model.encoder(patched)
        except AttributeError as error:
            raise TypeError("SAME codec requires pretransform and encoder modules") from error
        bottleneck = getattr(model, "bottleneck", None)
        if bottleneck is not None:
            latent = bottleneck.encode(latent)
            if isinstance(latent, tuple):
                latent = latent[0]
        return latent

    @torch.no_grad()
    def encode_raw(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode without applying optional channel statistics."""

        self._validate_waveform(waveform)
        self.autoencoder.eval()
        latent = (
            self._encode_semantic(waveform)
            if self.kind == "semantic"
            else self._encode_same(waveform)
        )
        self._validate_raw_latent(latent)
        return latent

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.normalizer.normalize(self.encode_raw(waveform))

    def _semantic_native_length(self, target_num_samples: int) -> int:
        model = self.autoencoder
        native_rate = int(getattr(model, "native_sample_rate", self.sample_rate))
        data_rate = int(getattr(model, "data_sample_rate", self.sample_rate))
        if native_rate == data_rate:
            return target_num_samples
        return int(math.ceil(target_num_samples * native_rate / data_rate))

    def _decode_semantic(self, latent: torch.Tensor, target_num_samples: int) -> torch.Tensor:
        model = self.autoencoder
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            raise TypeError("Semantic codec requires a decoder")
        semantic_features = latent.transpose(1, 2).contiguous()
        native_length = self._semantic_native_length(target_num_samples)
        # Calling forward is intentional: it applies the decoder's learned
        # projection before its convolutional synthesis stack.
        decoded = decoder(semantic_features, native_length)
        waveform_native = decoded[0] if isinstance(decoded, tuple) else decoded
        if hasattr(model, "_from_native_rate"):
            return model._from_native_rate(waveform_native, target_num_samples)
        return _match_length(waveform_native, target_num_samples)

    @staticmethod
    def _decode_same_bottleneck(latent: torch.Tensor, bottleneck: nn.Module) -> torch.Tensor:
        """Deterministically reproduce SoftNorm decode without random regularization."""

        value = latent
        running_std = getattr(bottleneck, "running_std", None)
        if running_std is not None:
            value = value * running_std.to(device=value.device, dtype=value.dtype)

        noise_augment_dim = int(getattr(bottleneck, "noise_augment_dim", 0))
        if noise_augment_dim > 0:
            # Noise augmentation is an optional decoder input.  Its deterministic
            # expectation is zero, which preserves decoder shape without RNG.
            zeros = torch.zeros(
                value.shape[0],
                noise_augment_dim,
                value.shape[-1],
                device=value.device,
                dtype=value.dtype,
            )
            value = torch.cat([value, zeros], dim=1)
        return value

    def _decode_same(self, latent: torch.Tensor, target_num_samples: int) -> torch.Tensor:
        model = self.autoencoder
        bottleneck = getattr(model, "bottleneck", None)
        decoder_latent = (
            self._decode_same_bottleneck(latent, bottleneck)
            if bottleneck is not None
            else latent
        )
        try:
            decoded = model.decoder(decoder_latent)
            waveform = model.pretransform.decode(decoded)
        except AttributeError as error:
            raise TypeError("SAME codec requires decoder and pretransform modules") from error
        if hasattr(model, "_match_length"):
            waveform = model._match_length(waveform, target_num_samples)
        else:
            waveform = _match_length(waveform, target_num_samples)
        if str(getattr(model, "output_activation", "none")) == "tanh":
            waveform = torch.tanh(waveform)
        return waveform

    @torch.no_grad()
    def decode_raw(self, latent: torch.Tensor, target_num_samples: int) -> torch.Tensor:
        """Decode an unnormalized latent deterministically."""

        target_num_samples = int(target_num_samples)
        if target_num_samples <= 0:
            raise ValueError("target_num_samples must be positive")
        self._validate_raw_latent(latent)
        self.autoencoder.eval()
        waveform = (
            self._decode_semantic(latent, target_num_samples)
            if self.kind == "semantic"
            else self._decode_same(latent, target_num_samples)
        )
        self._validate_waveform(waveform)
        if waveform.shape[-1] != target_num_samples:
            raise RuntimeError("Codec decoder failed to return the requested waveform length")
        return waveform

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, target_num_samples: int) -> torch.Tensor:
        return self.decode_raw(
            self.normalizer.denormalize(latent), target_num_samples=target_num_samples
        )


def _match_length(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    if waveform.shape[-1] > target_num_samples:
        return waveform[..., :target_num_samples]
    if waveform.shape[-1] < target_num_samples:
        return torch.nn.functional.pad(
            waveform, (0, target_num_samples - waveform.shape[-1])
        )
    return waveform


def load_frozen_codec(
    autoencoder_config: Mapping[str, Any],
    *,
    expected_sample_rate: int | None = None,
    expected_channels: int | None = None,
    map_location: str | torch.device = "cpu",
    model_factory: ModelFactory | None = None,
) -> FrozenAutoencoderCodec:
    """Construct a codec from the ``autoencoder`` section of a DiT config."""

    if not isinstance(autoencoder_config, Mapping):
        raise ValueError("autoencoder config must be a mapping")
    normalizer_config = autoencoder_config.get("normalizer", {})
    if normalizer_config is None:
        normalizer_config = {}
    if not isinstance(normalizer_config, Mapping):
        raise ValueError("autoencoder.normalizer must be a mapping")
    normalizer_enabled = normalizer_config.get("enabled", True)
    if not isinstance(normalizer_enabled, bool):
        raise ValueError("autoencoder.normalizer.enabled must be true or false")
    stats_path = autoencoder_config.get(
        "latent_stats_path",
        autoencoder_config.get("stats_path", normalizer_config.get("stats_path")),
    )
    if not normalizer_enabled:
        stats_path = None
    return FrozenAutoencoderCodec(
        autoencoder_config.get("config_path"),
        autoencoder_config.get("checkpoint_path"),
        kind=str(autoencoder_config.get("type", "auto")),
        expected_sample_rate=expected_sample_rate,
        expected_channels=expected_channels,
        latent_stats_path=stats_path,
        normalizer_eps=float(normalizer_config.get("eps", 1.0e-8)),
        map_location=map_location,
        model_factory=model_factory,
    )


FrozenCodecAdapter = FrozenAutoencoderCodec


__all__ = [
    "ChannelStatsAccumulator",
    "CodecKind",
    "FrozenAutoencoderCodec",
    "FrozenCodecAdapter",
    "LatentNormalizer",
    "load_frozen_codec",
]
