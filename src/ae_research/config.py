from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and validate the invariants used by the training code."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"data", "model", "loss", "training", "evaluation"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    data = config["data"]
    if int(data["sample_rate"]) <= 0:
        raise ValueError("data.sample_rate must be positive")
    if float(data["duration_seconds"]) <= 0:
        raise ValueError("data.duration_seconds must be positive")
    if int(data["channels"]) not in (1, 2):
        raise ValueError("data.channels must be 1 or 2")
    preprocessing = data.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("data.preprocessing must configure offline chunk preparation")
    for name in ("source_root", "source_manifest_dir"):
        if not str(preprocessing.get(name, "")).strip():
            raise ValueError(f"data.preprocessing.{name} must be set")
    if int(preprocessing.get("workers", 0)) <= 0:
        raise ValueError("data.preprocessing.workers must be positive")
    for name in ("drop_last", "overwrite"):
        if not isinstance(preprocessing.get(name), bool):
            raise ValueError(f"data.preprocessing.{name} must be true or false")

    model = config["model"]
    model_type = str(model.get("type", "semantic_mert_autoencoder"))
    if model_type == "semantic_mert_autoencoder":
        model_name = str(model["mert_name"])
        if model_name not in {"m-a-p/MERT-v1-95M", "m-a-p/MERT-v1-330M"}:
            raise ValueError(
                "model.mert_name must be m-a-p/MERT-v1-95M or m-a-p/MERT-v1-330M"
            )
        detail_aware = model.get("detail_aware", {})
        if not isinstance(detail_aware, dict):
            raise ValueError("model.detail_aware must be a mapping")
        if not isinstance(detail_aware.get("enabled", False), bool):
            raise ValueError("model.detail_aware.enabled must be true or false")
        if int(detail_aware.get("depth", 6)) <= 0:
            raise ValueError("model.detail_aware.depth must be positive")
        if float(detail_aware.get("mlp_ratio", 4.0)) <= 0:
            raise ValueError("model.detail_aware.mlp_ratio must be positive")
        for name in ("projection_dropout", "attention_dropout", "drop_path_rate"):
            value = float(detail_aware.get(name, 0.0))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"model.detail_aware.{name} must be in [0, 1)")
        layer_scale_init = detail_aware.get("layer_scale_init")
        if layer_scale_init is not None and float(layer_scale_init) < 0.0:
            raise ValueError("model.detail_aware.layer_scale_init must be non-negative")
    elif model_type == "same":
        if str(model.get("variant", "")) == "same_s":
            if int(data["sample_rate"]) != 44100:
                raise ValueError("official SAME-S config expects data.sample_rate=44100")
            if int(data["channels"]) != 2:
                raise ValueError("official SAME-S config expects data.channels=2")
        for name in ("latent_dim", "downsampling_ratio", "io_channels"):
            if int(model[name]) <= 0:
                raise ValueError(f"model.{name} must be positive")
        pretransform = model.get("pretransform")
        if not isinstance(pretransform, dict) or str(pretransform.get("type")) != "patched":
            raise ValueError("model.pretransform must configure patched pretransform")
        patch_config = pretransform.get("config", {})
        patch_size = int(patch_config.get("patch_size", 0))
        patch_channels = int(patch_config.get("channels", 0))
        if patch_size <= 0 or patch_channels <= 0:
            raise ValueError("model.pretransform.config patch_size/channels must be positive")
        if patch_channels != int(data["channels"]):
            raise ValueError("model.pretransform.config.channels must match data.channels")

        encoder = model.get("encoder")
        decoder = model.get("decoder")
        if not isinstance(encoder, dict) or str(encoder.get("type")) != "same":
            raise ValueError("model.encoder must configure SAME encoder")
        if not isinstance(decoder, dict) or str(decoder.get("type")) != "same":
            raise ValueError("model.decoder must configure SAME decoder")
        encoder_config = encoder.get("config", {})
        decoder_config = decoder.get("config", {})
        if int(encoder_config.get("in_channels", 0)) != patch_channels * patch_size:
            raise ValueError(
                "model.encoder.config.in_channels must equal channels * patch_size"
            )
        if int(decoder_config.get("out_channels", 0)) != patch_channels * patch_size:
            raise ValueError(
                "model.decoder.config.out_channels must equal channels * patch_size"
            )
        if int(encoder_config.get("latent_dim", 0)) != int(model["latent_dim"]):
            raise ValueError("model.encoder.config.latent_dim must match model.latent_dim")
        if int(decoder_config.get("latent_dim", 0)) != int(model["latent_dim"]):
            raise ValueError("model.decoder.config.latent_dim must match model.latent_dim")
        strides = [int(value) for value in encoder_config["strides"]]
        decoder_strides = [int(value) for value in decoder_config["strides"]]
        c_mults = [int(value) for value in encoder_config["c_mults"]]
        decoder_c_mults = [int(value) for value in decoder_config["c_mults"]]
        depths = [int(value) for value in encoder_config["transformer_depths"]]
        decoder_depths = [int(value) for value in decoder_config["transformer_depths"]]
        if not strides or len(strides) != len(c_mults) or len(strides) != len(depths):
            raise ValueError(
                "model.encoder.config strides, c_mults, and transformer_depths "
                "must be non-empty lists with the same length"
            )
        if strides != decoder_strides:
            raise ValueError("model.decoder.config.strides must match encoder strides")
        if len(decoder_c_mults) != len(strides) or len(decoder_depths) != len(strides):
            raise ValueError(
                "model.decoder.config c_mults/depths must match encoder depth"
            )
        if any(value <= 0 for value in strides + c_mults + depths):
            raise ValueError("model SAME strides, c_mults, and depths must be positive")
        chunk_size = int(encoder_config.get("chunk_size", 128))
        decoder_chunk_size = int(decoder_config.get("chunk_size", 128))
        if chunk_size <= 0:
            raise ValueError("model.encoder.config.chunk_size must be positive")
        if decoder_chunk_size <= 0:
            raise ValueError("model.decoder.config.chunk_size must be positive")
        if encoder_config.get("sliding_window") is None:
            for stride in strides:
                if chunk_size % stride != 0:
                    raise ValueError(
                        "model.encoder.config.chunk_size must be divisible by "
                        "every SAME stride when sliding_window is null"
                    )
        if decoder_config.get("sliding_window") is None:
            for stride in decoder_strides:
                if decoder_chunk_size % stride != 0:
                    raise ValueError(
                        "model.decoder.config.chunk_size must be divisible by "
                        "every SAME stride when sliding_window is null"
                    )
        bottleneck = model.get("bottleneck")
        if not isinstance(bottleneck, dict) or str(bottleneck.get("type")) != "softnorm":
            raise ValueError("model.bottleneck must configure softnorm")
        if int(bottleneck.get("config", {}).get("dim", 0)) != int(model["latent_dim"]):
            raise ValueError("model.bottleneck.config.dim must match model.latent_dim")
        expected_ratio = patch_size * math.prod(strides)
        if int(model["downsampling_ratio"]) != expected_ratio:
            raise ValueError(
                "model.downsampling_ratio must equal patch_size * product(strides)"
            )
        if str(model.get("output_activation", "none")) not in {"none", "tanh"}:
            raise ValueError("model.output_activation must be none or tanh")
    else:
        raise ValueError(
            "model.type must be semantic_mert_autoencoder or same"
        )

    loss = config["loss"]
    fft_sizes = [int(value) for value in loss["fft_sizes"]]
    if len(fft_sizes) != 7 or fft_sizes != [32, 64, 128, 256, 512, 1024, 2048]:
        raise ValueError("SAME baseline requires exactly the seven configured FFT sizes")
    stability_defaults = {
        "eps": 1e-7,
        "spectral_contrast_eps": 1e-4,
        "log_magnitude_std_floor": 1e-4,
        "complex_distance_eps": 1e-5,
        "phase_eps": 1e-3,
        "phase_weight_floor": 1e-3,
    }
    for name, default in stability_defaults.items():
        if float(loss.get(name, default)) <= 0:
            raise ValueError(f"loss.{name} must be positive")

    training = config["training"]
    if str(training.get("lr_scheduler")) != "warmup_cosine":
        raise ValueError("training.lr_scheduler must be warmup_cosine")
    warmup_steps = int(training.get("warmup_steps", -1))
    peak_lr = float(training.get("peak_lr", 0.0))
    min_lr = float(training.get("min_lr", -1.0))
    if warmup_steps < 0:
        raise ValueError("training.warmup_steps must be non-negative")
    if peak_lr <= 0:
        raise ValueError("training.peak_lr must be positive")
    if not 0 <= min_lr < peak_lr:
        raise ValueError("training.min_lr must be non-negative and smaller than peak_lr")


def merged_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config dictionaries without mutating either input."""
    result = copy.deepcopy(base)

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(result, overrides)
    validate_config(result)
    return result
