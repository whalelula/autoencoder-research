from __future__ import annotations

import copy
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
    model_name = str(model["mert_name"])
    if model_name not in {"m-a-p/MERT-v1-95M", "m-a-p/MERT-v1-330M"}:
        raise ValueError(
            "model.mert_name must be m-a-p/MERT-v1-95M or m-a-p/MERT-v1-330M"
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
