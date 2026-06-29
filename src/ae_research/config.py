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

    model = config["model"]
    model_name = str(model["mert_name"])
    if model_name not in {"m-a-p/MERT-v1-95M", "m-a-p/MERT-v1-330M"}:
        raise ValueError(
            "model.mert_name must be m-a-p/MERT-v1-95M or m-a-p/MERT-v1-330M"
        )

    fft_sizes = [int(value) for value in config["loss"]["fft_sizes"]]
    if len(fft_sizes) != 7 or fft_sizes != [32, 64, 128, 256, 512, 1024, 2048]:
        raise ValueError("SAME baseline requires exactly the seven configured FFT sizes")


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
