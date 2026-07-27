"""Configuration loading and validation for downstream DiT experiments.

The autoencoder training configuration intentionally remains independent from
this module.  A diffusion config points at an existing autoencoder config and
checkpoint, then describes only the frozen-codec downstream experiment.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DIT_DEPTH = 11
DEFAULT_TEXT_ENCODER = "google/t5gemma-s-s-ul2"
REQUIRED_SECTIONS = (
    "data",
    "autoencoder",
    "dit",
    "conditioning",
    "diffusion",
    "optimizer",
    "training",
    "evaluation",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _path(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not str(value).strip():
        qualifier = "a non-empty path when set" if optional else "a non-empty path"
        raise ValueError(f"{name} must be {qualifier}")
    return str(value)


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    minimum = 0 if allow_zero else 1
    if converted < minimum:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return converted


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    valid = converted >= 0.0 if allow_zero else converted > 0.0
    if not valid:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return converted


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _close(value: Any, expected: float, name: str) -> float:
    converted = _positive_float(value, name, allow_zero=expected == 0.0)
    if not math.isclose(converted, expected, rel_tol=1e-7, abs_tol=1e-12):
        raise ValueError(f"{name} must be {expected:g}")
    return converted


def _first(mapping: Mapping[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _apply_defaults(config: dict[str, Any]) -> None:
    """Add only architecture-defining defaults to an already copied config."""

    dit = config.get("dit")
    if isinstance(dit, dict):
        dit.setdefault("depth", DEFAULT_DIT_DEPTH)
        dit.setdefault("mlp_multiplier", 4.0)
        dit.setdefault("dropout", 0.0)

    conditioning = config.get("conditioning")
    if isinstance(conditioning, dict):
        text_encoder = conditioning.get("text_encoder")
        if text_encoder is None:
            fallback_names = ("model_name", "max_length", "batch_size", "cache_dir")
            text_encoder = {
                name: conditioning[name] for name in fallback_names if name in conditioning
            }
            conditioning["text_encoder"] = text_encoder
        if isinstance(text_encoder, dict):
            text_encoder.setdefault("model_name", DEFAULT_TEXT_ENCODER)
            text_encoder.setdefault("max_length", 256)
            text_encoder.setdefault("batch_size", 32)
            text_encoder.setdefault("frozen", True)

    diffusion = config.get("diffusion")
    if isinstance(diffusion, dict):
        diffusion.setdefault("prediction_type", "x_prediction")
        diffusion.setdefault("t_eps", 0.05)
        diffusion.setdefault(
            "timestep_sampler",
            {
                "type": "truncated_logit_normal",
                "mean": 0.0,
                "std": 1.0,
                "truncation": 0.075,
                "rescale": True,
                "flip": False,
            },
        )
        diffusion.setdefault(
            "sampling",
            {
                "steps": 50,
                "guidance_scale": 1.0,
                "guidance_interval": [0.1, 1.0],
            },
        )

    optimizer = config.get("optimizer")
    if isinstance(optimizer, dict):
        optimizer.setdefault("type", "muon_adamw")
        muon = optimizer.setdefault("muon", {})
        if isinstance(muon, dict):
            muon.setdefault("lr", 1.0e-5)
            muon.setdefault("momentum", 0.95)
        adamw = optimizer.setdefault("adamw", {})
        if isinstance(adamw, dict):
            adamw.setdefault("lr", 1.0e-6)
            adamw.setdefault("betas", [0.9, 0.95])
            adamw.setdefault("weight_decay", 0.01)

    training = config.get("training")
    if isinstance(training, dict):
        training.setdefault("lr_scheduler", "warmup_cosine")


def load_dit_config(path: str | Path) -> dict[str, Any]:
    """Load, default, and validate a DiT YAML configuration.

    Referenced data, autoencoder, and output paths are deliberately not checked
    for existence here.  This keeps configuration inspection and dry runs free
    from downloads and other external side effects.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    config = copy.deepcopy(loaded)
    _apply_defaults(config)
    validate_dit_config(config)
    return config


def _validate_no_cfg(*sections: tuple[str, Mapping[str, Any]]) -> None:
    bool_names = ("cfg", "use_cfg", "cfg_enabled", "classifier_free_guidance")
    numeric_names = ("cfg_dropout", "cfg_dropout_prob", "cfg_scale")
    for section_name, section in sections:
        for name in bool_names:
            if name in section and section[name] is not False:
                raise ValueError(f"{section_name}.{name} must be false; CFG is disabled")
        for name in numeric_names:
            if name not in section:
                continue
            value = float(section[name])
            expected = 1.0 if name == "cfg_scale" else 0.0
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{section_name}.{name} must be {expected:g}; CFG is disabled"
                )


def _validate_data(data: Mapping[str, Any]) -> None:
    _path(data.get("root"), "data.root")
    _positive_int(data.get("sample_rate"), "data.sample_rate")
    channels = _positive_int(data.get("channels"), "data.channels")
    if channels not in (1, 2):
        raise ValueError("data.channels must be 1 or 2")
    if "duration_seconds" in data:
        _positive_float(data["duration_seconds"], "data.duration_seconds")

    manifest_names = (
        "manifest_path",
        "manifest_dir",
        "dit_manifest_dir",
        "text_manifest_dir",
    )
    present_manifests = [name for name in manifest_names if name in data]
    if not present_manifests:
        raise ValueError(
            "data must set manifest_path, manifest_dir, dit_manifest_dir, or text_manifest_dir"
        )
    for name in present_manifests:
        _path(data[name], f"data.{name}")
    for name in ("root", "audio_root", "text_embedding_dir"):
        if name in data:
            _path(data[name], f"data.{name}")
    if "num_workers" in data:
        _positive_int(data["num_workers"], "data.num_workers", allow_zero=True)
    if "pin_memory" in data:
        _boolean(data["pin_memory"], "data.pin_memory")


def _validate_autoencoder(
    autoencoder: Mapping[str, Any], data: Mapping[str, Any]
) -> None:
    _path(autoencoder.get("config_path"), "autoencoder.config_path")
    _path(autoencoder.get("checkpoint_path"), "autoencoder.checkpoint_path")
    kind = str(autoencoder.get("type", "auto")).lower()
    if kind not in {"auto", "semantic", "semantic_mert_autoencoder", "same", "same_s"}:
        raise ValueError("autoencoder.type must be auto, semantic, or same")
    for name in ("latent_stats_path", "stats_path"):
        if name in autoencoder:
            _path(autoencoder[name], f"autoencoder.{name}", optional=True)
    normalizer = autoencoder.get("normalizer")
    if normalizer is not None:
        normalizer = _mapping(normalizer, "autoencoder.normalizer")
        if "enabled" in normalizer:
            _boolean(normalizer["enabled"], "autoencoder.normalizer.enabled")
        if "stats_path" in normalizer:
            _path(
                normalizer["stats_path"],
                "autoencoder.normalizer.stats_path",
                optional=True,
            )
        if "eps" in normalizer:
            _positive_float(normalizer["eps"], "autoencoder.normalizer.eps")
    if "sample_rate" in autoencoder and int(autoencoder["sample_rate"]) != int(
        data["sample_rate"]
    ):
        raise ValueError("autoencoder.sample_rate must match data.sample_rate")
    if "channels" in autoencoder and int(autoencoder["channels"]) != int(data["channels"]):
        raise ValueError("autoencoder.channels must match data.channels")


def _validate_dit(dit: Mapping[str, Any]) -> int:
    depth = _positive_int(dit.get("depth", DEFAULT_DIT_DEPTH), "dit.depth")
    model_dim = _positive_int(
        _first(dit, ("model_dim", "hidden_dim", "embed_dim")), "dit.model_dim"
    )
    num_heads = _positive_int(dit.get("num_heads"), "dit.num_heads")
    if model_dim % num_heads != 0:
        raise ValueError("dit.model_dim must be divisible by dit.num_heads")
    _positive_int(dit.get("text_dim"), "dit.text_dim")
    if "latent_dim" in dit:
        _positive_int(dit["latent_dim"], "dit.latent_dim")
    if any(name in dit for name in ("mlp_multiplier", "ff_multiplier", "ff_mult")):
        _positive_float(
            _first(dit, ("mlp_multiplier", "ff_multiplier", "ff_mult")),
            "dit.mlp_multiplier",
        )
    for name in ("patch_size", "timestep_features_dim"):
        if name in dit:
            _positive_int(dit[name], f"dit.{name}")
    if "memory_tokens" in dit:
        _positive_int(dit["memory_tokens"], "dit.memory_tokens", allow_zero=True)
    for name in ("dropout", "attention_dropout"):
        if name in dit:
            value = _positive_float(dit[name], f"dit.{name}", allow_zero=True)
            if value >= 1.0:
                raise ValueError(f"dit.{name} must be in [0, 1)")
    repa = dit.get("repa")
    if repa is not None:
        repa = _mapping(repa, "dit.repa")
        enabled = repa.get("enabled", False)
        _boolean(enabled, "dit.repa.enabled")
        if "layer" in repa:
            layer = _positive_int(repa["layer"], "dit.repa.layer", allow_zero=True)
            if layer >= depth:
                raise ValueError("dit.repa.layer must be smaller than dit.depth")
        if "loss_weight" in repa:
            weight = _positive_float(
                repa["loss_weight"], "dit.repa.loss_weight", allow_zero=True
            )
            if enabled and weight <= 0:
                raise ValueError(
                    "dit.repa.loss_weight must be positive when REPA is enabled"
                )
    return depth


def _validate_text_conditioning(conditioning: Mapping[str, Any]) -> None:
    text_encoder_value = conditioning.get("text_encoder")
    if text_encoder_value is None:
        text_encoder: Mapping[str, Any] = conditioning
    else:
        text_encoder = _mapping(text_encoder_value, "conditioning.text_encoder")
    model_name = str(text_encoder.get("model_name", "")).strip()
    if model_name != DEFAULT_TEXT_ENCODER:
        raise ValueError(
            "conditioning.text_encoder.model_name must select the smallest T5Gemma "
            f"checkpoint ({DEFAULT_TEXT_ENCODER})"
        )
    _positive_int(text_encoder.get("max_length"), "conditioning.text_encoder.max_length")
    _positive_int(text_encoder.get("batch_size", 1), "conditioning.text_encoder.batch_size")
    if "cache_dir" in text_encoder:
        _path(
            text_encoder["cache_dir"],
            "conditioning.text_encoder.cache_dir",
        )
    else:
        raise ValueError("conditioning.text_encoder.cache_dir must be set")
    frozen = text_encoder.get("frozen", text_encoder.get("freeze", True))
    if frozen is not True:
        raise ValueError("conditioning.text_encoder.frozen must be true")
    if "cross_attention" in conditioning and conditioning["cross_attention"] is not True:
        raise ValueError("conditioning.cross_attention must be true")
    if "duration_conditioning" in conditioning:
        if conditioning["duration_conditioning"] is not True:
            raise ValueError("conditioning.duration_conditioning must be true")


def _sampler_config(diffusion: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    value = diffusion.get("timestep_sampler")
    if isinstance(value, str):
        return value, diffusion
    sampler = _mapping(value, "diffusion.timestep_sampler")
    return str(sampler.get("type", "")), sampler


def _validate_optional_loss(
    diffusion: Mapping[str, Any], name: str, depth: int
) -> None:
    value = diffusion.get(name)
    if value is None:
        return
    config = _mapping(value, f"diffusion.{name}")
    enabled = config.get("enabled", False)
    _boolean(enabled, f"diffusion.{name}.enabled")
    weight = _positive_float(
        config.get("weight", 0.0), f"diffusion.{name}.weight", allow_zero=True
    )
    if enabled and weight == 0.0:
        raise ValueError(f"diffusion.{name}.weight must be positive when enabled")

    layer_values: list[Any] = []
    single_layer = _first(config, ("layer_index", "block_index"))
    if single_layer is not None:
        layer_values.append(single_layer)
    if "layer_indices" in config:
        values = config["layer_indices"]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"diffusion.{name}.layer_indices must be a non-empty list")
        layer_values.extend(values)
    for value in layer_values:
        index = _positive_int(
            value, f"diffusion.{name}.layer_index", allow_zero=True
        )
        if index >= depth:
            raise ValueError(f"diffusion.{name}.layer_index must be smaller than dit.depth")
    for field in ("projection_dim", "target_dim"):
        if field in config:
            _positive_int(config[field], f"diffusion.{name}.{field}")
    if "loss_type" in config and str(config["loss_type"]).lower() not in {
        "cosine",
        "mse",
        "smooth_l1",
    }:
        raise ValueError(f"diffusion.{name}.loss_type must be cosine, mse, or smooth_l1")


def _validate_diffusion(diffusion: Mapping[str, Any], depth: int) -> None:
    prediction_type = str(
        _first(diffusion, ("prediction_type", "objective"), "")
    ).lower().replace("-", "_")
    if prediction_type not in {"x", "x_prediction"}:
        raise ValueError("diffusion.prediction_type must be x_prediction")
    if "t_eps" in diffusion:
        _positive_float(diffusion["t_eps"], "diffusion.t_eps")

    sampler_type, sampler = _sampler_config(diffusion)
    sampler_type = sampler_type.lower().replace("-", "_")
    if sampler_type not in {"truncated_logit_normal", "trunc_logit_normal"}:
        raise ValueError(
            "diffusion.timestep_sampler.type must be truncated_logit_normal"
        )
    mean = _first(sampler, ("mean", "logit_mean"), 0.0)
    std = _first(sampler, ("std", "logit_std"), 1.0)
    truncation = _first(sampler, ("truncation", "min_t", "left_trunc"), 0.075)
    _close(mean, 0.0, "diffusion.timestep_sampler.mean")
    _close(std, 1.0, "diffusion.timestep_sampler.std")
    _close(truncation, 0.075, "diffusion.timestep_sampler.truncation")
    if sampler.get("rescale", True) is not True:
        raise ValueError("diffusion.timestep_sampler.rescale must be true")
    if sampler.get("flip", False) is not False:
        raise ValueError("diffusion.timestep_sampler.flip must be false")
    sampling = _mapping(diffusion.get("sampling"), "diffusion.sampling")
    _positive_int(sampling.get("steps"), "diffusion.sampling.steps")
    _positive_float(
        sampling.get("guidance_scale", 1.0),
        "diffusion.sampling.guidance_scale",
    )
    interval = sampling.get("guidance_interval", [0.0, 1.0])
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("diffusion.sampling.guidance_interval must have two values")
    start, end = (float(interval[0]), float(interval[1]))
    if not 0 <= start <= end <= 1:
        raise ValueError(
            "diffusion.sampling.guidance_interval must lie within [0, 1]"
        )


def _validate_optimizer(optimizer: Mapping[str, Any]) -> None:
    optimizer_type = str(optimizer.get("type", "")).lower().replace("+", "_")
    optimizer_type = optimizer_type.replace("-", "_")
    if optimizer_type not in {"muon_adamw", "hybrid", "hybrid_muon_adamw"}:
        raise ValueError("optimizer.type must be the Muon+AdamW hybrid optimizer")

    muon_value = optimizer.get("muon", {})
    adamw_value = optimizer.get("adamw", {})
    muon = _mapping(muon_value, "optimizer.muon")
    adamw = _mapping(adamw_value, "optimizer.adamw")
    _close(_first(optimizer, ("muon_lr",), muon.get("lr")), 1.0e-5, "optimizer.muon.lr")
    _close(
        _first(optimizer, ("muon_momentum",), muon.get("momentum")),
        0.95,
        "optimizer.muon.momentum",
    )
    _close(_first(optimizer, ("adamw_lr", "adam_lr"), adamw.get("lr")), 1.0e-6, "optimizer.adamw.lr")
    betas = _first(optimizer, ("adamw_betas", "adam_betas"), adamw.get("betas"))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optimizer.adamw.betas must contain two values")
    _close(betas[0], 0.9, "optimizer.adamw.betas[0]")
    _close(betas[1], 0.95, "optimizer.adamw.betas[1]")
    _close(
        _first(
            optimizer,
            ("adamw_weight_decay", "adam_weight_decay"),
            adamw.get("weight_decay"),
        ),
        0.01,
        "optimizer.adamw.weight_decay",
    )
    if "ns_steps" in muon:
        _positive_int(muon["ns_steps"], "optimizer.muon.ns_steps")
    if "nesterov" in muon:
        _boolean(muon["nesterov"], "optimizer.muon.nesterov")
    assignment = optimizer.get("muon_parameters", optimizer.get("parameter_assignment"))
    if assignment is not None and str(assignment).lower() not in {
        "qkv_ff",
        "qkv_and_ff",
        "attention_qkv_and_ff",
    }:
        raise ValueError("optimizer Muon parameters must be limited to attention QKV and FF")


def _validate_training(training: Mapping[str, Any]) -> None:
    scheduler_value = training.get("lr_scheduler", training.get("scheduler"))
    if isinstance(scheduler_value, Mapping):
        scheduler_type = scheduler_value.get("type")
    else:
        scheduler_type = scheduler_value
    if str(scheduler_type).lower().replace("-", "_") != "warmup_cosine":
        raise ValueError("training.lr_scheduler must be warmup_cosine")
    _path(training.get("output_dir"), "training.output_dir")
    for name in ("epochs", "batch_size", "grad_accumulation_steps", "max_steps"):
        if name in training:
            _positive_int(training[name], f"training.{name}")
    if "warmup_steps" in training:
        _positive_int(training["warmup_steps"], "training.warmup_steps", allow_zero=True)
    for name in (
        "validate_every_steps",
        "sample_every_steps",
        "log_every_steps",
        "loss_curve_every_steps",
    ):
        if name in training:
            _positive_int(training[name], f"training.{name}")
    sample_count = _first(
        training,
        ("num_validation_samples", "num_listen_samples", "validation_audio_samples"),
    )
    if sample_count is not None and int(sample_count) != 5:
        raise ValueError("training validation audio sample count must be 5")
    if "amp" in training:
        _boolean(training["amp"], "training.amp")
    if "listening" in training:
        listening = _mapping(training["listening"], "training.listening")
        for name in ("selection_seed", "noise_seed", "validation_seed"):
            if name in listening:
                _positive_int(listening[name], f"training.listening.{name}", allow_zero=True)


def _validate_evaluation(evaluation: Mapping[str, Any]) -> None:
    _path(evaluation.get("output_dir"), "evaluation.output_dir")
    for name in ("test_manifest", "test_manifest_path", "reference_dir"):
        if name in evaluation:
            _path(evaluation[name], f"evaluation.{name}")
    if "batch_size" in evaluation:
        _positive_int(evaluation["batch_size"], "evaluation.batch_size")
    if "sampling_steps" in evaluation:
        _positive_int(evaluation["sampling_steps"], "evaluation.sampling_steps")
    if "run_after_training" in evaluation:
        _boolean(evaluation["run_after_training"], "evaluation.run_after_training")
    gfad = evaluation.get("gfad")
    if gfad is not None:
        if isinstance(gfad, Mapping):
            enabled = gfad.get("enabled", True)
        else:
            enabled = gfad
        if not isinstance(enabled, bool):
            raise ValueError("evaluation.gfad.enabled must be true or false")
    metric = evaluation.get("metric", evaluation.get("metrics"))
    if metric is not None:
        metrics = [metric] if isinstance(metric, str) else list(metric)
        if not any(str(value).lower() in {"gfad", "g_fad"} for value in metrics):
            raise ValueError("evaluation metrics must include gFAD")


def validate_dit_config(config: Mapping[str, Any]) -> None:
    """Validate invariants required by the downstream DiT training design."""

    if not isinstance(config, Mapping):
        raise ValueError("DiT config must be a mapping")
    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        raise ValueError(f"Missing DiT config sections: {missing}")

    sections = {name: _mapping(config[name], name) for name in REQUIRED_SECTIONS}
    data = sections["data"]
    autoencoder = sections["autoencoder"]
    dit = sections["dit"]
    conditioning = sections["conditioning"]
    diffusion = sections["diffusion"]
    optimizer = sections["optimizer"]
    training = sections["training"]
    evaluation = sections["evaluation"]

    _validate_data(data)
    _validate_autoencoder(autoencoder, data)
    depth = _validate_dit(dit)
    _validate_text_conditioning(conditioning)
    _validate_diffusion(diffusion, depth)
    _validate_optimizer(optimizer)
    _validate_training(training)
    _validate_evaluation(evaluation)
    _validate_no_cfg(
        ("conditioning", conditioning),
        ("diffusion", diffusion),
        ("training", training),
    )


# Short aliases are convenient when importing this module directly, while the
# package exports retain the explicit DiT names to avoid confusion with the AE
# configuration loader.
load_config = load_dit_config
validate_config = validate_dit_config


__all__ = [
    "DEFAULT_DIT_DEPTH",
    "DEFAULT_TEXT_ENCODER",
    "REQUIRED_SECTIONS",
    "load_config",
    "load_dit_config",
    "validate_config",
    "validate_dit_config",
]
