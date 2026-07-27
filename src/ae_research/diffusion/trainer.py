from __future__ import annotations

import copy
import csv
import importlib
import json
import math
import os
import random
import re
import shutil
import uuid
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import yaml
from torch import nn
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .dit import AudioDiffusionTransformer
from .flow_matching import (
    XPredictionObjective,
    euler_sample,
    flow_interpolate,
    sample_truncated_logit_normal,
)
from .optim import build_muon_adamw


LOSS_NAMES = ("total", "x_prediction", "repa_internal_guidance")


def _reset_fresh_output_dir(output_dir: Path) -> None:
    """Remove DiT artifacts before starting a non-resumed training run."""
    for relative in (
        "dit_history.csv",
        "loss_curves.png",
        "resolved_dit_config.yaml",
        "tensorboard_dit",
        "checkpoints",
        "listening",
    ):
        path = output_dir / relative
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warmup_cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    """Return one LR multiplier shared by the Muon and AdamW groups.

    ``step`` is the zero-based optimizer-update index. The first warmup update
    therefore uses ``1 / warmup_steps`` of each group's own base learning rate.
    Applying this same multiplier to both groups preserves their LR ratio.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps + 1) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _prompt_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _prompt_from_record(record: Mapping[str, Any]) -> str:
    for key in ("text", "prompt", "tag_text", "condition_text", "tags"):
        prompt = _prompt_from_value(record.get(key))
        if prompt:
            return prompt
    return ""


def _dataset_prompt(dataset: Any, index: int) -> str:
    records = getattr(dataset, "records", None)
    if records is not None:
        record = records[index]
        if isinstance(record, Mapping):
            prompt = _prompt_from_record(record)
            if prompt:
                return prompt
    item = dataset[index]
    if not isinstance(item, Mapping):
        raise TypeError("Diffusion dataset items must be mappings")
    return _prompt_from_record(item)


def select_fixed_listening_indices(
    dataset: Any,
    *,
    count: int = 5,
    seed: int,
) -> list[int]:
    """Select deterministic indices whose text prompts are all different."""
    if count <= 0:
        raise ValueError("count must be positive")
    if len(dataset) < count:
        raise ValueError(
            f"Validation dataset has {len(dataset)} items; {count} are required "
            "for fixed listening samples"
        )
    candidates = list(range(len(dataset)))
    random.Random(seed).shuffle(candidates)
    indices: list[int] = []
    prompts: set[str] = set()
    for index in candidates:
        prompt = _dataset_prompt(dataset, index)
        if not prompt:
            continue
        canonical = " ".join(prompt.casefold().split())
        if canonical in prompts:
            continue
        prompts.add(canonical)
        indices.append(index)
        if len(indices) == count:
            return indices
    raise ValueError(
        f"Validation dataset does not contain {count} distinct non-empty text "
        "conditions. Rebuild tag caches with ae-prepare-dit-tags."
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_checkpoint_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: Mapping[str, Any],
    epoch: int,
    batch_in_epoch: int,
    epoch_complete: bool,
    global_step: int,
    best_val: float,
    best_step: int,
    listening_indices: Sequence[int],
    listening_seeds: Mapping[str, int],
    last_validation_step: int,
    optimizer_parameter_names: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Build the complete, self-contained DiT resume state."""
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": copy.deepcopy(dict(config)),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "epoch_complete": bool(epoch_complete),
        "global_step": int(global_step),
        "best_val": float(best_val),
        "best_step": int(best_step),
        "listening_indices": [int(index) for index in listening_indices],
        "listening_seeds": {
            str(name): int(seed) for name, seed in listening_seeds.items()
        },
        "last_validation_step": int(last_validation_step),
        "optimizer_parameter_names": {
            str(name): tuple(str(item) for item in values)
            for name, values in (optimizer_parameter_names or {}).items()
        },
        "rng_state": _rng_state(),
    }


def atomic_save_checkpoint(state: Mapping[str, Any], path: str | Path) -> None:
    """Durably replace ``path`` only after a readable checkpoint is written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    required = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "config",
        "epoch",
        "global_step",
        "listening_indices",
        "listening_seeds",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Checkpoint state is missing keys: {sorted(missing)}")
    try:
        torch.save(dict(state), temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        loaded = torch.load(temporary, map_location="cpu", weights_only=False)
        missing_after_save = required.difference(loaded)
        if missing_after_save:
            raise RuntimeError(
                "Checkpoint became incomplete while saving: "
                f"{sorted(missing_after_save)}"
            )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint save produced an empty file: {destination}")


class DiffusionHistoryWriter:
    FIELDS = [
        "split",
        "epoch",
        "step",
        "total",
        "x_prediction",
        "repa_internal_guidance",
        "muon_lr",
        "adamw_lr",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.FIELDS).writeheader()

    def append(
        self,
        *,
        split: str,
        epoch: int,
        step: int,
        metrics: Mapping[str, float],
    ) -> None:
        row: dict[str, Any] = {name: "" for name in self.FIELDS}
        row.update({"split": split, "epoch": int(epoch), "step": int(step)})
        for name in self.FIELDS[3:]:
            if name in metrics:
                row[name] = float(metrics[name])
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.FIELDS).writerow(row)


def plot_diffusion_history(history_path: str | Path, output_path: str | Path) -> None:
    """Plot train and validation losses from the persisted DiT history."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history_path = Path(history_path)
    if not history_path.is_file():
        return
    with history_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(LOSS_NAMES), figsize=(15, 4.5))
    for axis, metric in zip(axes, LOSS_NAMES):
        for split, style in (("train", "-"), ("val", "--")):
            selected = [
                (int(row["step"]), float(row[metric]))
                for row in rows
                if row["split"] == split and row.get(metric)
            ]
            if selected:
                steps, values = zip(*selected)
                axis.plot(steps, values, style, label=split, alpha=0.85)
        axis.set_title(metric)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    figure.tight_layout()
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        figure.savefig(temporary, format="png", dpi=160)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def _codec_kind(autoencoder_config: Mapping[str, Any]) -> str:
    candidates = (
        autoencoder_config.get("type"),
        autoencoder_config.get("model_type"),
        autoencoder_config.get("kind"),
        autoencoder_config.get("model", {}).get("type")
        if isinstance(autoencoder_config.get("model"), Mapping)
        else None,
    )
    value = " ".join(str(item).casefold() for item in candidates if item is not None)
    if "same" in value:
        return "same"
    if "semantic" in value or "mert" in value:
        return "semantic"
    return value.strip() or "unknown"


def _normalizer_apply(normalizer: Any, latent: torch.Tensor, *, inverse: bool) -> torch.Tensor:
    if normalizer is None:
        return latent
    method_names = ("denormalize", "decode", "inverse") if inverse else (
        "normalize",
        "encode",
    )
    for name in method_names:
        method = getattr(normalizer, name, None)
        if callable(method):
            return method(latent)
    if isinstance(normalizer, nn.Identity):
        return normalizer(latent)
    if callable(normalizer) and not inverse:
        return normalizer(latent)
    direction = "denormalize" if inverse else "normalize"
    raise TypeError(f"codec.normalizer does not provide a {direction} operation")


def _extract_encoded_latent(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for key in ("latent", "modulated_features", "semantic_features"):
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
    raise TypeError("codec.encode() must return a tensor or a mapping containing latent")


def _extract_decoded_audio(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for key in ("audio", "waveform", "reconstruction"):
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
    if isinstance(value, Sequence) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError("codec.decode() must return a waveform tensor")


def _validate_cached_conditions(batch: Mapping[str, Any]) -> None:
    missing = {"text_embedding", "text_mask"}.difference(batch)
    if missing:
        raise FileNotFoundError(
            "Diffusion text-condition cache is missing fields "
            f"{sorted(missing)}. Run ae-prepare-dit-tags before ae-train-dit; "
            "the training command never downloads or runs T5Gemma."
        )
    if not isinstance(batch["text_embedding"], torch.Tensor):
        raise TypeError("Cached text_embedding must be a tensor")
    if not isinstance(batch["text_mask"], torch.Tensor):
        raise TypeError("Cached text_mask must be a tensor")


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _seed_loader_for_epoch(loader: Any, seed: int) -> None:
    """Give a shuffled loader a repeatable epoch permutation for mid-epoch resume."""
    generator = torch.Generator().manual_seed(int(seed))
    if hasattr(loader, "generator"):
        loader.generator = generator
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "generator"):
        sampler.generator = generator


def _optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        algorithm = str(group.get("algorithm", f"group_{index}")).casefold()
        result[f"{algorithm}_lr"] = float(group["lr"])
    if "muon_lr" not in result and optimizer.param_groups:
        result["muon_lr"] = float(optimizer.param_groups[0]["lr"])
    if "adamw_lr" not in result and len(optimizer.param_groups) > 1:
        result["adamw_lr"] = float(optimizer.param_groups[1]["lr"])
    return result


class DiTTrainer:
    """Train the downstream text-conditioned latent DiT without touching the AE."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: str | torch.device | None = None,
        codec: Any | None = None,
        model: AudioDiffusionTransformer | None = None,
        train_loader: Any | None = None,
        val_loader: Any | None = None,
    ) -> None:
        self.config: dict[str, Any] = copy.deepcopy(dict(config))
        self.seed = int(self.config.get("seed", 42))
        _seed_everything(self.seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.data_config = self.config["data"]
        self.autoencoder_config = self.config["autoencoder"]
        self.dit_config = self.config["dit"]
        self.train_config = self.config["training"]
        self.optimizer_config = self.config.get(
            "optimizer", self.train_config.get("optimizer")
        )
        if not isinstance(self.optimizer_config, Mapping):
            raise ValueError("DiT config must contain an optimizer mapping")

        self.codec_kind = _codec_kind(self.autoencoder_config)
        repa_config = self.dit_config.setdefault("repa", {})
        if not isinstance(repa_config, dict):
            raise ValueError("dit.repa must be a mapping")
        if "enabled" not in repa_config:
            repa_config["enabled"] = self.codec_kind == "semantic"
        self.repa_enabled = bool(repa_config["enabled"])
        if self.codec_kind == "same" and self.repa_enabled:
            raise ValueError(
                "RAEv2 self-REPA/internal guidance requires a semantic clean latent; "
                "disable dit.repa.enabled for SAME."
            )

        self.output_dir = Path(self.train_config["output_dir"])
        resume_from = self.train_config.get("resume_from")
        if not resume_from:
            _reset_fresh_output_dir(self.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "listening"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self._write_resolved_config()

        self.codec = codec if codec is not None else self._build_codec()
        if isinstance(self.codec, nn.Module):
            self.codec = self.codec.to(self.device)
        self._freeze_codec()
        self.latent_dim = int(self.codec.latent_dim)
        model_dim = int(self.dit_config["model_dim"])
        if self.codec_kind == "semantic" and model_dim < self.latent_dim:
            warnings.warn(
                "Semantic DiT model_dim is smaller than codec.latent_dim "
                f"({model_dim} < {self.latent_dim}). RAE reports an irreducible "
                "bottleneck in this regime; treat this configuration as experimental.",
                stacklevel=2,
            )
        self.model = model or AudioDiffusionTransformer.from_config(
            self.dit_config, latent_dim=self.latent_dim
        )
        self.model = self.model.to(self.device)

        batch_size = int(self.train_config["batch_size"])
        self.train_loader = train_loader or self._build_loader(
            split="train", batch_size=batch_size, shuffle=True
        )
        self.val_loader = val_loader or self._build_loader(
            split="val", batch_size=batch_size, shuffle=False
        )
        if len(self.train_loader) == 0:
            raise RuntimeError("Diffusion training loader produced no batches")
        if len(self.val_loader) == 0:
            raise RuntimeError("Diffusion validation loader produced no batches")

        diffusion_config = self.config["diffusion"]
        timestep_config = diffusion_config.get("timestep_sampler", {})
        if not isinstance(timestep_config, Mapping):
            raise ValueError("diffusion.timestep_sampler must be a mapping")
        self.t_eps = float(diffusion_config.get("t_eps", 0.05))
        self.timestep_minimum = float(
            timestep_config.get(
                "truncation", timestep_config.get("minimum", 0.075)
            )
        )
        self.rescale_timesteps = bool(timestep_config.get("rescale", True))
        base_weight = float(
            repa_config.get(
                "loss_weight", diffusion_config.get("base_loss_weight", 1.0)
            )
        )
        self.objective = XPredictionObjective(
            t_eps=self.t_eps, base_loss_weight=base_weight
        ).to(self.device)

        self.optimizer, self.optimizer_parameter_names = build_muon_adamw(
            self.model, dict(self.optimizer_config)
        )
        accumulation = int(self.train_config.get("grad_accumulation_steps", 1))
        if accumulation <= 0:
            raise ValueError("training.grad_accumulation_steps must be positive")
        updates_per_epoch = math.ceil(len(self.train_loader) / accumulation)
        self.total_optimizer_steps = updates_per_epoch * int(
            self.train_config["epochs"]
        )
        scheduler_config = self.train_config.get(
            "scheduler", self.config.get("scheduler", {})
        )
        warmup_steps = int(
            scheduler_config.get(
                "warmup_steps", self.train_config.get("warmup_steps", 0)
            )
        )
        min_ratio = self._scheduler_min_ratio(scheduler_config)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine_multiplier(
                step,
                total_steps=self.total_optimizer_steps,
                warmup_steps=warmup_steps,
                min_ratio=min_ratio,
            ),
        )

        precision = str(
            self.train_config.get(
                "mixed_precision",
                "fp16" if self.train_config.get("amp", True) else "none",
            )
        ).casefold()
        self.amp_enabled = self.device.type == "cuda" and precision in {
            "fp16",
            "float16",
            "bf16",
            "bfloat16",
        }
        self.amp_dtype = (
            torch.bfloat16 if precision in {"bf16", "bfloat16"} else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled and self.amp_dtype == torch.float16,
        )

        self.writer = SummaryWriter(self.output_dir / "tensorboard_dit")
        self.history = DiffusionHistoryWriter(self.output_dir / "dit_history.csv")
        listening_config = self.train_config.get("listening", {})
        self.listening_seeds = {
            "selection": int(listening_config.get("selection_seed", self.seed + 30_000)),
            "noise": int(listening_config.get("noise_seed", self.seed + 20_000)),
            "validation": int(
                listening_config.get("validation_seed", self.seed + 10_000)
            ),
        }
        self.listening_indices = select_fixed_listening_indices(
            self.val_loader.dataset,
            count=5,
            seed=self.listening_seeds["selection"],
        )
        self.global_step = 0
        self.start_epoch = 1
        self.resume_batch_in_epoch = 0
        self.best_val = float("inf")
        self.best_step = 0
        self.last_validation_step = -1
        self.current_epoch = 0
        self.current_batch_in_epoch = 0
        if resume_from:
            self.load_checkpoint(resume_from)

    def _write_resolved_config(self) -> None:
        path = self.output_dir / "resolved_dit_config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    self.config,
                    handle,
                    sort_keys=False,
                    allow_unicode=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _build_codec(self) -> Any:
        module = importlib.import_module("ae_research.diffusion.codec")
        factory = getattr(module, "load_frozen_codec")
        return factory(
            self.autoencoder_config,
            expected_sample_rate=int(self.data_config["sample_rate"]),
            expected_channels=int(self.data_config["channels"]),
            map_location=self.device,
        )

    def _build_loader(self, *, split: str, batch_size: int, shuffle: bool) -> Any:
        module = importlib.import_module("ae_research.diffusion.data")
        factory = getattr(module, "create_dit_dataloader")
        manifest_dir = Path(
            self.data_config.get("dit_manifest_dir")
            or self.data_config.get("text_manifest_dir")
            or self.data_config["manifest_dir"]
        )
        text_config = self.config["conditioning"]["text_encoder"]
        embedding_dir = (
            text_config.get("cache_dir")
            or self.data_config.get("text_embedding_dir")
        )
        if not embedding_dir:
            raise ValueError(
                "Set conditioning.text_encoder.cache_dir or "
                "data.text_embedding_dir"
            )
        return factory(
            manifest_dir / f"{split}.jsonl",
            self.data_config,
            text_embedding_dir=embedding_dir,
            batch_size=batch_size,
            split=split,
            expected_text_model=text_config.get("model_name"),
            expected_text_dim=self.dit_config.get("text_dim"),
            expected_text_max_length=text_config.get("max_length"),
            shuffle=shuffle,
        )

    def _freeze_codec(self) -> None:
        candidates = [self.codec]
        for name in ("model", "autoencoder", "encoder", "decoder"):
            value = getattr(self.codec, name, None)
            if value is not None:
                candidates.append(value)
        for value in candidates:
            if isinstance(value, nn.Module):
                value.requires_grad_(False)
                value.eval()

    def _scheduler_min_ratio(self, scheduler_config: Mapping[str, Any]) -> float:
        for key in ("min_lr_ratio", "min_ratio"):
            if key in scheduler_config:
                return float(scheduler_config[key])
            if key in self.train_config:
                return float(self.train_config[key])
        if "min_lr" in scheduler_config or "min_lr" in self.train_config:
            minimum = float(
                scheduler_config.get("min_lr", self.train_config.get("min_lr"))
            )
            base = max(float(group["lr"]) for group in self.optimizer.param_groups)
            return minimum / base
        return 0.05

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    @torch.no_grad()
    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        self._freeze_codec()
        value = _extract_encoded_latent(self.codec.encode(audio))
        if value.ndim != 3:
            raise ValueError(
                f"codec.encode() must produce [batch, channels, frames], got {value.shape}"
            )
        if value.shape[1] != self.latent_dim:
            raise ValueError(
                f"codec latent has {value.shape[1]} channels, expected {self.latent_dim}"
            )
        if not torch.isfinite(value).all():
            raise FloatingPointError("codec produced non-finite normalized latents")
        return value.detach().float()

    @torch.no_grad()
    def _decode(self, normalized_latent: torch.Tensor, target_samples: int) -> torch.Tensor:
        decoded = self.codec.decode(normalized_latent, int(target_samples))
        audio = _extract_decoded_audio(decoded).float()
        if audio.ndim != 3:
            raise ValueError("codec.decode() must produce [batch, channels, samples]")
        return audio

    def _conditions(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_cached_conditions(batch)
        embedding = batch["text_embedding"].to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        mask = batch["text_mask"].to(
            self.device, dtype=torch.bool, non_blocking=True
        )
        batch_size = embedding.shape[0]
        duration_value = batch.get("duration")
        if isinstance(duration_value, torch.Tensor):
            duration = duration_value.to(
                self.device, dtype=torch.float32, non_blocking=True
            ).reshape(batch_size)
        elif duration_value is not None:
            duration = torch.as_tensor(
                duration_value, device=self.device, dtype=torch.float32
            ).reshape(batch_size)
        else:
            duration = torch.full(
                (batch_size,),
                float(self.data_config["duration_seconds"]),
                device=self.device,
            )
        return embedding, mask, duration

    def _sample_timestep(
        self, batch_size: int, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return sample_truncated_logit_normal(
            batch_size,
            device=self.device,
            minimum=self.timestep_minimum,
            rescale=self.rescale_timesteps,
            generator=generator,
        )

    def _forward_batch(
        self,
        batch: Mapping[str, Any],
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        audio = batch["audio"].to(self.device, dtype=torch.float32, non_blocking=True)
        clean = self._encode(audio)
        embedding, mask, duration = self._conditions(batch)
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        timestep = self._sample_timestep(clean.shape[0], generator=generator)
        noisy = flow_interpolate(clean, noise, timestep)
        with self._autocast():
            outputs = self.model(
                noisy,
                timestep,
                duration=duration,
                text_embedding=embedding,
                text_mask=mask,
            )
            losses = self.objective(
                outputs,
                noisy_latent=noisy,
                clean_latent=clean,
                timestep=timestep,
            )
        if "repa_internal_guidance" not in losses:
            losses["repa_internal_guidance"] = losses["total"].detach().new_zeros(())
        for name in LOSS_NAMES:
            if not torch.isfinite(losses[name]).all():
                raise FloatingPointError(f"Non-finite {name} loss")
        return losses, int(clean.shape[0])

    def _write_metrics(
        self,
        *,
        split: str,
        epoch: int,
        metrics: Mapping[str, float],
    ) -> None:
        values = {**metrics, **_optimizer_learning_rates(self.optimizer)}
        for name in (*LOSS_NAMES, "muon_lr", "adamw_lr"):
            if name in values:
                self.writer.add_scalar(f"{split}/{name}", values[name], self.global_step)
        self.history.append(
            split=split,
            epoch=epoch,
            step=self.global_step,
            metrics=values,
        )
        print(
            f"[{split}] epoch={epoch} step={self.global_step} "
            + " ".join(
                f"{name}={float(values[name]):.6g}"
                for name in (*LOSS_NAMES, "muon_lr", "adamw_lr")
                if name in values
            )
        )

    def _update_loss_curve(self) -> None:
        plot_diffusion_history(
            self.output_dir / "dit_history.csv",
            self.output_dir / "loss_curves.png",
        )

    def _checkpoint_state(self, *, epoch_complete: bool) -> dict[str, Any]:
        return build_checkpoint_state(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            epoch=self.current_epoch,
            batch_in_epoch=self.current_batch_in_epoch,
            epoch_complete=epoch_complete,
            global_step=self.global_step,
            best_val=self.best_val,
            best_step=self.best_step,
            listening_indices=self.listening_indices,
            listening_seeds=self.listening_seeds,
            last_validation_step=self.last_validation_step,
            optimizer_parameter_names=self.optimizer_parameter_names,
        )

    def save_checkpoint(self, path: str | Path, *, epoch_complete: bool) -> None:
        atomic_save_checkpoint(
            self._checkpoint_state(epoch_complete=epoch_complete), path
        )

    def save_interrupted_checkpoints(self) -> None:
        """Save the latest resumable state without replacing a validated best."""
        state = self._checkpoint_state(epoch_complete=False)
        atomic_save_checkpoint(state, self.checkpoint_dir / "last.pt")
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.is_file():
            atomic_save_checkpoint(state, best_path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model", "optimizer", "scheduler", "scaler", "config"}
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(f"Resume checkpoint is missing keys: {sorted(missing)}")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", 0))
        epoch = int(checkpoint.get("epoch", 1))
        if bool(checkpoint.get("epoch_complete", False)):
            self.start_epoch = epoch + 1
            self.resume_batch_in_epoch = 0
        else:
            self.start_epoch = epoch
            self.resume_batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))
        self.best_val = float(checkpoint.get("best_val", float("inf")))
        self.best_step = int(checkpoint.get("best_step", 0))
        self.last_validation_step = int(checkpoint.get("last_validation_step", -1))
        if "listening_indices" in checkpoint:
            self.listening_indices = [
                int(index) for index in checkpoint["listening_indices"]
            ]
        if "listening_seeds" in checkpoint:
            self.listening_seeds = {
                str(name): int(seed)
                for name, seed in checkpoint["listening_seeds"].items()
            }
        _restore_rng_state(checkpoint.get("rng_state"))

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        sums: defaultdict[str, float] = defaultdict(float)
        sample_count = 0
        generator = _make_generator(
            self.device, self.listening_seeds["validation"]
        )
        maximum = self.train_config.get("max_validation_batches")
        for batch_index, batch in enumerate(self.val_loader, start=1):
            if maximum is not None and batch_index > int(maximum):
                break
            losses, batch_size = self._forward_batch(batch, generator=generator)
            for name in LOSS_NAMES:
                sums[name] += float(losses[name].detach()) * batch_size
            sample_count += batch_size
        if sample_count == 0:
            raise RuntimeError("Validation loader produced no samples")
        metrics = {name: sums[name] / sample_count for name in LOSS_NAMES}
        self.last_validation_step = self.global_step
        self._write_metrics(split="val", epoch=self.current_epoch, metrics=metrics)
        self.save_listening_samples()
        if metrics["total"] < self.best_val:
            self.best_val = metrics["total"]
            self.best_step = self.global_step
            self.save_checkpoint(
                self.checkpoint_dir / "best.pt", epoch_complete=False
            )
        if was_training:
            self.model.train()
        return metrics

    def _listening_batch(self) -> Mapping[str, Any]:
        items = [
            self.val_loader.dataset[index] for index in self.listening_indices
        ]
        collate_fn = getattr(self.val_loader, "collate_fn", None)
        batch = collate_fn(items) if callable(collate_fn) else default_collate(items)
        if not isinstance(batch, Mapping):
            raise TypeError("Diffusion validation dataset must collate to a mapping")
        _validate_cached_conditions(batch)
        return batch

    @torch.no_grad()
    def save_listening_samples(self) -> Path:
        was_training = self.model.training
        self.model.eval()
        batch = self._listening_batch()
        audio = batch["audio"].to(self.device, dtype=torch.float32)
        reference_latent = self._encode(audio)
        embedding, mask, duration = self._conditions(batch)
        sampling = self.config["diffusion"].get("sampling", {})
        interval = sampling.get("guidance_interval", (0.0, 1.0))
        if isinstance(interval, Mapping):
            interval = (interval.get("min", 0.0), interval.get("max", 1.0))
        generator = _make_generator(self.device, self.listening_seeds["noise"])
        with self._autocast():
            generated_latent = euler_sample(
                self.model,
                shape=reference_latent.shape,
                duration=duration,
                text_embedding=embedding,
                text_mask=mask,
                steps=int(sampling.get("steps", 50)),
                t_eps=float(sampling.get("t_eps", self.t_eps)),
                guidance_scale=float(sampling.get("guidance_scale", 1.0)),
                guidance_interval=(float(interval[0]), float(interval[1])),
                generator=generator,
            )
        target_samples = audio.shape[-1]
        generated = self._decode(generated_latent.float(), target_samples)
        output_dir = self.sample_dir / f"step_{self.global_step:08d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_rate = int(self.data_config["sample_rate"])
        manifest: dict[str, Any] = {
            "epoch": self.current_epoch,
            "step": self.global_step,
            "selection_seed": self.listening_seeds["selection"],
            "noise_seed": self.listening_seeds["noise"],
            "samples": [],
        }
        for slot, dataset_index in enumerate(self.listening_indices):
            track_values = batch.get("track_id", [str(dataset_index)] * 5)
            track_id = str(track_values[slot])
            prompt = _prompt_from_record(
                {
                    key: batch[key][slot]
                    for key in ("text", "prompt", "tag_text", "condition_text", "tags")
                    if key in batch
                }
            )
            if not prompt:
                prompt = _dataset_prompt(self.val_loader.dataset, dataset_index)
            safe_track = re.sub(r"[^A-Za-z0-9_.-]+", "_", track_id)[:80]
            generated_filename = f"{slot:02d}_{safe_track}_generated.wav"
            reference_filename = f"{slot:02d}_{safe_track}_reference.wav"
            torchaudio.save(
                output_dir / generated_filename,
                generated[slot].detach().cpu().clamp(-1.0, 1.0),
                sample_rate,
            )
            torchaudio.save(
                output_dir / reference_filename,
                audio[slot].detach().cpu().clamp(-1.0, 1.0),
                sample_rate,
            )
            manifest["samples"].append(
                {
                    "slot": slot,
                    "dataset_index": int(dataset_index),
                    "track_id": track_id,
                    "text": prompt,
                    "audio": generated_filename,
                    "generated_audio": generated_filename,
                    "reference_audio": reference_filename,
                }
            )
        _atomic_json_dump(manifest, output_dir / "prompts.json")
        if was_training:
            self.model.train()
        return output_dir

    def _run_post_training_evaluation(self) -> None:
        evaluation = self.config.get("evaluation", {})
        if not bool(evaluation.get("run_after_training", False)):
            return
        try:
            module = importlib.import_module("ae_research.diffusion.evaluation")
            evaluate_dit = getattr(module, "evaluate_dit")
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "evaluation.run_after_training is true, but the DiT evaluator "
                "could not be imported."
            ) from error
        checkpoint = self.checkpoint_dir / "best.pt"
        evaluate_dit(
            config=self.config,
            checkpoint_path=checkpoint,
            device=str(self.device),
        )

    def train(self) -> dict[str, float | int]:
        accumulation = int(self.train_config.get("grad_accumulation_steps", 1))
        log_every = int(self.train_config.get("log_every_steps", 20))
        loss_curve_every = int(
            self.train_config.get("loss_curve_every_steps", log_every)
        )
        validate_every = int(self.train_config.get("validate_every_steps", 200))
        checkpoint_every = int(
            self.train_config.get("checkpoint_every_steps", 2_000)
        )
        grad_clip = float(self.train_config.get("grad_clip_norm", 1.0))
        epochs = int(self.train_config["epochs"])
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        interrupted = False

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                self.current_epoch = epoch
                self.current_batch_in_epoch = 0
                _seed_loader_for_epoch(self.train_loader, self.seed + epoch)
                rolling: defaultdict[str, float] = defaultdict(float)
                rolling_samples = 0
                progress = tqdm(self.train_loader, desc=f"dit epoch {epoch}")
                for batch_index, batch in enumerate(progress, start=1):
                    if epoch == self.start_epoch and batch_index <= self.resume_batch_in_epoch:
                        continue
                    losses, batch_size = self._forward_batch(batch)
                    group_start = ((batch_index - 1) // accumulation) * accumulation + 1
                    group_size = min(
                        accumulation, len(self.train_loader) - group_start + 1
                    )
                    self.scaler.scale(losses["total"] / group_size).backward()
                    for name in LOSS_NAMES:
                        rolling[name] += float(losses[name].detach()) * batch_size
                    rolling_samples += batch_size

                    update = batch_index % accumulation == 0 or batch_index == len(
                        self.train_loader
                    )
                    if not update:
                        continue
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        grad_clip,
                        error_if_nonfinite=True,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self.scheduler.step()
                    self.current_batch_in_epoch = batch_index
                    progress.set_postfix(total=f"{float(losses['total'].detach()):.4f}")

                    log_due = log_every > 0 and self.global_step % log_every == 0
                    curve_due = (
                        loss_curve_every > 0
                        and self.global_step % loss_curve_every == 0
                    )
                    if log_due or curve_due:
                        metrics = {
                            name: rolling[name] / max(rolling_samples, 1)
                            for name in LOSS_NAMES
                        }
                        self._write_metrics(
                            split="train", epoch=epoch, metrics=metrics
                        )
                        if curve_due:
                            self._update_loss_curve()
                        rolling.clear()
                        rolling_samples = 0
                    if validate_every > 0 and self.global_step % validate_every == 0:
                        self.validate()
                    if (
                        checkpoint_every > 0
                        and self.global_step % checkpoint_every == 0
                    ):
                        self.save_checkpoint(
                            self.checkpoint_dir
                            / f"step_{self.global_step:08d}.pt",
                            epoch_complete=False,
                        )

                self.resume_batch_in_epoch = 0
                if rolling_samples:
                    metrics = {
                        name: rolling[name] / rolling_samples for name in LOSS_NAMES
                    }
                    self._write_metrics(split="train", epoch=epoch, metrics=metrics)
                self._update_loss_curve()
                self.current_batch_in_epoch = len(self.train_loader)
                self.save_checkpoint(
                    self.checkpoint_dir / "last.pt", epoch_complete=True
                )
        except KeyboardInterrupt:
            interrupted = True
            self.save_interrupted_checkpoints()
            raise
        finally:
            if interrupted:
                self.writer.flush()

        if self.global_step <= 0:
            raise RuntimeError("DiT training completed without an optimizer update")
        if self.last_validation_step != self.global_step:
            self.validate()
        self._update_loss_curve()
        self.save_checkpoint(self.checkpoint_dir / "last.pt", epoch_complete=True)
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.is_file():
            self.save_checkpoint(best_path, epoch_complete=True)
        self.writer.close()
        self._run_post_training_evaluation()
        result: dict[str, float | int] = {
            "best_val_total": self.best_val,
            "best_step": self.best_step,
            "global_step": self.global_step,
        }
        print(json.dumps(result, indent=2))
        return result


Trainer = DiTTrainer


__all__ = [
    "DiTTrainer",
    "DiffusionHistoryWriter",
    "Trainer",
    "atomic_save_checkpoint",
    "build_checkpoint_state",
    "plot_diffusion_history",
    "select_fixed_listening_indices",
]
