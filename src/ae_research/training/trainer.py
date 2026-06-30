from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ae_research.data.dataset import create_dataloader
from ae_research.losses import SameObjective
from ae_research.metrics import si_sdr
from ae_research.models import SemanticAudioAutoencoder

from .history import HistoryWriter, plot_history


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_metrics(sums: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in sums.items()}


def _warmup_cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    """Return the LR multiplier used by the zero-indexed optimizer update."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps + 1) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _nonfinite_details(
    named_tensors: Iterable[tuple[str, torch.Tensor | None]],
) -> list[str]:
    details = []
    for name, value in named_tensors:
        if value is None or torch.isfinite(value).all():
            continue
        nan_count = int(torch.isnan(value).sum().item())
        inf_count = int(torch.isinf(value).sum().item())
        details.append(
            f"{name}: shape={tuple(value.shape)}, "
            f"nan={nan_count}, inf={inf_count}"
        )
    return details


class Trainer:
    def __init__(self, config: dict[str, Any], device: str | None = None) -> None:
        self.config = config
        _seed_everything(int(config.get("seed", 42)))
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.train_config = config["training"]
        self.output_dir = Path(self.train_config["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "resolved_config.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

        data_config = config["data"]
        manifest_dir = Path(data_config["manifest_dir"])
        batch_size = int(self.train_config["batch_size"])
        self.train_loader = create_dataloader(
            manifest_dir / "train.jsonl",
            data_config,
            batch_size=batch_size,
            split="train",
        )
        self.val_loader = create_dataloader(
            manifest_dir / "val.jsonl",
            data_config,
            batch_size=batch_size,
            split="val",
            shuffle=False,
        )
        self.model = SemanticAudioAutoencoder(
            config["model"],
            audio_channels=int(data_config["channels"]),
            data_sample_rate=int(data_config["sample_rate"]),
        ).to(self.device)
        self.objective = SameObjective(
            config["loss"], sample_rate=int(data_config["sample_rate"])
        ).to(self.device)
        peak_lr = float(self.train_config["peak_lr"])
        self.optimizer = torch.optim.AdamW(
            self.model.decoder.parameters(),
            lr=peak_lr,
            weight_decay=float(self.train_config["weight_decay"]),
        )
        accumulation = int(self.train_config["grad_accumulation_steps"])
        updates_per_epoch = math.ceil(len(self.train_loader) / accumulation)
        self.total_optimizer_steps = (
            updates_per_epoch * int(self.train_config["epochs"])
        )
        warmup_steps = int(self.train_config["warmup_steps"])
        min_ratio = float(self.train_config["min_lr"]) / peak_lr
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine_multiplier(
                step,
                total_steps=self.total_optimizer_steps,
                warmup_steps=warmup_steps,
                min_ratio=min_ratio,
            ),
        )
        self.amp_enabled = bool(self.train_config["amp"]) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        self.history = HistoryWriter(self.output_dir / "history.csv")
        self.global_step = 0
        self.start_epoch = 1
        self.best_val = float("inf")
        resume = self.train_config.get("resume_from")
        if resume:
            self.load_checkpoint(resume)

    def _forward(self, audio: torch.Tensor) -> tuple[dict, dict]:
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            outputs = self.model(audio)
        # STFT and statistics are deliberately evaluated in float32.
        losses = self.objective(
            outputs["reconstruction"], audio, outputs["latent"]
        )
        return outputs, losses

    def _abort_nonfinite(
        self,
        *,
        kind: str,
        epoch: int,
        batch_index: int,
        track_ids: list[str],
        details: list[str],
    ) -> None:
        message = (
            f"Non-finite {kind} detected before optimizer step "
            f"{self.global_step + 1} (epoch={epoch}, batch={batch_index}, "
            f"tracks={track_ids}).\n" + "\n".join(details)
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.writer.flush()
        (self.output_dir / "nonfinite_error.txt").write_text(
            message + "\n", encoding="utf-8"
        )
        raise FloatingPointError(message)

    def _check_losses(
        self,
        losses: dict[str, torch.Tensor],
        *,
        epoch: int,
        batch_index: int,
        track_ids: list[str],
        split: str,
    ) -> None:
        details = _nonfinite_details(losses.items())
        if details:
            self._abort_nonfinite(
                kind=f"{split} loss",
                epoch=epoch,
                batch_index=batch_index,
                track_ids=track_ids,
                details=details,
            )

    def _check_gradients(
        self,
        *,
        epoch: int,
        batch_index: int,
        track_ids: list[str],
    ) -> None:
        try:
            torch.nn.utils.clip_grad_norm_(
                self.model.decoder.parameters(),
                max_norm=float("inf"),
                error_if_nonfinite=True,
            )
        except RuntimeError:
            details = _nonfinite_details(
                (name, parameter.grad)
                for name, parameter in self.model.decoder.named_parameters()
            )
            self._abort_nonfinite(
                kind="gradient",
                epoch=epoch,
                batch_index=batch_index,
                track_ids=track_ids,
                details=details or ["Non-finite total gradient norm"],
            )

    def _check_parameters(
        self,
        *,
        epoch: int,
        batch_index: int,
        track_ids: list[str],
    ) -> None:
        details = _nonfinite_details(self.model.decoder.named_parameters())
        if details:
            self._abort_nonfinite(
                kind="decoder parameter",
                epoch=epoch,
                batch_index=batch_index,
                track_ids=track_ids,
                details=details,
            )

    def _write_metrics(
        self, split: str, epoch: int, metrics: dict[str, float]
    ) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(f"{split}/{name}", value, self.global_step)
        self.history.append(split, epoch, self.global_step, metrics)
        readable = " ".join(
            f"{name}={value:.4f}"
            for name, value in metrics.items()
            if name in {"total", "mrstft", "kl", "si_sdr"}
        )
        print(f"[{split}] epoch={epoch} step={self.global_step} {readable}")

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        sums: defaultdict[str, float] = defaultdict(float)
        count = 0
        max_batches = self.train_config.get("max_validation_batches")
        for batch_index, batch in enumerate(self.val_loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            audio = batch["audio"].to(self.device, non_blocking=True)
            outputs, losses = self._forward(audio)
            self._check_losses(
                losses,
                epoch=epoch,
                batch_index=batch_index + 1,
                track_ids=[str(value) for value in batch["track_id"]],
                split="validation",
            )
            values = {
                **{key: float(value.detach()) for key, value in losses.items()},
                "si_sdr": float(si_sdr(outputs["reconstruction"], audio)),
            }
            for key, value in values.items():
                sums[key] += value
            count += 1
        if count == 0:
            raise RuntimeError("Validation loader produced no batches")
        metrics = _mean_metrics(sums, count)
        self._write_metrics("val", epoch, metrics)
        if metrics["total"] < self.best_val:
            self.best_val = metrics["total"]
            self.save_checkpoint(self.checkpoint_dir / "best.pt", epoch)
        self.model.train()
        return metrics

    @torch.no_grad()
    def save_listening_samples(self, epoch: int) -> None:
        self.model.eval()
        batch = next(iter(self.val_loader))
        audio = batch["audio"].to(self.device)
        outputs = self.model(audio)
        limit = min(int(self.train_config["num_listen_samples"]), audio.shape[0])
        epoch_dir = self.sample_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        sample_rate = int(self.config["data"]["sample_rate"])
        for index in range(limit):
            track_id = batch["track_id"][index]
            torchaudio.save(
                epoch_dir / f"{index:02d}_{track_id}_reference.wav",
                audio[index].detach().cpu().clamp(-1, 1),
                sample_rate,
            )
            torchaudio.save(
                epoch_dir / f"{index:02d}_{track_id}_reconstruction.wav",
                outputs["reconstruction"][index].detach().cpu().clamp(-1, 1),
                sample_rate,
            )
        self.model.train()

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        state = {
            "decoder": self.model.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val": self.best_val,
            "config": self.config,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        torch.save(state, temporary)
        temporary.replace(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Checkpoint save failed or produced an empty file: {path}")

    def _assert_resume_checkpoints(self) -> None:
        missing = [
            str(path)
            for path in (
                self.checkpoint_dir / "best.pt",
                self.checkpoint_dir / "last.pt",
            )
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise RuntimeError(
                "Training did not produce the required resume checkpoints: "
                + ", ".join(missing)
            )

    def _save_resume_checkpoints(self, epoch: int) -> None:
        self.save_checkpoint(self.checkpoint_dir / "last.pt", epoch)
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.is_file() or best_path.stat().st_size == 0:
            self.save_checkpoint(best_path, epoch)
        self._assert_resume_checkpoints()

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.decoder.load_state_dict(checkpoint["decoder"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", 0))
        if "scheduler" not in checkpoint and self.global_step > 0:
            self.scheduler.last_epoch = self.global_step
            learning_rates = [
                base_lr * lr_lambda(self.global_step)
                for base_lr, lr_lambda in zip(
                    self.scheduler.base_lrs,
                    self.scheduler.lr_lambdas,
                )
            ]
            for group, learning_rate in zip(
                self.optimizer.param_groups, learning_rates
            ):
                group["lr"] = learning_rate
            self.scheduler._last_lr = learning_rates
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_val = float(checkpoint.get("best_val", float("inf")))

    def train(self) -> None:
        accumulation = int(self.train_config["grad_accumulation_steps"])
        log_every = int(self.train_config["log_every_steps"])
        validate_every = int(self.train_config["validate_every_steps"])
        checkpoint_every = int(self.train_config["checkpoint_every_steps"])
        grad_clip = float(self.train_config["grad_clip_norm"])
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        current_epoch = self.start_epoch - 1

        try:
            for epoch in range(self.start_epoch, int(self.train_config["epochs"]) + 1):
                current_epoch = epoch
                rolling: defaultdict[str, float] = defaultdict(float)
                rolling_count = 0
                progress = tqdm(self.train_loader, desc=f"epoch {epoch}")
                for batch_index, batch in enumerate(progress, start=1):
                    audio = batch["audio"].to(self.device, non_blocking=True)
                    outputs, losses = self._forward(audio)
                    track_ids = [str(value) for value in batch["track_id"]]
                    self._check_losses(
                        losses,
                        epoch=epoch,
                        batch_index=batch_index,
                        track_ids=track_ids,
                        split="training",
                    )
                    scaled_loss = losses["total"] / accumulation
                    self.scaler.scale(scaled_loss).backward()
                    if not self.amp_enabled:
                        self._check_gradients(
                            epoch=epoch,
                            batch_index=batch_index,
                            track_ids=track_ids,
                        )

                    for key, value in losses.items():
                        rolling[key] += float(value.detach())
                    rolling["si_sdr"] += float(
                        si_sdr(outputs["reconstruction"].detach(), audio)
                    )
                    rolling_count += 1

                    is_update = (
                        batch_index % accumulation == 0
                        or batch_index == len(self.train_loader)
                    )
                    if not is_update:
                        continue
                    self.scaler.unscale_(self.optimizer)
                    self._check_gradients(
                        epoch=epoch,
                        batch_index=batch_index,
                        track_ids=track_ids,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.decoder.parameters(),
                        grad_clip,
                        error_if_nonfinite=True,
                    )
                    used_learning_rate = self.optimizer.param_groups[0]["lr"]
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self._check_parameters(
                        epoch=epoch,
                        batch_index=batch_index,
                        track_ids=track_ids,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self.scheduler.step()
                    progress.set_postfix(total=f"{float(losses['total'].detach()):.3f}")

                    if self.global_step % log_every == 0:
                        metrics = _mean_metrics(rolling, rolling_count)
                        metrics["learning_rate"] = used_learning_rate
                        self._write_metrics("train", epoch, metrics)
                        rolling.clear()
                        rolling_count = 0
                    if validate_every > 0 and self.global_step % validate_every == 0:
                        self.validate(epoch)
                    if checkpoint_every > 0 and self.global_step % checkpoint_every == 0:
                        self.save_checkpoint(
                            self.checkpoint_dir / f"step_{self.global_step:08d}.pt",
                            epoch,
                        )

                if rolling_count:
                    metrics = _mean_metrics(rolling, rolling_count)
                    metrics["learning_rate"] = used_learning_rate
                    self._write_metrics("train", epoch, metrics)
                self.validate(epoch)
                self._save_resume_checkpoints(epoch)
                if epoch % int(self.train_config["sample_every_epochs"]) == 0:
                    self.save_listening_samples(epoch)
                plot_history(
                    self.output_dir / "history.csv",
                    self.output_dir / "loss_curves.png",
                )
            self._assert_resume_checkpoints()
        except KeyboardInterrupt:
            if current_epoch >= self.start_epoch and self.global_step > 0:
                self.optimizer.zero_grad(set_to_none=True)
                self._save_resume_checkpoints(current_epoch)
            raise
        finally:
            self.writer.close()
        print(json.dumps({"best_val_total": self.best_val}, indent=2))
