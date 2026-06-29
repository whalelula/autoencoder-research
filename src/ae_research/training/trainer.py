from __future__ import annotations

import json
import random
from collections import defaultdict
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
        self.optimizer = torch.optim.AdamW(
            self.model.decoder.parameters(),
            lr=float(self.train_config["learning_rate"]),
            weight_decay=float(self.train_config["weight_decay"]),
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
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val": self.best_val,
            "config": self.config,
        }
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".part")
        torch.save(state, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.decoder.load_state_dict(checkpoint["decoder"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", 0))
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

        for epoch in range(self.start_epoch, int(self.train_config["epochs"]) + 1):
            rolling: defaultdict[str, float] = defaultdict(float)
            rolling_count = 0
            progress = tqdm(self.train_loader, desc=f"epoch {epoch}")
            for batch_index, batch in enumerate(progress, start=1):
                audio = batch["audio"].to(self.device, non_blocking=True)
                outputs, losses = self._forward(audio)
                scaled_loss = losses["total"] / accumulation
                self.scaler.scale(scaled_loss).backward()

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
                torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                progress.set_postfix(total=f"{float(losses['total'].detach()):.3f}")

                if self.global_step % log_every == 0:
                    self._write_metrics(
                        "train", epoch, _mean_metrics(rolling, rolling_count)
                    )
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
                self._write_metrics(
                    "train", epoch, _mean_metrics(rolling, rolling_count)
                )
            self.validate(epoch)
            self.save_checkpoint(self.checkpoint_dir / "last.pt", epoch)
            if epoch % int(self.train_config["sample_every_epochs"]) == 0:
                self.save_listening_samples(epoch)
            plot_history(
                self.output_dir / "history.csv",
                self.output_dir / "loss_curves.png",
            )
        self.writer.close()
        print(json.dumps({"best_val_total": self.best_val}, indent=2))

