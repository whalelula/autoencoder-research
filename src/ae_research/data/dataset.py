from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


class AudioManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        *,
        data_root: str | Path,
        sample_rate: int,
        duration_seconds: float,
        channels: int,
        random_crop: bool,
        peak_normalize: bool = False,
    ) -> None:
        self.records = read_manifest(manifest)
        self.data_root = Path(data_root)
        self.sample_rate = int(sample_rate)
        self.num_samples = round(sample_rate * duration_seconds)
        self.channels = int(channels)
        self.random_crop = random_crop
        self.peak_normalize = peak_normalize

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

    def _fix_channels(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.channels == 1:
            return waveform.mean(dim=0, keepdim=True)
        if waveform.shape[0] == 1:
            return waveform.repeat(2, 1)
        return waveform[:2]

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[-1]
        if length > self.num_samples:
            if self.random_crop:
                start = int(torch.randint(length - self.num_samples + 1, (1,)).item())
            else:
                start = (length - self.num_samples) // 2
            waveform = waveform[..., start : start + self.num_samples]
        elif length < self.num_samples:
            padding = self.num_samples - length
            left = 0 if self.random_crop else padding // 2
            waveform = F.pad(waveform, (left, padding - left))
        return waveform

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self._resolve_path(record["path"])
        try:
            waveform, source_rate = torchaudio.load(path)
        except Exception as torchaudio_error:
            try:
                samples, source_rate = sf.read(
                    path, dtype="float32", always_2d=True
                )
                waveform = torch.from_numpy(samples.T.copy())
            except Exception as soundfile_error:
                raise RuntimeError(
                    f"Could not decode {path} with torchaudio or soundfile. MP3 "
                    "loading requires FFmpeg/TorchCodec or an MP3-enabled libsndfile; "
                    f"converting the dataset to FLAC is a safe fallback. torchaudio: "
                    f"{torchaudio_error}; soundfile: {soundfile_error}"
                ) from soundfile_error
        waveform = self._fix_channels(waveform)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        waveform = self._crop_or_pad(waveform)
        if self.peak_normalize:
            peak = waveform.abs().amax().clamp_min(1e-8)
            waveform = waveform / peak
        return {
            "audio": waveform.clamp(-1.0, 1.0),
            "track_id": str(record["track_id"]),
            "path": str(path),
        }


def create_dataloader(
    manifest: str | Path,
    data_config: dict[str, Any],
    *,
    batch_size: int,
    split: str,
    shuffle: bool | None = None,
) -> DataLoader:
    is_train = split == "train"
    dataset = AudioManifestDataset(
        manifest,
        data_root=data_config["root"],
        sample_rate=int(data_config["sample_rate"]),
        duration_seconds=float(data_config["duration_seconds"]),
        channels=int(data_config["channels"]),
        random_crop=is_train and bool(data_config["train_random_crop"]),
        peak_normalize=bool(data_config.get("peak_normalize", False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]),
        drop_last=False,
        persistent_workers=int(data_config["num_workers"]) > 0,
    )
