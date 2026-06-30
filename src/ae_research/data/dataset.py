from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
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
    ) -> None:
        self.records = read_manifest(manifest)
        self.data_root = Path(data_root)
        self.sample_rate = int(sample_rate)
        self.num_samples = round(sample_rate * duration_seconds)
        self.channels = int(channels)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

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
        if source_rate != self.sample_rate:
            raise ValueError(
                f"Preprocessed audio has sample rate {source_rate}, expected "
                f"{self.sample_rate}: {path}"
            )
        if waveform.shape[0] != self.channels:
            raise ValueError(
                f"Preprocessed audio has {waveform.shape[0]} channels, expected "
                f"{self.channels}: {path}"
            )
        if waveform.shape[-1] != self.num_samples:
            raise ValueError(
                f"Preprocessed audio has {waveform.shape[-1]} samples, expected "
                f"{self.num_samples}: {path}"
            )
        if not torch.isfinite(waveform).all():
            raise ValueError(f"Preprocessed audio contains non-finite samples: {path}")
        return {
            "audio": waveform,
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
