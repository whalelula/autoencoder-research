from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


def _read_audio(path: Path) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(path)
    except Exception as torchaudio_error:
        try:
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            return torch.from_numpy(samples.T.copy()), int(sample_rate)
        except Exception as soundfile_error:
            raise RuntimeError(
                f"Could not decode {path}. torchaudio: {torchaudio_error}; "
                f"soundfile: {soundfile_error}"
            ) from soundfile_error


def _to_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    if channels == 1:
        return waveform.mean(dim=0, keepdim=True)
    if channels != 2:
        raise ValueError("channels must be 1 or 2")
    if waveform.shape[0] == 1:
        return waveform.repeat(2, 1)
    return waveform[:2]


def _resolve_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def _write_flac(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    samples = waveform.squeeze(0).numpy() if waveform.shape[0] == 1 else waveform.T.numpy()
    sf.write(temporary, samples, sample_rate, format="FLAC")
    os.replace(temporary, path)


def preprocess_record(
    record: dict[str, Any],
    *,
    input_root: Path,
    output_root: Path,
    sample_rate: int,
    chunk_samples: int,
    channels: int,
    split: str,
    drop_last: bool,
    overwrite: bool,
) -> list[dict[str, Any]]:
    source = _resolve_path(input_root, str(record["path"]))
    waveform, source_rate = _read_audio(source)
    waveform = _to_channels(waveform, channels)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)

    total_samples = waveform.shape[-1]
    chunk_count = total_samples // chunk_samples
    if not drop_last and total_samples % chunk_samples:
        chunk_count += 1
    if chunk_count == 0:
        return []

    processed = []
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_samples
        chunk = waveform[..., start : start + chunk_samples]
        if chunk.shape[-1] < chunk_samples:
            if drop_last:
                continue
            chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

        track_id = str(record["track_id"])
        relative_path = Path("audio") / split / f"{track_id}_{chunk_index:05d}.flac"
        destination = output_root / relative_path
        if overwrite or not destination.exists():
            _write_flac(destination, chunk.contiguous().cpu().clamp(-1.0, 1.0), sample_rate)

        new_record = dict(record)
        new_record["source_track_id"] = record["track_id"]
        new_record["track_id"] = f"{track_id}_{chunk_index:05d}"
        new_record["path"] = relative_path.as_posix()
        new_record["chunk_index"] = chunk_index
        new_record["chunk_start_seconds"] = start / sample_rate
        new_record["duration"] = chunk_samples / sample_rate
        processed.append(new_record)
    return processed


def preprocess_manifest(
    manifest_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    output_manifest_dir: Path,
    sample_rate: int,
    chunk_seconds: float,
    channels: int = 1,
    workers: int = 1,
    drop_last: bool = True,
    overwrite: bool = False,
) -> int:
    split = manifest_path.stem
    chunk_samples = round(sample_rate * chunk_seconds)
    if chunk_samples <= 0:
        raise ValueError("chunk_seconds must produce at least one sample")

    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_manifest_dir / manifest_path.name
    temporary_manifest = output_manifest.with_suffix(output_manifest.suffix + ".part")
    processed_count = 0

    with temporary_manifest.open("w", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [
                pool.submit(
                    preprocess_record,
                    record,
                    input_root=input_root,
                    output_root=output_root,
                    sample_rate=sample_rate,
                    chunk_samples=chunk_samples,
                    channels=channels,
                    split=split,
                    drop_last=drop_last,
                    overwrite=overwrite,
                )
                for record in records
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Preprocessing {split}",
            ):
                for record in future.result():
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed_count += 1

    os.replace(temporary_manifest, output_manifest)
    if processed_count == 0:
        raise RuntimeError(f"No chunks were written for {manifest_path}")
    return processed_count
