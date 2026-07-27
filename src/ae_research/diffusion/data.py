from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ae_research.data.dataset import AudioManifestDataset

from .text import (
    TEXT_CACHE_FORMAT_VERSION,
    TEXT_CACHE_METADATA,
    text_embedding_path,
)


def read_text_cache_metadata(
    cache_dir: str | Path,
    *,
    expected_model_name: str | None = None,
    expected_hidden_dim: int | None = None,
    expected_max_length: int | None = None,
) -> dict[str, Any]:
    path = Path(cache_dir) / TEXT_CACHE_METADATA
    if not path.is_file():
        raise FileNotFoundError(
            f"Text embedding cache metadata not found: {path}. "
            "Run ae-prepare-dit-tags without --manifests-only first."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid text embedding cache metadata: {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Text embedding cache metadata must be an object: {path}")
    if int(metadata.get("format_version", -1)) != TEXT_CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported text cache format in {path}: "
            f"{metadata.get('format_version')!r}"
        )
    model_name = str(metadata.get("model_name", ""))
    hidden_dim = int(metadata.get("hidden_dim", 0))
    max_length = int(metadata.get("max_length", 0))
    if not model_name or hidden_dim <= 0 or max_length <= 0:
        raise ValueError(f"Incomplete text embedding cache metadata: {path}")
    if expected_model_name is not None and model_name != str(expected_model_name):
        raise ValueError(
            f"Text cache model mismatch: cache={model_name!r}, "
            f"expected={expected_model_name!r}"
        )
    if expected_hidden_dim is not None and hidden_dim != int(expected_hidden_dim):
        raise ValueError(
            f"Text cache hidden dimension mismatch: cache={hidden_dim}, "
            f"expected={int(expected_hidden_dim)}"
        )
    if expected_max_length is not None and max_length != int(expected_max_length):
        raise ValueError(
            f"Text cache max length mismatch: cache={max_length}, "
            f"expected={int(expected_max_length)}"
        )
    return metadata


def load_text_embedding(
    cache_dir: str | Path,
    key: str,
    *,
    metadata: dict[str, Any],
    expected_text: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = text_embedding_path(cache_dir, key)
    if not path.is_file():
        raise FileNotFoundError(
            f"Text embedding cache entry not found: {path}. "
            "Run ae-prepare-dit-tags to build the complete cache."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Could not read text embedding cache entry: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Text embedding cache entry must be a mapping: {path}")

    expected_model = str(metadata["model_name"])
    expected_hidden = int(metadata["hidden_dim"])
    expected_max_length = int(metadata["max_length"])
    if payload.get("format_version") != TEXT_CACHE_FORMAT_VERSION:
        raise ValueError(f"Text embedding cache format mismatch: {path}")
    if payload.get("model_name") != expected_model:
        raise ValueError(
            f"Text embedding model mismatch in {path}: "
            f"{payload.get('model_name')!r} != {expected_model!r}"
        )
    if int(payload.get("hidden_dim", -1)) != expected_hidden:
        raise ValueError(f"Text embedding hidden dimension metadata mismatch: {path}")
    if int(payload.get("max_length", -1)) != expected_max_length:
        raise ValueError(f"Text embedding max length metadata mismatch: {path}")
    if expected_text is not None and payload.get("text") != expected_text:
        raise ValueError(
            f"Manifest text does not match cached text for key {key!r}: {path}"
        )

    embedding = payload.get("text_embedding")
    mask = payload.get("text_mask")
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise ValueError(f"text_embedding must have shape [tokens, hidden]: {path}")
    if embedding.shape[1] != expected_hidden:
        raise ValueError(
            f"Text embedding shape mismatch in {path}: got {tuple(embedding.shape)}, "
            f"expected hidden_dim={expected_hidden}"
        )
    if not 0 < embedding.shape[0] <= expected_max_length:
        raise ValueError(
            f"Text embedding token length must be in [1, {expected_max_length}]: {path}"
        )
    if not isinstance(mask, torch.Tensor) or mask.shape != embedding.shape[:1]:
        raise ValueError(f"text_mask must have shape [tokens]: {path}")
    mask = mask.bool()
    if not mask.any():
        raise ValueError(f"Text embedding cache entry has no valid tokens: {path}")
    if not torch.isfinite(embedding).all():
        raise ValueError(f"Text embedding cache entry contains non-finite values: {path}")
    return embedding, mask


class DiTManifestDataset(Dataset[dict[str, Any]]):
    """Fixed-length audio paired with precomputed frozen text token embeddings."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        data_root: str | Path,
        text_embedding_dir: str | Path,
        sample_rate: int,
        duration_seconds: float,
        channels: int,
        expected_text_model: str | None = None,
        expected_text_dim: int | None = None,
        expected_text_max_length: int | None = None,
        cache_embeddings_in_memory: bool = True,
    ) -> None:
        self.audio_dataset = AudioManifestDataset(
            manifest,
            data_root=data_root,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            channels=channels,
        )
        self.records = self.audio_dataset.records
        self.text_embedding_dir = Path(text_embedding_dir)
        self.cache_metadata = read_text_cache_metadata(
            self.text_embedding_dir,
            expected_model_name=expected_text_model,
            expected_hidden_dim=expected_text_dim,
            expected_max_length=expected_text_max_length,
        )
        self.text_embedding_dim = int(self.cache_metadata["hidden_dim"])
        self.text_max_length = int(self.cache_metadata["max_length"])
        self.cache_embeddings_in_memory = bool(cache_embeddings_in_memory)
        self._embedding_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        referenced_keys = set()
        for index, record in enumerate(self.records):
            key = record.get("text_embedding_key")
            text = record.get("text")
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"Manifest record {index} has no text_embedding_key: {manifest}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Manifest record {index} has no text: {manifest}")
            referenced_keys.add(key)
        missing = [
            str(text_embedding_path(self.text_embedding_dir, key))
            for key in sorted(referenced_keys)
            if not text_embedding_path(self.text_embedding_dir, key).is_file()
        ]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"Text embedding cache is missing {len(missing)} referenced entries. "
                f"First entries: {preview}"
            )

    def __len__(self) -> int:
        return len(self.audio_dataset)

    def _text_embedding(
        self, *, key: str, text: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        value = load_text_embedding(
            self.text_embedding_dir,
            key,
            metadata=self.cache_metadata,
            expected_text=text,
        )
        if self.cache_embeddings_in_memory:
            self._embedding_cache[key] = value
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        audio_item = self.audio_dataset[index]
        key = str(record["text_embedding_key"])
        text = str(record["text"])
        embedding, mask = self._text_embedding(key=key, text=text)
        return {
            **audio_item,
            "source_track_id": str(record.get("source_track_id", record["track_id"])),
            "text": text,
            "text_embedding_key": key,
            "text_embedding": embedding,
            "text_mask": mask,
            "duration": float(
                record.get(
                    "duration",
                    self.audio_dataset.num_samples / self.audio_dataset.sample_rate,
                )
            ),
        }


def collate_dit_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty DiT batch")
    embeddings = [item["text_embedding"] for item in batch]
    masks = [item["text_mask"].bool() for item in batch]
    hidden_dims = {int(embedding.shape[-1]) for embedding in embeddings}
    dtypes = {embedding.dtype for embedding in embeddings}
    if len(hidden_dims) != 1:
        raise ValueError("Every text embedding in a batch must have the same hidden dim")
    if len(dtypes) != 1:
        raise ValueError("Every text embedding in a batch must have the same dtype")
    for embedding, mask in zip(embeddings, masks):
        if embedding.ndim != 2 or mask.shape != embedding.shape[:1]:
            raise ValueError("Invalid text embedding or mask shape in DiT batch")
        if not mask.any():
            raise ValueError("Every text condition must contain at least one valid token")

    return {
        "audio": torch.stack([item["audio"] for item in batch]),
        "track_id": [str(item["track_id"]) for item in batch],
        "source_track_id": [str(item["source_track_id"]) for item in batch],
        "path": [str(item["path"]) for item in batch],
        "text": [str(item["text"]) for item in batch],
        "text_embedding_key": [str(item["text_embedding_key"]) for item in batch],
        "text_embedding": pad_sequence(embeddings, batch_first=True, padding_value=0.0),
        "text_mask": pad_sequence(masks, batch_first=True, padding_value=False),
        "duration": torch.tensor(
            [float(item["duration"]) for item in batch], dtype=torch.float32
        ),
    }


def create_dit_dataloader(
    manifest: str | Path,
    data_config: dict[str, Any],
    *,
    text_embedding_dir: str | Path,
    batch_size: int,
    split: str,
    expected_text_model: str | None = None,
    expected_text_dim: int | None = None,
    expected_text_max_length: int | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    is_train = split == "train"
    dataset = DiTManifestDataset(
        manifest,
        data_root=data_config["root"],
        text_embedding_dir=text_embedding_dir,
        sample_rate=int(data_config["sample_rate"]),
        duration_seconds=float(data_config["duration_seconds"]),
        channels=int(data_config["channels"]),
        expected_text_model=expected_text_model,
        expected_text_dim=expected_text_dim,
        expected_text_max_length=expected_text_max_length,
    )
    workers = int(data_config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=is_train if shuffle is None else bool(shuffle),
        num_workers=workers,
        pin_memory=bool(data_config["pin_memory"]),
        drop_last=False,
        persistent_workers=workers > 0,
        collate_fn=collate_dit_batch,
    )


__all__ = [
    "DiTManifestDataset",
    "collate_dit_batch",
    "create_dit_dataloader",
    "load_text_embedding",
    "read_text_cache_metadata",
]
