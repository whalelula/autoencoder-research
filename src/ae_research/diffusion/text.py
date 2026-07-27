from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn


DEFAULT_T5GEMMA_MODEL = "google/t5gemma-s-s-ul2"
DEFAULT_TEXT_MAX_LENGTH = 256
TEXT_CACHE_FORMAT_VERSION = 1
TEXT_CACHE_METADATA = "cache_metadata.json"
DEFAULT_SPLITS = ("train", "val", "test")

_CATEGORY_ORDER = ("genre", "instrument", "mood-theme")
_CATEGORY_ALIASES = {
    "genre": "genre",
    "instrument": "instrument",
    "mood": "mood-theme",
    "mood-theme": "mood-theme",
    "moodtheme": "mood-theme",
}
_SAFE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_WORD_SEPARATOR = re.compile(r"[-_/]+")
_WHITESPACE = re.compile(r"\s+")


def _normalise_category(value: str) -> str | None:
    value = value.strip().lower().replace("_", "-").replace("/", "-")
    value = _WHITESPACE.sub("", value)
    return _CATEGORY_ALIASES.get(value)


def _readable_tag_value(value: str) -> str:
    value = _WORD_SEPARATOR.sub(" ", value.strip())
    return _WHITESPACE.sub(" ", value).strip()


def normalize_mtg_tags(tags: Iterable[str] | None) -> str:
    """Convert MTG-Jamendo tags into a stable, human-readable text prompt.

    Only genre, instrument, and mood/theme tags are retained. Category prefixes
    are removed, values are de-duplicated after separator normalisation, and the
    result is ordered by category and then alphabetically within each category.
    """

    grouped: dict[str, dict[str, str]] = {category: {} for category in _CATEGORY_ORDER}
    for raw_tag in tags or ():
        if not isinstance(raw_tag, str) or "---" not in raw_tag:
            continue
        raw_category, raw_value = raw_tag.split("---", 1)
        category = _normalise_category(raw_category)
        if category is None:
            continue
        value = _readable_tag_value(raw_value)
        if not value:
            continue
        grouped[category].setdefault(value.casefold(), value)

    values = []
    for category in _CATEGORY_ORDER:
        values.extend(
            grouped[category][key]
            for key in sorted(grouped[category], key=str.casefold)
        )
    return ", ".join(values) if values else "music"


def source_track_id(record: dict[str, Any]) -> str:
    """Return the source-level id used to share one text embedding across chunks."""

    value = record.get("source_track_id", record.get("track_id"))
    if value is None or not str(value).strip():
        raise ValueError("Manifest record must contain track_id or source_track_id")
    return str(value)


def text_embedding_key(value: str | int) -> str:
    """Create a deterministic cache filename stem without allowing path traversal."""

    raw = str(value).strip()
    if _SAFE_KEY.fullmatch(raw) and raw not in {".", ".."}:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"source-{digest}"


def text_embedding_path(cache_dir: str | Path, key: str) -> Path:
    if not _SAFE_KEY.fullmatch(key) or key in {".", ".."}:
        raise ValueError(f"Invalid text_embedding_key: {key!r}")
    return Path(cache_dir) / f"{key}.pt"


class TextEncoder(Protocol):
    model_name: str
    max_length: int
    hidden_dim: int

    def encode(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]: ...


def _config_hidden_dim(*configs: Any) -> int:
    for config in configs:
        if config is None:
            continue
        for name in ("hidden_size", "d_model", "model_dim"):
            if isinstance(config, dict):
                value = config.get(name)
            else:
                value = getattr(config, name, None)
            if value is not None and int(value) > 0:
                return int(value)
    raise ValueError("Could not infer the T5Gemma encoder hidden dimension")


class FrozenT5GemmaEncoder(nn.Module):
    """Frozen T5Gemma encoder loaded only when this class is instantiated.

    Importing this module does not import Transformers or access Hugging Face.
    ``local_files_only=True`` provides a strict offline/cache-only preparation mode.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_T5GEMMA_MODEL,
        *,
        max_length: int = DEFAULT_TEXT_MAX_LENGTH,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:  # pragma: no cover - exercised without the dit extra
            raise RuntimeError(
                'T5Gemma preparation requires the DiT extra: pip install -e ".[dit]"'
            ) from error

        self.model_name = str(model_name)
        self.max_length = int(max_length)
        load_options = {"local_files_only": bool(local_files_only)}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **load_options,
        )
        full_model = AutoModel.from_pretrained(self.model_name, **load_options)
        get_encoder = getattr(full_model, "get_encoder", None)
        encoder = get_encoder() if callable(get_encoder) else getattr(full_model, "encoder", None)
        if encoder is None:
            raise TypeError(f"{self.model_name} does not expose an encoder module")
        nested_config = getattr(getattr(full_model, "config", None), "encoder", None)
        self.hidden_dim = _config_hidden_dim(
            getattr(encoder, "config", None),
            nested_config,
            getattr(full_model, "config", None),
        )
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        if device is not None:
            self.encoder.to(torch.device(device))

    @property
    def device(self) -> torch.device:
        try:
            return next(self.encoder.parameters()).device
        except StopIteration:  # pragma: no cover - real T5Gemma always has parameters
            return torch.device("cpu")

    def train(self, mode: bool = True) -> "FrozenT5GemmaEncoder":
        super().train(False)
        self.encoder.eval()
        return self

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(text) if str(text).strip() else "music" for text in texts]
        if not texts:
            raise ValueError("texts must contain at least one prompt")
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        embeddings = outputs.last_hidden_state
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                "T5Gemma returned an unexpected hidden shape: "
                f"{tuple(embeddings.shape)} (expected hidden_dim={self.hidden_dim})"
            )
        return embeddings, attention_mask.bool()

    def forward(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(texts)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Manifest record must be an object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def _atomic_write_bytes(path: Path, writer) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    def write(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    _atomic_write_bytes(path, write)


def _atomic_write_manifest(path: Path, records: Sequence[dict[str, Any]]) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    _atomic_write_bytes(path, write)


def _atomic_torch_save(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, lambda temporary: torch.save(value, temporary))


def _augment_manifests(
    manifest_dir: Path,
    *,
    splits: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[str, str]]]:
    augmented: dict[str, list[dict[str, Any]]] = {}
    conditions: dict[str, tuple[str, str]] = {}
    key_sources: dict[str, str] = {}
    for split in splits:
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
        split_records = []
        for record in _read_manifest(path):
            source_id = source_track_id(record)
            text = normalize_mtg_tags(record.get("tags"))
            key = text_embedding_key(source_id)
            previous = conditions.get(source_id)
            if previous is not None and previous != (key, text):
                raise ValueError(
                    f"Source track {source_id!r} has inconsistent tags/text across records"
                )
            previous_source = key_sources.get(key)
            if previous_source is not None and previous_source != source_id:
                raise RuntimeError(
                    f"Text embedding key collision: {previous_source!r} and {source_id!r}"
                )
            conditions[source_id] = (key, text)
            key_sources[key] = source_id
            updated = dict(record)
            updated["text"] = text
            updated["text_embedding_key"] = key
            split_records.append(updated)
        augmented[split] = split_records
    return augmented, conditions


def _load_cached_payload(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _payload_matches(
    payload: dict[str, Any] | None,
    *,
    source_id: str,
    text: str,
    model_name: str,
    max_length: int,
    hidden_dim: int,
) -> bool:
    if payload is None:
        return False
    embedding = payload.get("text_embedding")
    mask = payload.get("text_mask")
    return bool(
        payload.get("format_version") == TEXT_CACHE_FORMAT_VERSION
        and payload.get("source_track_id") == source_id
        and payload.get("text") == text
        and payload.get("model_name") == model_name
        and int(payload.get("max_length", -1)) == max_length
        and int(payload.get("hidden_dim", -1)) == hidden_dim
        and isinstance(embedding, torch.Tensor)
        and embedding.ndim == 2
        and embedding.shape[1] == hidden_dim
        and 0 < embedding.shape[0] <= max_length
        and isinstance(mask, torch.Tensor)
        and mask.shape == embedding.shape[:1]
        and bool(mask.bool().any())
    )


def prepare_dit_text_data(
    manifest_dir: str | Path,
    output_manifest_dir: str | Path,
    cache_dir: str | Path,
    *,
    encoder: TextEncoder | None = None,
    manifests_only: bool = False,
    batch_size: int = 8,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """Write text-augmented manifests and an optional de-duplicated embedding cache."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not splits:
        raise ValueError("splits must not be empty")
    manifest_dir = Path(manifest_dir)
    output_manifest_dir = Path(output_manifest_dir)
    cache_dir = Path(cache_dir)
    augmented, conditions = _augment_manifests(manifest_dir, splits=splits)

    written = 0
    reused = 0
    hidden_dim: int | None = None
    if not manifests_only:
        if encoder is None:
            raise ValueError("encoder is required unless manifests_only=True")
        model_name = str(encoder.model_name)
        max_length = int(encoder.max_length)
        hidden_dim = int(encoder.hidden_dim)
        if max_length <= 0 or hidden_dim <= 0:
            raise ValueError("encoder max_length and hidden_dim must be positive")
        cache_dir.mkdir(parents=True, exist_ok=True)

        pending: list[tuple[str, str, str]] = []
        for source_id, (key, text) in sorted(conditions.items()):
            path = text_embedding_path(cache_dir, key)
            payload = _load_cached_payload(path)
            if _payload_matches(
                payload,
                source_id=source_id,
                text=text,
                model_name=model_name,
                max_length=max_length,
                hidden_dim=hidden_dim,
            ):
                reused += 1
            else:
                pending.append((source_id, key, text))

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            embeddings, masks = encoder.encode([text for _, _, text in batch])
            if embeddings.ndim != 3 or embeddings.shape[0] != len(batch):
                raise ValueError(
                    "Text encoder must return embeddings shaped [batch, tokens, hidden]"
                )
            if embeddings.shape[-1] != hidden_dim:
                raise ValueError(
                    f"Text encoder returned hidden_dim={embeddings.shape[-1]}, "
                    f"expected {hidden_dim}"
                )
            if masks.shape != embeddings.shape[:2]:
                raise ValueError("Text encoder mask must have shape [batch, tokens]")
            for index, (source_id, key, text) in enumerate(batch):
                valid = masks[index].bool()
                if not valid.any():
                    raise ValueError(f"Text encoder produced no valid tokens for {text!r}")
                embedding = embeddings[index][valid].detach().cpu().contiguous()
                if embedding.shape[0] > max_length:
                    raise ValueError("Text encoder returned more tokens than max_length")
                text_mask = torch.ones(embedding.shape[0], dtype=torch.bool)
                payload = {
                    "format_version": TEXT_CACHE_FORMAT_VERSION,
                    "source_track_id": source_id,
                    "text": text,
                    "model_name": model_name,
                    "max_length": max_length,
                    "hidden_dim": hidden_dim,
                    "text_embedding": embedding,
                    "text_mask": text_mask,
                }
                _atomic_torch_save(text_embedding_path(cache_dir, key), payload)
                written += 1

        metadata = {
            "format_version": TEXT_CACHE_FORMAT_VERSION,
            "model_name": model_name,
            "max_length": max_length,
            "hidden_dim": hidden_dim,
            "num_embeddings": len(conditions),
        }
        _atomic_write_json(cache_dir / TEXT_CACHE_METADATA, metadata)

    output_paths = {}
    for split, records in augmented.items():
        output = output_manifest_dir / f"{split}.jsonl"
        _atomic_write_manifest(output, records)
        output_paths[split] = str(output)

    return {
        "manifests": output_paths,
        "num_records": sum(len(records) for records in augmented.values()),
        "num_unique_source_tracks": len(conditions),
        "embeddings_written": written,
        "embeddings_reused": reused,
        "hidden_dim": hidden_dim,
        "manifests_only": bool(manifests_only),
    }


__all__ = [
    "DEFAULT_T5GEMMA_MODEL",
    "DEFAULT_TEXT_MAX_LENGTH",
    "FrozenT5GemmaEncoder",
    "TEXT_CACHE_FORMAT_VERSION",
    "TEXT_CACHE_METADATA",
    "normalize_mtg_tags",
    "prepare_dit_text_data",
    "source_track_id",
    "text_embedding_key",
    "text_embedding_path",
]
