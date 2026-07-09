from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def read_manifest_records(path: str | Path) -> list[dict[str, Any]]:
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


def sample_manifest_records(
    records: list[dict[str, Any]], *, sample_count: int, seed: int
) -> list[dict[str, Any]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(records):
        raise ValueError(
            f"Cannot sample {sample_count} records from manifest with {len(records)} records"
        )
    indices = sorted(random.Random(seed).sample(range(len(records)), k=sample_count))
    return [records[index] for index in indices]


def write_sample_manifest(
    source_manifest: str | Path,
    output_manifest_dir: str | Path,
    *,
    sample_count: int,
    seed: int,
    split: str = "test",
) -> Path:
    records = read_manifest_records(source_manifest)
    sampled = sample_manifest_records(records, sample_count=sample_count, seed=seed)
    output_manifest_dir = Path(output_manifest_dir)
    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_manifest_dir / f"{split}.jsonl"
    with output_manifest.open("w", encoding="utf-8") as handle:
        for record in sampled:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_manifest


def sample_manifest_track_ids(
    source_manifest: str | Path, *, sample_count: int, seed: int
) -> set[str]:
    records = read_manifest_records(source_manifest)
    sampled = sample_manifest_records(records, sample_count=sample_count, seed=seed)
    return {str(record["track_id"]) for record in sampled}
