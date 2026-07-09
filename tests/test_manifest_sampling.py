from __future__ import annotations

import json

import pytest

from ae_research.data.sampling import (
    sample_manifest_records,
    sample_manifest_track_ids,
    write_sample_manifest,
)


def test_sample_manifest_records_are_deterministic():
    records = [{"track_id": str(index), "path": f"{index}.flac"} for index in range(10)]

    first = sample_manifest_records(records, sample_count=4, seed=123)
    second = sample_manifest_records(records, sample_count=4, seed=123)

    assert first == second
    assert len(first) == 4
    assert [int(record["track_id"]) for record in first] == sorted(
        int(record["track_id"]) for record in first
    )


def test_sample_manifest_rejects_too_many_records():
    with pytest.raises(ValueError, match="Cannot sample"):
        sample_manifest_records([{"track_id": "a"}], sample_count=2, seed=42)


def test_write_sample_manifest(tmp_path):
    source = tmp_path / "test.jsonl"
    records = [{"track_id": str(index), "path": f"{index}.flac"} for index in range(5)]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    output = write_sample_manifest(
        source,
        tmp_path / "sample_manifest",
        sample_count=2,
        seed=42,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert output.name == "test.jsonl"
    assert len(lines) == 2


def test_sample_manifest_track_ids_are_deterministic(tmp_path):
    source = tmp_path / "test.jsonl"
    records = [{"track_id": str(index), "path": f"{index}.flac"} for index in range(8)]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    first = sample_manifest_track_ids(source, sample_count=3, seed=7)
    second = sample_manifest_track_ids(source, sample_count=3, seed=7)
    different_seed = sample_manifest_track_ids(source, sample_count=3, seed=8)

    assert first == second
    assert len(first) == 3
    assert first != different_seed
