from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from ae_research.diffusion.data import (  # noqa: E402
    DiTManifestDataset,
    collate_dit_batch,
    read_text_cache_metadata,
)
from ae_research.diffusion.text import (  # noqa: E402
    normalize_mtg_tags,
    prepare_dit_text_data,
    text_embedding_path,
)


class FakeTextEncoder:
    model_name = "test/t5gemma"
    max_length = 8
    hidden_dim = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts):  # noqa: ANN001, ANN201
        texts = list(texts)
        self.calls.append(texts)
        lengths = [min(self.max_length, len(text.split()) + 1) for text in texts]
        embeddings = torch.zeros(len(texts), max(lengths), self.hidden_dim)
        masks = torch.zeros(len(texts), max(lengths), dtype=torch.bool)
        for index, length in enumerate(lengths):
            embeddings[index, :length] = float(index + 1)
            masks[index, :length] = True
        return embeddings, masks


def _write_manifests(
    manifest_dir: Path, records: dict[str, list[dict]]
) -> None:
    manifest_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        with (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records[split]:
                handle.write(json.dumps(record) + "\n")


def test_normalize_mtg_tags_is_stable_and_human_readable():
    tags = [
        "mood/theme---high-energy",
        "instrument---electric_guitar",
        "genre---post-rock",
        "genre---ambient",
        "instrument---electric-guitar",
        "mood_theme---dreamy",
        "genre---ambient",
        "tempo---fast",
        "malformed",
    ]
    assert normalize_mtg_tags(tags) == (
        "ambient, post rock, electric guitar, dreamy, high energy"
    )
    assert normalize_mtg_tags([]) == "music"
    assert normalize_mtg_tags(None) == "music"


def test_prepare_text_cache_deduplicates_source_tracks_and_reuses_entries(tmp_path):
    source = tmp_path / "source"
    records = {
        "train": [
            {
                "track_id": "10_00000",
                "source_track_id": 10,
                "path": "audio/train/10_00000.flac",
                "tags": ["genre---post-rock"],
            }
        ],
        "val": [
            {
                "track_id": "10_00001",
                "source_track_id": 10,
                "path": "audio/val/10_00001.flac",
                "tags": ["genre---post-rock"],
            }
        ],
        "test": [
            {
                "track_id": "20_00000",
                "source_track_id": 20,
                "path": "audio/test/20_00000.flac",
                "tags": ["instrument---electric-guitar"],
            }
        ],
    }
    _write_manifests(source, records)
    output = tmp_path / "dit_manifests"
    cache = tmp_path / "embeddings"
    encoder = FakeTextEncoder()

    result = prepare_dit_text_data(
        source,
        output,
        cache,
        encoder=encoder,
        batch_size=4,
    )

    assert result["num_records"] == 3
    assert result["num_unique_source_tracks"] == 2
    assert result["embeddings_written"] == 2
    assert encoder.calls == [["post rock", "electric guitar"]]
    train = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    validation = json.loads((output / "val.jsonl").read_text(encoding="utf-8"))
    assert train["text"] == "post rock"
    assert train["text_embedding_key"] == validation["text_embedding_key"] == "10"
    assert len(list(cache.glob("*.pt"))) == 2
    assert not list(tmp_path.rglob("*.tmp"))

    fresh_encoder = FakeTextEncoder()
    reused = prepare_dit_text_data(
        source,
        output,
        cache,
        encoder=fresh_encoder,
        batch_size=4,
    )
    assert reused["embeddings_written"] == 0
    assert reused["embeddings_reused"] == 2
    assert fresh_encoder.calls == []


def test_manifests_only_does_not_require_an_encoder_or_create_cache(tmp_path):
    source = tmp_path / "source"
    record = {
        "track_id": "1_00000",
        "source_track_id": 1,
        "path": "unused.flac",
        "tags": ["genre---ambient"],
    }
    _write_manifests(
        source,
        {"train": [record], "val": [record], "test": [record]},
    )
    output = tmp_path / "dit_manifests"
    cache = tmp_path / "not_created"

    result = prepare_dit_text_data(
        source,
        output,
        cache,
        manifests_only=True,
    )

    assert result["manifests_only"] is True
    assert not cache.exists()
    written = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    assert written["text"] == "ambient"
    assert written["text_embedding_key"] == "1"


def _make_audio_dataset(tmp_path: Path):  # noqa: ANN201
    data_root = tmp_path / "data"
    source_manifests = tmp_path / "source_manifests"
    sample_rate = 8_000
    sample_count = 80
    records = []
    for index, tags in enumerate(
        (["genre---rock"], ["instrument---electric-guitar"]), start=1
    ):
        relative = Path("audio") / "train" / f"{index}.wav"
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(path, torch.zeros(1, sample_count), sample_rate)
        records.append(
            {
                "track_id": f"{index}_00000",
                "source_track_id": index,
                "path": relative.as_posix(),
                "duration": sample_count / sample_rate,
                "tags": tags,
            }
        )
    _write_manifests(
        source_manifests,
        {"train": records, "val": [records[0]], "test": [records[1]]},
    )
    output_manifests = tmp_path / "dit_manifests"
    cache = tmp_path / "embeddings"
    prepare_dit_text_data(
        source_manifests,
        output_manifests,
        cache,
        encoder=FakeTextEncoder(),
    )
    return data_root, output_manifests, cache, sample_rate, sample_count


def test_dit_dataset_validates_cache_and_collate_pads_text(tmp_path):
    data_root, manifests, cache, sample_rate, sample_count = _make_audio_dataset(
        tmp_path
    )
    dataset = DiTManifestDataset(
        manifests / "train.jsonl",
        data_root=data_root,
        text_embedding_dir=cache,
        sample_rate=sample_rate,
        duration_seconds=sample_count / sample_rate,
        channels=1,
        expected_text_model=FakeTextEncoder.model_name,
        expected_text_dim=FakeTextEncoder.hidden_dim,
        expected_text_max_length=FakeTextEncoder.max_length,
    )

    batch = collate_dit_batch([dataset[0], dataset[1]])

    assert batch["audio"].shape == (2, 1, sample_count)
    assert batch["text_embedding"].shape == (2, 3, FakeTextEncoder.hidden_dim)
    assert batch["text_mask"].dtype == torch.bool
    assert batch["text_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["duration"].tolist() == pytest.approx([0.01, 0.01])

    with pytest.raises(ValueError, match="model mismatch"):
        read_text_cache_metadata(cache, expected_model_name="different/model")


def test_dit_dataset_rejects_missing_and_wrong_shape_cache_entries(tmp_path):
    data_root, manifests, cache, sample_rate, sample_count = _make_audio_dataset(
        tmp_path
    )
    first_key = "1"
    first_path = text_embedding_path(cache, first_key)
    first_path.unlink()
    with pytest.raises(FileNotFoundError, match="missing 1 referenced"):
        DiTManifestDataset(
            manifests / "train.jsonl",
            data_root=data_root,
            text_embedding_dir=cache,
            sample_rate=sample_rate,
            duration_seconds=sample_count / sample_rate,
            channels=1,
        )

    prepare_dit_text_data(
        tmp_path / "source_manifests",
        manifests,
        cache,
        encoder=FakeTextEncoder(),
    )
    payload = torch.load(first_path, map_location="cpu", weights_only=True)
    payload["text_embedding"] = torch.zeros(2, FakeTextEncoder.hidden_dim + 1)
    payload["text_mask"] = torch.ones(2, dtype=torch.bool)
    torch.save(payload, first_path)
    dataset = DiTManifestDataset(
        manifests / "train.jsonl",
        data_root=data_root,
        text_embedding_dir=cache,
        sample_rate=sample_rate,
        duration_seconds=sample_count / sample_rate,
        channels=1,
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        _ = dataset[0]
