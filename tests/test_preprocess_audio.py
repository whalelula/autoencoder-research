from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from ae_research.data.preprocess import preprocess_manifest


def test_preprocess_manifest_writes_fixed_length_flac_chunks(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "processed"
    manifest_dir = input_root / "manifests"
    audio_dir = input_root / "audio"
    manifest_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)

    source_rate = 8000
    waveform = torch.stack(
        (
            torch.linspace(-0.5, 0.5, source_rate * 6),
            torch.linspace(0.5, -0.5, source_rate * 6),
        )
    )
    torchaudio.save(audio_dir / "track.wav", waveform, source_rate)
    (manifest_dir / "train.jsonl").write_text(
        json.dumps(
            {
                "track_id": 123,
                "artist_id": 1,
                "album_id": 2,
                "original_path": "track.wav",
                "duration": 6.0,
                "tags": ["genre---test"],
                "path": "audio/track.wav",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = preprocess_manifest(
        manifest_dir / "train.jsonl",
        input_root=input_root,
        output_root=output_root,
        output_manifest_dir=output_root / "manifests",
        sample_rate=24000,
        chunk_seconds=3.0,
        channels=1,
        workers=1,
    )

    assert count == 2
    records = [
        json.loads(line)
        for line in (output_root / "manifests" / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["track_id"] for record in records] == ["123_00000", "123_00001"]

    chunk_path = output_root / records[0]["path"]
    chunk, sample_rate = torchaudio.load(chunk_path)
    assert chunk_path.suffix == ".flac"
    assert sample_rate == 24000
    assert tuple(chunk.shape) == (1, 72000)
