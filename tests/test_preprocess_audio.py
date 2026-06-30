from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from ae_research.data.dataset import AudioManifestDataset  # noqa: E402
from ae_research.data.preprocess import (  # noqa: E402
    ensure_preprocessed_dataset,
    preprocess_manifest,
)
from ae_research.cli import preprocess_audio as preprocess_cli  # noqa: E402


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

    dataset = AudioManifestDataset(
        output_root / "manifests" / "train.jsonl",
        data_root=output_root,
        sample_rate=24000,
        duration_seconds=3.0,
        channels=1,
    )
    assert tuple(dataset[0]["audio"].shape) == (1, 72000)

    wrong_duration = AudioManifestDataset(
        output_root / "manifests" / "train.jsonl",
        data_root=output_root,
        sample_rate=24000,
        duration_seconds=2.0,
        channels=1,
    )
    with pytest.raises(ValueError, match="samples"):
        wrong_duration[0]


def test_ensure_preprocessed_dataset_reuses_complete_chunks(tmp_path):
    output_root = tmp_path / "processed"
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True)

    for split in ("train", "val", "test"):
        chunk = output_root / "audio" / split / "1_00000.flac"
        chunk.parent.mkdir(parents=True)
        chunk.touch()
        (manifest_dir / f"{split}.jsonl").write_text(
            json.dumps({"track_id": f"{split}-1", "path": chunk.relative_to(output_root).as_posix()})
            + "\n",
            encoding="utf-8",
        )

    counts, prepared = ensure_preprocessed_dataset(
        {
            "root": str(output_root),
            "manifest_dir": str(manifest_dir),
            "sample_rate": 24000,
            "duration_seconds": 3.0,
            "channels": 1,
            "preprocessing": {
                "source_root": str(tmp_path / "source"),
                "source_manifest_dir": str(tmp_path / "source" / "manifests"),
                "workers": 1,
                "drop_last": True,
                "overwrite": False,
            },
        }
    )

    assert counts == {"train": 1, "val": 1, "test": 1}
    assert prepared == []


def test_preprocess_cli_supports_config_mode(monkeypatch, capsys):
    config = {"data": {"sentinel": True}}
    monkeypatch.setattr(preprocess_cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        preprocess_cli,
        "ensure_preprocessed_dataset",
        lambda data: ({"train": 7, "val": 1, "test": 2}, ["train", "val", "test"]),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["ae-preprocess-audio", "--config", "configs/smoke_test.yaml"],
    )

    preprocess_cli.main()

    output = capsys.readouterr().out
    assert "prepared train, val, test" in output
    assert "train=7, val=1, test=2" in output
