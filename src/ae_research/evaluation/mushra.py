from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio


def _load_at_rate(path: Path, sample_rate: int) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(path)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform


def _save(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(path, waveform.clamp(-1, 1), sample_rate)


def prepare_mushra(
    reference_dir: str | Path,
    systems: dict[str, str | Path],
    output_dir: str | Path,
    *,
    sample_rate: int,
    seed: int = 42,
    max_trials: int | None = None,
) -> None:
    """Create blinded MUSHRA stimuli, low/mid anchors, manifest, and score sheet."""
    reference_dir = Path(reference_dir)
    output_dir = Path(output_dir)
    stimuli_root = output_dir / "stimuli"
    references = sorted(reference_dir.glob("*.wav"))
    if max_trials is not None:
        references = references[:max_trials]
    if not references:
        raise ValueError(f"No WAV references found in {reference_dir}")
    if not systems:
        raise ValueError("At least one --system name=directory is required")

    rng = random.Random(seed)
    public_trials = []
    secret_key: dict[str, dict[str, str]] = {}
    score_rows = []
    for trial_index, reference_path in enumerate(references):
        trial_id = f"trial_{trial_index:04d}"
        trial_dir = stimuli_root / trial_id
        reference = _load_at_rate(reference_path, sample_rate)
        explicit_reference = trial_dir / "reference.wav"
        _save(explicit_reference, reference, sample_rate)

        named_stimuli: list[tuple[str, torch.Tensor]] = [
            ("hidden_reference", reference),
            (
                "anchor_7khz",
                torchaudio.functional.lowpass_biquad(
                    reference, sample_rate, min(7000.0, sample_rate * 0.45)
                ),
            ),
            (
                "anchor_3.5khz",
                torchaudio.functional.lowpass_biquad(
                    reference, sample_rate, min(3500.0, sample_rate * 0.45)
                ),
            ),
        ]
        for system_name, system_dir_value in systems.items():
            candidate = Path(system_dir_value) / reference_path.name
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"System {system_name!r} has no file matching {reference_path.name}"
                )
            waveform = _load_at_rate(candidate, sample_rate)
            target_length = reference.shape[-1]
            if waveform.shape[-1] > target_length:
                waveform = waveform[..., :target_length]
            elif waveform.shape[-1] < target_length:
                waveform = torch.nn.functional.pad(
                    waveform, (0, target_length - waveform.shape[-1])
                )
            named_stimuli.append((system_name, waveform))
        rng.shuffle(named_stimuli)

        public_stimuli = []
        trial_key = {}
        for stimulus_index, (system_name, waveform) in enumerate(named_stimuli):
            stimulus_id = f"stimulus_{stimulus_index:02d}"
            relative_path = Path("stimuli") / trial_id / f"{stimulus_id}.wav"
            _save(output_dir / relative_path, waveform, sample_rate)
            public_stimuli.append(
                {"stimulus_id": stimulus_id, "path": relative_path.as_posix()}
            )
            trial_key[stimulus_id] = system_name
            score_rows.append(
                {
                    "listener_id": "",
                    "trial_id": trial_id,
                    "stimulus_id": stimulus_id,
                    "score": "",
                }
            )
        public_trials.append(
            {
                "trial_id": trial_id,
                "reference": explicit_reference.relative_to(output_dir).as_posix(),
                "stimuli": public_stimuli,
            }
        )
        secret_key[trial_id] = trial_key

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"trials": public_trials}, indent=2), encoding="utf-8"
    )
    (output_dir / "key.json").write_text(
        json.dumps(secret_key, indent=2), encoding="utf-8"
    )
    with (output_dir / "scores_template.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("listener_id", "trial_id", "stimulus_id", "score")
        )
        writer.writeheader()
        writer.writerows(score_rows)
    (output_dir / "README.txt").write_text(
        "Distribute manifest.json and stimuli/ without key.json. Each trained listener "
        "rates every blinded stimulus from 0 to 100 while the explicit reference remains "
        "available. Duplicate score rows per listener, fill listener_id/score, then combine "
        "them as scores.csv. The organizer retains key.json.\n",
        encoding="utf-8",
    )


def summarize_mushra(
    scores_path: str | Path,
    *,
    key_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, dict[str, float | int]]:
    scores_path = Path(scores_path)
    key_path = Path(key_path) if key_path else scores_path.parent / "key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    with scores_path.open("r", newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if not row.get("score", "").strip():
                continue
            score = float(row["score"])
            if not 0.0 <= score <= 100.0:
                raise ValueError(f"Score outside [0,100] at line {line_number}")
            try:
                system_name = key[row["trial_id"]][row["stimulus_id"]]
            except KeyError as exc:
                raise ValueError(f"Unknown blinded stimulus at line {line_number}") from exc
            grouped[system_name].append(score)
    if not grouped:
        raise ValueError(f"No completed scores in {scores_path}")

    summary = {}
    for system_name, values in sorted(grouped.items()):
        count = len(values)
        std = statistics.stdev(values) if count > 1 else 0.0
        summary[system_name] = {
            "n": count,
            "mean": statistics.mean(values),
            "std": std,
            "ci95": 1.96 * std / math.sqrt(count),
        }
    destination = Path(output_path) if output_path else scores_path.parent / "summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
