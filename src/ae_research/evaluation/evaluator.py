from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio
from tqdm import tqdm

from ae_research.data.dataset import create_dataloader
from ae_research.losses import MultiResolutionSTFTLoss
from ae_research.metrics import LogMelL1, si_sdr
from ae_research.models import SemanticAudioAutoencoder


def _run_rfad(
    reference_dir: Path, reconstruction_dir: Path, model_name: str, output_dir: Path
) -> float | None:
    executable = shutil.which("fadtk")
    if executable is None:
        raise RuntimeError("fadtk executable not found; install the project with .[eval]")
    result = subprocess.run(
        [executable, model_name, str(reference_dir), str(reconstruction_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    text = result.stdout + "\n" + result.stderr
    (output_dir / f"rfad_{model_name}.txt").write_text(text, encoding="utf-8")
    candidates = []
    for line in text.splitlines():
        if "fad" in line.lower():
            candidates.extend(
                float(value)
                for value in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", line)
            )
    return candidates[-1] if candidates else None


@torch.no_grad()
def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
    run_rfad: bool = False,
    fad_model: str = "vggish",
) -> dict[str, Any]:
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data_config = config["data"]
    eval_config = config["evaluation"]
    output_dir = Path(eval_config["output_dir"])
    reference_dir = output_dir / "reference"
    reconstruction_dir = output_dir / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(eval_config["export_audio"]):
        reference_dir.mkdir(parents=True, exist_ok=True)
        reconstruction_dir.mkdir(parents=True, exist_ok=True)
        for generated_dir in (reference_dir, reconstruction_dir):
            for stale_file in generated_dir.glob("*.wav"):
                stale_file.unlink()

    loader = create_dataloader(
        Path(data_config["manifest_dir"]) / "test.jsonl",
        data_config,
        batch_size=int(eval_config["batch_size"]),
        split="test",
        shuffle=False,
    )
    model = SemanticAudioAutoencoder(
        config["model"],
        audio_channels=int(data_config["channels"]),
        data_sample_rate=int(data_config["sample_rate"]),
    ).to(selected_device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder"])
    if model.detail_aware is not None:
        if "detail_aware" not in checkpoint:
            raise KeyError(
                "Checkpoint has no Detail-Aware Module state. "
                "Evaluate with a DAM checkpoint or disable model.detail_aware."
            )
        model.detail_aware.load_state_dict(checkpoint["detail_aware"])
    model.eval()

    mrstft = MultiResolutionSTFTLoss(
        config["loss"]["fft_sizes"],
        sample_rate=int(data_config["sample_rate"]),
        hop_ratio=float(config["loss"]["hop_ratio"]),
        use_k_weighting=bool(config["loss"]["use_k_weighting"]),
        stereo_representations=bool(config["loss"]["stereo_representations"]),
        eps=float(config["loss"]["eps"]),
    ).to(selected_device)
    mel = LogMelL1(
        int(data_config["sample_rate"]),
        n_fft=int(eval_config["mel_n_fft"]),
        hop_length=int(eval_config["mel_hop_length"]),
        n_mels=int(eval_config["mel_n_mels"]),
    ).to(selected_device)

    sums: defaultdict[str, float] = defaultdict(float)
    batches = 0
    samples = 0
    max_batches = eval_config.get("max_batches")
    for batch_index, batch in enumerate(tqdm(loader, desc="test evaluation")):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        audio = batch["audio"].to(selected_device)
        outputs = model(audio)
        reconstruction = outputs["reconstruction"]
        spectral, components = mrstft(reconstruction, audio)
        batch_size = audio.shape[0]
        values = {
            "SI-SDR": float(si_sdr(reconstruction, audio)),
            "MEL": float(mel(reconstruction, audio)),
            "MR-STFT": float(spectral),
            **{f"MR-STFT/{key}": float(value) for key, value in components.items()},
        }
        for key, value in values.items():
            sums[key] += value * batch_size
        batches += 1
        samples += batch_size

        if bool(eval_config["export_audio"]):
            sample_rate = int(data_config["sample_rate"])
            for index, track_id in enumerate(batch["track_id"]):
                filename = f"{track_id}.wav"
                torchaudio.save(
                    reference_dir / filename,
                    audio[index].cpu().clamp(-1, 1),
                    sample_rate,
                )
                torchaudio.save(
                    reconstruction_dir / filename,
                    reconstruction[index].cpu().clamp(-1, 1),
                    sample_rate,
                )
    if batches == 0:
        raise RuntimeError("Test loader produced no batches")

    summary: dict[str, Any] = {
        "num_samples": samples,
        **{key: value / samples for key, value in sums.items()},
        "rFAD": None,
        "MUSHRA": "pending_human_test",
    }
    if run_rfad:
        if not bool(eval_config["export_audio"]):
            raise ValueError("evaluation.export_audio must be true to compute rFAD")
        summary["rFAD"] = _run_rfad(
            reference_dir, reconstruction_dir, fad_model, output_dir
        )

    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    commands = (
        f"# rFAD\nfadtk vggish {reference_dir} {reconstruction_dir}\n\n"
        "# After a downstream generator exists:\n"
        f"fadtk vggish {reference_dir} /path/to/generated  # gFAD\n"
        f"fadtk clap-laion-music {reference_dir} /path/to/generated  # FAD-CLAP\n"
        "# MuQ-Eval: run the official scorer over /path/to/generated.\n"
    )
    (output_dir / "external_metrics_commands.txt").write_text(
        commands, encoding="utf-8"
    )
    return summary
