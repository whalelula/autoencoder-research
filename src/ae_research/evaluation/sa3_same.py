from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torchaudio
from tqdm import tqdm

from ae_research.data.dataset import create_dataloader
from ae_research.data.sampling import sample_manifest_track_ids, write_sample_manifest
from ae_research.evaluation.evaluator import _run_rfad
from ae_research.losses import MultiResolutionSTFTLoss
from ae_research.metrics import (
    BANDWISE_SPECTRAL_METRIC_NAMES,
    BandwiseSpectralErrors,
    LogMelL1,
    si_sdr,
)

SA3_SAME_MODELS = ("same-s", "same-l")


def _load_sa3_autoencoder(model_name: str, device: torch.device):
    try:
        from stable_audio_3 import AutoencoderModel
    except ImportError as exc:
        raise RuntimeError(
            "stable-audio-3 is required for SAME-S/SAME-L evaluation. "
            "Install Stability-AI/stable-audio-3 in this environment, or run this "
            "baseline from a separate environment with this project installed editable."
        ) from exc
    return AutoencoderModel.from_pretrained(model_name, device=str(device))


def _match_reference_format(
    waveform: torch.Tensor,
    *,
    source_rate: int,
    target_rate: int,
    target_channels: int,
    target_samples: int,
) -> torch.Tensor:
    if waveform.ndim != 3:
        raise ValueError(f"Expected waveform shape [B, C, T], got {waveform.shape}")
    if waveform.shape[1] != target_channels:
        if target_channels == 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        elif waveform.shape[1] == 1:
            waveform = waveform.repeat(1, target_channels, 1)
        else:
            raise ValueError(
                f"Cannot convert {waveform.shape[1]} channels to {target_channels}"
            )
    if source_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
    if waveform.shape[-1] > target_samples:
        waveform = waveform[..., :target_samples]
    elif waveform.shape[-1] < target_samples:
        waveform = torch.nn.functional.pad(
            waveform, (0, target_samples - waveform.shape[-1])
        )
    return waveform


def _prepare_audio_dirs(
    output_dir: Path, model_names: Sequence[str], export_audio: bool
) -> tuple[Path, dict[str, Path]]:
    reference_dir = output_dir / "reference"
    reconstruction_dirs = {name: output_dir / name for name in model_names}
    output_dir.mkdir(parents=True, exist_ok=True)
    if export_audio:
        for directory in (reference_dir, *reconstruction_dirs.values()):
            directory.mkdir(parents=True, exist_ok=True)
            for stale_file in directory.glob("*.wav"):
                stale_file.unlink()
    return reference_dir, reconstruction_dirs


@torch.no_grad()
def evaluate_sa3_same(
    *,
    data_root: str | Path,
    manifest_dir: str | Path | None = None,
    model_names: Sequence[str] = SA3_SAME_MODELS,
    device: str | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    sample_rate: int = 24_000,
    duration_seconds: float = 5.0,
    channels: int = 1,
    export_audio: bool = True,
    max_batches: int | None = None,
    max_audio_samples: int | None = None,
    sample_count: int | None = None,
    sample_seed: int = 42,
    sample_manifest_dir: str | Path | None = None,
    run_rfad: bool = False,
    fad_model: str = "vggish",
    mel_n_fft: int = 1024,
    mel_hop_length: int = 256,
    mel_n_mels: int = 128,
    fft_sizes: Sequence[int] = (32, 64, 128, 256, 512, 1024, 2048),
    hop_ratio: float = 0.25,
    use_k_weighting: bool = True,
    stereo_representations: bool = True,
    eps: float = 1.0e-7,
    chunked: bool = False,
    chunk_size: int = 128,
    overlap: int = 32,
) -> dict[str, Any]:
    invalid = sorted(set(model_names) - set(SA3_SAME_MODELS))
    if invalid:
        raise ValueError(f"Unknown SA3 SAME model(s): {invalid}")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if run_rfad and not export_audio:
        raise ValueError("Audio export must be enabled to compute rFAD")
    if run_rfad and max_audio_samples is not None:
        raise ValueError("--run-rfad cannot be combined with --max-audio-samples")
    output_path = Path(output_dir or "outputs/evaluation/sa3_same")
    reference_dir, reconstruction_dirs = _prepare_audio_dirs(
        output_path, model_names, export_audio
    )

    data_root = Path(data_root)
    manifest_dir = Path(manifest_dir) if manifest_dir is not None else data_root / "manifests"
    sampled_manifest_dir = None
    if sample_count is not None:
        sampled_manifest_dir = Path(sample_manifest_dir or output_path / "sample_manifest")
        write_sample_manifest(
            manifest_dir / "test.jsonl",
            sampled_manifest_dir,
            sample_count=int(sample_count),
            seed=int(sample_seed),
            split="test",
        )
        manifest_dir = sampled_manifest_dir
    manifest_path = manifest_dir / "test.jsonl"
    export_track_ids = None
    if max_audio_samples is not None:
        export_track_ids = sample_manifest_track_ids(
            manifest_path,
            sample_count=int(max_audio_samples),
            seed=int(sample_seed),
        )
    data_config = {
        "root": str(data_root),
        "sample_rate": int(sample_rate),
        "duration_seconds": float(duration_seconds),
        "channels": int(channels),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    loader = create_dataloader(
        manifest_path,
        data_config,
        batch_size=int(batch_size),
        split="test",
        shuffle=False,
    )
    sample_rate = int(sample_rate)
    channels = int(channels)
    target_samples = round(sample_rate * float(duration_seconds))

    mrstft = MultiResolutionSTFTLoss(
        tuple(int(value) for value in fft_sizes),
        sample_rate=sample_rate,
        hop_ratio=float(hop_ratio),
        use_k_weighting=bool(use_k_weighting),
        stereo_representations=bool(stereo_representations),
        eps=float(eps),
    ).to(selected_device)
    mel = LogMelL1(
        sample_rate,
        n_fft=int(mel_n_fft),
        hop_length=int(mel_hop_length),
        n_mels=int(mel_n_mels),
    ).to(selected_device)
    band_errors = BandwiseSpectralErrors(
        sample_rate,
        n_fft=int(mel_n_fft),
        hop_length=int(mel_hop_length),
        n_mels=int(mel_n_mels),
    ).to(selected_device)

    autoencoders = {
        name: _load_sa3_autoencoder(name, selected_device) for name in model_names
    }
    sums: dict[str, defaultdict[str, float]] = {
        name: defaultdict(float) for name in model_names
    }
    samples = 0
    exported = 0
    effective_max_batches = int(max_batches) if max_batches is not None else None

    for batch_index, batch in enumerate(tqdm(loader, desc="SA3 SAME evaluation")):
        if effective_max_batches is not None and batch_index >= int(effective_max_batches):
            break
        audio = batch["audio"].to(selected_device)
        batch_size = audio.shape[0]
        reconstructions: dict[str, torch.Tensor] = {}

        for name, autoencoder in autoencoders.items():
            latents = autoencoder.encode(
                audio,
                sample_rate,
                chunked=chunked,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            decoded = autoencoder.decode(
                latents,
                chunked=chunked,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            reconstruction = _match_reference_format(
                decoded.float(),
                source_rate=int(autoencoder.sample_rate),
                target_rate=sample_rate,
                target_channels=channels,
                target_samples=target_samples,
            )
            reconstructions[name] = reconstruction

            spectral, components = mrstft(reconstruction, audio)
            values = {
                "SI-SDR": float(si_sdr(reconstruction, audio)),
                "MEL": float(mel(reconstruction, audio)),
                "MR-STFT": float(spectral),
                **{f"MR-STFT/{key}": float(value) for key, value in components.items()},
            }
            for key, value in band_errors(reconstruction, audio).items():
                values[key] = None if value is None else float(value)
            for key, value in values.items():
                if value is None:
                    continue
                sums[name][key] += value * batch_size

        if export_audio:
            for index, track_id in enumerate(batch["track_id"]):
                track_id = str(track_id)
                if export_track_ids is not None and track_id not in export_track_ids:
                    continue
                filename = f"{track_id}.wav"
                torchaudio.save(
                    reference_dir / filename,
                    audio[index].cpu().clamp(-1, 1),
                    sample_rate,
                )
                for name, reconstruction in reconstructions.items():
                    torchaudio.save(
                        reconstruction_dirs[name] / filename,
                        reconstruction[index].cpu().clamp(-1, 1),
                        sample_rate,
                    )
                exported += 1
        samples += batch_size

    if samples == 0:
        raise RuntimeError("Test loader produced no batches")

    summary: dict[str, Any] = {
        "num_samples": samples,
        "num_exported_audio_samples": exported,
        "sample_rate": sample_rate,
        "channels": channels,
        "manifest_dir": str(manifest_dir),
        "sample_count": sample_count,
        "sample_seed": int(sample_seed) if sample_count is not None else None,
        "audio_sample_seed": int(sample_seed) if max_audio_samples is not None else None,
        "models": {},
    }
    for name in model_names:
        model_summary = {
            key: value / samples for key, value in sorted(sums[name].items())
        }
        for key in BANDWISE_SPECTRAL_METRIC_NAMES:
            model_summary.setdefault(key, None)
        model_summary["rFAD"] = None
        if run_rfad:
            rfad_output_dir = output_path / f"rfad_{name}"
            rfad_output_dir.mkdir(parents=True, exist_ok=True)
            model_summary["rFAD"] = _run_rfad(
                reference_dir, reconstruction_dirs[name], fad_model, rfad_output_dir
            )
        summary["models"][name] = model_summary

    (output_path / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mushra_systems = " ".join(
        f"--system {name}={reconstruction_dirs[name]}" for name in model_names
    )
    (output_path / "mushra_command.txt").write_text(
        "ae-mushra prepare "
        f"--reference-dir {reference_dir} "
        f"{mushra_systems} "
        f"--output-dir {output_path / 'mushra'} "
        f"--sample-rate {sample_rate}\n",
        encoding="utf-8",
    )
    return summary
