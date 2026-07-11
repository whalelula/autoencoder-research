from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torchaudio
from tqdm import tqdm

from ae_research.data.dataset import create_dataloader
from ae_research.data.sampling import sample_manifest_track_ids
from ae_research.evaluation.evaluator import _run_rfad
from ae_research.evaluation.sa3_same import _match_reference_format
from ae_research.losses import MultiResolutionSTFTLoss
from ae_research.metrics import (
    BANDWISE_SPECTRAL_METRIC_NAMES,
    BandwiseSpectralErrors,
    LogMelL1,
    si_sdr,
)


def _load_stable_audio_pretransform(
    pretrained_name: str,
    device: torch.device,
    *,
    half: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
        from stable_audio_tools import get_pretrained_model
        from stable_audio_tools.models.factory import create_pretransform_from_config
        from stable_audio_tools.models.pretrained import (
            create_model_from_config,
            load_ckpt_state_dict,
        )
    except ImportError as exc:
        raise RuntimeError(
            "stable-audio-tools is required for Stable Audio VAE evaluation. "
            "Install Stability-AI/stable-audio-tools in this environment, or run "
            "this evaluator from a separate environment with this project installed editable."
        ) from exc

    pretrained_path = Path(pretrained_name)
    if pretrained_path.exists():
        config_path = pretrained_path / "model_config.json"
        if not config_path.exists():
            raise RuntimeError(f"Missing model_config.json in {pretrained_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        for filename in ("model.safetensors", "model.ckpt"):
            checkpoint_path = pretrained_path / filename
            if checkpoint_path.exists():
                break
        else:
            raise RuntimeError(
                f"Missing model.safetensors or model.ckpt in {pretrained_path}"
            )
    else:
        config_path = Path(
            hf_hub_download(pretrained_name, filename="model_config.json", repo_type="model")
        )
        with config_path.open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        try:
            checkpoint_path = Path(
                hf_hub_download(
                    pretrained_name, filename="model.safetensors", repo_type="model"
                )
            )
        except Exception:
            checkpoint_path = Path(
                hf_hub_download(pretrained_name, filename="model.ckpt", repo_type="model")
            )

    pretransform_config = model_config.get("model", {}).get("pretransform")
    if pretransform_config is not None:
        autoencoder = create_pretransform_from_config(
            pretransform_config, sample_rate=int(model_config["sample_rate"])
        )
        state_dict = load_ckpt_state_dict(str(checkpoint_path))
        pretransform_model_prefix = "pretransform.model."
        pretransform_state_dict = {
            key.removeprefix(pretransform_model_prefix): value
            for key, value in state_dict.items()
            if key.startswith(pretransform_model_prefix)
        }
        if not pretransform_state_dict:
            raise RuntimeError(
                f"No pretransform weights found in checkpoint: {checkpoint_path}"
            )
        missing, unexpected = autoencoder.model.load_state_dict(
            pretransform_state_dict, strict=False
        )
        if missing:
            raise RuntimeError(f"Missing pretransform checkpoint keys: {missing[:10]}")
        unexpected_without_loss = [
            key for key in unexpected if not key.startswith("loss.")
        ]
        if unexpected_without_loss:
            raise RuntimeError(
                "Unexpected pretransform checkpoint keys: "
                f"{unexpected_without_loss[:10]}"
            )
    else:
        if pretrained_path.exists():
            model = create_model_from_config(model_config)
            model.load_state_dict(load_ckpt_state_dict(str(checkpoint_path)))
        else:
            model, model_config = get_pretrained_model(pretrained_name)
        model = model.to(device)
        model.eval()
        if half:
            model = model.half()
        pretransform = getattr(model, "pretransform", None)
        autoencoder = pretransform if pretransform is not None else model
        if not hasattr(autoencoder, "encode") or not hasattr(autoencoder, "decode"):
            raise RuntimeError(
                f"Model {pretrained_name!r} does not expose an encode/decode "
                "autoencoder or pretransform."
            )

    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    if half:
        autoencoder = autoencoder.half()
    return autoencoder, model_config


def _model_sample_rate(model_config: dict[str, Any], autoencoder: torch.nn.Module) -> int:
    value = model_config.get("sample_rate", getattr(autoencoder, "sample_rate", None))
    if value is None:
        raise RuntimeError("Could not determine Stable Audio model sample rate")
    return int(value)


def _model_channels(model_config: dict[str, Any], autoencoder: torch.nn.Module) -> int:
    value = getattr(autoencoder, "io_channels", None)
    if value is None:
        value = model_config.get("audio_channels", model_config.get("io_channels"))
    if value is None:
        raise RuntimeError("Could not determine Stable Audio autoencoder channel count")
    return int(value)


def _prepare_audio_dirs(
    output_dir: Path, system_name: str, export_audio: bool
) -> tuple[Path, Path]:
    reference_dir = output_dir / "reference"
    reconstruction_dir = output_dir / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)
    if export_audio:
        for directory in (reference_dir, reconstruction_dir):
            directory.mkdir(parents=True, exist_ok=True)
            for stale_file in directory.glob("*.wav"):
                stale_file.unlink()
    return reference_dir, reconstruction_dir


@torch.no_grad()
def evaluate_stable_audio_vae(
    *,
    data_root: str | Path,
    manifest_dir: str | Path | None = None,
    pretrained_name: str = "stabilityai/stable-audio-open-1.0",
    system_name: str = "stable-audio-open-1.0-vae-latent",
    device: str | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    sample_rate: int = 24_000,
    duration_seconds: float = 5.0,
    channels: int = 1,
    export_audio: bool = True,
    max_batches: int | None = None,
    max_audio_samples: int | None = None,
    sample_seed: int = 42,
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
    half: bool = False,
) -> dict[str, Any]:
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if run_rfad and not export_audio:
        raise ValueError("Audio export must be enabled to compute rFAD")

    output_path = Path(
        output_dir or "outputs/evaluation/stable_audio_open_1_0_vae_latent"
    )
    reference_dir, reconstruction_dir = _prepare_audio_dirs(
        output_path, system_name, export_audio
    )
    rfad_temp_dir: tempfile.TemporaryDirectory[str] | None = None
    rfad_reference_dir = reference_dir
    rfad_reconstruction_dir = reconstruction_dir
    if run_rfad and max_audio_samples is not None:
        rfad_temp_dir = tempfile.TemporaryDirectory(
            prefix="rfad_audio_", dir=str(output_path)
        )
        rfad_root = Path(rfad_temp_dir.name)
        rfad_reference_dir = rfad_root / "reference"
        rfad_reconstruction_dir = rfad_root / "reconstruction"
        rfad_reference_dir.mkdir(parents=True, exist_ok=True)
        rfad_reconstruction_dir.mkdir(parents=True, exist_ok=True)

    try:
        data_root = Path(data_root)
        manifest_dir = Path(manifest_dir) if manifest_dir is not None else data_root / "manifests"
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

        autoencoder, model_config = _load_stable_audio_pretransform(
            pretrained_name, selected_device, half=half
        )
        model_sample_rate = _model_sample_rate(model_config, autoencoder)
        model_channels = _model_channels(model_config, autoencoder)
        model_samples = round(model_sample_rate * float(duration_seconds))

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

        sums: defaultdict[str, float] = defaultdict(float)
        latent_shapes: set[tuple[int, ...]] = set()
        samples = 0
        exported = 0
        effective_max_batches = int(max_batches) if max_batches is not None else None

        for batch_index, batch in enumerate(tqdm(loader, desc="Stable Audio VAE evaluation")):
            if effective_max_batches is not None and batch_index >= int(effective_max_batches):
                break
            audio = batch["audio"].to(selected_device)
            current_batch_size = audio.shape[0]

            model_audio = _match_reference_format(
                audio,
                source_rate=sample_rate,
                target_rate=model_sample_rate,
                target_channels=model_channels,
                target_samples=model_samples,
            )
            if half:
                model_audio = model_audio.half()
            latents = autoencoder.encode(model_audio)
            latent_shapes.add(tuple(int(dim) for dim in latents.shape[1:]))
            decoded = autoencoder.decode(latents)
            reconstruction = _match_reference_format(
                decoded.float(),
                source_rate=model_sample_rate,
                target_rate=sample_rate,
                target_channels=channels,
                target_samples=target_samples,
            )

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
                sums[key] += value * current_batch_size

            for index, track_id in enumerate(batch["track_id"]):
                track_id = str(track_id)
                filename = f"{track_id}.wav"
                if run_rfad and max_audio_samples is not None:
                    torchaudio.save(
                        rfad_reference_dir / filename,
                        audio[index].cpu().clamp(-1, 1),
                        sample_rate,
                    )
                    torchaudio.save(
                        rfad_reconstruction_dir / filename,
                        reconstruction[index].cpu().clamp(-1, 1),
                        sample_rate,
                    )
                if not export_audio:
                    continue
                if export_track_ids is not None and track_id not in export_track_ids:
                    continue
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
                exported += 1
            samples += current_batch_size

        if samples == 0:
            raise RuntimeError("Test loader produced no batches")

        summary: dict[str, Any] = {
            "num_samples": samples,
            "num_exported_audio_samples": exported,
            "audio_sample_seed": int(sample_seed) if max_audio_samples is not None else None,
            **{key: value / samples for key, value in sorted(sums.items())},
            "rFAD": None,
            "MUSHRA": "pending_human_test",
        }
        for key in BANDWISE_SPECTRAL_METRIC_NAMES:
            summary.setdefault(key, None)
        if run_rfad:
            rfad_output_dir = output_path / f"rfad_{system_name}"
            rfad_output_dir.mkdir(parents=True, exist_ok=True)
            summary["rFAD"] = _run_rfad(
                rfad_reference_dir, rfad_reconstruction_dir, fad_model, rfad_output_dir
            )

        (output_path / "metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if max_audio_samples is None:
            rfad_command = f"fadtk vggish {reference_dir} {reconstruction_dir}"
        else:
            rfad_command = (
                "# rFAD used temporary full-set audio exports. To reproduce it "
                "manually, rerun without --max-audio-samples or export full audio dirs."
            )
        commands = (
            f"# rFAD\n{rfad_command}\n\n"
            "# After a downstream generator exists:\n"
            f"fadtk vggish {reference_dir} /path/to/generated  # gFAD\n"
            f"fadtk clap-laion-music {reference_dir} /path/to/generated  # FAD-CLAP\n"
            "# MuQ-Eval: run the official scorer over /path/to/generated.\n"
        )
        (output_path / "external_metrics_commands.txt").write_text(
            commands,
            encoding="utf-8",
        )
        stale_mushra_command = output_path / "mushra_command.txt"
        if stale_mushra_command.exists():
            stale_mushra_command.unlink()
        return summary
    finally:
        if rfad_temp_dir is not None:
            rfad_temp_dir.cleanup()
