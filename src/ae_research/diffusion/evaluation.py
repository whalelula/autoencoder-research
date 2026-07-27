from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import torchaudio
from tqdm import tqdm

from .codec import load_frozen_codec
from .data import create_dit_dataloader
from .dit import AudioDiffusionTransformer
from .flow_matching import euler_sample


def run_gfad(
    reference_dir: str | Path,
    generated_dir: str | Path,
    *,
    model_name: str,
    output_path: str | Path,
) -> float:
    """Run FADtk on real test audio versus unpaired generated audio."""
    executable = shutil.which("fadtk")
    if executable is None:
        raise RuntimeError(
            'fadtk executable not found; install the evaluation extra with "pip '
            'install -e .[eval]"'
        )
    command = [
        executable,
        str(model_name),
        str(Path(reference_dir)),
        str(Path(generated_dir)),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + "\n" + result.stderr
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"FADtk failed with exit code {result.returncode}; see {output_path}"
        )
    candidates: list[float] = []
    for line in combined.splitlines():
        if "fad" not in line.lower():
            continue
        candidates.extend(
            float(value)
            for value in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", line)
        )
    if not candidates:
        raise RuntimeError(f"Could not parse a FAD value from FADtk output: {output_path}")
    return candidates[-1]


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config["conditioning"].get("text_encoder", {})
    return value if isinstance(value, dict) else {}


def _manifest_dir(data_config: dict[str, Any]) -> Path:
    value = (
        data_config.get("dit_manifest_dir")
        or data_config.get("text_manifest_dir")
        or data_config.get("manifest_dir")
    )
    return Path(value)


def _embedding_dir(
    data_config: dict[str, Any], text_config: dict[str, Any]
) -> Path:
    value = text_config.get("cache_dir") or data_config.get("text_embedding_dir")
    if value is None:
        value = _manifest_dir(data_config).parent / "text_embeddings"
    return Path(value)


def _sampling_options(config: dict[str, Any]) -> dict[str, Any]:
    diffusion = config["diffusion"]
    evaluation = config["evaluation"]
    sampling = diffusion.get("sampling", {})
    if not isinstance(sampling, dict):
        sampling = {}
    guidance = diffusion.get("internal_guidance", {})
    if not isinstance(guidance, dict):
        guidance = {}
    interval = guidance.get("interval", sampling.get("guidance_interval", [0.0, 1.0]))
    return {
        "steps": int(evaluation.get("sampling_steps", sampling.get("steps", 50))),
        "t_eps": float(diffusion.get("t_eps", 1e-5)),
        "guidance_scale": float(
            evaluation.get(
                "internal_guidance_scale",
                guidance.get("scale", sampling.get("guidance_scale", 1.0)),
            )
        ),
        "guidance_interval": (float(interval[0]), float(interval[1])),
    }


@torch.no_grad()
def evaluate_dit(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
    run_gfad_metric: bool | None = None,
) -> dict[str, Any]:
    """Generate the complete test split and optionally compute gFAD."""
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data_config = config["data"]
    text_config = _text_config(config)
    evaluation = config["evaluation"]
    manifest_path = _manifest_dir(data_config) / "test.jsonl"
    embedding_dir = _embedding_dir(data_config, text_config)
    loader = create_dit_dataloader(
        manifest_path,
        data_config,
        text_embedding_dir=embedding_dir,
        batch_size=int(evaluation.get("batch_size", 8)),
        split="test",
        expected_text_model=text_config.get("model_name"),
        expected_text_dim=config["dit"].get("text_dim"),
        expected_text_max_length=text_config.get("max_length"),
        shuffle=False,
    )

    codec = load_frozen_codec(
        config["autoencoder"],
        expected_sample_rate=int(data_config["sample_rate"]),
        expected_channels=int(data_config["channels"]),
        map_location=selected_device,
    ).to(selected_device)
    model_config = dict(config["dit"])
    model_config.setdefault("text_dim", int(loader.dataset.text_embedding_dim))
    model = AudioDiffusionTransformer.from_config(
        model_config, latent_dim=int(codec.latent_dim)
    ).to(selected_device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = Path(evaluation["output_dir"])
    reference_dir = output_dir / "reference"
    generated_dir = output_dir / "generated"
    for directory in (reference_dir, generated_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.wav"):
            stale.unlink()
    prompt_records = []
    sample_rate = int(data_config["sample_rate"])
    generator = torch.Generator(device=selected_device).manual_seed(
        int(evaluation.get("seed", config.get("seed", 42)))
    )
    options = _sampling_options(config)
    generated_count = 0
    max_batches = evaluation.get("max_batches")
    for batch_index, batch in enumerate(tqdm(loader, desc="DiT test generation")):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        audio = batch["audio"].to(selected_device, non_blocking=True)
        clean_latent = codec.encode(audio)
        text_embedding = batch["text_embedding"].to(selected_device)
        text_mask = batch["text_mask"].to(selected_device)
        duration = batch["duration"].to(selected_device)
        generated_latent = euler_sample(
            model,
            shape=clean_latent.shape,
            duration=duration,
            text_embedding=text_embedding,
            text_mask=text_mask,
            generator=generator,
            **options,
        )
        generated = codec.decode(
            generated_latent, target_num_samples=audio.shape[-1]
        )
        for item_index in range(audio.shape[0]):
            track_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", batch["track_id"][item_index])
            filename = f"{generated_count:06d}_{track_id}.wav"
            torchaudio.save(
                reference_dir / filename,
                audio[item_index].detach().cpu().float().clamp(-1, 1),
                sample_rate,
            )
            torchaudio.save(
                generated_dir / filename,
                generated[item_index].detach().cpu().float().clamp(-1, 1),
                sample_rate,
            )
            prompt_records.append(
                {
                    "filename": filename,
                    "track_id": batch["track_id"][item_index],
                    "source_track_id": batch["source_track_id"][item_index],
                    "text": batch["text"][item_index],
                }
            )
            generated_count += 1
    if generated_count == 0:
        raise RuntimeError("Test loader produced no generated samples")
    with (output_dir / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for record in prompt_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    gfad_config = evaluation.get("gfad", {})
    if isinstance(gfad_config, bool):
        gfad_config = {"enabled": gfad_config}
    if run_gfad_metric is None:
        run_gfad_metric = bool(gfad_config.get("enabled", True))
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "num_generated": generated_count,
        "reference_dir": str(reference_dir),
        "generated_dir": str(generated_dir),
        "gFAD": None,
    }
    if run_gfad_metric:
        fad_model = str(gfad_config.get("model", evaluation.get("fad_model", "vggish")))
        summary["gFAD"] = run_gfad(
            reference_dir,
            generated_dir,
            model_name=fad_model,
            output_path=output_dir / f"gfad_{fad_model}.txt",
        )
        summary["gFAD_model"] = fad_model
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["evaluate_dit", "run_gfad"]
