from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from ae_research.config import load_config

SA3_SAME_MODELS = ("same-s", "same-l")
SAME_ALIASES = {"same", "sa3-same", "sa3_same"}
SAO_ALIASES = {
    "sao",
    "sao-vae",
    "sao_vae",
    "stable-audio-vae",
    "stable_audio_vae",
    "stable-audio-open-vae",
    "stable_audio_open_vae",
}
CHECKPOINT_ALIASES = {
    "checkpoint",
    "ours",
    "mert95",
    "mert95m",
    "mert330",
    "mert330m",
    "dam_mert95",
    "dam-mert95",
    "dam_mert95m",
    "dam-mert95m",
    "dam_mert330",
    "dam-mert330",
    "dam_mert330m",
    "dam-mert330m",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate project checkpoints, SA3 SAME baselines, or Stable Audio Open "
            "VAE baselines."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        help=(
            "Model/baseline to evaluate. Use checkpoint/dam_mert330 for a project "
            "checkpoint, same for SAME-S and SAME-L, same-s or same-l for one SAME "
            "variant, or sao for Stable Audio Open VAE. Repeat only for SAME variants."
        ),
    )

    checkpoint = parser.add_argument_group("project checkpoint models")
    checkpoint.add_argument("--config", help="Training/evaluation config for checkpoint models.")
    checkpoint.add_argument("--checkpoint", help="Checkpoint path for checkpoint models.")

    data = parser.add_argument_group("dataset")
    data.add_argument(
        "--data-root",
        help="Preprocessed dataset root, e.g. data/MTG-Jamendo-1000-24k-mono-5s.",
    )
    data.add_argument("--manifest-dir", help="Manifest directory. Defaults to DATA_ROOT/manifests.")
    data.add_argument("--sample-rate", type=int, help="Dataset sample rate.")
    data.add_argument("--duration-seconds", type=float, help="Dataset clip duration.")
    data.add_argument("--channels", type=int, help="Dataset audio channels.")

    runtime = parser.add_argument_group("runtime and output")
    runtime.add_argument("--device")
    runtime.add_argument("--output-dir", help="Evaluation output directory.")
    runtime.add_argument("--batch-size", type=int)
    runtime.add_argument("--num-workers", type=int)
    runtime.add_argument("--no-pin-memory", action="store_true")
    runtime.add_argument("--no-export-audio", action="store_true")
    runtime.add_argument("--max-batches", type=int)
    runtime.add_argument(
        "--max-audio-samples",
        type=int,
        help="Limit exported listening WAVs while still computing metrics on all batches.",
    )
    runtime.add_argument("--run-rfad", action="store_true")
    runtime.add_argument("--fad-model", default="vggish")

    same = parser.add_argument_group("SA3 SAME options")
    same.add_argument(
        "--sample-count",
        type=int,
        help=(
            "Deterministically sample this many records from test.jsonl before "
            "SAME evaluation and write the sampled manifest for reuse."
        ),
    )
    same.add_argument("--sample-seed", type=int, default=42)
    same.add_argument(
        "--sample-manifest-dir",
        help="Where to write/read sampled test.jsonl. Defaults to OUTPUT_DIR/sample_manifest.",
    )
    same.add_argument("--chunked", action="store_true", help="Use SA3 chunked encode/decode.")
    same.add_argument("--chunk-size", type=int, default=128)
    same.add_argument("--overlap", type=int, default=32)

    sao = parser.add_argument_group("Stable Audio Open VAE options")
    sao.add_argument(
        "--pretrained-name",
        default="stabilityai/stable-audio-open-1.0",
        help="Stable Audio Tools pretrained name, local directory, or HF repo id.",
    )
    sao.add_argument("--system-name", default="stable-audio-open-1.0-vae-latent")
    sao.add_argument("--half", action="store_true")
    return parser


def _normalise_models(models: Sequence[str] | None, args: argparse.Namespace) -> list[str]:
    if not models:
        if args.config or args.checkpoint:
            return ["checkpoint"]
        raise SystemExit(
            "ae-evaluate requires --model unless --config/--checkpoint imply checkpoint mode."
        )
    return [model.strip().lower() for model in models]


def _same_model_names(models: Sequence[str]) -> tuple[str, ...] | None:
    selected: list[str] = []
    for model in models:
        if model in SAME_ALIASES:
            selected.extend(SA3_SAME_MODELS)
        elif model in SA3_SAME_MODELS:
            selected.append(model)
        else:
            return None
    return tuple(dict.fromkeys(selected))


def _is_sao_model(models: Sequence[str]) -> bool:
    return len(models) == 1 and models[0] in SAO_ALIASES


def _is_checkpoint_model(models: Sequence[str]) -> bool:
    return len(models) == 1 and models[0] in CHECKPOINT_ALIASES


def _require_data_root(args: argparse.Namespace) -> str:
    if not args.data_root:
        raise SystemExit("--data-root is required for SAME and SAO baseline evaluation.")
    return args.data_root


def _apply_checkpoint_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    updated = deepcopy(config)
    data_config = updated["data"]
    eval_config = updated["evaluation"]
    if args.output_dir is not None:
        eval_config["output_dir"] = args.output_dir
    if args.batch_size is not None:
        eval_config["batch_size"] = int(args.batch_size)
    if args.max_batches is not None:
        eval_config["max_batches"] = int(args.max_batches)
    if args.no_export_audio:
        eval_config["export_audio"] = False
    if args.sample_rate is not None:
        data_config["sample_rate"] = int(args.sample_rate)
    if args.duration_seconds is not None:
        data_config["duration_seconds"] = float(args.duration_seconds)
    if args.channels is not None:
        data_config["channels"] = int(args.channels)
    if args.num_workers is not None:
        data_config["num_workers"] = int(args.num_workers)
    if args.no_pin_memory:
        data_config["pin_memory"] = False
    return updated


def _evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    if not args.config or not args.checkpoint:
        raise SystemExit("--config and --checkpoint are required for checkpoint models.")
    from ae_research.evaluation import evaluate_checkpoint

    config = _apply_checkpoint_overrides(load_config(args.config), args)
    return evaluate_checkpoint(
        config,
        Path(args.checkpoint),
        device=args.device,
        run_rfad=args.run_rfad,
        fad_model=args.fad_model,
    )


def _evaluate_same(args: argparse.Namespace, model_names: tuple[str, ...]) -> dict[str, Any]:
    data_root = _require_data_root(args)
    from ae_research.evaluation.sa3_same import evaluate_sa3_same

    return evaluate_sa3_same(
        data_root=data_root,
        manifest_dir=args.manifest_dir,
        model_names=model_names,
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size if args.batch_size is not None else 4,
        num_workers=args.num_workers if args.num_workers is not None else 4,
        pin_memory=not args.no_pin_memory,
        sample_rate=args.sample_rate if args.sample_rate is not None else 24_000,
        duration_seconds=args.duration_seconds if args.duration_seconds is not None else 5.0,
        channels=args.channels if args.channels is not None else 1,
        export_audio=not args.no_export_audio,
        max_batches=args.max_batches,
        max_audio_samples=args.max_audio_samples,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
        sample_manifest_dir=args.sample_manifest_dir,
        run_rfad=args.run_rfad,
        fad_model=args.fad_model,
        chunked=args.chunked,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


def _evaluate_sao(args: argparse.Namespace) -> dict[str, Any]:
    data_root = _require_data_root(args)
    from ae_research.evaluation.stable_audio_vae import evaluate_stable_audio_vae

    return evaluate_stable_audio_vae(
        data_root=data_root,
        manifest_dir=args.manifest_dir,
        pretrained_name=args.pretrained_name,
        system_name=args.system_name,
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size if args.batch_size is not None else 1,
        num_workers=args.num_workers if args.num_workers is not None else 4,
        pin_memory=not args.no_pin_memory,
        sample_rate=args.sample_rate if args.sample_rate is not None else 24_000,
        duration_seconds=args.duration_seconds if args.duration_seconds is not None else 5.0,
        channels=args.channels if args.channels is not None else 1,
        export_audio=not args.no_export_audio,
        max_batches=args.max_batches,
        max_audio_samples=args.max_audio_samples,
        run_rfad=args.run_rfad,
        fad_model=args.fad_model,
        half=args.half,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    models = _normalise_models(args.model, args)
    same_models = _same_model_names(models)
    if same_models is not None:
        result = _evaluate_same(args, same_models)
    elif _is_sao_model(models):
        result = _evaluate_sao(args)
    elif _is_checkpoint_model(models):
        result = _evaluate_checkpoint(args)
    else:
        valid = sorted(SAME_ALIASES | set(SA3_SAME_MODELS) | SAO_ALIASES | CHECKPOINT_ALIASES)
        raise SystemExit(
            f"Unknown or incompatible --model selection {models!r}. Valid values include: "
            f"{', '.join(valid)}"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
