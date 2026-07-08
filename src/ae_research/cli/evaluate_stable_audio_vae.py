from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stable Audio VAE/pretransform on a preprocessed test set."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Preprocessed dataset root, e.g. data/MTG-Jamendo-1000-24k-mono-5s.",
    )
    parser.add_argument(
        "--manifest-dir",
        help="Manifest directory. Defaults to DATA_ROOT/manifests.",
    )
    parser.add_argument(
        "--pretrained-name",
        default="stabilityai/stable-audio-open-1.0",
        help="Stable Audio Tools pretrained name or HF repo id.",
    )
    parser.add_argument("--system-name", default="stable-audio-vae")
    parser.add_argument("--device")
    parser.add_argument("--output-dir", default="outputs/evaluation/stable_audio_vae")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--no-export-audio", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--max-audio-samples",
        type=int,
        help="Limit exported listening WAVs while still computing metrics on all batches.",
    )
    parser.add_argument("--run-rfad", action="store_true")
    parser.add_argument("--fad-model", default="vggish")
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    from ae_research.evaluation.stable_audio_vae import evaluate_stable_audio_vae

    result = evaluate_stable_audio_vae(
        data_root=args.data_root,
        manifest_dir=args.manifest_dir,
        pretrained_name=args.pretrained_name,
        system_name=args.system_name,
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
        channels=args.channels,
        export_audio=not args.no_export_audio,
        max_batches=args.max_batches,
        max_audio_samples=args.max_audio_samples,
        run_rfad=args.run_rfad,
        fad_model=args.fad_model,
        half=args.half,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
