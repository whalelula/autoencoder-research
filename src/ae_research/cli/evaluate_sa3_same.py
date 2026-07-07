from __future__ import annotations

import argparse
import json

SA3_SAME_MODELS = ("same-s", "same-l")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SA3 SAME-S/SAME-L on a preprocessed 24 kHz mono 5s test set."
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
    parser.add_argument("--device")
    parser.add_argument(
        "--model",
        action="append",
        choices=SA3_SAME_MODELS,
        help="SA3 SAME model to evaluate. Repeat to select multiple; defaults to both.",
    )
    parser.add_argument("--output-dir", default="outputs/evaluation/sa3_same")
    parser.add_argument("--batch-size", type=int, default=4)
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
    parser.add_argument(
        "--sample-count",
        type=int,
        help=(
            "Deterministically sample this many records from test.jsonl before "
            "evaluation and write the sampled manifest for reuse."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-manifest-dir",
        help="Where to write/read sampled test.jsonl. Defaults to OUTPUT_DIR/sample_manifest.",
    )
    parser.add_argument("--run-rfad", action="store_true")
    parser.add_argument("--fad-model", default="vggish")
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Use SA3 chunked encode/decode to reduce peak memory.",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    from ae_research.evaluation.sa3_same import evaluate_sa3_same

    result = evaluate_sa3_same(
        data_root=args.data_root,
        manifest_dir=args.manifest_dir,
        model_names=tuple(args.model or SA3_SAME_MODELS),
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
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
        sample_manifest_dir=args.sample_manifest_dir,
        run_rfad=args.run_rfad,
        fad_model=args.fad_model,
        chunked=args.chunked,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
