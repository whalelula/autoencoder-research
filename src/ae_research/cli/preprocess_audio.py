from __future__ import annotations

import argparse
from pathlib import Path

from ae_research.config import load_config
from ae_research.data.preprocess import preprocess_manifest
from ae_research.data.preprocess import ensure_preprocessed_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert manifest audio to fixed-length 24 kHz mono FLAC chunks."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=Path,
        help="Read all offline preprocessing settings from a training YAML config.",
    )
    source.add_argument("--input-root", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        help="Split to preprocess. Repeatable; defaults to train/val/test.",
    )
    parser.add_argument(
        "--keep-last",
        action="store_true",
        help="Pad and keep the final partial chunk instead of dropping it.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config is not None:
        if args.manifest_dir is not None or args.output_root is not None or args.split:
            parser.error(
                "--config cannot be combined with --manifest-dir, --output-root, or --split"
            )
        config = load_config(args.config)
        counts, prepared = ensure_preprocessed_dataset(config["data"])
        action = (
            f"prepared {', '.join(prepared)}"
            if prepared
            else "reused existing chunks"
        )
        summary = ", ".join(f"{split}={count}" for split, count in counts.items())
        print(f"Offline dataset ready ({action}; {summary}).")
        return

    if args.manifest_dir is None or args.output_root is None:
        parser.error(
            "--manifest-dir and --output-root are required with --input-root"
        )

    splits = args.split or ["train", "val", "test"]
    manifest_output = args.output_root / "manifests"
    counts = {}
    for split in splits:
        counts[split] = preprocess_manifest(
            args.manifest_dir / f"{split}.jsonl",
            input_root=args.input_root,
            output_root=args.output_root,
            output_manifest_dir=manifest_output,
            sample_rate=args.sample_rate,
            chunk_seconds=args.chunk_seconds,
            channels=args.channels,
            workers=args.workers,
            drop_last=not args.keep_last,
            overwrite=args.overwrite,
        )
    print(
        "Prepared FLAC chunks in "
        f"{args.output_root.resolve()}: "
        + ", ".join(f"{split}={count}" for split, count in counts.items())
    )


if __name__ == "__main__":
    main()
