from __future__ import annotations

import argparse
from pathlib import Path

from ae_research.data.preprocess import preprocess_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert manifest audio to fixed-length 24 kHz mono FLAC chunks."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    args = build_parser().parse_args()
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
