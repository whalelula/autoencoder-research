from __future__ import annotations

import argparse
import json
from pathlib import Path


def _system(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--system must use name=/path/to/wavs")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--system must use name=/path/to/wavs")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or summarize a MUSHRA test.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--reference-dir", type=Path, required=True)
    prepare.add_argument("--system", action="append", type=_system, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--sample-rate", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-trials", type=int)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--scores", type=Path, required=True)
    summarize.add_argument("--key", type=Path)
    summarize.add_argument("--output", type=Path)
    args = parser.parse_args()
    from ae_research.evaluation.mushra import prepare_mushra, summarize_mushra

    if args.command == "prepare":
        systems = dict(args.system)
        if len(systems) != len(args.system):
            raise SystemExit("Every --system name must be unique")
        prepare_mushra(
            args.reference_dir,
            systems,
            args.output_dir,
            sample_rate=args.sample_rate,
            seed=args.seed,
            max_trials=args.max_trials,
        )
        print(f"MUSHRA package written to {args.output_dir.resolve()}")
    else:
        summary = summarize_mushra(
            args.scores, key_path=args.key, output_path=args.output
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
