from __future__ import annotations

import argparse
from pathlib import Path

from ae_research.data.sampling import write_sample_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic sampled manifest for listening comparisons."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    output_manifest = write_sample_manifest(
        args.source_manifest,
        args.output_manifest_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        split=args.split,
    )
    print(f"Wrote sampled manifest: {output_manifest}")


if __name__ == "__main__":
    main()
