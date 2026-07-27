from __future__ import annotations

import argparse
import json

from ae_research.diffusion.config import load_dit_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the DiT test split and compute gFAD."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", help="Example: cuda, cuda:1, or cpu")
    gfad = parser.add_mutually_exclusive_group()
    gfad.add_argument("--run-gfad", action="store_true")
    gfad.add_argument("--skip-gfad", action="store_true")
    args = parser.parse_args()
    config = load_dit_config(args.config)
    run_gfad = True if args.run_gfad else False if args.skip_gfad else None

    from ae_research.diffusion.evaluation import evaluate_dit

    summary = evaluate_dit(
        config,
        args.checkpoint,
        device=args.device,
        run_gfad_metric=run_gfad,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
