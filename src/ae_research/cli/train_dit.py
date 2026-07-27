from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the downstream text-conditioned audio DiT."
    )
    parser.add_argument("--config", required=True, help="DiT YAML config path")
    parser.add_argument("--device", help="Example: cuda, cuda:1, or cpu")
    args = parser.parse_args()

    # These imports are deliberately lazy. This command never downloads data,
    # preprocesses audio, or runs T5Gemma; all text caches must already exist.
    from ae_research.diffusion.config import load_dit_config
    from ae_research.diffusion.trainer import DiTTrainer

    config = load_dit_config(args.config)
    DiTTrainer(config, device=args.device).train()


if __name__ == "__main__":
    main()
