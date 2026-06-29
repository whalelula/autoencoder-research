from __future__ import annotations

import argparse

from ae_research.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen-MERT autoencoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", help="Example: cuda, cuda:1, or cpu")
    args = parser.parse_args()
    from ae_research.training import Trainer

    Trainer(load_config(args.config), device=args.device).train()


if __name__ == "__main__":
    main()
