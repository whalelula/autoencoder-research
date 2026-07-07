from __future__ import annotations

import argparse

from ae_research.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an audio autoencoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", help="Example: cuda, cuda:1, or cpu")
    args = parser.parse_args()
    config = load_config(args.config)

    from ae_research.data.preprocess import ensure_preprocessed_dataset

    counts, prepared = ensure_preprocessed_dataset(config["data"])
    action = f"prepared {', '.join(prepared)}" if prepared else "reused existing chunks"
    summary = ", ".join(f"{split}={count}" for split, count in counts.items())
    print(f"Offline dataset ready ({action}; {summary}).")

    from ae_research.training import Trainer

    Trainer(config, device=args.device).train()


if __name__ == "__main__":
    main()
