from __future__ import annotations

import argparse
import json

from ae_research.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a decoder checkpoint on test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-rfad", action="store_true")
    parser.add_argument("--fad-model", default="vggish")
    args = parser.parse_args()
    from ae_research.evaluation import evaluate_checkpoint

    result = evaluate_checkpoint(
        load_config(args.config),
        args.checkpoint,
        device=args.device,
        run_rfad=args.run_rfad,
        fad_model=args.fad_model,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
