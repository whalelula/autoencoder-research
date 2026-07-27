from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ae_research.diffusion.text import (
    DEFAULT_T5GEMMA_MODEL,
    DEFAULT_TEXT_MAX_LENGTH,
    FrozenT5GemmaEncoder,
    prepare_dit_text_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create text-conditioned DiT manifests and precompute frozen "
            "T5Gemma tag embeddings."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", help="Text encoder device, e.g. cuda, cuda:1, or cpu")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only an already-cached T5Gemma tokenizer/model; never access the network.",
    )
    parser.add_argument(
        "--manifests-only",
        action="store_true",
        help="Write normalized text manifests without importing/loading Transformers.",
    )
    return parser


def _text_encoder_config(config: dict[str, Any]) -> dict[str, Any]:
    direct = config.get("text_encoder")
    if isinstance(direct, dict):
        return direct
    conditioning = config.get("conditioning")
    if isinstance(conditioning, dict) and isinstance(conditioning.get("text_encoder"), dict):
        return conditioning["text_encoder"]
    return {}


def _preparation_paths(
    config: dict[str, Any], text_config: dict[str, Any]
) -> tuple[Path, Path, Path]:
    data_config = config["data"]
    manifest_dir = Path(data_config["manifest_dir"])
    output_manifest_value = (
        data_config.get("dit_manifest_dir")
        or data_config.get("text_manifest_dir")
        or text_config.get("manifest_dir")
        or manifest_dir.parent / "dit_manifests"
    )
    cache_value = (
        text_config.get("cache_dir")
        or data_config.get("text_embedding_dir")
        or manifest_dir.parent / "text_embeddings"
    )
    return manifest_dir, Path(output_manifest_value), Path(cache_value)


def main() -> None:
    args = build_parser().parse_args()
    # Kept lazy so importing this CLI does not couple basic data tests to the DiT
    # config module while that module evolves independently.
    from ae_research.diffusion.config import load_dit_config

    config = load_dit_config(args.config)
    text_config = _text_encoder_config(config)
    manifest_dir, output_manifest_dir, cache_dir = _preparation_paths(
        config, text_config
    )
    model_name = str(text_config.get("model_name", DEFAULT_T5GEMMA_MODEL))
    max_length = int(text_config.get("max_length", DEFAULT_TEXT_MAX_LENGTH))
    batch_size = int(text_config.get("batch_size", 8))
    local_files_only = bool(
        args.local_files_only or text_config.get("local_files_only", False)
    )

    encoder = None
    if not args.manifests_only:
        encoder = FrozenT5GemmaEncoder(
            model_name,
            max_length=max_length,
            device=args.device,
            local_files_only=local_files_only,
        )
    result = prepare_dit_text_data(
        manifest_dir,
        output_manifest_dir,
        cache_dir,
        encoder=encoder,
        manifests_only=args.manifests_only,
        batch_size=batch_size,
    )
    result.update(
        {
            "source_manifest_dir": str(manifest_dir),
            "output_manifest_dir": str(output_manifest_dir),
            "text_embedding_dir": str(cache_dir),
            "model_name": None if args.manifests_only else model_name,
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
