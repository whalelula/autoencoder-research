from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from ae_research.data.mtg_jamendo import (
    DEFAULT_METADATA_URL,
    download_with_resume,
    fetch_jamendo_tracks,
    parse_mtg_metadata,
    select_tracks,
    split_tracks,
    write_manifests,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample MTG-Jamendo metadata, download selected audio, and split 7:1:2."
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--metadata-url", default=DEFAULT_METADATA_URL)
    parser.add_argument("--num-tracks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--match-all-tags", action="store_true")
    parser.add_argument("--min-duration", type=float, default=30.0)
    parser.add_argument("--client-id", default=os.getenv("JAMENDO_CLIENT_ID"))
    parser.add_argument("--audio-format", choices=("mp31", "mp32", "ogg", "flac"), default="mp32")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Use files from an already-unpacked official archive instead of the Jamendo API.",
    )
    parser.add_argument("--group-by-artist", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write selected.jsonl but do not create train/val/test manifests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.output_root
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata or metadata_dir / "raw_30s.tsv"
    if not metadata_path.exists():
        print(f"Downloading metadata: {args.metadata_url}")
        download_with_resume(args.metadata_url, metadata_path)

    selected = select_tracks(
        parse_mtg_metadata(metadata_path),
        args.num_tracks,
        args.seed,
        args.tag,
        args.match_all_tags,
        args.min_duration,
    )

    if args.metadata_only:
        write_manifests({"selected": selected}, root / "manifests", root=root)
        print(f"Selected {len(selected)} tracks; metadata-only manifest written.")
        return

    if args.audio_root:
        available = []
        missing = []
        for track in selected:
            path = args.audio_root / track.original_path
            if path.is_file():
                available.append(replace(track, path=str(path.resolve())))
            else:
                missing.append(path)
        if missing:
            preview = "\n".join(str(path) for path in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} selected archive files are missing. First examples:\n{preview}"
            )
        downloaded = available
    else:
        if not args.client_id:
            raise SystemExit(
                "Jamendo client id required. Set JAMENDO_CLIENT_ID, pass --client-id, "
                "or use --audio-root/--metadata-only."
            )
        downloaded, failures = fetch_jamendo_tracks(
            selected,
            args.client_id,
            root / "audio",
            args.audio_format,
            args.workers,
        )
        if failures:
            failure_path = root / "metadata" / "download_failures.tsv"
            with failure_path.open("w", encoding="utf-8") as handle:
                for track, message in failures:
                    handle.write(f"{track.track_id}\t{message}\n")
            print(f"Warning: {len(failures)} downloads failed; details: {failure_path}")
        if not downloaded:
            raise RuntimeError("Every selected audio download failed; no manifests were written")

    splits = split_tracks(downloaded, args.seed, group_by_artist=args.group_by_artist)
    write_manifests(splits, root / "manifests", root=root)
    counts = ", ".join(f"{name}={len(items)}" for name, items in splits.items())
    print(f"Prepared {len(downloaded)} tracks ({counts}) in {root.resolve()}")


if __name__ == "__main__":
    main()

