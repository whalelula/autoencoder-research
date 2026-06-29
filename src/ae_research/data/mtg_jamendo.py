from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import requests
from tqdm import tqdm

DEFAULT_METADATA_URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/"
    "master/data/raw_30s.tsv"
)
JAMENDO_TRACKS_API = "https://api.jamendo.com/v3.0/tracks/"


@dataclass(frozen=True)
class Track:
    track_id: int
    artist_id: int
    album_id: int
    original_path: str
    duration: float
    tags: tuple[str, ...]
    path: str | None = None
    license_ccurl: str | None = None
    split: str | None = None


def _valid_audio_container(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        prefix = handle.read(4)
    suffix = path.suffix.lower()
    if suffix == ".flac":
        return prefix == b"fLaC"
    if suffix == ".ogg":
        return prefix == b"OggS"
    if suffix == ".mp3":
        return prefix[:3] == b"ID3" or (
            len(prefix) >= 2 and prefix[0] == 0xFF and prefix[1] & 0xE0 == 0xE0
        )
    return False


def _numeric_id(value: str) -> int:
    try:
        return int(value.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid MTG-Jamendo id: {value!r}") from exc


def parse_mtg_metadata(path: str | Path) -> list[Track]:
    """Parse MTG's variable-width TSV (all columns after DURATION are tags)."""
    tracks: list[Track] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if not header or [item.upper() for item in header[:5]] != [
            "TRACK_ID",
            "ARTIST_ID",
            "ALBUM_ID",
            "PATH",
            "DURATION",
        ]:
            raise ValueError(f"Unexpected MTG-Jamendo metadata header in {path}")
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) < 5:
                raise ValueError(f"Malformed row {line_number} in {path}: {row!r}")
            tracks.append(
                Track(
                    track_id=_numeric_id(row[0]),
                    artist_id=_numeric_id(row[1]),
                    album_id=_numeric_id(row[2]),
                    original_path=row[3],
                    duration=float(row[4]),
                    tags=tuple(tag for tag in row[5:] if tag),
                )
            )
    if not tracks:
        raise ValueError(f"No tracks found in {path}")
    return tracks


def select_tracks(
    tracks: Sequence[Track],
    num_tracks: int,
    seed: int,
    tags: Sequence[str] = (),
    match_all_tags: bool = False,
    min_duration: float = 0.0,
) -> list[Track]:
    if num_tracks <= 0:
        raise ValueError("num_tracks must be positive")
    wanted = set(tags)
    eligible = []
    for track in tracks:
        if track.duration < min_duration:
            continue
        present = set(track.tags)
        tag_match = wanted.issubset(present) if match_all_tags else bool(wanted & present)
        if wanted and not tag_match:
            continue
        eligible.append(track)
    if len(eligible) < num_tracks:
        raise ValueError(
            f"Requested {num_tracks} tracks but only {len(eligible)} match the filters"
        )
    rng = random.Random(seed)
    return rng.sample(eligible, num_tracks)


def split_tracks(
    tracks: Sequence[Track],
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    group_by_artist: bool = False,
) -> dict[str, list[Track]]:
    """Make deterministic train/val/test splits.

    Track-level mode gives exact floor(0.7N), floor(0.1N), remainder counts.
    Artist-level mode is leakage-safe and greedily approximates those targets.
    """
    if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0, abs_tol=1e-8):
        raise ValueError("ratios must contain three values summing to 1")
    names = ("train", "val", "test")
    rng = random.Random(seed)

    if not group_by_artist:
        shuffled = list(tracks)
        rng.shuffle(shuffled)
        train_end = math.floor(len(shuffled) * ratios[0])
        val_end = train_end + math.floor(len(shuffled) * ratios[1])
        chunks = (shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:])
    else:
        groups: dict[int, list[Track]] = {}
        for track in tracks:
            groups.setdefault(track.artist_id, []).append(track)
        artist_groups = list(groups.values())
        rng.shuffle(artist_groups)
        artist_groups.sort(key=len, reverse=True)
        targets = [len(tracks) * ratio for ratio in ratios]
        assigned: list[list[Track]] = [[], [], []]
        for group in artist_groups:
            split_index = min(
                range(3),
                key=lambda index: (
                    len(assigned[index]) / max(targets[index], 1.0),
                    len(assigned[index]),
                ),
            )
            assigned[split_index].extend(group)
        chunks = tuple(assigned)

    return {
        name: [replace(track, split=name) for track in chunk]
        for name, chunk in zip(names, chunks)
    }


def download_with_resume(
    url: str,
    destination: str | Path,
    *,
    retries: int = 4,
    timeout: tuple[float, float] = (15.0, 120.0),
    progress: bool = True,
    session: requests.Session | None = None,
) -> Path:
    """Download through a persistent .part file and atomically rename on success."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    requester = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requester.get(url, headers=headers, stream=True, timeout=timeout) as response:
                if offset and response.status_code == 200:
                    print(
                        f"{destination.name}: server did not honor Range; "
                        "restarting from byte zero."
                    )
                    partial.unlink()
                    offset = 0
                elif offset and response.status_code != 206:
                    raise RuntimeError(
                        f"Server refused resume for {url} (HTTP {response.status_code})"
                    )
                response.raise_for_status()
                content_length = int(response.headers.get("content-length", 0))
                total = offset + content_length if content_length else None
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle, tqdm(
                    total=total,
                    initial=offset,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                    disable=not progress,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            bar.update(len(chunk))
            if not partial.exists() or partial.stat().st_size == 0:
                raise IOError(f"Downloaded file is empty: {url}")
            os.replace(partial, destination)
            return destination
        except (OSError, requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"Download failed after {retries} attempts: {url} -> {destination}. "
        f"Keep/remove {partial} before retrying as appropriate."
    ) from last_error


def fetch_jamendo_track(
    track: Track,
    client_id: str,
    audio_dir: str | Path,
    audio_format: str = "mp32",
) -> Track:
    response = requests.get(
        JAMENDO_TRACKS_API,
        params={
            "client_id": client_id,
            "format": "json",
            "id": track.track_id,
            "audioformat": audio_format,
            "audiodlformat": audio_format,
        },
        timeout=(15, 60),
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    if not results:
        raise RuntimeError(f"Jamendo API returned no result for track {track.track_id}")
    result = results[0]
    url = result.get("audiodownload") or result.get("audio")
    if not url:
        raise RuntimeError(f"No downloadable audio URL for track {track.track_id}")
    suffix = {"mp31": ".mp3", "mp32": ".mp3", "ogg": ".ogg", "flac": ".flac"}[audio_format]
    destination = Path(audio_dir) / f"{track.track_id}{suffix}"
    if destination.exists() and not _valid_audio_container(destination):
        destination.unlink()
    if not destination.exists():
        download_with_resume(url, destination, progress=False)
    if not _valid_audio_container(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded audio for track {track.track_id} has an invalid {suffix} container"
        )
    return replace(
        track,
        path=str(destination),
        license_ccurl=result.get("license_ccurl"),
    )


def fetch_jamendo_tracks(
    tracks: Sequence[Track],
    client_id: str,
    audio_dir: str | Path,
    audio_format: str = "mp32",
    workers: int = 4,
) -> tuple[list[Track], list[tuple[Track, str]]]:
    completed: list[Track] = []
    failures: list[tuple[Track, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(fetch_jamendo_track, track, client_id, audio_dir, audio_format): track
            for track in tracks
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading selected tracks"
        ):
            track = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # keep other downloads useful and report every failure
                failures.append((track, str(exc)))
    completed.sort(key=lambda item: item.track_id)
    return completed, failures


def write_manifests(
    splits: dict[str, Sequence[Track]],
    manifest_dir: str | Path,
    root: str | Path | None = None,
) -> None:
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    root_path = Path(root).resolve() if root is not None else None
    for split_name, tracks in splits.items():
        output = manifest_dir / f"{split_name}.jsonl"
        temporary = output.with_suffix(output.suffix + ".part")
        with temporary.open("w", encoding="utf-8") as handle:
            for track in tracks:
                record = asdict(track)
                record["tags"] = list(track.tags)
                if track.path and root_path:
                    resolved = Path(track.path).resolve()
                    try:
                        record["path"] = resolved.relative_to(root_path).as_posix()
                    except ValueError:
                        record["path"] = str(resolved)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
