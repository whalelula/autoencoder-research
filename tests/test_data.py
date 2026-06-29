from __future__ import annotations

from ae_research.data.mtg_jamendo import (
    Track,
    parse_mtg_metadata,
    select_tracks,
    split_tracks,
)


def _tracks(count: int = 20) -> list[Track]:
    return [
        Track(
            track_id=index,
            artist_id=index // 2,
            album_id=index // 3,
            original_path=f"{index % 100:02d}/{index}.mp3",
            duration=30.0 + index,
            tags=("genre---electronic",) if index % 2 else ("instrument---piano",),
        )
        for index in range(count)
    ]


def test_parse_variable_width_metadata(tmp_path):
    metadata = tmp_path / "raw.tsv"
    metadata.write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "track_0001\tartist_002\talbum_03\t01/1.mp3\t31.5"
        "\tgenre---rock\tinstrument---guitar\n",
        encoding="utf-8",
    )
    track = parse_mtg_metadata(metadata)[0]
    assert track.track_id == 1
    assert track.artist_id == 2
    assert track.tags == ("genre---rock", "instrument---guitar")


def test_exact_7_1_2_split_is_deterministic():
    tracks = _tracks(10)
    first = split_tracks(tracks, seed=7)
    second = split_tracks(tracks, seed=7)
    assert {name: len(items) for name, items in first.items()} == {
        "train": 7,
        "val": 1,
        "test": 2,
    }
    assert [item.track_id for item in first["train"]] == [
        item.track_id for item in second["train"]
    ]


def test_artist_group_split_has_no_leakage():
    split = split_tracks(_tracks(30), seed=2, group_by_artist=True)
    artists = {
        name: {track.artist_id for track in tracks} for name, tracks in split.items()
    }
    assert artists["train"].isdisjoint(artists["val"])
    assert artists["train"].isdisjoint(artists["test"])
    assert artists["val"].isdisjoint(artists["test"])


def test_metadata_filtering():
    selected = select_tracks(
        _tracks(20),
        num_tracks=4,
        seed=1,
        tags=["instrument---piano"],
        min_duration=30,
    )
    assert len(selected) == 4
    assert all("instrument---piano" in track.tags for track in selected)

