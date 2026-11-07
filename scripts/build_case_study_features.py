"""Build the Spotify / Tidal recommendation feature tables for the case study.

Every row is looked up in the project datasets (no hand-typed numbers):
  1. Music Info.csv      (preferred: model catalogue, has all 12 features)
  2. spotify_songs.csv   (fallback: no time_signature column -> left empty)
  3. not found           -> all features empty, source = "not_found"

The case study fills a missing time_signature with the Music Info median (same rule as
hybrid.load_spotify) and skips rows whose other audio features are missing.

Run from the repository root:
  python scripts/build_case_study_features.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

FEATURES = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "time_signature"]
COLUMNS = ["rank", "query", "track_id", "name", "artist", *FEATURES, "duration_ms", "source"]

# (query as written in the .txt list, title to match, artist to match)
# Artist = the original/best-known recording. Ambiguous titles are noted in docs/case_study.md.
SPOTIFY_RECS = [
    ("california dreamin'", "California Dreamin'", "The Mamas & the Papas"),
    ("wicked game", "Wicked Game", "Chris Isaak"),
    ("bohemian rhapsody", "Bohemian Rhapsody", "Queen"),
    ("everybody wants to rule the world", "Everybody Wants to Rule the World", "Tears for Fears"),
    ("dream on", "Dream On", "Aerosmith"),
    ("bitter sweet symphony", "Bitter Sweet Symphony", "The Verve"),
    ("ain't no sunshine", "Ain't No Sunshine", "Bill Withers"),
    ("careless whisper", "Careless Whisper", "George Michael"),
    ("i'm still standing", "I'm Still Standing", "Elton John"),
    ("smooth operator", "Smooth Operator", "Sade"),
]
TIDAL_RECS = [
    ("forever young", "Forever Young", "Alphaville"),
    ("mamma mia", "Mamma Mia", "ABBA"),
    ("total eclipse of the heart", "Total Eclipse of the Heart", "Bonnie Tyler"),
    ("i got you", "I Got You", "Split Enz"),  # ambiguous title, see docs/case_study.md
    ("let's dance", "Let's Dance", "David Bowie"),
    ("born in usa", "Born in the U.S.A.", "Bruce Springsteen"),
    ("i want you back", "I Want You Back", "The Jackson 5"),
    ("uptown funk", "Uptown Funk", "Mark Ronson"),
    ("i love rock'N roll", "I Love Rock 'n' Roll", "Joan Jett"),
    ("living on a prayer", "Livin' On A Prayer", "Bon Jovi"),
]


def _norm(text) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+-\s+.*$", "", text)  # drop " - 2011 Mix", " - Remastered" suffixes
    return re.sub(r"[^a-z0-9]", "", text)


def _artist_key(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower().replace("&", "and"))


def lookup(title: str, artist: str, music: pd.DataFrame, spotify: pd.DataFrame) -> dict:
    t, a = _norm(title), _artist_key(artist)
    hit = music[(music["_name"] == t) & music["_artist"].str.contains(a, regex=False)]
    if len(hit):
        row = hit.iloc[0]
        return {"track_id": row["track_id"], "name": row["name"], "artist": row["artist"],
                **{f: row[f] for f in FEATURES}, "duration_ms": row.get("duration_ms"), "source": "Music Info.csv"}
    hit = spotify[(spotify["_name"] == t) & spotify["_artist"].str.contains(a, regex=False)]
    if len(hit):
        row = hit.sort_values("track_popularity", ascending=False).iloc[0]
        return {"track_id": row["track_id"], "name": row["track_name"], "artist": row["track_artist"],
                **{f: row.get(f) for f in FEATURES}, "duration_ms": row.get("duration_ms"),
                "source": "spotify_songs.csv"}
    return {"track_id": None, "name": title, "artist": artist, **{f: None for f in FEATURES},
            "duration_ms": None, "source": "not_found"}


def build(recs, music, spotify) -> pd.DataFrame:
    rows = [{"rank": i, "query": q, **lookup(t, a, music, spotify)} for i, (q, t, a) in enumerate(recs, 1)]
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--music", default="data/Music Info.csv")
    p.add_argument("--spotify", default="data/spotify_songs.csv.zip")
    p.add_argument("--output", default="data/case_study")
    args = p.parse_args(argv)

    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    spotify = pd.read_csv(args.spotify, encoding="utf-8-sig", dtype={"track_id": str})
    if "time_signature" not in spotify.columns:
        spotify["time_signature"] = pd.NA
    music["_name"], music["_artist"] = music["name"].map(_norm), music["artist"].map(_artist_key)
    spotify["_name"], spotify["_artist"] = spotify["track_name"].map(_norm), spotify["track_artist"].map(_artist_key)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for label, recs in (("spotify", SPOTIFY_RECS), ("tidal", TIDAL_RECS)):
        df = build(recs, music, spotify)
        df.to_csv(out / f"{label}_recommended_features.csv", index=False, encoding="utf-8-sig")
        print(f"\n{label}: {out / f'{label}_recommended_features.csv'}")
        print(df[["rank", "name", "artist", "source"]].to_string(index=False))


if __name__ == "__main__":
    main()
