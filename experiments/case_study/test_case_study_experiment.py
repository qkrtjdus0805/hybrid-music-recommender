"""Unit tests for the 12D audio-space case study and feature builder.

Run: python -m unittest experiments/case_study/test_case_study_experiment.py -v
"""
from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from collaborative import CollaborativeRecommender
from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender
from experiments.case_study.case_study_experiment import (
    compute_12d_metrics,
    prepare_external_recs,
    recommend_final_from_seeds,
    resolve_seed_tracks,
)
from scripts.build_case_study_features import build, lookup
from tests.test_hybrid import make_fixture


class CaseStudyTests(unittest.TestCase):
    def setUp(self):
        self.music, self.train, _ = make_fixture(seed=7)
        self.cb = ContentBasedRecommender(n_clusters=3, random_state=7).fit(self.music, self.train)
        with contextlib.redirect_stdout(io.StringIO()):
            self.cf = CollaborativeRecommender(method="als", n_factors=4, n_epochs=2, random_state=7).fit(
                self.train, self.music
            )

    def test_resolve_seed_tracks_matches_by_id_and_artist_name(self):
        first = self.music.iloc[0]
        second = self.music.iloc[1]
        seeds_in = pd.DataFrame(
            [
                {"track_id": first["track_id"], "artist": "", "name": ""},
                {"track_id": "", "artist": second["artist"].upper(), "name": f"  {second['name']}  "},
            ]
        )
        resolved = resolve_seed_tracks(seeds_in, self.music)
        self.assertEqual(resolved["track_id"].tolist(), [first["track_id"], second["track_id"]])

    def test_recommend_final_from_seeds_excludes_seeds_and_respects_cap(self):
        seeds = self.music.iloc[:3].copy()
        recs = recommend_final_from_seeds(self.cb, self.cf, seeds, top_n=10)
        self.assertEqual(len(recs), 10)
        self.assertTrue(set(seeds["track_id"]).isdisjoint(set(recs["track_id"])))
        self.assertLessEqual(recs["artist"].value_counts().max(), 5)

    def test_prepare_external_recs_fills_time_signature_and_drops_missing_rows(self):
        ext = self.music.iloc[:3][["name", "artist", *AUDIO_FEATURE_COLS]].copy()
        ext = ext.drop(columns=["time_signature"])
        missing_row = pd.DataFrame([{"name": "Unknown", "artist": "Nobody"}])
        combined = pd.concat([ext, missing_row], ignore_index=True)
        with contextlib.redirect_stdout(io.StringIO()):
            cleaned = prepare_external_recs(combined, self.cb, "Test")
        self.assertEqual(len(cleaned), 3)
        self.assertFalse(cleaned[AUDIO_FEATURE_COLS].isna().any().any())

    def test_compute_12d_metrics_bounds(self):
        seeds = self.music.iloc[:3]
        recs = self.music.iloc[3:8]
        metrics = compute_12d_metrics(self.cb, seeds, recs)
        self.assertGreaterEqual(metrics["profile_similarity_01"], 0.0)
        self.assertLessEqual(metrics["profile_similarity_01"], 1.0)
        self.assertGreaterEqual(metrics["intra_list_diversity_01"], 0.0)
        self.assertLessEqual(metrics["intra_list_diversity_01"], 1.0)
        self.assertEqual(set(metrics["mean_raw_vector"].keys()), set(AUDIO_FEATURE_COLS))

    def test_feature_builder_prefers_music_info_then_spotify_then_not_found(self):
        msd = self.music.iloc[:2].copy()
        msd["duration_ms"] = 200000
        msd["_name"] = ["songone", "songtwo"]
        msd["_artist"] = ["artistone", "artisttwo"]
        sp = msd.iloc[:1].rename(columns={"name": "track_name", "artist": "track_artist"}).copy()
        sp["track_name"] = "Song Three"
        sp["track_artist"] = "Artist Three"
        sp["track_popularity"] = 80
        sp["time_signature"] = pd.NA
        sp["_name"] = ["songthree"]
        sp["_artist"] = ["artistthree"]

        r1 = lookup("Song One", "Artist One", msd, sp)
        r2 = lookup("Song Three", "Artist Three", msd, sp)
        r3 = lookup("Missing", "Ghost", msd, sp)
        self.assertEqual(r1["source"], "Music Info.csv")
        self.assertEqual(r2["source"], "spotify_songs.csv")
        self.assertTrue(pd.isna(r2["time_signature"]))
        self.assertEqual(r3["source"], "not_found")
        self.assertIsNone(r3["danceability"])

        table = build([("song one", "Song One", "Artist One"), ("missing", "Missing", "Ghost")], msd, sp)
        self.assertEqual(len(table), 2)
        self.assertEqual(table["source"].tolist(), ["Music Info.csv", "not_found"])


if __name__ == "__main__":
    unittest.main()
