"""Unit tests for the 80/20 Spotify Playlist Continuation Validation.
Run: python -m unittest test_spotify_playlist_experiment.py -v
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from collaborative import CollaborativeRecommender
from content_based import ContentBasedRecommender
from hybrid import load_spotify
from experiments.val3_spotify_80_20.spotify_playlist_experiment import (
    MODELS,
    evaluate_spotify_80_20,
    link_catalogs,
    make_norm_key,
    split_playlists_80_20,
)
from tests.test_hybrid import make_fixture, make_spotify


class Spotify8020Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train, _ = make_fixture()
        cls.cb = ContentBasedRecommender(n_clusters=3).fit(cls.music, cls.train)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.cf = CollaborativeRecommender(n_factors=8, n_epochs=3).fit(cls.train, cls.music)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "spotify.csv"
        make_spotify(cls.music, linked=True).to_csv(cls.path, index=False)
        cls.spotify = load_spotify(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_make_norm_key_format(self):
        artists = pd.Series(["  The Killers ", "Oasis"])
        names = pd.Series(["Mr. Brightside  ", " Wonderwall"])
        keys = make_norm_key(artists, names)
        self.assertEqual(keys.tolist(), ["the killers - mr. brightside", "oasis - wonderwall"])

    def test_link_catalogs_maps_warm_and_keeps_cold_ids(self):
        unlinked = make_spotify(self.music, linked=False).iloc[:2].copy()
        combined = pd.concat([make_spotify(self.music, linked=True), unlinked], ignore_index=True)
        path = Path(self.tmp.name) / "mixed.csv"
        combined.to_csv(path, index=False)
        sp_df = load_spotify(path)
        sp_linked, link = link_catalogs(sp_df, self.music)
        self.assertGreater(len(link), 0)
        self.assertTrue(sp_linked["unified_id"].str.startswith("TR_").any())
        self.assertTrue(sp_linked["unified_id"].str.startswith("SP_").any())

    def test_split_playlists_80_20_ratios_and_disjoint(self):
        sp_linked, _ = link_catalogs(self.spotify, self.music)
        splits = split_playlists_80_20(sp_linked, key_col="norm_key", train_ratio=0.80, min_tracks=10, seed=42)
        self.assertEqual({pid for pid, _, _ in splits}, {"P1", "P2"})
        for pid, query, hidden in splits:
            total = {"P1": 14, "P2": 12}[pid]
            expected_train = round(total * 0.80)
            self.assertEqual(len(query), expected_train)
            self.assertEqual(len(hidden), total - expected_train)
            self.assertTrue(set(query).isdisjoint(hidden))

    def test_evaluate_spotify_80_20_returns_four_models(self):
        report = evaluate_spotify_80_20(
            self.cb, self.cf, self.spotify, self.music, train_ratio=0.80, seed=42
        )
        self.assertEqual(report["n_playlists_evaluated"], 2)
        self.assertEqual(tuple(report["models"].keys()), MODELS)
        for m in MODELS:
            for k in ("@5", "@10", "@20"):
                self.assertIn(k, report["models"][m])
                for metric in ("hit_rate", "recall", "precision", "ndcg", "warm_recall", "cold_recall", "cold_share"):
                    val = report["models"][m][k][metric]
                    self.assertTrue(0.0 <= val <= 1.0, f"{m} {k} {metric} = {val}")


if __name__ == "__main__":
    unittest.main()