"""Checks for the artist affinity experiment. No real data needed.
Run: python -m unittest test_artist_experiment.py -v
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from artist_experiment import (ArtistAffinity, ArtistCB, duplicate_share, mock_topk, run_artist_experiment)
from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender
from hybrid import HybridRecommender, mask_mock_cold
from test_hybrid import make_fixture

SMALL_CF = {"n_factors": 8, "n_epochs": 3}


def artist_world():
    """Three artists with six songs each; U0 and U1 each like one artist; the last songs are new."""
    rng = np.random.default_rng(3)
    n = 18
    music = pd.DataFrame(rng.random((n, len(AUDIO_FEATURE_COLS))), columns=AUDIO_FEATURE_COLS)
    music.insert(0, "track_id", [f"T{i:02}" for i in range(n)])
    music["name"] = [f"Song {i}" for i in range(n)]
    music["artist"] = [["Alpha", "Beta", "Gamma"][i % 3] for i in range(n)]
    music["genre"] = "rock"
    train = pd.DataFrame(
        [{"user_id": "U0", "track_id": f"T{i:02}", "playcount": 5} for i in (0, 3, 6)]
        + [{"user_id": "U0", "track_id": "T01", "playcount": 1}]
        + [{"user_id": "U1", "track_id": f"T{i:02}", "playcount": 5} for i in (1, 4, 7)])
    return music, train


class AffinityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train = artist_world()
        cls.cb = ContentBasedRecommender(n_clusters=3).fit(cls.music, cls.train)
        cls.affinity = ArtistAffinity(cls.cb, cls.train)

    def test_top_artist_scores_one(self):
        a = self.affinity.scores("U0")
        self.assertAlmostEqual(a[0], 1.0)
        self.assertAlmostEqual(a[12], 1.0)  # T12 is by Alpha, never heard

    def test_less_heard_artist_scores_lower(self):
        a = self.affinity.scores("U0")
        self.assertTrue(0.0 < a[1] < 1.0)

    def test_unheard_artist_scores_zero(self):
        self.assertEqual(self.affinity.scores("U0")[2], 0.0)

    def test_artist_matching_ignores_case_and_spaces(self):
        music = self.music.copy()
        music.loc[12, "artist"] = "  ALPHA "
        cb = ContentBasedRecommender(n_clusters=3).fit(music, self.train)
        self.assertAlmostEqual(ArtistAffinity(cb, self.train).scores("U0")[12], 1.0)

    def test_missing_artist_scores_zero(self):
        music = self.music.copy()
        music.loc[12, "artist"] = np.nan
        cb = ContentBasedRecommender(n_clusters=3).fit(music, self.train)
        self.assertEqual(ArtistAffinity(cb, self.train).scores("U0")[12], 0.0)

    def test_unknown_user_scores_zero(self):
        self.assertTrue((self.affinity.scores("NOPE") == 0).all())

    def test_beta_zero_equals_cb(self):
        wrapped = ArtistCB(self.cb, self.affinity, 0.0)
        np.testing.assert_allclose(wrapped.score_user_items("U0"), self.cb.score_user_items("U0"))

    def test_beta_mixes_cb_and_artist(self):
        wrapped = ArtistCB(self.cb, self.affinity, 0.3)
        expected = 0.7 * self.cb.score_user_items("U0") + 0.3 * self.affinity.scores("U0")
        np.testing.assert_allclose(wrapped.score_user_items("U0"), expected)

    def test_wrapper_works_inside_hybrid(self):
        from collaborative import CollaborativeRecommender
        with contextlib.redirect_stdout(io.StringIO()):
            cf = CollaborativeRecommender(**SMALL_CF).fit(self.train, self.music)
        h = HybridRecommender(ArtistCB(self.cb, self.affinity, 0.5), cf)
        s = h.score_user_items("U0")
        self.assertEqual(len(s["S_Hybrid"]), len(self.music))
        self.assertTrue(np.isfinite(s["S_Hybrid"]).all())

    def test_mock_topk_splits_known_and_unknown_artists(self):
        test = pd.DataFrame([{"user_id": "U0", "track_id": "T12", "playcount": 1},   # Alpha: known to U0
                             {"user_id": "U0", "track_id": "T14", "playcount": 1}])  # Gamma: unknown to U0
        wrapped = ArtistCB(self.cb, self.affinity, 0.9)
        r = mock_topk(wrapped, test, np.array(["T12", "T14"]), 3, 10, 42)
        self.assertEqual(r["n_users"], 1)
        self.assertAlmostEqual(r["known_artist_answer_share"], 0.5)
        self.assertEqual(r["recall_known_artist"], 1.0)
        self.assertEqual(r["recall_unknown_artist"], 0.0)

    def test_duplicate_share_counts_same_title_and_artist(self):
        music = self.music.copy()
        music.loc[12, ["name", "artist"]] = ["Song 0", "Alpha"]
        self.assertEqual(duplicate_share(music, np.array(["T12", "T14"]), {"T00"}), 0.5)


class ExperimentTests(unittest.TestCase):
    def test_selection_uses_validation_only(self):
        music, train, test = make_fixture()
        music["artist"] = [f"Artist {i % 4}" for i in range(len(music))]
        scrambled = test.copy()
        scrambled["track_id"] = np.random.default_rng(7).choice(music["track_id"].to_numpy(), len(scrambled))
        kwargs = dict(scalers=("standard",), betas=(0.0, 0.5), mock_cold_fraction=0.2, seed=42, max_users=20,
                      val_users=20, cf_kwargs=SMALL_CF, n_clusters=3)
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_artist_experiment(music, train, test, **kwargs)
            second = run_artist_experiment(music, train, scrambled, **kwargs)
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["validation"], second["validation"])
        self.assertEqual(len(first["validation"]), 2)
        self.assertIn("baseline", first["test"])
        self.assertIn("selected", first["test"])
        for field in ("cb_ndcg10", "cb_mock_recall20", "hybrid_ndcg10", "hybrid_cold_share", "fixed_ndcg10",
                      "topk_recall20", "recall_known_artist", "recall_unknown_artist", "shuffled_percentile_all"):
            self.assertIn(field, first["test"]["selected"])
        self.assertIn("mock_cold_duplicate_share", first)


if __name__ == "__main__":
    unittest.main()
