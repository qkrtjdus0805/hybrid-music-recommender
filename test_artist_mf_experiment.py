"""Checks for estimating a new song's MF score from its artist. No real data needed.
Run: python -m unittest test_artist_mf_experiment.py -v
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from artist_experiment import ArtistAffinity, ArtistCB
from artist_mf_experiment import (BETAS, GAMMAS, ArtistMF, choose_config, paired_bootstrap, parse_args,
                                  per_user_metrics, run_artist_mf_experiment)
from collaborative import CollaborativeRecommender
from content_based import ContentBasedRecommender
from hybrid import HybridRecommender, mask_mock_cold
from test_artist_experiment import artist_world
from test_hybrid import make_fixture

SMALL_CF = {"n_factors": 8, "n_epochs": 3}


class ArtistMFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, train = artist_world()
        extra = pd.DataFrame([{"user_id": "U2", "track_id": "T04", "playcount": 9},  # gives U0's unheard songs a range
                              {"user_id": "U2", "track_id": "T01", "playcount": 2},
                              {"user_id": "U3", "track_id": "T04", "playcount": 1},
                              {"user_id": "U3", "track_id": "T07", "playcount": 1}])
        cls.train = pd.concat([train, extra], ignore_index=True)
        cls.cb = ContentBasedRecommender(n_clusters=3).fit(cls.music, cls.train)
        cls.affinity = ArtistAffinity(cls.cb, cls.train)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.cf = CollaborativeRecommender(**SMALL_CF).fit(cls.train, cls.music)
        cls.ids = cls.cb.music_info["track_id"].to_numpy(dtype=str)
        cls.model = ArtistMF(cls.cf, cls.ids, cls.affinity.codes, gamma=0.5)
        cls.base = cls.cf.score_user_items("U0", cls.ids)
        cls.warm = cls.cf.get_item_listener_counts(cls.ids) > 0

    def test_trained_songs_keep_their_scores(self):
        s = self.model.score_user_items("U0", self.ids)
        np.testing.assert_allclose(s[self.warm], self.base[self.warm])

    def test_new_song_gets_mean_of_same_artist_trained_songs(self):
        s = self.model.score_user_items("U0", self.ids)
        same_artist = self.warm & (self.affinity.codes == self.affinity.codes[12])
        self.assertAlmostEqual(s[12], self.base[same_artist].mean())

    def test_new_artist_stays_zero(self):
        s = self.model.score_user_items("U0", self.ids)
        self.assertEqual(s[14], 0.0)  # Gamma has no trained songs

    def test_alpha_unchanged_for_trained_songs(self):
        a = self.model.compute_dynamic_alpha(self.ids)
        np.testing.assert_allclose(a[self.warm], self.cf.compute_dynamic_alpha(self.ids)[self.warm])

    def test_alpha_uses_artist_listeners_for_new_songs(self):
        a = self.model.compute_dynamic_alpha(self.ids)
        same_artist = self.warm & (self.affinity.codes == self.affinity.codes[12])
        n_eff = 0.5 * self.cf.get_item_listener_counts(self.ids)[same_artist].mean()
        self.assertAlmostEqual(a[12], 0.25 + 0.75 / (1 + n_eff / 25.0))
        self.assertLess(a[12], 1.0)

    def test_alpha_is_one_for_new_artist_and_zero_gamma(self):
        self.assertAlmostEqual(self.model.compute_dynamic_alpha(self.ids)[14], 1.0)
        none = ArtistMF(self.cf, self.ids, self.affinity.codes, gamma=0.0)
        np.testing.assert_allclose(none.compute_dynamic_alpha(self.ids)[~self.warm], 1.0)

    def test_true_listener_counts_are_kept(self):
        np.testing.assert_array_equal(self.model.get_item_listener_counts(self.ids),
                                      self.cf.get_item_listener_counts(self.ids))

    def test_wrong_candidate_order_raises(self):
        with self.assertRaises(ValueError):
            self.model.score_user_items("U0", self.ids[::-1])

    def test_new_song_score_exceeds_fixed_alpha_cap(self):
        """Estimated S_MF removes the 0-to-alpha limit a zero S_MF creates."""
        h = HybridRecommender(ArtistCB(self.cb, self.affinity, 0.9), self.model)
        s = h.score_user_items("U0")
        np.testing.assert_allclose(s["S_Hybrid"], s["alpha"] * s["S_CB"] + (1 - s["alpha"]) * s["S_MF"])
        self.assertEqual(h.n_listeners[12], 0)
        self.assertGreater(s["S_Hybrid"][12], s["alpha"][12] * s["S_CB"][12])


class ArgumentTests(unittest.TestCase):
    def test_grid_defaults_match_the_module(self):
        args = parse_args([])
        self.assertEqual(tuple(args.betas), BETAS)
        self.assertEqual(tuple(args.gammas), GAMMAS)

    def test_grid_can_be_widened(self):
        args = parse_args(["--betas", "0.5", "0.9", "--gammas", "1", "2.5", "--output", "elsewhere"])
        self.assertEqual(args.betas, [0.5, 0.9])
        self.assertEqual(args.gammas, [1.0, 2.5])
        self.assertEqual(args.output, "elsewhere")


class SelectionTests(unittest.TestCase):
    def test_picks_best_new_song_recall_within_accuracy_floor(self):
        rows = {"a": {"ndcg10": 0.20, "mock_recall20": 0.01},
                "b": {"ndcg10": 0.19, "mock_recall20": 0.09},
                "c": {"ndcg10": 0.05, "mock_recall20": 0.50}}
        label, met = choose_config(rows, baseline_ndcg10=0.20)
        self.assertEqual((label, met), ("b", True))

    def test_falls_back_to_best_accuracy_when_nothing_qualifies(self):
        rows = {"a": {"ndcg10": 0.10, "mock_recall20": 0.01}, "b": {"ndcg10": 0.05, "mock_recall20": 0.50}}
        self.assertEqual(choose_config(rows, baseline_ndcg10=0.20), ("a", False))

    def test_bootstrap_interval_of_constant_difference(self):
        ci = paired_bootstrap(np.full(50, 0.3), np.full(50, 0.1), seed=1)
        self.assertAlmostEqual(ci["mean_diff"], 0.2)
        self.assertAlmostEqual(ci["ci_low"], 0.2)
        self.assertAlmostEqual(ci["ci_high"], 0.2)

    def test_bootstrap_ignores_missing_pairs(self):
        ci = paired_bootstrap(np.array([1.0, np.nan, 1.0]), np.array([0.0, 5.0, 0.0]), seed=1)
        self.assertEqual(ci["n"], 2)
        self.assertAlmostEqual(ci["mean_diff"], 1.0)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train, cls.test = make_fixture()
        cls.music["artist"] = [f"Artist {i % 4}" for i in range(len(cls.music))]

    def test_per_user_metrics_marks_users_without_new_song_answers(self):
        masked, cold = mask_mock_cold(self.train, 0.2, 42)
        cb = ContentBasedRecommender(n_clusters=3).fit(self.music, masked)
        with contextlib.redirect_stdout(io.StringIO()):
            cf = CollaborativeRecommender(**SMALL_CF).fit(masked, self.music)
        metrics = per_user_metrics(HybridRecommender(cb, cf), self.test, cold, 100, 42)
        self.assertEqual(list(metrics.columns), ["ndcg10", "recall10", "mock_recall20"])
        self.assertEqual(len(metrics), self.test["user_id"].nunique())
        self.assertTrue(metrics["ndcg10"].between(0, 1).all())
        self.assertTrue(metrics["mock_recall20"].isna().any() or metrics["mock_recall20"].notna().all())

    def test_selection_uses_validation_only(self):
        scrambled = self.test.copy()
        scrambled["track_id"] = np.random.default_rng(7).choice(self.music["track_id"].to_numpy(), len(scrambled))
        kwargs = dict(betas=(0.5,), gammas=(0.0, 0.5), mock_cold_fraction=0.2, seed=42, max_users=20, val_users=20,
                      cf_kwargs=SMALL_CF, n_clusters=3)
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_artist_mf_experiment(self.music, self.train, self.test, **kwargs)
            second = run_artist_mf_experiment(self.music, self.train, scrambled, **kwargs)
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["validation"], second["validation"])
        self.assertEqual(set(first["test"]), {"mf", "fixed_artist", "dynamic_artist", "selected"})
        for field in ("ndcg10", "recall10", "mock_recall20", "recall20_1_4", "recall20_5_24", "exposure10_25plus",
                      "coverage10", "top10_cold_share", "hybrid_effect_verifiable"):
            self.assertIn(field, first["test"]["selected"])
        self.assertIn("recall_known_artist", first["test"]["selected"])
        self.assertIn("ndcg10_vs_mf", first["confidence_intervals"])


if __name__ == "__main__":
    unittest.main()
