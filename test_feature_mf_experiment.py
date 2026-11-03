"""Checks for predicting a new song's MF vector from its music features. No real data needed.
Run: python -m unittest test_feature_mf_experiment.py -v
"""
import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np

from collaborative import CollaborativeRecommender
from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender
from feature_mf_experiment import (FeatureMF, choose_feature_config, feature_matrix, fit_feature_regression,
                                   run_feature_mf_experiment)
from final_model import ArtistAffinity, ArtistCB, FinalRecommender
from hybrid import mask_mock_cold
from test_hybrid import make_fixture

SMALL_CF = {"n_factors": 8, "n_epochs": 3}


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def stub_cf(q, trained):
    """Only what the regression reads: factors and listener counts for trained songs."""
    ids = np.array([f"T{i}" for i in range(len(trained))])
    trained_ids = ids[trained]
    return ids, SimpleNamespace(Q=q, i_map={t: i for i, t in enumerate(trained_ids)},
                                get_item_listener_counts=lambda _ids: trained.astype(int))


class RegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.features = rng.normal(size=(300, 5))
        self.weights = rng.normal(size=(5, 3))
        self.trained = np.arange(300) < 250
        self.truth = self.features @ self.weights

    def test_recovers_a_linear_mapping_for_new_songs(self):
        ids, cf = stub_cf(self.truth[self.trained], self.trained)
        predicted, r2 = fit_feature_regression(cf, ids, self.features, seed=42)
        np.testing.assert_allclose(predicted[~self.trained], self.truth[~self.trained], atol=0.05)
        self.assertGreater(r2, 0.99)

    def test_noise_features_report_low_r2(self):
        ids, cf = stub_cf(self.truth[self.trained], self.trained)
        noise = np.random.default_rng(1).normal(size=self.features.shape)
        _, r2 = fit_feature_regression(cf, ids, noise, seed=42)
        self.assertLess(r2, 0.1)

    def test_new_song_targets_are_never_used(self):
        ids, cf = stub_cf(self.truth[self.trained], self.trained)
        changed = self.features.copy()
        first, _ = fit_feature_regression(cf, ids, changed, seed=42)
        changed[~self.trained] += 100.0  # features of new songs change only their own predictions
        second, _ = fit_feature_regression(cf, ids, changed, seed=42)
        np.testing.assert_allclose(first[self.trained], second[self.trained])


class FeatureMFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        music, train, _ = make_fixture()
        cls.masked, _ = mask_mock_cold(train, 0.2, 42)
        cls.cb = ContentBasedRecommender(n_clusters=3).fit(music, cls.masked)
        cls.affinity = ArtistAffinity(cls.cb, cls.masked)
        cls.cf = quiet(CollaborativeRecommender(**SMALL_CF).fit, cls.masked, music)
        cls.ids = cls.cb.music_info["track_id"].to_numpy(dtype=str)
        cls.warm = cls.cf.get_item_listener_counts(cls.ids) > 0
        cls.predicted = np.random.default_rng(0).normal(size=(len(cls.ids), SMALL_CF["n_factors"]))
        cls.model = FeatureMF(cls.cf, cls.ids, cls.predicted, n_new=25.0)

    def test_trained_songs_keep_their_scores(self):
        base = self.cf.score_user_items("U1", self.ids)
        np.testing.assert_allclose(self.model.score_user_items("U1", self.ids)[self.warm], base[self.warm])

    def test_new_songs_use_the_trained_dot_product_range(self):
        p = self.cf.P[self.cf.u_map["U1"]]
        heard = np.array([t in self.cf.user_listened_tracks["U1"] for t in self.ids])
        pool = np.array([self.cf.i_map[t] for t in self.ids[self.warm & ~heard]])
        dots = self.cf.Q[pool] @ p
        expected = np.clip((self.predicted @ p - dots.min()) / np.ptp(dots), 0.0, 1.0)
        s = self.model.score_user_items("U1", self.ids)
        np.testing.assert_allclose(s[~self.warm], expected[~self.warm])

    def test_alpha_for_new_songs_comes_from_the_shared_formula(self):
        a = self.model.compute_dynamic_alpha(self.ids)
        np.testing.assert_allclose(a[~self.warm], 0.25 + 0.75 / (1 + 25.0 / 25.0))
        np.testing.assert_allclose(a[self.warm], self.cf.compute_dynamic_alpha(self.ids)[self.warm])

    def test_true_listener_counts_are_kept(self):
        np.testing.assert_array_equal(self.model.get_item_listener_counts(self.ids),
                                      self.cf.get_item_listener_counts(self.ids))

    def test_wrong_candidate_order_raises(self):
        with self.assertRaises(ValueError):
            self.model.score_user_items("U1", self.ids[::-1])

    def test_unknown_artist_new_song_can_outrank_with_zero_artist_weight(self):
        h = FinalRecommender(ArtistCB(self.cb, self.affinity, 0.0), self.model)
        s = h.score_user_items("U1")
        self.assertTrue((s["S_MF"][~self.warm] > 0).any())
        np.testing.assert_allclose(s["S_Hybrid"], s["alpha"] * s["S_CB"] + (1 - s["alpha"]) * s["S_MF"])

    def test_feature_sets_have_the_expected_columns(self):
        audio = feature_matrix(self.cb, "audio")
        self.assertEqual(audio.shape[1], len(AUDIO_FEATURE_COLS))
        self.assertGreater(feature_matrix(self.cb, "audio_tags").shape[1], audio.shape[1])
        with self.assertRaises(ValueError):
            feature_matrix(self.cb, "nope")


class SelectionTests(unittest.TestCase):
    def test_picks_best_unknown_artist_recall_within_the_floor(self):
        rows = {"a": {"ndcg10": 0.20, "recall20_unknown_artist": 0.01},
                "b": {"ndcg10": 0.19, "recall20_unknown_artist": 0.05},
                "c": {"ndcg10": 0.05, "recall20_unknown_artist": 0.50}}
        self.assertEqual(choose_feature_config(rows, 0.20), ("b", True))

    def test_reports_the_best_unknown_recall_when_nothing_qualifies(self):
        rows = {"a": {"ndcg10": 0.10, "recall20_unknown_artist": 0.01},
                "b": {"ndcg10": 0.05, "recall20_unknown_artist": 0.50}}
        self.assertEqual(choose_feature_config(rows, 0.20), ("b", False))

    def test_missing_recall_counts_as_zero(self):
        rows = {"a": {"ndcg10": 0.20, "recall20_unknown_artist": None},
                "b": {"ndcg10": 0.20, "recall20_unknown_artist": 0.01}}
        self.assertEqual(choose_feature_config(rows, 0.20), ("b", True))


class ExperimentTests(unittest.TestCase):
    def test_selection_uses_validation_only(self):
        music, train, test = make_fixture()
        scrambled = test.copy()
        scrambled["track_id"] = np.random.default_rng(7).choice(music["track_id"].to_numpy(), len(scrambled))
        kwargs = dict(betas=(0.5,), new_song_listeners=(25.0,), mock_cold_fraction=0.2, seed=42, max_users=20,
                      val_users=20, cf_kwargs=SMALL_CF, n_clusters=3)
        first = quiet(run_feature_mf_experiment, music, train, test, **kwargs)
        second = quiet(run_feature_mf_experiment, music, train, scrambled, **kwargs)
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["validation"], second["validation"])
        self.assertEqual(set(first["selected"]), {"audio", "audio_tags"})
        self.assertEqual(set(first["test"]), {"final", "audio", "audio_tags"})
        self.assertIn("recall20_unknown_artist", first["test"]["audio"])
        self.assertIn("holdout_r2", first)
        self.assertIn("ndcg10_vs_final", first["confidence_intervals"]["audio"])
        self.assertIn("unknown_recall20_vs_final", first["confidence_intervals"]["audio"])


if __name__ == "__main__":
    unittest.main()
