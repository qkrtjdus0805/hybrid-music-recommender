"""Checks for the CB scaler experiment. No real data needed.
Run: python -m unittest experiments.exp2_scaler.test_scaler_experiment -v
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender
from hybrid import mask_mock_cold
from experiments.exp2_scaler.scaler_experiment import SCALERS, build_scaler, cold_diagnostics, make_cb, run_experiment
from tests.test_hybrid import make_fixture

SMALL_CF = {"n_factors": 8, "n_epochs": 3}


class ScalerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train, cls.test = make_fixture()
        cls.masked, cls.cold = mask_mock_cold(cls.train, 0.2, 42)

    def test_standard_matches_stage1_features(self):
        plain = ContentBasedRecommender(n_clusters=3).fit(self.music)
        cb = make_cb("standard", self.music, 42, n_clusters=3)
        np.testing.assert_allclose(cb.track_features, plain.track_features)

    def test_every_scaler_produces_finite_scores(self):
        for name in SCALERS:
            cb = make_cb(name, self.music, 42, n_clusters=3)
            cb.fit_user_profiles(self.masked)
            scores = cb.score_user_items("U1")
            self.assertEqual(len(scores), len(self.music), name)
            self.assertTrue(np.isfinite(scores).all(), name)

    def test_no_categorical_drops_three_columns(self):
        out = build_scaler("no_categorical", 42).fit_transform(self.music[AUDIO_FEATURE_COLS].to_numpy())
        self.assertEqual(out.shape[1], len(AUDIO_FEATURE_COLS) - 3)

    def test_onehot_adds_columns_and_handles_new_values(self):
        music = self.music.copy()
        music["key"] = np.arange(len(music)) % 12
        music["time_signature"] = 4
        scaler = build_scaler("categorical_onehot", 42)
        out = scaler.fit_transform(music[AUDIO_FEATURE_COLS].to_numpy())
        self.assertEqual(out.shape[1], len(AUDIO_FEATURE_COLS) - 2 + 12 + 1)
        fresh = music[AUDIO_FEATURE_COLS].to_numpy()[:1].copy()
        fresh[0, AUDIO_FEATURE_COLS.index("time_signature")] = 7
        self.assertEqual(scaler.transform(fresh).shape[1], out.shape[1])

    def test_unknown_scaler_raises(self):
        with self.assertRaises(ValueError):
            build_scaler("nope", 42)

    def test_diagnostics_detect_a_personal_signal(self):
        """Each user's profile points at their own answers, so another user's profile must do worse."""
        ids = [f"T{i:03}" for i in range(60)]
        rng = np.random.default_rng(0)
        music = pd.DataFrame(rng.normal(size=(60, len(AUDIO_FEATURE_COLS))), columns=AUDIO_FEATURE_COLS)
        music.insert(0, "track_id", ids)
        music["genre"] = "rock"
        groups =[ids[:20], ids[20:40], ids[40:]]
        cb = ContentBasedRecommender(n_clusters=3).fit(music)
        centres = rng.normal(size=(3, cb.track_features.shape[1])) * 3
        for g, centre in enumerate(centres):
            cb.track_features[g * 20:(g + 1) * 20] = centre + rng.normal(scale=0.2, size=(20, centres.shape[1]))
        train = pd.DataFrame([{"user_id": f"U{g}", "track_id": t, "playcount": 3}
                              for g, tracks in enumerate(groups) for t in tracks[:10]])
        cb.fit_user_profiles(train)
        test = pd.DataFrame([{"user_id": f"U{g}", "track_id": t, "playcount": 1}
                             for g, tracks in enumerate(groups) for t in tracks[10:15]])
        cold = np.array([t for tracks in groups for t in tracks[10:15]])
        d = cold_diagnostics(cb, test, cold, 100, 42)
        self.assertEqual(d["n_users"], 3)
        self.assertGreater(d["percentile_all"], d["shuffled_percentile_all"])
        self.assertGreater(d["percentile_cold_pool"], d["shuffled_percentile_cold_pool"])

    def test_experiment_selects_on_validation_only(self):
        rng = np.random.default_rng(7)
        scrambled = self.test.copy()
        scrambled["track_id"] = rng.choice(self.music["track_id"].to_numpy(), len(scrambled))
        kwargs = dict(scalers=("standard", "robust"), mock_cold_fraction=0.2, seed=42, max_users=20, val_users=20,
                      cf_kwargs=SMALL_CF, n_clusters=3)
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_experiment(self.music, self.train, self.test, **kwargs)
            second = run_experiment(self.music, self.train, scrambled, **kwargs)
        self.assertEqual(first["selected_scaler"], second["selected_scaler"])
        self.assertEqual(first["validation"], second["validation"])
        self.assertEqual(set(first["test"]), {"standard", "robust"})
        for field in ("cb_ndcg10", "cb_mock_percentile", "shuffled_percentile_all", "hybrid_ndcg10",
                      "hybrid_cold_share", "fixed_ndcg10", "fixed_mock_recall20"):
            self.assertIn(field, first["test"]["standard"])


if __name__ == "__main__":
    unittest.main()
