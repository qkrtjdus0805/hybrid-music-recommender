"""Checks for the final model. No real data needed.
Run: python -m unittest test_final_model.py -v
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from collaborative import CollaborativeRecommender
from content_based import ContentBasedRecommender
from final_model import (FINAL_BETA, FINAL_CAP, FINAL_GAMMA, ArtistAffinity, ArtistCB, ArtistMF, build_final_model,
                         evaluate_final, list_diversity, main)
from hybrid import HybridRecommender, mask_mock_cold
from test_hybrid import make_fixture

SMALL_CF = {"n_factors": 8, "n_epochs": 3}


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class FinalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train, cls.test = make_fixture()
        cls.model = quiet(build_final_model, cls.music, cls.train, 42, SMALL_CF, 3)

    def test_settings_are_the_selected_ones(self):
        self.assertEqual((FINAL_BETA, FINAL_GAMMA, FINAL_CAP), (0.9, 5.0, 5))
        self.assertEqual(self.model.cb.beta, 0.9)

    def test_recommendations_keep_the_artist_cap(self):
        music = self.music.copy()
        music["artist"] = ["A" if i % 2 else "B" for i in range(len(music))]
        model = quiet(build_final_model, music, self.train, 42, SMALL_CF, 3)
        recs = model.recommend_user("U1", 10)
        self.assertEqual(len(recs), 10)
        self.assertEqual(recs["artist"].value_counts().to_dict(), {"A": 5, "B": 5})

    def test_artist_cap_scales_with_list_length(self):
        music = self.music.copy()
        music["artist"] = ["A" if i % 2 else "B" for i in range(len(music))]
        model = quiet(build_final_model, music, self.train, 42, SMALL_CF, 3)
        recs = model.recommend_user("U1", 4)
        self.assertEqual(recs["artist"].value_counts().max(), 2)

    def test_scores_match_a_manual_composition(self):
        cb = ContentBasedRecommender(n_clusters=3).fit(self.music, self.train)
        cf = quiet(CollaborativeRecommender(method="als", random_state=42, **SMALL_CF).fit, self.train, self.music)
        affinity = ArtistAffinity(cb, self.train)
        ids = cb.music_info["track_id"].to_numpy(dtype=str)
        manual = HybridRecommender(ArtistCB(cb, affinity, 0.9), ArtistMF(cf, ids, affinity.codes, 5.0))
        np.testing.assert_allclose(self.model.score_user_items("U1")["S_Hybrid"],
                                   manual.score_user_items("U1")["S_Hybrid"])

    def test_new_songs_are_not_capped_by_alpha(self):
        s = self.model.score_user_items("U1")
        new = self.model.n_listeners == 0
        self.assertTrue((s["alpha"][new] <= 1.0).all())
        self.assertTrue((s["S_MF"][new] > 0).any())

    def test_recommendations_exclude_heard_songs(self):
        recs = self.model.recommend_user("U1", 5)
        self.assertEqual(len(recs), 5)
        self.assertTrue(set(recs["track_id"]).isdisjoint(self.model.cb.user_listened_tracks["U1"]))

    def test_evaluation_trains_without_mock_cold_tracks(self):
        report, model = quiet(evaluate_final, self.music, self.train, self.test, 0.2, 42, 20, SMALL_CF, 3)
        _, cold = mask_mock_cold(self.train, 0.2, 42)
        self.assertTrue(set(model.cf.items).isdisjoint(cold))
        self.assertEqual(report["beta"], 0.9)
        self.assertEqual(report["gamma"], 5.0)
        self.assertEqual(report["cap"], 5)
        self.assertEqual(report["n_mock_cold"], len(cold))
        self.assertIn("Hybrid_no_cap", report["models"])

    def test_evaluation_reports_the_capped_list_as_final(self):
        report, _ = quiet(evaluate_final, self.music, self.train, self.test, 0.2, 42, 20, SMALL_CF, 3)
        self.assertAlmostEqual(report["no_cap"]["ndcg10"], report["models"]["Hybrid_no_cap"]["overall"]["@10"]["ndcg"])
        self.assertEqual(report["list_diversity"]["final"]["mean_distinct_artists"],
                         report["final"]["mean_distinct_artists"])
        for field in ("recall20_known_artist", "recall20_unknown_artist", "known_artist_answer_share"):
            self.assertIn(field, report["final"])
        self.assertEqual(set(report["confidence_intervals"]),
                         {"ndcg10_vs_mf", "ndcg10_vs_fixed_artist", "ndcg10_vs_no_cap",
                          "mock_recall20_vs_fixed_artist", "mock_recall20_vs_no_cap"})


class _StubRanker:
    """Four songs by three artists, ranked by the given score; the fourth song has no artist."""

    candidate_ids = np.array(["a", "b", "c", "d"])

    def score_user_items(self, _user):
        return {"S_Hybrid": np.array([0.9, 0.8, 0.7, 0.6])}

    def seen_mask(self, _user):
        return np.zeros(4, dtype=bool)


class _StubAffinity:
    codes = np.array([0, 0, 1, -1])

    def scores(self, _user):
        return np.array([1.0, 1.0, 0.0, 0.0])


class ListDiversityTests(unittest.TestCase):
    def test_counts_distinct_artists_and_heard_share(self):
        d = list_diversity(_StubRanker(), _StubAffinity(), ["U"], k=3)
        self.assertEqual(d["mean_distinct_artists"], 2.0)
        self.assertAlmostEqual(d["heard_artist_share"], 2 / 3)
        self.assertEqual(d["n_users"], 1)

    def test_missing_artist_is_not_an_artist(self):
        d = list_diversity(_StubRanker(), _StubAffinity(), ["U"], k=4)
        self.assertEqual(d["mean_distinct_artists"], 2.0)
        self.assertAlmostEqual(d["heard_artist_share"], 2 / 4)

    def test_report_names_the_real_components_and_compares_lists(self):
        music, train, test = make_fixture()
        report, _ = quiet(evaluate_final, music, train, test, 0.2, 42, 20, SMALL_CF, 3)
        self.assertEqual(set(report["models"]), {"CB_artist", "MF_estimated", "Hybrid_no_cap"})
        self.assertEqual(set(report["list_diversity"]), {"final", "mf", "fixed_artist"})
        self.assertLessEqual(report["list_diversity"]["final"]["mean_distinct_artists"], 10)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        music, train, test = make_fixture()
        music.to_csv(self.dir / "music.csv", index=False)
        train.to_csv(self.dir / "train.csv", index=False)
        test.to_csv(self.dir / "test.csv", index=False)
        self.common = ["--music", str(self.dir / "music.csv"), "--train-history", str(self.dir / "train.csv"),
                       "--output", str(self.dir / "out"), "--factors", "8", "--epochs", "3", "--clusters", "3"]

    def test_evaluate_writes_metrics(self):
        quiet(main, ["evaluate", *self.common, "--test-history", str(self.dir / "test.csv"), "--max-users", "6",
                     "--mock-cold-fraction", "0.2"])
        metrics = json.loads((self.dir / "out" / "final_evaluation_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["gamma"], 5.0)
        self.assertTrue((self.dir / "out" / "mock_cold_tracks.csv").exists())

    def test_recommend_writes_top_n(self):
        quiet(main, ["recommend", *self.common, "--user", "U1", "--top-n", "4"])
        lines = (self.dir / "out" / "personalized_top_final.csv").read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(len(lines), 5)

    def test_recommend_unknown_user_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, "NOPE"):
            quiet(main, ["recommend", *self.common, "--user", "NOPE"])
        self.assertFalse((self.dir / "out" / "personalized_top_final.csv").exists())


if __name__ == "__main__":
    unittest.main()
