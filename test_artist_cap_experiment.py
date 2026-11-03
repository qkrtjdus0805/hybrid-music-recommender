"""Checks for limiting how many songs one artist can place in a list. No real data needed.
Run: python -m unittest test_artist_cap_experiment.py -v
"""
import contextlib
import io
import unittest

import numpy as np

from artist_cap_experiment import cap_top_k, choose_cap, evaluate_capped, run_cap_experiment
from final_model import build_final_model
from hybrid import evaluate_hybrid, mask_mock_cold, top_k_indices
from test_hybrid import make_fixture

SMALL_CF = {"n_factors": 8, "n_epochs": 3}
IDS = np.array(["a", "b", "c", "d", "e"])
SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
NONE_SEEN = np.zeros(5, dtype=bool)


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class CapTests(unittest.TestCase):
    def test_no_cap_equals_the_plain_ranking(self):
        codes = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(cap_top_k(SCORES, IDS, NONE_SEEN, codes, None, 3),
                                      top_k_indices(SCORES, IDS, NONE_SEEN, 3))

    def test_cap_skips_extra_songs_by_one_artist(self):
        codes = np.array([0, 0, 0, 1, 2])
        self.assertEqual(cap_top_k(SCORES, IDS, NONE_SEEN, codes, 1, 3).tolist(), [0, 3, 4])

    def test_cap_two_keeps_two(self):
        codes = np.array([0, 0, 0, 1, 2])
        self.assertEqual(cap_top_k(SCORES, IDS, NONE_SEEN, codes, 2, 3).tolist(), [0, 1, 3])

    def test_songs_without_an_artist_are_not_limited(self):
        codes = np.array([-1, -1, -1, 1, 1])
        self.assertEqual(cap_top_k(SCORES, IDS, NONE_SEEN, codes, 1, 3).tolist(), [0, 1, 2])

    def test_list_is_filled_when_too_few_artists_exist(self):
        codes = np.zeros(5, dtype=int)
        self.assertEqual(cap_top_k(SCORES, IDS, NONE_SEEN, codes, 1, 3).tolist(), [0, 1, 2])

    def test_heard_songs_stay_excluded(self):
        codes = np.arange(5)
        seen = np.array([True, False, False, False, False])
        self.assertEqual(cap_top_k(SCORES, IDS, seen, codes, 1, 2).tolist(), [1, 2])


class ChooseCapTests(unittest.TestCase):
    def test_picks_the_tightest_cap_that_keeps_both_metrics(self):
        rows = {"none": {"ndcg10": 0.10, "mock_recall20": 0.10}, "1": {"ndcg10": 0.05, "mock_recall20": 0.10},
                "2": {"ndcg10": 0.095, "mock_recall20": 0.095}, "3": {"ndcg10": 0.10, "mock_recall20": 0.10}}
        self.assertEqual(choose_cap(rows), "2")

    def test_keeps_no_cap_when_nothing_qualifies(self):
        rows = {"none": {"ndcg10": 0.10, "mock_recall20": 0.10}, "1": {"ndcg10": 0.05, "mock_recall20": 0.10}}
        self.assertEqual(choose_cap(rows), "none")

    def test_a_better_score_still_qualifies(self):
        rows = {"none": {"ndcg10": 0.10, "mock_recall20": 0.10}, "1": {"ndcg10": 0.20, "mock_recall20": 0.30}}
        self.assertEqual(choose_cap(rows), "1")


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train, cls.test = make_fixture()
        cls.masked, cls.cold = mask_mock_cold(cls.train, 0.2, 42)
        cls.model = quiet(build_final_model, cls.music, cls.masked, 42, SMALL_CF, 3)
        cls.rows, cls.per_user = evaluate_capped(cls.model, cls.test, cls.cold, (None, 1, 3), 100, 42)

    def test_no_cap_reproduces_the_standard_evaluation(self):
        full = evaluate_hybrid(self.model, self.test, self.cold, max_users=100, seed=42)
        model = full["models"]["Hybrid"]
        row = self.rows["none"]
        self.assertAlmostEqual(row["ndcg10"], model["overall"]["@10"]["ndcg"])
        self.assertAlmostEqual(row["recall10"], model["overall"]["@10"]["recall"])
        self.assertAlmostEqual(row["mock_recall20"], model["bucket_recall"]["@20"]["mock_cold"])
        self.assertAlmostEqual(row["exposure10_25plus"], model["exposure"]["@10"]["25+"])
        self.assertAlmostEqual(row["coverage10"], model["catalog_coverage"]["@10"])
        self.assertAlmostEqual(row["top10_cold_share"], full["top10_cold_share"])

    def test_tighter_cap_gives_more_artists(self):
        self.assertGreaterEqual(self.rows["1"]["mean_distinct_artists"], self.rows["none"]["mean_distinct_artists"])
        self.assertLessEqual(self.rows["3"]["mean_distinct_artists"], 10)

    def test_cap_one_gives_ten_artists_when_enough_exist(self):
        music = self.music.copy()
        music["artist"] = [f"Artist {i}" for i in range(len(music))]
        model = quiet(build_final_model, music, self.masked, 42, SMALL_CF, 3)
        rows, _ = evaluate_capped(model, self.test, self.cold, (1,), 100, 42)
        self.assertEqual(rows["1"]["mean_distinct_artists"], 10.0)

    def test_per_user_table_aligns_across_caps(self):
        self.assertEqual(list(self.per_user["none"].index), list(self.per_user["1"].index))
        self.assertEqual(list(self.per_user["none"].columns), ["ndcg10", "mock_recall20"])


class ExperimentTests(unittest.TestCase):
    def test_selection_uses_validation_only(self):
        music, train, test = make_fixture()
        scrambled = test.copy()
        scrambled["track_id"] = np.random.default_rng(7).choice(music["track_id"].to_numpy(), len(scrambled))
        kwargs = dict(caps=(None, 2), mock_cold_fraction=0.2, seed=42, max_users=20, val_users=20,
                      cf_kwargs=SMALL_CF, n_clusters=3)
        first = quiet(run_cap_experiment, music, train, test, **kwargs)
        second = quiet(run_cap_experiment, music, train, scrambled, **kwargs)
        self.assertEqual(first["selected_cap"], second["selected_cap"])
        self.assertEqual(first["validation"], second["validation"])
        self.assertEqual(set(first["test"]), {"none", "2"})
        self.assertIn("confidence_intervals", first)


if __name__ == "__main__":
    unittest.main()
