"""Checks for limiting how many songs one artist can place in a list. No real data needed.
Run: python -m unittest experiments.exp5_artist_cap.test_artist_cap_experiment -v
"""
import contextlib
import io
import unittest

import numpy as np

from experiments.exp5_artist_cap.artist_cap_experiment import cap_top_k, choose_cap, evaluate_capped, run_cap_experiment
from final_model import _ranked_prefix, build_final_model
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


def full_sort_cap(scores, ids, exclude, codes, cap, k):
    """The plain definition: sort every song, then walk down the list."""
    order = np.lexsort((ids, -scores))
    picked, skipped, used = [], [], {}
    for position in order[~exclude[order]]:
        code = codes[position]
        if code >= 0 and used.get(code, 0) >= cap:
            skipped.append(position)
            continue
        used[code] = used.get(code, 0) + 1
        picked.append(position)
        if len(picked) == k:
            break
    return picked + skipped[:k - len(picked)]


class RankedPrefixTests(unittest.TestCase):
    def test_prefix_matches_the_start_of_a_full_sort(self):
        rng = np.random.default_rng(0)
        ids = np.array([f"T{i:05d}" for i in rng.permutation(5000)])
        scores = rng.integers(0, 40, 5000) / 40  # many ties, so the id order decides
        exclude = rng.random(5000) < 0.2
        order = np.lexsort((ids, -scores))
        order = order[~exclude[order]]
        prefix = _ranked_prefix(scores, ids, exclude, 100)
        self.assertGreaterEqual(len(prefix), 100)
        np.testing.assert_array_equal(prefix, order[:len(prefix)])

    def test_prefix_is_short_when_scores_are_distinct(self):
        rng = np.random.default_rng(1)
        ids = np.array([f"T{i:05d}" for i in range(5000)])
        prefix = _ranked_prefix(rng.random(5000), ids, np.zeros(5000, dtype=bool), 100)
        self.assertEqual(len(prefix), 100)

    def test_size_beyond_the_catalog_returns_every_allowed_song(self):
        ids = np.array(["a", "b", "c"])
        exclude = np.array([False, True, False])
        self.assertEqual(_ranked_prefix(np.array([0.1, 0.9, 0.5]), ids, exclude, 10).tolist(), [2, 0])

    def test_nothing_allowed_returns_an_empty_prefix(self):
        ids = np.array(["a", "b"])
        self.assertEqual(len(_ranked_prefix(np.array([0.1, 0.9]), ids, np.ones(2, dtype=bool), 5)), 0)

    def test_cap_top_k_matches_the_full_sort_definition(self):
        rng = np.random.default_rng(2)
        for n_artists, cap, k in ((2, 5, 20), (3, 1, 10), (400, 5, 10), (1, 5, 20)):  # few artists force the fallback
            ids = np.array([f"T{i:05d}" for i in rng.permutation(3000)])
            scores = rng.integers(0, 30, 3000) / 30
            exclude = rng.random(3000) < 0.1
            codes = rng.integers(-1, n_artists, 3000)
            self.assertEqual(cap_top_k(scores, ids, exclude, codes, cap, k).tolist(),
                             [int(p) for p in full_sort_cap(scores, ids, exclude, codes, cap, k)])

    def test_one_dominant_artist_needs_more_than_the_first_prefix(self):
        ids = np.array([f"T{i:05d}" for i in range(3000)])
        scores = np.linspace(1, 0, 3000)
        codes = np.zeros(3000, dtype=int)
        codes[-10:] = np.arange(1, 11)  # the only other artists sit at the very bottom
        result = cap_top_k(scores, ids, np.zeros(3000, dtype=bool), codes, 2, 10)
        self.assertEqual(result.tolist(), [0, 1] + list(range(2990, 2998)))


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
        self.assertEqual(list(self.per_user["none"].columns), ["ndcg10", "mock_recall20", "unknown_recall20"])


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
