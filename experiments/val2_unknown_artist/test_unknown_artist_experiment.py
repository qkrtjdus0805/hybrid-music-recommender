"""Checks for evaluating recommendations among artists the user has never heard. No real data needed.
Run: python -m unittest experiments.val2_unknown_artist.test_unknown_artist_experiment -v
"""
import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from final_model import paired_bootstrap
from hybrid import assign_buckets
from tests.test_hybrid import make_fixture
from experiments.val2_unknown_artist.unknown_artist_experiment import evaluate_unknown, run_unknown_artist_experiment, unknown_candidates

SMALL_CF = {"n_factors": 8, "n_epochs": 3}
IDS = np.array(["a", "b", "c", "d", "e", "f"])
CODES = np.array([0, 0, 1, 1, 2, -1])  # f has no artist
LABELS = assign_buckets(IDS, np.array([30, 30, 30, 0, 2, 30]), {"d"})


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def history(rows):
    return pd.DataFrame(rows, columns=["user_id", "track_id", "playcount"])


def allowed_of(user):
    """U1 knows artist 0, U2 knows artists 0 and 1. Song f has no artist, so it is never allowed."""
    return {"U1": np.array([0, 0, 1, 1, 1, 0], dtype=bool), "U2": np.array([0, 0, 0, 0, 1, 0], dtype=bool),
            "U3": np.array([0, 0, 1, 1, 1, 0], dtype=bool)}[user]


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.affinity = SimpleNamespace(codes=CODES, scores=lambda user: np.array([0.5, 0.5, 0, 0, 0, 0]))

    def test_heard_artists_and_missing_artists_are_removed(self):
        seen = np.zeros(6, dtype=bool)
        self.assertEqual(unknown_candidates(self.affinity, seen, "U1").tolist(), [False, False, True, True, True, False])

    def test_heard_songs_are_removed(self):
        seen = np.array([0, 0, 1, 0, 0, 0], dtype=bool)
        self.assertEqual(unknown_candidates(self.affinity, seen, "U1").tolist(), [False, False, False, True, True, False])


class EvaluationTests(unittest.TestCase):
    def evaluate(self, scorers, test, users=("U1", "U2")):
        return evaluate_unknown(scorers, IDS, CODES, LABELS, allowed_of, test, users, max_users=100, seed=42)

    def test_answers_by_heard_artists_do_not_count(self):
        test = history([("U1", "a", 1), ("U1", "c", 1)])
        rows, _ = self.evaluate({"m": lambda u: np.array([9, 8, 7, 1, 1, 1.0])}, test)
        self.assertEqual(rows["m"]["recall10"], 1.0)  # only c is an answer, and a and b cannot be listed

    def test_a_perfect_scorer_gets_full_marks(self):
        test = history([("U1", "c", 1), ("U1", "e", 1)])
        rows, _ = self.evaluate({"m": lambda u: np.array([0, 0, 9, 1, 8, 0.0])}, test)
        self.assertEqual(rows["m"]["ndcg10"], 1.0)
        self.assertEqual(rows["m"]["recall20"], 1.0)

    def test_a_scorer_that_prefers_other_songs_ranks_the_answer_lower(self):
        test = history([("U1", "e", 1)])
        rows, _ = self.evaluate({"m": lambda u: np.array([0, 0, 9, 8, 1, 0.0])}, test)
        self.assertAlmostEqual(rows["m"]["ndcg10"], 1 / np.log2(4))

    def test_users_without_unknown_artist_answers_are_left_out(self):
        test = history([("U1", "c", 1), ("U2", "a", 1)])
        rows, per_user = self.evaluate({"m": lambda u: np.ones(6)}, test)
        self.assertEqual(rows["m"]["n_users"], 1)
        self.assertEqual(list(per_user["m"].index), ["U1"])

    def test_only_eligible_users_are_evaluated(self):
        test = history([("U1", "c", 1), ("U3", "c", 1)])
        _, per_user = self.evaluate({"m": lambda u: np.ones(6)}, test)
        self.assertEqual(list(per_user["m"].index), ["U1"])

    def test_buckets_and_exposure_use_the_allowed_list(self):
        test = history([("U1", "d", 1), ("U1", "e", 1)])  # d is a new song, e has 2 listeners
        rows, per_user = self.evaluate({"m": lambda u: np.array([0, 0, 1, 9, 2, 0.0])}, test)
        row = rows["m"]
        self.assertEqual(row["new_recall20"], 1.0)
        self.assertEqual(row["recall20_1_4"], 1.0)
        self.assertAlmostEqual(row["exposure10_new"], 1 / 3)
        self.assertAlmostEqual(row["exposure10_1_4"], 1 / 3)
        self.assertAlmostEqual(row["exposure10_25plus"], 1 / 3)
        self.assertEqual(row["distinct_artists"], 2.0)
        self.assertEqual(per_user["m"].loc["U1", "new_recall20"], 1.0)

    def test_new_song_recall_is_nan_for_users_without_such_answers(self):
        test = history([("U1", "e", 1)])
        rows, per_user = self.evaluate({"m": lambda u: np.ones(6)}, test)
        self.assertIsNone(rows["m"]["new_recall20"])
        self.assertTrue(np.isnan(per_user["m"].loc["U1", "new_recall20"]))

    def test_every_model_sees_the_same_users(self):
        test = history([("U1", "c", 1), ("U2", "e", 1)])
        _, per_user = self.evaluate({"a": lambda u: np.ones(6), "b": lambda u: np.arange(6.0)}, test)
        self.assertEqual(list(per_user["a"].index), list(per_user["b"].index))
        self.assertFalse(np.isnan(paired_bootstrap(per_user["a"]["ndcg10"], per_user["b"]["ndcg10"])["mean_diff"]))


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        music, train, test = make_fixture(n_tracks=60, n_users=10, tracks_per_user=20)
        cls.music, cls.train, cls.test = music, train, test
        cls.kwargs = dict(new_song_listeners=(25.0, 225.0), mock_cold_fraction=0.2, seed=42, max_users=20, val_users=20,
                          cf_kwargs=SMALL_CF, n_clusters=3)
        cls.result = quiet(run_unknown_artist_experiment, music, train, test, **cls.kwargs)

    def test_selection_uses_validation_only(self):
        scrambled = self.test.copy()
        scrambled["track_id"] = np.random.default_rng(7).choice(self.music["track_id"].to_numpy(), len(scrambled))
        second = quiet(run_unknown_artist_experiment, self.music, self.train, scrambled, **self.kwargs)
        self.assertEqual(self.result["selected_n_new"], second["selected_n_new"])
        self.assertEqual(self.result["validation"], second["validation"])

    def test_reports_every_model_on_test(self):
        expected = {"Random", "Popularity", "CB", "MF", "Feature_audio", "Feature_audio_tags", "Hybrid",
                    "Hybrid_audio", "Hybrid_audio_tags"}
        self.assertEqual(set(self.result["test"]), expected)
        self.assertIn("ndcg10_vs_MF", self.result["confidence_intervals"]["Hybrid_audio"])
        self.assertIn("holdout_r2", self.result)


if __name__ == "__main__":
    unittest.main()
