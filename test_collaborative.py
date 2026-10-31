"""Quick checks for the Stage 2 collaborative model.
Run: python -m unittest test_collaborative.py -v
"""
import unittest
import numpy as np
import pandas as pd
from collaborative import CollaborativeRecommender
from content_based import split_history_by_user


class Stage2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.tracks = pd.DataFrame({"track_id": [f"TR_{i:03}" for i in range(36)],
                                   "name": [f"Song {i}" for i in range(36)]})
        history = pd.DataFrame([
            {"user_id": u, "track_id": f"TR_{t:03}", "playcount": int(rng.integers(1, 20))}
            for u in ("U1", "U2", "U3", "U4") for t in rng.choice(36, 15, replace=False)])
        cls.train, cls.test = split_history_by_user(history)
        cls.als = CollaborativeRecommender(n_factors=8, n_epochs=3).fit(cls.train, cls.tracks)
        cls.sgd = CollaborativeRecommender("sgd", n_factors=8, n_epochs=3).fit(cls.train, cls.tracks)

    def test_default_method_is_als(self):
        self.assertEqual(CollaborativeRecommender().method, "als")

    def test_scores_in_range(self):
        for model in (self.als, self.sgd):
            s = model.score_user_items("U1")
            self.assertEqual(len(s), len(self.tracks))
            self.assertTrue(np.all((s >= 0) & (s <= 1)))

    def test_cold_start_is_zero(self):
        s = self.als.score_user_items("U1", ["UNSEEN_1", "UNSEEN_2"])
        self.assertTrue(np.all(s == 0.0))
        a = self.als.compute_dynamic_alpha(["UNSEEN_1"])
        self.assertAlmostEqual(a[0], 1.0)

    def test_recommend_excludes_listened(self):
        recs = self.als.recommend_user("U1", top_n=5)
        listened = set(self.train.loc[self.train["user_id"] == "U1", "track_id"])
        self.assertTrue(set(recs["track_id"]).isdisjoint(listened))

    def test_evaluate_runs(self):
        for model in (self.als, self.sgd):
            m = model.evaluate(self.test, k_values=(5,))
            self.assertTrue(0 <= m["@5"]["ndcg"] <= 1)
        self.assertIn("rmse", self.sgd.evaluate(self.test, k_values=(5,)))


if __name__ == "__main__":
    unittest.main()
