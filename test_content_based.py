"""Small deterministic checks for the Stage 1 content-based module.
Run: python -m unittest test_content_based.py -v
"""
import unittest
import numpy as np
import pandas as pd
from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender, split_history_by_user


class Stage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(10)
        n = 36
        tracks = pd.DataFrame({
            "track_id": [f"TR_{i:03}" for i in range(n)],
            "name": [f"Song {i}" for i in range(n)],
            "artist": [f"Artist {i % 4}" for i in range(n)],
            "genre": ["Rock" if i % 2 else "Pop" for i in range(n)],
            "tags": ["guitar live" if i % 2 else "dance upbeat" for i in range(n)],
        })
        for col in AUDIO_FEATURE_COLS:
            tracks[col] = rng.uniform(0.05, 1.0, size=n)
        tracks["tempo"] *= 160
        tracks["loudness"] *= -30
        history = pd.DataFrame([
            {"user_id": user, "track_id": f"TR_{i:03}", "playcount": i + 1}
            for user in ("U1", "U2", "U3") for i in range(12)
        ])
        cls.tracks = tracks
        cls.train, cls.test = split_history_by_user(history, seed=42)
        cls.model = ContentBasedRecommender(n_clusters=4).fit(tracks, cls.train)

    def test_userwise_split_has_no_overlap(self):
        a = set(map(tuple, self.train[["user_id", "track_id"]].values))
        b = set(map(tuple, self.test[["user_id", "track_id"]].values))
        self.assertFalse(a.intersection(b))
        self.assertEqual(len(a) + len(b), 36)

    def test_scores_are_finite_and_in_range(self):
        scores = self.model.score_user_items("U1")
        self.assertEqual(len(scores), len(self.tracks))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_unheard_recommendations(self):
        recs = self.model.recommend_user("U1", top_n=10)
        listened = set(self.train.loc[self.train["user_id"] == "U1", "track_id"])
        self.assertEqual(len(recs), 10)
        self.assertTrue(set(recs.track_id).isdisjoint(listened))
        self.assertTrue(recs.S_CB.is_monotonic_decreasing)

    def test_new_songs_reuse_scaler_and_clusters(self):
        new_songs = self.tracks.iloc[:3].drop(columns=["genre", "tags"])
        feat, cluster = self.model.transform_tracks(new_songs)
        self.assertEqual(feat.shape, (3, self.model.track_features.shape[1]))
        self.assertEqual(cluster.shape, (3,))
        scores = self.model.score_user_items("U1", feat)
        self.assertEqual(len(scores), 3)

    def test_cluster_metrics(self):
        m = self.model.evaluate_clustering(sample_size=25)
        self.assertEqual(sum(m["cluster_sizes"]), len(self.tracks))
        self.assertIsNotNone(m["silhouette_sample"])


if __name__ == "__main__":
    unittest.main()
