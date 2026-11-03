"""Checks for the Stage 3 hybrid model. No real data needed.
Run: python -m unittest test_hybrid.py -v
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from collaborative import CollaborativeRecommender
from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender, split_history_by_user
from hybrid import (HybridRecommender, _percentile_unseen, assign_buckets, evaluate_hybrid, evaluate_spotify, fold_in_raw_scores,
                    link_to_msd, load_spotify, main, make_shared_split, mask_mock_cold, normalize_key,
                    ranking_metrics, run_msd_evaluation, select_config, split_playlists, top_k_indices)


def make_fixture(n_tracks: int = 40, n_users: int = 6, tracks_per_user: int = 15, seed: int = 0):
    """Small synthetic catalogue; tracks after index 29 are never listened to."""
    rng = np.random.default_rng(seed)
    ids = [f"TR_{i:03}" for i in range(n_tracks)]
    music = pd.DataFrame(rng.random((n_tracks, len(AUDIO_FEATURE_COLS))), columns=AUDIO_FEATURE_COLS)
    music.insert(0, "track_id", ids)
    music.insert(1, "name", [f"Song {i}" for i in range(n_tracks)])
    music.insert(2, "artist", [f"Artist {i % 7}" for i in range(n_tracks)])
    music["genre"] = ["rock" if i % 2 else "pop" for i in range(n_tracks)]
    music["tags"] = ["indie, 90s" if i % 3 else "dance" for i in range(n_tracks)]
    rows = [{"user_id": f"U{u}", "track_id": ids[t], "playcount": int(rng.integers(1, 20))}
            for u in range(1, n_users + 1)
            for t in rng.choice(30, tracks_per_user, replace=False)]
    train, test = split_history_by_user(pd.DataFrame(rows))
    return music, train, test


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train, cls.test = make_fixture()

    def test_mock_cold_removes_train_rows_and_is_deterministic(self):
        masked, cold = mask_mock_cold(self.train, 0.10, 42)
        n_tracks = self.train["track_id"].nunique()
        self.assertEqual(len(cold), max(1, round(n_tracks * 0.10)))
        self.assertTrue(set(masked["track_id"]).isdisjoint(cold))
        self.assertEqual(set(masked["track_id"]) | set(cold), set(self.train["track_id"]))
        np.testing.assert_array_equal(cold, mask_mock_cold(self.train, 0.10, 42)[1])

    def test_mock_cold_rejects_bad_fraction(self):
        for bad in (0.0, 1.0):
            with self.assertRaises(ValueError):
                mask_mock_cold(self.train, bad, 42)

    def test_make_shared_split_has_no_overlap(self):
        history = pd.concat([self.train, self.test])[["track_id", "user_id", "playcount"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            history.to_csv(path, index=False)
            train, test = make_shared_split(path, set(self.music["track_id"]), 1.0, 42)
        pairs = lambda df: set(zip(df["user_id"], df["track_id"]))
        self.assertFalse(pairs(train) & pairs(test))
        self.assertEqual(len(train) + len(test), len(history))


def fit_models(music, train):
    cb = ContentBasedRecommender(n_clusters=3).fit(music, train)
    with contextlib.redirect_stdout(io.StringIO()):
        cf = CollaborativeRecommender(n_factors=8, n_epochs=3).fit(train, music)
    return cb, cf


class HybridModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train, cls.test = make_fixture()
        cls.cb, cls.cf = fit_models(cls.music, cls.train)
        cls.hybrid = HybridRecommender(cls.cb, cls.cf)

    def test_scores_in_range_and_formula(self):
        s = self.hybrid.score_user_items("U1")
        np.testing.assert_allclose(s["S_Hybrid"], s["alpha"] * s["S_CB"] + (1 - s["alpha"]) * s["S_MF"])
        self.assertTrue(((s["S_Hybrid"] >= 0) & (s["S_Hybrid"] <= 1)).all())

    def test_candidate_order_matches_both_models(self):
        s = self.hybrid.score_user_items("U1")
        ids = self.hybrid.candidate_ids
        np.testing.assert_allclose(s["S_MF"], self.cf.score_user_items("U1", ids))
        np.testing.assert_allclose(s["S_CB"], self.cb.score_user_items("U1"))

    def test_cold_song_equals_cb(self):
        s = self.hybrid.score_user_items("U1")
        cold = self.cf.get_item_listener_counts(self.hybrid.candidate_ids) == 0
        self.assertTrue(cold.any())
        np.testing.assert_allclose(s["alpha"][cold], 1.0)
        np.testing.assert_allclose(s["S_Hybrid"][cold], s["S_CB"][cold])

    def test_alpha_decreases_with_listeners(self):
        s = self.hybrid.score_user_items("U1")
        n = self.cf.get_item_listener_counts(self.hybrid.candidate_ids)
        order = np.argsort(n)
        self.assertTrue((np.diff(s["alpha"][order]) <= 1e-12).all())

    def test_fixed_alpha_caps_cold_score(self):
        h = HybridRecommender(self.cb, self.cf, alpha_mode="fixed", fixed_alpha=0.5)
        s = h.score_user_items("U1")
        cold = self.cf.get_item_listener_counts(h.candidate_ids) == 0
        self.assertTrue((s["S_Hybrid"][cold] <= 0.5).all())
        np.testing.assert_allclose(s["alpha"], 0.5)

    def test_minmax_flat_scores_become_zero(self):
        h = HybridRecommender(self.cb, self.cf, calibration="minmax")
        flat = np.full(len(h.candidate_ids), 0.7)
        with mock.patch.object(self.cb, "score_user_items", return_value=flat):
            s = h.score_user_items("U1")
        self.assertFalse(np.isnan(s["S_Hybrid"]).any())
        np.testing.assert_array_equal(s["S_CB"], 0.0)

    def test_minmax_uses_unseen_range(self):
        h = HybridRecommender(self.cb, self.cf, calibration="minmax")
        s = h.score_user_items("U1")
        unseen = ~h.seen_mask("U1")
        self.assertAlmostEqual(s["S_CB"][unseen].min(), 0.0)
        self.assertAlmostEqual(s["S_CB"][unseen].max(), 1.0)

    def test_percentile_handles_ties_and_seen(self):
        seen = np.array([False, False, False, False, True])
        pct = _percentile_unseen(np.array([.1, .2, .2, .9, .5]), seen)
        np.testing.assert_allclose(pct[:4], [0.0, 0.5, 0.5, 1.0])
        self.assertTrue(0.0 <= pct[4] <= 1.0)

    def test_percentile_pool_leaves_others_at_zero(self):
        seen = np.array([False, False, False])
        pool = np.array([False, True, True])
        np.testing.assert_allclose(_percentile_unseen(np.array([.9, .2, .4]), seen, pool), [0.0, 0.0, 1.0])

    def test_percentile_flat_scores_become_zero(self):
        pct = _percentile_unseen(np.full(4, 0.7), np.zeros(4, dtype=bool))
        np.testing.assert_array_equal(pct, 0.0)

    def test_rank_calibration_scales_both_scores(self):
        h = HybridRecommender(self.cb, self.cf, calibration="rank")
        s = h.score_user_items("U1")
        unseen = ~h.seen_mask("U1")
        warm = h.n_listeners > 0
        self.assertAlmostEqual(s["S_CB"][unseen].min(), 0.0)
        self.assertAlmostEqual(s["S_CB"][unseen].max(), 1.0)
        self.assertAlmostEqual(s["S_MF"][unseen & warm].max(), 1.0)
        np.testing.assert_array_equal(s["S_MF"][~warm], 0.0)
        np.testing.assert_allclose(s["S_Hybrid"], s["alpha"] * s["S_CB"] + (1 - s["alpha"]) * s["S_MF"])

    def test_rank_calibration_keeps_cb_order(self):
        h = HybridRecommender(self.cb, self.cf, calibration="rank")
        unseen = ~h.seen_mask("U1")
        raw = self.cb.score_user_items("U1")[unseen]
        pct = h.score_user_items("U1")["S_CB"][unseen]
        np.testing.assert_array_equal(np.argsort(raw, kind="stable"), np.argsort(pct, kind="stable"))

    def test_recommend_excludes_seen_and_has_columns(self):
        recs = self.hybrid.recommend_user("U1", 5)
        self.assertEqual(len(recs), 5)
        self.assertTrue(set(recs["track_id"]).isdisjoint(self.cb.user_listened_tracks["U1"]))
        self.assertTrue({"track_id", "n_listeners", "alpha", "S_CB", "S_MF", "S_Hybrid"} <= set(recs.columns))
        self.assertTrue((np.diff(recs["S_Hybrid"]) <= 1e-12).all())

    def test_top_k_ties_break_by_track_id(self):
        idx = top_k_indices(np.array([.5, .5, .9]), np.array(["b", "a", "c"]), np.array([False] * 3), 3)
        self.assertEqual(idx.tolist(), [2, 1, 0])

    def test_top_k_skips_excluded(self):
        idx = top_k_indices(np.array([.5, .9]), np.array(["a", "b"]), np.array([False, True]), 2)
        self.assertEqual(idx.tolist(), [0])

    def test_unknown_user_raises(self):
        with self.assertRaises(KeyError):
            self.hybrid.score_user_items("NOPE")

    def test_invalid_options_raise(self):
        with self.assertRaises(ValueError):
            HybridRecommender(self.cb, self.cf, alpha_mode="wrong")
        with self.assertRaises(ValueError):
            HybridRecommender(self.cb, self.cf, calibration="wrong")


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, train, cls.test = make_fixture()
        cls.masked, cls.cold = mask_mock_cold(train, 0.2, 42)
        cls.cb, cls.cf = fit_models(cls.music, cls.masked)
        cls.hybrid = HybridRecommender(cls.cb, cls.cf)

    def test_metrics_hand_example(self):
        m = ranking_metrics(["a", "b", "c"], {"b", "z"}, 3)
        self.assertAlmostEqual(m["precision"], 1 / 3)
        self.assertAlmostEqual(m["recall"], 1 / 2)
        self.assertAlmostEqual(m["ndcg"], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))

    def test_buckets(self):
        b = assign_buckets(["m", "z", "a", "b", "c"], np.array([0, 0, 4, 5, 25]), {"m"})
        self.assertEqual(b.tolist(), ["mock_cold", "sample_zero", "1-4", "5-24", "25+"])

    def test_cold_tracks_stay_in_relevant(self):
        r = evaluate_hybrid(self.hybrid, self.test, self.cold, (5,), 100, 42)
        self.assertGreater(r["bucket_test_rows"]["mock_cold"], 0)
        self.assertIn("mock_cold", r["models"]["Hybrid"]["bucket_recall"]["@5"])

    def test_mf_never_hits_cold_before_warm_exhausted(self):
        r = evaluate_hybrid(self.hybrid, self.test, self.cold, (5,), 100, 42)
        self.assertEqual(r["models"]["MF"]["bucket_recall"]["@5"]["mock_cold"], 0.0)

    def test_exposure_shares_sum_to_one(self):
        r = evaluate_hybrid(self.hybrid, self.test, self.cold, (5,), 100, 42)
        self.assertAlmostEqual(sum(r["models"]["CB"]["exposure"]["@5"].values()), 1.0)
        self.assertTrue(0 < r["models"]["CB"]["catalog_coverage"]["@5"] <= 1)

    def test_mock_cold_percentile_is_high_for_an_oracle(self):
        ids = self.hybrid.candidate_ids
        oracle = np.where(np.isin(ids, self.cold), 1.0, np.linspace(0.0, 0.5, len(ids)))
        with mock.patch.object(self.cb, "score_user_items", return_value=oracle):
            r = evaluate_hybrid(self.hybrid, self.test, self.cold, (5,), 100, 42)
        self.assertGreater(r["models"]["CB"]["mock_cold_percentile"], 0.8)

    def test_mock_cold_percentile_is_low_for_mf(self):
        r = evaluate_hybrid(self.hybrid, self.test, self.cold, (5,), 100, 42)
        self.assertLess(r["models"]["MF"]["mock_cold_percentile"], 0.5)
        self.assertTrue(0.0 <= r["models"]["CB"]["mock_cold_percentile"] <= 1.0)

    def test_dropped_user_counted(self):
        extra = pd.DataFrame([{"user_id": "GONE", "track_id": "TR_000", "playcount": 1}])
        r = evaluate_hybrid(self.hybrid, pd.concat([self.test, extra]), self.cold, (5,), 100, 42)
        self.assertEqual(r["n_users_dropped"], 1)
        self.assertEqual(r["n_users"], self.test["user_id"].nunique())

    def test_max_users_limits_evaluated_users(self):
        r = evaluate_hybrid(self.hybrid, self.test, self.cold, (5,), 2, 42)
        self.assertEqual(r["n_users"], 2)

    def test_identical_to_cb_is_not_verifiable(self):
        h = HybridRecommender(self.cb, self.cf, alpha_mode="fixed", fixed_alpha=1.0)
        r = evaluate_hybrid(h, self.test, self.cold, (5, 10), 100, 42)
        self.assertEqual(r["top10_overlap_with_cb"], 1.0)
        self.assertFalse(r["hybrid_effect_verifiable"])

    def test_different_from_cb_is_verifiable(self):
        h = HybridRecommender(self.cb, self.cf, alpha_mode="fixed", fixed_alpha=0.0)
        r = evaluate_hybrid(h, self.test, self.cold, (5, 10), 100, 42)
        self.assertLess(r["top10_overlap_with_cb"], 0.9)
        self.assertTrue(r["hybrid_effect_verifiable"])


SMALL_CF = {"n_factors": 8, "n_epochs": 3}


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train, cls.test = make_fixture()
        cls.masked, cls.cold = mask_mock_cold(cls.train, 0.2, 42)

    def test_select_config_picks_allowed_values(self):
        cb = ContentBasedRecommender(n_clusters=3).fit(self.music)
        cfg = quiet(select_config, cb, self.masked, self.music, "als", 42, 50, SMALL_CF)
        self.assertIn(cfg["calibration"], ("none", "minmax", "rank"))
        self.assertIn(cfg["fixed_alpha"], (0.25, 0.5, 0.75))
        self.assertEqual(set(cfg["validation"]["calibration"]), {"none", "minmax", "rank"})
        self.assertIn("top10_cold_share", cfg)

    def test_selected_config_does_not_depend_on_test_rows(self):
        """Replacing the TEST answers must not change anything chosen on validation."""
        rng = np.random.default_rng(7)
        scrambled = self.test.copy()
        scrambled["track_id"] = rng.choice(self.music["track_id"].to_numpy(), len(scrambled))
        first, _ = quiet(run_msd_evaluation, self.music, self.train, self.test, "als", 0.2, 42, 100, SMALL_CF)
        second, _ = quiet(run_msd_evaluation, self.music, self.train, scrambled, "als", 0.2, 42, 100, SMALL_CF)
        self.assertEqual(first["selected_config"], second["selected_config"])

    def test_unknown_cli_user_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.music.to_csv(tmp / "music.csv", index=False)
            self.train.to_csv(tmp / "train.csv", index=False)
            self.test.to_csv(tmp / "test.csv", index=False)
            with self.assertRaisesRegex(ValueError, "NOPE"):
                quiet(main, ["evaluate", "--music", str(tmp / "music.csv"), "--train-history", str(tmp / "train.csv"),
                             "--test-history", str(tmp / "test.csv"), "--output", str(tmp / "out"),
                             "--factors", "8", "--epochs", "3", "--user", "NOPE"])
            self.assertFalse((tmp / "out" / "hybrid_evaluation_metrics.json").exists())

    def test_run_msd_evaluation_reports_all_models_and_settings(self):
        result, h = quiet(run_msd_evaluation, self.music, self.train, self.test, "als", 0.2, 42, 100, SMALL_CF)
        self.assertEqual(set(result["models"]), {"CB", "MF", "Fixed", "Hybrid"})
        for key in ("selected_config", "seed", "mf_method", "n_train_rows", "n_test_rows",
                    "n_mock_cold", "calibration_comparison"):
            self.assertIn(key, result)
        self.assertEqual(h.calibration, result["selected_config"]["calibration"])
        self.assertEqual(h.alpha_mode, "dynamic")
        self.assertEqual(set(result["calibration_comparison"]), {"none", "minmax", "rank"})
        self.assertIn("mock_cold_percentile", result["calibration_comparison"]["rank"])
        self.assertEqual(result["n_train_rows"], len(self.masked))
        self.assertEqual(result["n_mock_cold"], len(self.cold))

    def test_both_models_train_without_mock_cold_tracks(self):
        _, h = quiet(run_msd_evaluation, self.music, self.train, self.test, "als", 0.2, 42, 100, SMALL_CF)
        _, cold = mask_mock_cold(self.train, 0.2, 42)
        listened = set().union(*h.cb.user_listened_tracks.values())
        self.assertTrue(listened.isdisjoint(cold))
        self.assertTrue(set(h.cf.items).isdisjoint(cold))

    def test_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.music.to_csv(tmp / "music.csv", index=False)
            self.train.to_csv(tmp / "train.csv", index=False)
            self.test.to_csv(tmp / "test.csv", index=False)
            quiet(main, ["evaluate", "--music", str(tmp / "music.csv"), "--train-history", str(tmp / "train.csv"),
                         "--test-history", str(tmp / "test.csv"), "--output", str(tmp / "out"),
                         "--factors", "8", "--epochs", "3", "--max-users", "6"])
            for name in ("hybrid_evaluation_metrics.json", "personalized_top10_hybrid.csv", "mock_cold_tracks.csv"):
                self.assertTrue((tmp / "out" / name).exists(), name)
            metrics = json.loads((tmp / "out" / "hybrid_evaluation_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("Hybrid", metrics["models"])

    def test_cli_make_split_writes_shared_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.music.to_csv(tmp / "music.csv", index=False)
            pd.concat([self.train, self.test]).to_csv(tmp / "history.csv", index=False)
            quiet(main, ["make-split", "--music", str(tmp / "music.csv"), "--history", str(tmp / "history.csv"),
                         "--user-fraction", "1.0", "--output", str(tmp)])
            train = pd.read_csv(tmp / "shared_train_history.csv")
            test = pd.read_csv(tmp / "shared_test_history.csv")
        self.assertEqual(len(train) + len(test), len(self.train) + len(self.test))


def make_spotify(music: pd.DataFrame, linked: bool = True) -> pd.DataFrame:
    """Two playlists big enough to use and one too small; rows use the real file's column names."""
    rng = np.random.default_rng(1)
    rows = []
    sizes = {"P1": 14, "P2": 12, "SMALL": 9}
    for pid, size in sizes.items():
        for pos in rng.choice(30, size, replace=False):
            source = music.iloc[pos]
            row = {c: source[c] for c in AUDIO_FEATURE_COLS if c != "time_signature"}
            row.update({"track_id": f"SP_{pos:03}", "playlist_id": pid, "playlist_genre": "rock",
                        "playlist_subgenre": "indie rock",
                        "track_name": source["name"] if linked else f"Unknown {pos}",
                        "track_artist": source["artist"] if linked else "Nobody"})
            rows.append(row)
    rows.append({**rows[0]})  # the same song twice in P1
    return pd.DataFrame(rows)


class SpotifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music, cls.train, _ = make_fixture()
        cls.cb, cls.cf = fit_models(cls.music, cls.train)
        cls.config = {"calibration": "none", "fixed_alpha": 0.5}
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "spotify.csv"
        make_spotify(cls.music).to_csv(cls.path, index=False)
        cls.spotify = load_spotify(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_normalize_key_matches_variants(self):
        self.assertEqual(normalize_key("Wonderwall (Remastered)", "OASIS!"), normalize_key("wonderwall", "Oasis"))

    def test_normalize_key_keeps_korean(self):
        self.assertEqual(normalize_key("봄날", "방탄소년단"), "봄날|방탄소년단")

    def test_normalize_key_rejects_empty_values(self):
        for bad in ((None, "a"), ("a", float("nan")), ("(Live)", "a"), ("", "")):
            self.assertIsNone(normalize_key(*bad), bad)

    def test_load_spotify_maps_schema_without_time_signature(self):
        self.assertTrue({"name", "artist", "genre", "tags", "key", "song_key"} <= set(self.spotify.columns))
        self.assertEqual(self.spotify["genre"].iloc[0], "rock indie rock")
        features, _ = self.cb.transform_tracks(self.spotify)
        self.assertEqual(len(features), len(self.spotify))

    def test_load_spotify_missing_column_names_it(self):
        path = Path(self.tmp.name) / "broken.csv"
        make_spotify(self.music).drop(columns="track_artist").to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "track_artist"):
            load_spotify(path)

    def test_split_playlists_filters_and_dedups(self):
        splits = split_playlists(self.spotify, 10, 42)
        self.assertEqual({pid for pid, _, _ in splits}, {"P1", "P2"})
        for pid, query, hidden in splits:
            self.assertTrue(set(query).isdisjoint(hidden))
            self.assertLessEqual(abs(len(query) - len(hidden)), 1)
            self.assertEqual(len(set(query)) + len(set(hidden)), {"P1": 14, "P2": 12}[pid])

    def test_link_to_msd_finds_matching_songs(self):
        link = link_to_msd(self.spotify, self.music)
        self.assertEqual(link[normalize_key("Song 3", "Artist 3")], "TR_003")

    def test_fold_in_matches_als_user_update(self):
        ids = list(self.cf.items[:2])
        Q = self.cf.Q
        F = Q[:2]
        r = np.log(2.0)
        w = self.cf.alpha_conf * r
        A = Q.T @ Q + self.cf.reg * np.eye(Q.shape[1]) + (F.T * w) @ F
        expected = Q @ np.linalg.solve(A, F.T @ np.full(2, w + 1.0))
        row = csr_matrix((np.full(2, r), ([0, 0], [0, 1])), shape=(1, len(self.cf.items)))
        knn = ((row @ self.cf.MT) @ self.cf.M).toarray().ravel()
        np.testing.assert_allclose(fold_in_raw_scores(self.cf, ids) - knn, expected)

    def test_fold_in_without_links_returns_none(self):
        self.assertIsNone(fold_in_raw_scores(self.cf, []))
        self.assertIsNone(fold_in_raw_scores(self.cf, ["NOT_TRAINED"]))

    def test_no_linked_playlist_is_cb(self):
        path = Path(self.tmp.name) / "unlinked.csv"
        make_spotify(self.music, linked=False).to_csv(path, index=False)
        r = evaluate_spotify(self.cb, self.cf, load_spotify(path), self.music, self.config)
        self.assertEqual(r["n_playlists_no_mf_link"], r["n_playlists"])
        self.assertEqual(r["top10_overlap_with_cb"], 1.0)
        self.assertFalse(r["hybrid_effect_verifiable"])
        self.assertEqual(r["link_rate_msd"], 0.0)

    def test_unlinked_playlists_rank_like_cb_even_with_linked_candidates(self):
        """Linked songs from a too-small playlist stay candidates, so alpha < 1 exists in the catalogue."""
        unlinked = make_spotify(self.music, linked=False)
        unlinked = unlinked.loc[unlinked["playlist_id"] != "SMALL"]
        linked = make_spotify(self.music, linked=True)
        linked = linked.loc[linked["playlist_id"] == "SMALL"]
        path = Path(self.tmp.name) / "mixed.csv"
        pd.concat([unlinked, linked]).to_csv(path, index=False)
        for calibration in ("none", "minmax", "rank"):
            r = evaluate_spotify(self.cb, self.cf, load_spotify(path), self.music, {"calibration": calibration})
            self.assertGreater(r["link_rate_msd"], 0.0)
            self.assertEqual(r["n_playlists_no_mf_link"], r["n_playlists"])
            self.assertEqual(r["top10_overlap_with_cb"], 1.0)
            self.assertFalse(r["hybrid_effect_verifiable"])

    def test_report_fields(self):
        r = evaluate_spotify(self.cb, self.cf, self.spotify, self.music, self.config)
        for field in ("link_rate_msd", "link_rate_mf", "playlists_with_mf_link_rate",
                      "playlists_with_msd_link_rate", "n_spotify_rows",
                      "top10_overlap_with_cb", "hybrid_effect_verifiable"):
            self.assertIn(field, r)
        self.assertEqual(r["n_spotify_rows"], len(self.spotify))
        self.assertEqual(r["playlists_with_msd_link_rate"], 1.0)
        self.assertEqual(set(r["models"]), {"CB", "CB_audio_only", "MF", "Hybrid"})
        self.assertEqual(r["n_playlists"], 2)
        self.assertGreater(r["link_rate_msd"], 0.9)

    def test_cli_writes_spotify_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            music, train, test = make_fixture()
            music.to_csv(tmp / "music.csv", index=False)
            train.to_csv(tmp / "train.csv", index=False)
            test.to_csv(tmp / "test.csv", index=False)
            quiet(main, ["evaluate", "--music", str(tmp / "music.csv"), "--train-history", str(tmp / "train.csv"),
                         "--test-history", str(tmp / "test.csv"), "--output", str(tmp / "out"),
                         "--factors", "8", "--epochs", "3", "--max-users", "6", "--spotify", str(self.path)])
            phase2 = json.loads((tmp / "out" / "spotify_evaluation_metrics.json").read_text(encoding="utf-8"))
            for field in ("spotify_file", "music_file", "train_file", "seed", "mf_method"):
                self.assertIn(field, phase2)


if __name__ == "__main__":
    unittest.main()
