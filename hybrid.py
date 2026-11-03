"""Stage 3 — Weighted Hybrid: S_Hybrid(u, i) = alpha(i) * S_CB(u, i) + (1 - alpha(i)) * S_MF(u, i).

README_hybrid.md describes the experiments and results; final_model.py holds the final model.
Unit tests: test_hybrid.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

from collaborative import CollaborativeRecommender
from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender, load_history, split_history_by_user


def make_shared_split(history_path: str | Path, valid_track_ids: set[str], user_fraction: float = 0.05,
                      seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample users from the raw history and hold out 20% of each user's tracks."""
    history = load_history(history_path, valid_track_ids, user_fraction=user_fraction)
    return split_history_by_user(history, test_fraction=0.2, seed=seed)


def mask_mock_cold(train: pd.DataFrame, fraction: float = 0.10,
                   seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Erase every TRAIN row of a random share of tracks, making them new songs with known answers."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    tracks = np.unique(train["track_id"].astype(str))
    n_cold = max(1, round(len(tracks) * fraction))
    cold = np.sort(np.random.default_rng(seed).choice(tracks, size=n_cold, replace=False))
    masked = train.loc[~train["track_id"].astype(str).isin(cold)].reset_index(drop=True)
    return masked, cold


def top_k_indices(scores: np.ndarray, track_ids: np.ndarray, exclude: np.ndarray, k: int) -> np.ndarray:
    """Best-first positions of up to k non-excluded candidates; ties go to the smaller track_id."""
    order = np.lexsort((track_ids, -scores))
    return order[~exclude[order]][:k]


def _minmax_unseen(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Per-listener min-max over unheard candidates, the same way S_MF is scaled."""
    unseen = scores[~seen] if not seen.all() else scores
    low, span = unseen.min(), np.ptp(unseen)
    if span <= 1e-9:
        return np.zeros_like(scores)
    return np.clip((scores - low) / span, 0.0, 1.0)


def _percentile_unseen(scores: np.ndarray, seen: np.ndarray, pool: np.ndarray | None = None) -> np.ndarray:
    """Per-listener percentile rank in [0, 1] over unheard songs in pool; songs outside pool stay 0."""
    in_pool = ~seen if pool is None else pool & ~seen
    out = np.zeros_like(scores, dtype=float)
    values = scores[in_pool]
    if len(values) < 2 or np.ptp(values) <= 1e-9:
        return out
    out[in_pool] = (rankdata(values) - 1) / (len(values) - 1)
    heard = seen if pool is None else pool & seen
    out[heard] = np.clip(np.searchsorted(np.sort(values), scores[heard]) / (len(values) - 1), 0.0, 1.0)
    return out


CALIBRATIONS = ("none", "minmax", "rank")


class HybridRecommender:
    """Combines fitted Stage 1 and Stage 2 models over the Music Info catalogue."""

    def __init__(self, cb, cf, alpha_mode: str = "dynamic", fixed_alpha: float = 0.5,
                 calibration: str = "none"):
        if alpha_mode not in ("dynamic", "fixed"):
            raise ValueError("alpha_mode must be 'dynamic' or 'fixed'")
        if calibration not in CALIBRATIONS:
            raise ValueError(f"calibration must be one of {CALIBRATIONS}")
        self.cb, self.cf = cb, cf
        self.alpha_mode, self.fixed_alpha, self.calibration = alpha_mode, fixed_alpha, calibration
        self.candidate_ids = cb.music_info["track_id"].astype(str).to_numpy()
        self.position = {t: i for i, t in enumerate(self.candidate_ids)}
        self.n_listeners = cf.get_item_listener_counts(self.candidate_ids)
        self._alpha = (cf.compute_dynamic_alpha(self.candidate_ids) if alpha_mode == "dynamic"
                       else np.full(len(self.candidate_ids), float(fixed_alpha)))

    def seen_mask(self, user_id: str) -> np.ndarray:
        """True for candidates the user listened to in TRAIN."""
        user_id = str(user_id)
        seen = self.cb.user_listened_tracks.get(user_id, set()) | self.cf.user_listened_tracks.get(user_id, set())
        mask = np.zeros(len(self.candidate_ids), dtype=bool)
        mask[[self.position[t] for t in seen if t in self.position]] = True
        return mask

    def score_user_items(self, user_id: str) -> dict[str, np.ndarray]:
        user_id = str(user_id)
        s_cb = self.cb.score_user_items(user_id)
        s_mf = self.cf.score_user_items(user_id, self.candidate_ids)
        if self.calibration == "minmax":
            s_cb = _minmax_unseen(s_cb, self.seen_mask(user_id))
        elif self.calibration == "rank":
            seen = self.seen_mask(user_id)
            s_cb = _percentile_unseen(s_cb, seen)
            s_mf = _percentile_unseen(s_mf, seen, self.n_listeners > 0)  # no MF signal for cold songs: stay 0
        hybrid = self._alpha * s_cb + (1 - self._alpha) * s_mf
        return {"S_CB": s_cb, "S_MF": s_mf, "alpha": self._alpha, "S_Hybrid": hybrid}

    def recommend_user(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        """Top-N unheard tracks by S_Hybrid, with every score that produced them."""
        scores = self.score_user_items(user_id)
        idx = top_k_indices(scores["S_Hybrid"], self.candidate_ids, self.seen_mask(user_id), top_n)
        return self._recommendation_table(idx, scores)

    def _recommendation_table(self, idx: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
        music = self.cb.music_info
        keep = [c for c in ["track_id", "name", "artist", "genre"] if c in music]
        out = music.iloc[idx][keep].reset_index(drop=True)
        out["n_listeners"] = self.n_listeners[idx]
        for name in ("alpha", "S_CB", "S_MF", "S_Hybrid"):
            out[name] = scores[name][idx]
        return out


BUCKETS = ("mock_cold", "sample_zero", "1-4", "5-24", "25+")
MODEL_SCORES = {"CB": "S_CB", "MF": "S_MF", "Hybrid": "S_Hybrid"}
OVERLAP_LIMIT = 0.9  # Top-10 overlap at or above this means Hybrid is effectively CB


def ranking_metrics(ranked: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """Precision, Recall and NDCG at k, using the same formulas as Stage 2."""
    hits = np.array([t in relevant for t in ranked[:k]], dtype=float)
    dcg = np.sum(hits / np.log2(np.arange(2, len(hits) + 2)))
    idcg = np.sum(1 / np.log2(np.arange(2, min(k, len(relevant)) + 2)))
    return {"precision": float(hits.sum() / k), "recall": float(hits.sum() / len(relevant)),
            "ndcg": float(dcg / idcg)}


def assign_buckets(track_ids, n_i: np.ndarray, mock_cold: set[str]) -> np.ndarray:
    """Label each track by its TRAIN listener count; erased tracks get their own label."""
    n_i = np.asarray(n_i)
    labels = np.select([n_i == 0, n_i < 5, n_i < 25], ["sample_zero", "1-4", "5-24"], "25+").astype(object)
    is_mock = np.array([str(t) in mock_cold for t in track_ids], dtype=bool)
    labels[is_mock] = "mock_cold"
    return labels


def _clean_test(test: pd.DataFrame) -> pd.DataFrame:
    df = test[["user_id", "track_id", "playcount"]].dropna().copy()
    df[["user_id", "track_id"]] = df[["user_id", "track_id"]].astype(str)
    df["playcount"] = pd.to_numeric(df["playcount"], errors="coerce")
    return df.loc[df["playcount"] > 0]


def evaluate_hybrid(hybrid: HybridRecommender, test: pd.DataFrame, mock_cold: np.ndarray,
                    k_values=(5, 10, 20), max_users: int = 2000, seed: int = 42) -> dict:
    """Score CB, MF and Hybrid on the same users, candidates and answers, keeping cold songs as answers."""
    positions = hybrid.position
    df = _clean_test(test)
    df = df.loc[df["track_id"].isin(positions)]
    relevant_by_user = df.groupby("user_id")["track_id"].agg(set)
    known = [u for u in relevant_by_user.index
             if u in hybrid.cb.user_profiles and u in hybrid.cf.u_map]
    n_dropped = len(relevant_by_user) - len(known)
    if not known:
        raise ValueError("No test users exist in the trained models")
    users = np.sort(np.array(known))
    if len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))

    labels = assign_buckets(hybrid.candidate_ids, hybrid.n_listeners, {str(t) for t in mock_cold})
    label_of = dict(zip(hybrid.candidate_ids, labels))
    report_ks = sorted(k_values)
    top_k = max(report_ks + [10])
    is_cold = hybrid.n_listeners == 0

    overall = {m: {k: [] for k in report_ks} for m in MODEL_SCORES}
    bucket_hits = {m: {k: {b: [] for b in BUCKETS} for k in report_ks} for m in MODEL_SCORES}
    exposure = {m: {k: collections.Counter() for k in report_ks} for m in MODEL_SCORES}
    recommended = {m: {k: set() for k in report_ks} for m in MODEL_SCORES}
    mock_percentiles = {m: [] for m in MODEL_SCORES}
    overlaps, cold_shares = [], []

    for user in users:
        scores = hybrid.score_user_items(user)
        seen = hybrid.seen_mask(user)
        relevant = relevant_by_user[user]
        relevant_buckets = collections.defaultdict(set)
        for track in relevant:
            relevant_buckets[label_of[track]].add(track)
        tops = {m: top_k_indices(scores[col], hybrid.candidate_ids, seen, top_k)
                for m, col in MODEL_SCORES.items()}
        cb_top10, hybrid_top10 = tops["CB"][:10], tops["Hybrid"][:10]
        overlaps.append(len(set(cb_top10) & set(hybrid_top10)) / max(len(hybrid_top10), 1))
        cold_shares.append(float(is_cold[hybrid_top10].mean()) if len(hybrid_top10) else 0.0)
        mock_answers = [positions[t] for t in relevant_buckets.get("mock_cold", ())]
        if mock_answers:
            for model, col in MODEL_SCORES.items():
                mock_percentiles[model].append(_percentile_unseen(scores[col], seen)[mock_answers].mean())
        for model, idx in tops.items():
            ranked = hybrid.candidate_ids[idx]
            for k in report_ks:
                head = ranked[:k]
                overall[model][k].append(ranking_metrics(list(ranked), relevant, k))
                exposure[model][k].update(labels[idx[:k]])
                recommended[model][k].update(head)
                head_set = set(head)
                for bucket, answers in relevant_buckets.items():
                    bucket_hits[model][k][bucket].append(len(head_set & answers) / len(answers))

    models = {}
    for model in MODEL_SCORES:
        models[model] = {"overall": {}, "bucket_recall": {}, "exposure": {}, "catalog_coverage": {},
                         # Mean percentile of mock-cold answers among unheard songs. Ties lower it (S_MF is
                         # 0 for most songs), so compare with another listener's profile, not with 0.5.
                         "mock_cold_percentile": (float(np.mean(mock_percentiles[model]))
                                                  if mock_percentiles[model] else None)}
        for k in report_ks:
            models[model]["overall"][f"@{k}"] = {
                name: float(np.mean([row[name] for row in overall[model][k]]))
                for name in ("precision", "recall", "ndcg")}
            models[model]["bucket_recall"][f"@{k}"] = {
                b: float(np.mean(v)) for b, v in bucket_hits[model][k].items() if v}
            slots = sum(exposure[model][k].values())
            models[model]["exposure"][f"@{k}"] = {b: exposure[model][k][b] / slots for b in BUCKETS}
            models[model]["catalog_coverage"][f"@{k}"] = len(recommended[model][k]) / len(hybrid.candidate_ids)
    overlap = float(np.mean(overlaps))
    return {
        "models": models,
        "top10_overlap_with_cb": overlap,
        "top10_cold_share": float(np.mean(cold_shares)),
        "hybrid_effect_verifiable": bool(overlap < OVERLAP_LIMIT),
        "n_users": int(len(users)),
        "n_users_dropped": int(n_dropped),
        "bucket_test_rows": {b: int((df["track_id"].map(label_of) == b).sum()) for b in BUCKETS},
    }


FIXED_ALPHAS = (0.25, 0.5, 0.75)


def _ndcg10(result: dict) -> float:
    return result["models"]["Hybrid"]["overall"]["@10"]["ndcg"]


def select_config(cb, masked_train: pd.DataFrame, music: pd.DataFrame, method: str, seed: int = 42,
                  max_users: int = 500, cf_kwargs: dict | None = None) -> dict:
    """Choose calibration and fixed alpha by NDCG@10 on a validation split carved from TRAIN."""
    fit_part, validation = split_history_by_user(masked_train, test_fraction=0.10, seed=seed)
    cb.fit_user_profiles(fit_part)
    cf = CollaborativeRecommender(method=method, random_state=seed, **(cf_kwargs or {})).fit(fit_part, music)
    no_cold: np.ndarray = np.array([], dtype=str)

    by_calibration = {
        cal: evaluate_hybrid(HybridRecommender(cb, cf, calibration=cal), validation, no_cold, (10,), max_users, seed)
        for cal in CALIBRATIONS}
    calibration = max(by_calibration, key=lambda cal: _ndcg10(by_calibration[cal]))
    by_alpha = {
        a: evaluate_hybrid(HybridRecommender(cb, cf, "fixed", a, calibration), validation, no_cold, (10,),
                           max_users, seed)
        for a in FIXED_ALPHAS}
    fixed_alpha = max(FIXED_ALPHAS, key=lambda a: _ndcg10(by_alpha[a]))
    return {
        "calibration": calibration,
        "fixed_alpha": fixed_alpha,
        "top10_cold_share": by_calibration[calibration]["top10_cold_share"],
        "validation": {
            "calibration": {c: {"ndcg@10": _ndcg10(r), "top10_cold_share": r["top10_cold_share"]}
                            for c, r in by_calibration.items()},
            "fixed_alpha": {str(a): _ndcg10(r) for a, r in by_alpha.items()},
        },
    }


def run_msd_evaluation(music: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, method: str = "als",
                       mock_cold_fraction: float = 0.10, seed: int = 42, max_users: int = 2000,
                       cf_kwargs: dict | None = None) -> tuple[dict, HybridRecommender]:
    """Phase 1: erase mock-cold tracks, choose settings on validation, then evaluate on TEST once."""
    masked, cold = mask_mock_cold(train, mock_cold_fraction, seed)
    print(f"Mock-cold: {len(cold):,} tracks erased; TRAIN rows {len(train):,} -> {len(masked):,}")
    cb = ContentBasedRecommender(n_clusters=6, random_state=seed).fit(music)

    print("Choosing settings on validation...")
    config = select_config(cb, masked, music, method, seed, cf_kwargs=cf_kwargs)
    print(f"Selected: calibration={config['calibration']}, fixed_alpha={config['fixed_alpha']}")

    print("Training on all of TRAIN...")
    cb.fit_user_profiles(masked)
    cf = CollaborativeRecommender(method=method, random_state=seed, **(cf_kwargs or {})).fit(masked, music)

    results = {cal: evaluate_hybrid(HybridRecommender(cb, cf, calibration=cal), test, cold,
                                    max_users=max_users, seed=seed)
               for cal in CALIBRATIONS}
    selected = results[config["calibration"]]
    fixed = evaluate_hybrid(HybridRecommender(cb, cf, "fixed", config["fixed_alpha"], config["calibration"]),
                            test, cold, max_users=max_users, seed=seed)
    report = dict(selected)
    report["models"] = {**selected["models"], "Fixed": fixed["models"]["Hybrid"]}
    report["calibration_comparison"] = {
        cal: {"overall": r["models"]["Hybrid"]["overall"], "bucket_recall": r["models"]["Hybrid"]["bucket_recall"],
              "exposure": r["models"]["Hybrid"]["exposure"], "top10_overlap_with_cb": r["top10_overlap_with_cb"],
              "top10_cold_share": r["top10_cold_share"], "hybrid_effect_verifiable": r["hybrid_effect_verifiable"],
              "mock_cold_percentile": r["models"]["Hybrid"]["mock_cold_percentile"]}
        for cal, r in results.items()}
    report.update({
        "selected_config": config, "seed": seed, "mf_method": method, "mock_cold_fraction": mock_cold_fraction,
        "n_mock_cold": int(len(cold)), "n_train_rows": int(len(masked)), "n_test_rows": int(len(test)),
    })
    hybrid = HybridRecommender(cb, cf, calibration=config["calibration"])
    hybrid.mock_cold = cold
    return report, hybrid


SPOTIFY_REQUIRED = ["track_id", "track_name", "track_artist", "playlist_id", "playlist_genre", "playlist_subgenre",
                    *[c for c in AUDIO_FEATURE_COLS if c != "time_signature"]]


def normalize_key(name, artist) -> str | None:
    """Comparable 'title|artist' key, or None when either part has no usable text."""
    parts = []
    for value in (name, artist):
        if value is None or pd.isna(value):
            return None
        text = unicodedata.normalize("NFKC", str(value)).lower()
        text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
        text = " ".join(re.sub(r"[^\w\s]", " ", text).split())
        if not text:
            return None
        parts.append(text)
    return "|".join(parts)


def load_spotify(path: str | Path) -> pd.DataFrame:
    """Read the Spotify songs file into the Music Info schema used by Stage 1."""
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"track_id": str, "playlist_id": str})
    missing = [c for c in SPOTIFY_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Spotify file is missing columns: {missing}")
    df = raw.rename(columns={"track_name": "name", "track_artist": "artist"})
    df["genre"] = df["playlist_genre"].fillna("").astype(str) + " " + df["playlist_subgenre"].fillna("").astype(str)
    df["tags"] = ""
    if "time_signature" not in df.columns:
        df["time_signature"] = np.nan  # Stage 1 fills gaps with its TRAIN median
    df["song_key"] = [normalize_key(n, a) for n, a in zip(df["name"], df["artist"])]
    return df


def link_to_msd(spotify: pd.DataFrame, music: pd.DataFrame) -> pd.Series:
    """Map each Spotify song_key to the MSD track with the same title and artist."""
    catalogue = music.dropna(subset=["track_id"]).sort_values("track_id")
    msd = {}
    for track_id, name, artist in zip(catalogue["track_id"], catalogue["name"], catalogue["artist"]):
        key = normalize_key(name, artist)
        if key is not None:
            msd.setdefault(key, str(track_id))
    linked = {k: msd[k] for k in spotify["song_key"].dropna().unique() if k in msd}
    return pd.Series(linked, dtype=object)


def split_playlists(spotify: pd.DataFrame, min_tracks: int = 10,
                    seed: int = 42) -> list[tuple[str, list[str], list[str]]]:
    """Split each large-enough playlist's unique songs in half: (playlist_id, input keys, hidden keys)."""
    rng = np.random.default_rng(seed)
    splits = []
    usable = spotify.dropna(subset=["song_key"])
    for playlist_id, keys in usable.groupby("playlist_id", sort=True)["song_key"]:
        unique = np.unique(keys.to_numpy(dtype=str))
        if len(unique) < min_tracks:
            continue
        shuffled = rng.permutation(unique).tolist()
        half = len(shuffled) // 2
        splits.append((str(playlist_id), shuffled[:half], shuffled[half:]))
    return splits


def fold_in_raw_scores(cf, msd_track_ids: list[str]) -> np.ndarray | None:
    """Raw MF scores (cf.items order) for an unseen listener; None when no song is in the MF model."""
    rows = sorted({cf.i_map[t] for t in msd_track_ids if t in cf.i_map})
    if not rows:
        return None
    Q, F, r = cf.Q, cf.Q[rows], np.log(2.0)  # no play counts in a playlist, so assume playcount = 1
    identity = cf.reg * np.eye(cf.n_factors)
    if cf.method == "als":
        w = cf.alpha_conf * r
        p = np.linalg.solve(Q.T @ Q + identity + (F.T * w) @ F, F.T @ np.full(len(rows), w + 1.0))
    else:
        p = np.linalg.solve(F.T @ F + identity, F.T @ np.full(len(rows), r))
    row = csr_matrix((np.full(len(rows), r), (np.zeros(len(rows), dtype=int), rows)), shape=(1, len(cf.items)))
    return Q @ p + ((row @ cf.MT) @ cf.M).toarray().ravel()


def _scale_mf(raw: np.ndarray | None, mf_index: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """S_MF for candidates: min-max over unheard songs in the MF model, 0 for every other song."""
    scores = np.zeros(len(mf_index))
    if raw is None:
        return scores
    linked = mf_index >= 0
    values = raw[mf_index[linked]]
    pool = raw[mf_index[linked & ~seen]] if (linked & ~seen).any() else values
    low, span = pool.min(), np.ptp(pool)
    if span > 1e-9:
        scores[linked] = np.clip((values - low) / span, 0.0, 1.0)
    return scores


def evaluate_spotify(cb, cf, spotify: pd.DataFrame, music: pd.DataFrame, config: dict,
                     k_values=(5, 10, 20), seed: int = 42) -> dict:
    """Phase 2: treat each playlist as a listener, hide half of its songs, and try to find them."""
    splits = split_playlists(spotify, seed=seed)
    if not splits:
        raise ValueError("No playlist has enough songs to evaluate")
    catalogue = (spotify.dropna(subset=["song_key"]).sort_values("track_id")
                 .drop_duplicates("song_key").reset_index(drop=True))
    keys = catalogue["song_key"].to_numpy(dtype=str)
    position = {k: i for i, k in enumerate(keys)}
    features, _ = cb.transform_tracks(catalogue)
    audio_only = features.copy()
    audio_only[:, len(AUDIO_FEATURE_COLS):] = 0.0

    link = link_to_msd(catalogue, music)
    msd_ids = [link.get(k) for k in keys]
    mf_index = np.array([cf.i_map.get(t, -1) if t is not None else -1 for t in msd_ids])
    alpha = cf.compute_dynamic_alpha([t if t is not None else "" for t in msd_ids])
    is_cold = cf.get_item_listener_counts([t if t is not None else "" for t in msd_ids]) == 0

    report_ks = sorted(k_values)
    top_k = max(report_ks + [10])
    names = ("CB", "CB_audio_only", "MF", "Hybrid")
    rows = {m: {k: [] for k in report_ks} for m in names}
    overlaps, cold_shares, n_no_link, n_msd_link = [], [], 0, 0
    for _, query, hidden in splits:
        q_idx = [position[k] for k in query]
        relevant = {k for k in hidden if k in position}
        seen = np.zeros(len(keys), dtype=bool)
        seen[q_idx] = True
        s_cb = cb._cosine_01(features[q_idx].mean(axis=0), features)
        s_audio = cb._cosine_01(audio_only[q_idx].mean(axis=0), audio_only)
        raw = fold_in_raw_scores(cf, [link[k] for k in query if k in link.index])
        n_no_link += raw is None
        s_mf = _scale_mf(raw, mf_index, seen)
        s_for_hybrid, s_mf_for_hybrid = s_cb, s_mf
        if config.get("calibration") == "minmax":
            s_for_hybrid = _minmax_unseen(s_cb, seen)
        elif config.get("calibration") == "rank":
            s_for_hybrid = _percentile_unseen(s_cb, seen)
            s_mf_for_hybrid = _percentile_unseen(s_mf, seen, mf_index >= 0)
        # Without any MF signal for this listener, alpha < 1 would only penalize songs known to MSD.
        s_hybrid = s_for_hybrid if raw is None else alpha * s_for_hybrid + (1 - alpha) * s_mf_for_hybrid
        n_msd_link += any(k in link.index for k in query)
        scores = {"CB": s_cb, "CB_audio_only": s_audio, "MF": s_mf, "Hybrid": s_hybrid}
        tops = {m: top_k_indices(s, keys, seen, top_k) for m, s in scores.items()}
        for model, idx in tops.items():
            for k in report_ks:
                rows[model][k].append(ranking_metrics(list(keys[idx]), relevant, k))
        cb_top, hybrid_top = tops["CB"][:10], tops["Hybrid"][:10]
        overlaps.append(len(set(cb_top) & set(hybrid_top)) / max(len(hybrid_top), 1))
        cold_shares.append(float(is_cold[hybrid_top].mean()) if len(hybrid_top) else 0.0)

    overlap = float(np.mean(overlaps))
    return {
        "models": {m: {"overall": {f"@{k}": {n: float(np.mean([r[n] for r in rows[m][k]]))
                                             for n in ("precision", "recall", "ndcg")} for k in report_ks}}
                   for m in names},
        "n_playlists": len(splits),
        "n_playlists_no_mf_link": int(n_no_link),
        "n_songs": len(keys),
        "n_spotify_rows": len(spotify),
        "n_rows_without_key": int(spotify["song_key"].isna().sum()),
        "playlists_with_msd_link_rate": n_msd_link / len(splits),
        "link_rate_msd": float(np.mean([t is not None for t in msd_ids])),
        "link_rate_mf": float(np.mean(mf_index >= 0)),
        "playlists_with_mf_link_rate": 1 - n_no_link / len(splits),
        "top10_overlap_with_cb": overlap,
        "top10_cold_share": float(np.mean(cold_shares)),
        "hybrid_effect_verifiable": bool(overlap < OVERLAP_LIMIT),
        "calibration": config.get("calibration"),
    }


def _read_music(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"track_id": str})


def _make_split_command(args) -> None:
    music = _read_music(args.music)
    train, test = make_shared_split(args.history, set(music["track_id"].dropna()), args.user_fraction, args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "shared_train_history.csv", index=False)
    test.to_csv(out / "shared_test_history.csv", index=False)
    print(f"TRAIN={len(train):,} rows, TEST={len(test):,} rows, users={train['user_id'].nunique():,}")


def _evaluate_command(args) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    music = _read_music(args.music)
    spotify = load_spotify(args.spotify) if args.spotify else None  # fail before the slow training
    valid = set(music["track_id"].dropna())
    train = load_history(args.train_history, valid, user_fraction=1.0)
    test = load_history(args.test_history, valid, user_fraction=1.0)
    if args.user and str(args.user) not in set(train["user_id"].astype(str)):
        raise ValueError(f"Unknown --user {args.user}: not in {args.train_history}")
    cf_kwargs = {}
    if args.factors is not None:
        cf_kwargs["n_factors"] = args.factors
    if args.epochs is not None:
        cf_kwargs["n_epochs"] = args.epochs

    report, hybrid = run_msd_evaluation(music, train, test, args.method, args.mock_cold_fraction, args.seed,
                                        args.max_users, cf_kwargs)
    report["music_file"], report["train_file"] = str(args.music), str(args.train_history)
    report["test_file"] = str(args.test_history)
    (out / "hybrid_evaluation_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"track_id": hybrid.mock_cold}).to_csv(out / "mock_cold_tracks.csv", index=False)

    user = str(args.user) if args.user else str(next(iter(hybrid.cb.user_profiles)))
    recs = hybrid.recommend_user(user, args.top_n)
    recs.to_csv(out / "personalized_top10_hybrid.csv", index=False, encoding="utf-8-sig")
    print(f"\nTop-{args.top_n} for user {user[:10]}...")
    print(recs.to_string(index=False))

    if spotify is not None:
        phase2 = evaluate_spotify(hybrid.cb, hybrid.cf, spotify, music, report["selected_config"], seed=args.seed)
        phase2.update({"spotify_file": str(args.spotify), "music_file": str(args.music),
                       "train_file": str(args.train_history), "seed": args.seed, "mf_method": args.method})
        (out / "spotify_evaluation_metrics.json").write_text(
            json.dumps(phase2, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results in {out.resolve()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 3: weighted hybrid")
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("make-split", help="Create the shared TRAIN/TEST files")
    split.add_argument("--music", default="Music Info.csv")
    split.add_argument("--history", default="User Listening History.csv")
    split.add_argument("--user-fraction", type=float, default=0.05)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--output", default=".")
    split.set_defaults(run=_make_split_command)

    evaluate = commands.add_parser("evaluate", help="Evaluate and write recommendations")
    evaluate.add_argument("--music", default="Music Info.csv")
    evaluate.add_argument("--train-history", default="shared_train_history.csv")
    evaluate.add_argument("--test-history", default="shared_test_history.csv")
    evaluate.add_argument("--mock-cold-fraction", type=float, default=0.10)
    evaluate.add_argument("--spotify", default=None, help="Optional Spotify songs CSV for phase 2")
    evaluate.add_argument("--method", choices=["als", "sgd"], default="als")
    evaluate.add_argument("--factors", type=int, default=None)
    evaluate.add_argument("--epochs", type=int, default=None)
    evaluate.add_argument("--max-users", type=int, default=2000)
    evaluate.add_argument("--user", default=None)
    evaluate.add_argument("--top-n", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--output", default="hybrid_output")
    evaluate.set_defaults(run=_evaluate_command)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
