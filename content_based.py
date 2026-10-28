"""Stage 1 — Content-Based Music Recommender (2026 ML term project).

Implements the four tasks from music_recommendation_plan.docx:
  1-1. StandardScaler on 12 precomputed audio features
  1-2. TF-IDF for genre/tags (max_features=100, tag_weight=0.4)
  1-3. K-Means++ clustering + inertia/sample silhouette
  1-4. Playcount-weighted user profiles + cosine similarity, scores in [0, 1]

Accepts Music Info.csv and User Listening History.csv or a .zip containing it.
Streaming, deterministic USER sampling is the default demo path for the ~9.7M-row
history. For fair collaborative/hybrid comparison, all team members must share
one common train/test split (see --train-history and README).

Original prototype and design: team's hybrid_music_recommender.txt and plan.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

AUDIO_FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature",
]
HISTORY_COLS = ["user_id", "track_id", "playcount"]


def _genre_tags(df: pd.DataFrame) -> list[str]:
    genre = df["genre"] if "genre" in df.columns else pd.Series("", index=df.index)
    tags = df["tags"] if "tags" in df.columns else pd.Series("", index=df.index)
    text = genre.fillna("").astype(str) + " " + tags.fillna("").astype(str)
    return text.str.replace(r"[,;|]", " ", regex=True).str.strip().tolist()


def _history_chunks(path: str | Path, chunksize: int = 300_000):
    """Read a large CSV in bounded chunks, including CSV inside a ZIP archive."""
    path = Path(path)
    options = {"usecols": HISTORY_COLS,
               "dtype": {"user_id": "str", "track_id": "str"},
               "chunksize": chunksize, "encoding": "utf-8-sig"}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_files = [n for n in archive.namelist()
                         if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
            if len(csv_files) != 1:
                raise ValueError(f"Expected one CSV in ZIP, got {csv_files}")
            with archive.open(csv_files[0]) as file:
                yield from pd.read_csv(file, **options)
    else:
        yield from pd.read_csv(path, **options)


def load_history(path: str | Path, valid_track_ids: set[str] | None = None,
                 user_fraction: float = 0.005, chunksize: int = 300_000) -> pd.DataFrame:
    """Deterministic user-level sample from raw history, no full 9.7M-row load.

    user_fraction=0.005 samples approximately 0.5% of *users*, retaining all
    their listening events. Use 1.0 for the full dataset (requires more RAM).
    User selection uses pandas' stable hashing and the same rule in each chunk.
    """
    if not 0.0 < user_fraction <= 1.0:
        raise ValueError("user_fraction must be in (0, 1]")
    pieces = []
    threshold = round(user_fraction * 1_000_000)
    for chunk in _history_chunks(path, chunksize):
        chunk = chunk.dropna(subset=["user_id", "track_id", "playcount"])
        if user_fraction < 1.0:
            hashes = pd.util.hash_pandas_object(chunk["user_id"], index=False).to_numpy(dtype="uint64")
            chunk = chunk.loc[(hashes % 1_000_000) < threshold]
        if valid_track_ids is not None and not chunk.empty:
            chunk = chunk.loc[chunk["track_id"].isin(valid_track_ids)]
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        raise ValueError("No listening history matched. Check the CSV, IDs, and user_fraction.")
    history = pd.concat(pieces, ignore_index=True)
    history["playcount"] = pd.to_numeric(history["playcount"], errors="coerce")
    history = history.dropna(subset=["playcount"])
    history = history.loc[history["playcount"] > 0].copy()
    # Ensure unique (user,track) pairs; combine duplicates rather than leaking them across splits.
    return history.groupby(["user_id", "track_id"], sort=False, as_index=False)["playcount"].sum()


def split_history_by_user(history: pd.DataFrame, test_fraction: float = 0.20,
                          seed: int = 42, min_tracks: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out 20% of each user's unique tracks; users with <min_tracks excluded.

    Use exactly the SAME split for content-based, collaborative and hybrid
    evaluation. Test interactions must not be used to learn user profiles.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    history = history.sort_values(["user_id", "track_id"]).reset_index(drop=True)
    counts = history.groupby("user_id")["track_id"].transform("size")
    history = history.loc[counts >= min_tracks].reset_index(drop=True)
    if history.empty:
        raise ValueError("No users have enough distinct songs for the split")
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(history), dtype=bool)
    for positions in history.groupby("user_id", sort=False).indices.values():
        n_test = max(1, min(len(positions) - 1, int(np.ceil(len(positions) * test_fraction))))
        test_mask[rng.choice(positions, size=n_test, replace=False)] = True
    return (history.loc[~test_mask].reset_index(drop=True),
            history.loc[test_mask].reset_index(drop=True))


class ContentBasedRecommender:
    """Stage 1 recommender, intentionally independent from Stage 2 and 3."""

    def __init__(self, n_clusters: int = 6, tag_weight: float = 0.4,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.tag_weight = tag_weight
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=100, token_pattern=r"(?u)\b[\w-]+\b")
        self.kmeans = KMeans(n_clusters=n_clusters, init="k-means++",
                             n_init=5, max_iter=200, random_state=random_state)
        self.feature_medians: pd.Series | None = None
        self.music_info: pd.DataFrame | None = None
        self.track_features: np.ndarray | None = None
        self.track_clusters: np.ndarray | None = None
        self.track_id_to_idx: dict[str, int] = {}
        self.user_profiles: dict[str, np.ndarray] = {}
        self.user_listened_tracks: dict[str, set[str]] = {}

    def _audio(self, tracks: pd.DataFrame, training: bool) -> np.ndarray:
        missing = sorted(set(AUDIO_FEATURE_COLS) - set(tracks.columns))
        if missing:
            raise ValueError(f"Missing required audio features: {missing}")
        values = tracks[AUDIO_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        if training:
            self.feature_medians = values.median().fillna(0)
        if self.feature_medians is None:
            raise RuntimeError("Fit on Music Info.csv before transforming new songs")
        return values.fillna(self.feature_medians).to_numpy(dtype=np.float64)

    def fit(self, music_info_df: pd.DataFrame,
            history_df: pd.DataFrame | None = None) -> "ContentBasedRecommender":
        """Learn the audio scaler, genre/tag TF-IDF, K-Means centers, optionally user profiles."""
        if "track_id" not in music_info_df.columns:
            raise ValueError("Music Info.csv must contain track_id")
        music = music_info_df.dropna(subset=["track_id"]).copy()
        music["track_id"] = music["track_id"].astype(str)
        music = music.drop_duplicates(subset="track_id").reset_index(drop=True)
        if len(music) < self.n_clusters:
            raise ValueError("Number of songs must be >= n_clusters")
        audio_scaled = self.scaler.fit_transform(self._audio(music, training=True)).astype("float32")
        text_vec = self.tfidf.fit_transform(_genre_tags(music)).toarray().astype("float32")
        self.track_features = np.hstack((audio_scaled, self.tag_weight * text_vec))
        self.track_clusters = self.kmeans.fit_predict(audio_scaled)
        self.track_id_to_idx = {tid: i for i, tid in enumerate(music["track_id"])}
        self.music_info = music
        self.user_profiles.clear()
        self.user_listened_tracks.clear()
        if history_df is not None:
            self.fit_user_profiles(history_df)
        return self

    def fit_user_profiles(self, train_history: pd.DataFrame) -> "ContentBasedRecommender":
        """Weighted mean of a user's listened tracks; use TRAIN rows ONLY."""
        if self.track_features is None:
            raise RuntimeError("Call fit(music_info_df) first")
        missing = set(HISTORY_COLS) - set(train_history.columns)
        if missing:
            raise ValueError(f"History missing columns: {sorted(missing)}")
        hist = train_history[HISTORY_COLS].dropna().copy()
        hist["user_id"] = hist["user_id"].astype(str)
        hist["track_id"] = hist["track_id"].astype(str)
        hist["playcount"] = pd.to_numeric(hist["playcount"], errors="coerce")
        hist = hist.loc[hist["playcount"] > 0]
        hist["item_idx"] = hist["track_id"].map(self.track_id_to_idx)
        hist = hist.dropna(subset=["item_idx"])
        if hist.empty:
            raise ValueError("No training interactions match music track IDs")
        # Aggregate duplicates before using log1p(playcount).
        hist = hist.groupby(["user_id", "track_id", "item_idx"], as_index=False, sort=False)["playcount"].sum()
        users, rows = np.unique(hist["user_id"].to_numpy(), return_inverse=True)
        cols = hist["item_idx"].to_numpy(dtype=np.int32)
        weights = np.log1p(hist["playcount"].to_numpy(dtype=np.float64))
        # Sparse weight matrix avoids user-by-item dense memory cost.
        weighted_matrix = csr_matrix((weights, (rows, cols)),
                                     shape=(len(users), len(self.track_features)))
        weighted_sums = weighted_matrix @ self.track_features
        denominators = np.asarray(weighted_matrix.sum(axis=1)).ravel()
        profiles = np.asarray(weighted_sums) / np.maximum(denominators[:, None], 1e-12)
        self.user_profiles = dict(zip(users.tolist(), profiles))
        self.user_listened_tracks = hist.groupby("user_id")["track_id"].agg(set).to_dict()
        return self

    def transform_tracks(self, tracks_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform unobserved songs with training scaler/TF-IDF; assign existing clusters."""
        if self.track_features is None:
            raise RuntimeError("Call fit first")
        audio = self.scaler.transform(self._audio(tracks_df, training=False)).astype("float32")
        tags = self.tfidf.transform(_genre_tags(tracks_df)).toarray().astype("float32")
        features = np.hstack((audio, self.tag_weight * tags))
        return features, self.kmeans.predict(audio)

    @staticmethod
    def _cosine_01(user_vector: np.ndarray, candidate_features: np.ndarray) -> np.ndarray:
        """Cosine range [-1,+1] -> score range [0,1]."""
        product = candidate_features @ user_vector
        denom = np.linalg.norm(candidate_features, axis=1) * np.linalg.norm(user_vector)
        sim = np.divide(product, denom, out=np.zeros(len(product)), where=denom > 1e-12)
        return np.clip((sim + 1.0) / 2.0, 0.0, 1.0)

    def score_user_items(self, user_id: str, candidate_features: np.ndarray | None = None) -> np.ndarray:
        """Stage 3-compatible API: one S_CB(u,i) score per candidate, in existing row order."""
        user_id = str(user_id)
        if user_id not in self.user_profiles:
            raise KeyError(f"Unknown user: {user_id}. Check TRAIN history.")
        if candidate_features is None:
            candidate_features = self.track_features
        return self._cosine_01(self.user_profiles[user_id], candidate_features)

    def recommend_user(self, user_id: str, top_n: int = 10,
                       candidate_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return top N unheard tracks with names, K-Means cluster, content-based score."""
        if self.music_info is None:
            raise RuntimeError("Call fit first")
        if candidate_df is None:
            music = self.music_info
            features, clusters = self.track_features, self.track_clusters
        else:
            music = candidate_df.reset_index(drop=True)
            features, clusters = self.transform_tracks(music)
        scores = self.score_user_items(user_id, features)
        keep = [c for c in ["track_id", "name", "artist", "genre"] if c in music]
        out = music[keep].copy()
        out["cluster"] = clusters
        out["S_CB"] = scores
        if "track_id" in out:
            listened = self.user_listened_tracks.get(str(user_id), set())
            out = out.loc[~out["track_id"].astype(str).isin(listened)]
        return out.nlargest(top_n, "S_CB").reset_index(drop=True)

    def recommend_similar(self, track_id: str, top_n: int = 10) -> pd.DataFrame:
        """For debugging without history only: similar songs, NOT personalized."""
        idx = self.track_id_to_idx[track_id]
        similarities = self._cosine_01(self.track_features[idx], self.track_features)
        cols = [c for c in ["track_id", "name", "artist", "genre"] if c in self.music_info]
        out = self.music_info[cols].copy()
        out["cluster"] = self.track_clusters
        out["similarity"] = similarities
        return out.loc[out["track_id"] != track_id].nlargest(top_n, "similarity").reset_index(drop=True)

    def compare_cluster_counts(self, k_values=(4, 6, 8, 10), sample_size: int = 1200) -> pd.DataFrame:
        """Elbow/Silhouette comparison for the report; doesn't change the fitted model.

        Reuses the fitted StandardScaler. Each K-Means fits on ALL audio features,
        but silhouette is measured on a random sample for manageable computation.
        """
        if self.music_info is None:
            raise RuntimeError("Call fit first")
        audio = self.scaler.transform(self._audio(self.music_info, training=False)).astype("float32")
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(audio), size=min(sample_size, len(audio)), replace=False)
        results = []
        for k in k_values:
            if k < 2 or k >= len(audio):
                raise ValueError(f"K={k} is not valid for {len(audio)} songs")
            km = KMeans(n_clusters=int(k), init="k-means++", n_init=5,
                        max_iter=200, random_state=self.random_state)
            km.fit(audio)
            sampled_labels = km.predict(audio[idx])
            sil = (float(silhouette_score(audio[idx], sampled_labels))
                   if 1 < len(set(sampled_labels)) < len(idx) else float("nan"))
            results.append({"K": int(k), "inertia": float(km.inertia_),
                            "silhouette_sample": sil})
        return pd.DataFrame(results)

    def evaluate_clustering(self, sample_size: int = 1200) -> dict:
        """Silhouette on sampled songs, plus full-training inertia and group sizes."""
        if self.music_info is None:
            raise RuntimeError("Call fit first")
        X = self.scaler.transform(self._audio(self.music_info, training=False))
        rng = np.random.default_rng(self.random_state)
        indexes = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
        labels = self.track_clusters[indexes]
        sil = float(silhouette_score(X[indexes], labels)) if 1 < len(set(labels)) < len(indexes) else None
        return {"n_songs": len(X), "n_clusters": self.n_clusters,
                "inertia": float(self.kmeans.inertia_), "silhouette_sample": sil,
                "silhouette_sample_size": len(indexes),
                "cluster_sizes": np.bincount(self.track_clusters, minlength=self.n_clusters).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: music content-based model")
    parser.add_argument("--music", default="Music Info.csv")
    parser.add_argument("--history", default=None, help="History .csv or .zip, for a demo")
    parser.add_argument("--train-history", default=None, help="Shared TRAIN history CSV from team; takes priority")
    parser.add_argument("--user-fraction", type=float, default=0.005,
                        help="Demo: sample this fraction of users from ZIP (default 0.005).")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--user", default=None, help="Optional known user ID")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--compare-k", action="store_true", help="Compare K=4,6,8,10 for report")
    parser.add_argument("--output", default="content_based_output")
    args = parser.parse_args()

    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    print(f"[1-1 ~ 1-3] Loading {len(music):,} songs; fitting scaler, TF-IDF, K-Means...")
    model = ContentBasedRecommender(n_clusters=args.clusters).fit(music)
    metrics = model.evaluate_clustering()
    print(f"K-Means: inertia={metrics['inertia']:.2f}, sampled silhouette={metrics['silhouette_sample']:.4f}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "clustering_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    cluster_counts = (model.music_info[["track_id", "name", "artist"]]
                      .assign(cluster=model.track_clusters))
    cluster_counts.to_csv(out / "song_clusters.csv", index=False, encoding="utf-8-sig")
    if args.compare_k:
        print("[1-3] Comparing K=4,6,8,10 (inertia / silhouette)...")
        comparison = model.compare_cluster_counts()
        print(comparison.to_string(index=False))
        comparison.to_csv(out / "kmeans_k_comparison.csv", index=False)

    history_path = args.train_history or args.history
    if history_path:
        fraction = 1.0 if args.train_history else args.user_fraction
        print(f"[1-4] Reading listening history from {history_path} (user_fraction={fraction})...")
        history = load_history(history_path, set(model.track_id_to_idx), user_fraction=fraction)
        if args.train_history:
            train = history  # NEVER split a pre-prepared team's TRAIN file again.
            print(f"Using shared TRAIN history ({len(train):,} interactions)")
        else:
            train, test = split_history_by_user(history, test_fraction=0.2)
            print(f"Demo 80/20 user-wise split: TRAIN={len(train):,}, TEST={len(test):,}")
            # Local reproducibility exports; share these split files with teammates if used.
            train.to_csv(out / "demo_train_history.csv", index=False)
            test.to_csv(out / "demo_test_history.csv", index=False)
        model.fit_user_profiles(train)
        user = str(args.user) if args.user else next(iter(model.user_profiles))
        print(f"Matched profile users: {len(model.user_profiles):,}")
        recommendations = model.recommend_user(user, top_n=args.top_n)
        print(f"\nPersonalized Top-{args.top_n} (user={user[:10]}...):")
        print(recommendations[["name", "artist", "cluster", "S_CB"]].to_string(index=False))
        recommendations.to_csv(out / "personalized_top10.csv", index=False, encoding="utf-8-sig")
    else:
        example_track_id = str(model.music_info.iloc[0]["track_id"])
        print("No user history provided; showing song-to-song similarity only.")
        print(model.recommend_similar(example_track_id, args.top_n).to_string(index=False))
    print(f"\nSaved results in {out.resolve()}")


if __name__ == "__main__":
    main()
