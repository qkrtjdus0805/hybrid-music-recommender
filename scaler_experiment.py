"""Experiment: does replacing Stage 1's StandardScaler help CB find new songs?

content_based.py stays unchanged: each variant swaps the public `scaler` attribute before fit.
The scaler is chosen on validation data; TEST is only read to report results.

Run:
  python scaler_experiment.py --music "Music Info.csv" \
      --train-history shared_train_history.csv --test-history shared_test_history.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, PowerTransformer, QuantileTransformer, RobustScaler,
                                   StandardScaler)

from collaborative import CollaborativeRecommender
from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender, load_history, split_history_by_user
from hybrid import HybridRecommender, _clean_test, _percentile_unseen, evaluate_hybrid, mask_mock_cold

CATEGORICAL = ["key", "mode", "time_signature"]
ONEHOT = ["key", "time_signature"]
SKEWED = ["loudness", "speechiness", "acousticness", "instrumentalness", "liveness"]  # |skew| > 1 in Music Info
SCALERS = ("standard", "robust", "minmax", "quantile_normal", "skew_aware", "categorical_onehot", "no_categorical")
FIXED_ALPHA = 0.25  # chosen on validation in the phase 1 run; not chosen again here


def _columns(names: list[str]) -> list[int]:
    return [AUDIO_FEATURE_COLS.index(n) for n in names]


def build_scaler(name: str, seed: int = 42):
    """Return an unfitted transformer for the 12 audio columns."""
    power = lambda: PowerTransformer(method="yeo-johnson", standardize=True)
    plain = [c for c in AUDIO_FEATURE_COLS if c not in SKEWED and c not in CATEGORICAL]
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "quantile_normal":
        return QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=seed)
    if name == "skew_aware":
        parts = [("skewed", power(), _columns(SKEWED)), ("rest", StandardScaler(), _columns(plain + CATEGORICAL))]
    elif name == "categorical_onehot":
        parts = [("skewed", power(), _columns(SKEWED)), ("rest", StandardScaler(), _columns(plain + ["mode"])),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _columns(ONEHOT))]
    elif name == "no_categorical":
        parts = [("skewed", power(), _columns(SKEWED)), ("rest", StandardScaler(), _columns(plain))]
    else:
        raise ValueError(f"Unknown scaler: {name}")
    return ColumnTransformer(parts, sparse_threshold=0.0)


def make_cb(name: str, music: pd.DataFrame, seed: int = 42, n_clusters: int = 6) -> ContentBasedRecommender:
    """Fit CB with a replacement scaler. One K-Means start is enough: S_CB does not use the clusters."""
    cb = ContentBasedRecommender(n_clusters=n_clusters, random_state=seed)
    cb.scaler = build_scaler(name, seed)
    cb.kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, max_iter=50, random_state=seed)
    return cb.fit(music)


def cold_diagnostics(cb: ContentBasedRecommender, test: pd.DataFrame, cold: np.ndarray, max_users: int = 2000,
                     seed: int = 42) -> dict:
    """Mean percentile of new-song answers, with each user's own profile and with another user's profile.

    If the two values match, CB is finding sound-alike mainstream songs, not this listener's taste.
    """
    ids = cb.music_info["track_id"].astype(str).to_numpy()
    position = {t: i for i, t in enumerate(ids)}
    cold_set = {str(t) for t in cold}
    df = _clean_test(test)
    df = df.loc[df["track_id"].isin(cold_set) & df["track_id"].isin(position) & df["user_id"].isin(cb.user_profiles)]
    if df.empty:
        raise ValueError("No test rows for new songs")
    answers = df.groupby("user_id")["track_id"].agg(list)
    rng = np.random.default_rng(seed)
    users = np.sort(answers.index.to_numpy(dtype=str))
    if len(users) > max_users:
        users = np.sort(rng.choice(users, max_users, replace=False))
    donors = np.sort(np.array(list(cb.user_profiles), dtype=str))

    listened = np.zeros(len(ids), dtype=bool)
    for tracks in cb.user_listened_tracks.values():
        listened[[position[t] for t in tracks if t in position]] = True
    pool = ~listened

    rows = {"percentile_all": [], "percentile_cold_pool": [], "shuffled_percentile_all": [],
            "shuffled_percentile_cold_pool": []}
    for user in users:
        seen = np.zeros(len(ids), dtype=bool)
        seen[[position[t] for t in cb.user_listened_tracks.get(user, ()) if t in position]] = True
        wanted = [position[t] for t in answers[user]]
        others = donors[donors != user]
        other = others[rng.integers(len(others))] if len(others) else user
        for prefix, profile in (("", user), ("shuffled_", other)):
            scores = cb.score_user_items(profile)
            rows[f"{prefix}percentile_all"].append(_percentile_unseen(scores, seen)[wanted].mean())
            rows[f"{prefix}percentile_cold_pool"].append(_percentile_unseen(scores, seen, pool)[wanted].mean())
    result = {name: float(np.mean(values)) for name, values in rows.items()}
    result["n_users"] = int(len(users))
    return result


class _CachedMF:
    """Keeps each user's S_MF so every scaler reuses one MF model without recomputing it."""

    def __init__(self, cf):
        self._cf = cf
        self._scores: dict[str, np.ndarray] = {}

    def __getattr__(self, name):
        return getattr(self._cf, name)

    def score_user_items(self, user_id, candidate_track_ids=None):
        user_id = str(user_id)
        if user_id not in self._scores:
            self._scores[user_id] = self._cf.score_user_items(user_id, candidate_track_ids).astype("float32")
        return self._scores[user_id].astype(float)


def _test_row(cb, cf, test, cold, max_users, seed) -> dict:
    dynamic = evaluate_hybrid(HybridRecommender(cb, cf), test, cold, max_users=max_users, seed=seed)
    fixed = evaluate_hybrid(HybridRecommender(cb, cf, "fixed", FIXED_ALPHA), test, cold, max_users=max_users, seed=seed)
    cb_model, hyb, fix = dynamic["models"]["CB"], dynamic["models"]["Hybrid"], fixed["models"]["Hybrid"]
    row = {
        "cb_ndcg10": cb_model["overall"]["@10"]["ndcg"],
        "cb_recall10": cb_model["overall"]["@10"]["recall"],
        "cb_mock_percentile": cb_model["mock_cold_percentile"],
        "cb_mock_recall20": cb_model["bucket_recall"]["@20"].get("mock_cold"),
        "hybrid_ndcg10": hyb["overall"]["@10"]["ndcg"],
        "hybrid_recall10": hyb["overall"]["@10"]["recall"],
        "hybrid_mock_recall20": hyb["bucket_recall"]["@20"].get("mock_cold"),
        "hybrid_cold_share": dynamic["top10_cold_share"],
        "fixed_ndcg10": fix["overall"]["@10"]["ndcg"],
        "fixed_recall10": fix["overall"]["@10"]["recall"],
        "fixed_mock_recall20": fix["bucket_recall"]["@20"].get("mock_cold"),
        "fixed_cold_share": fixed["top10_cold_share"],
    }
    row.update(cold_diagnostics(cb, test, cold, max_users, seed))
    return row


def run_experiment(music: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, scalers=SCALERS,
                   mock_cold_fraction: float = 0.10, seed: int = 42, max_users: int = 2000, val_users: int = 500,
                   cf_kwargs: dict | None = None, n_clusters: int = 6) -> dict:
    """Choose the scaler by new-song percentile on validation, then report every scaler on TEST."""
    masked, cold = mask_mock_cold(train, mock_cold_fraction, seed)
    fit_part, validation = split_history_by_user(masked, test_fraction=0.10, seed=seed)
    val_fit, val_cold = mask_mock_cold(fit_part, mock_cold_fraction, seed + 1)

    print("Training MF once on all of TRAIN...")
    cf = _CachedMF(CollaborativeRecommender(method="als", random_state=seed, **(cf_kwargs or {})).fit(masked, music))

    val_rows, test_rows = {}, {}
    for name in scalers:
        print(f"[{name}] fitting...", flush=True)
        cb = make_cb(name, music, seed, n_clusters)
        cb.fit_user_profiles(val_fit)
        val_rows[name] = cold_diagnostics(cb, validation, val_cold, val_users, seed)
        cb.fit_user_profiles(masked)
        test_rows[name] = _test_row(cb, cf, test, cold, max_users, seed)
        print(f"[{name}] validation={val_rows[name]['percentile_all']:.4f} "
              f"test={test_rows[name]['cb_mock_percentile']:.4f}", flush=True)
    selected = max(val_rows, key=lambda n: val_rows[n]["percentile_all"])
    return {"selected_scaler": selected, "selection_metric": "validation percentile_all", "validation": val_rows,
            "test": test_rows, "seed": seed, "fixed_alpha": FIXED_ALPHA, "n_mock_cold": int(len(cold)),
            "n_train_rows": int(len(masked)), "n_test_rows": int(len(test))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare scalers for the Stage 1 audio features")
    parser.add_argument("--music", default="Music Info.csv")
    parser.add_argument("--train-history", default="shared_train_history.csv")
    parser.add_argument("--test-history", default="shared_test_history.csv")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="hybrid_output/scaler")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    valid = set(music["track_id"].dropna())
    train = load_history(args.train_history, valid, user_fraction=1.0)
    test = load_history(args.test_history, valid, user_fraction=1.0)
    result = run_experiment(music, train, test, seed=args.seed, max_users=args.max_users)
    (out / "scaler_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table = pd.DataFrame(result["test"]).T
    table.to_csv(out / "scaler_comparison.csv", encoding="utf-8-sig")
    print(f"Selected on validation: {result['selected_scaler']}")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
