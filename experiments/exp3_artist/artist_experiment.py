"""Experiment: does knowing which artists a user likes help CB find new songs?

S_CB'(u, i) = (1 - beta) * S_CB(u, i) + beta * A(u, artist(i))
A is the user's log-playcount per artist, divided by the user's largest artist value, from TRAIN only.

The scaler and beta are chosen by new-song Recall@20 on validation (ties: percentile); TEST only reports.
content_based.py and collaborative.py stay unchanged.

Run:
  python -m experiments.exp3_artist.artist_experiment --music "Music Info.csv" \
      --train-history shared_train_history.csv --test-history shared_test_history.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collaborative import CollaborativeRecommender
from content_based import load_history, split_history_by_user
from final_model import ArtistAffinity, ArtistCB
from hybrid import _clean_test, _percentile_unseen, mask_mock_cold, normalize_key, top_k_indices
from experiments.exp2_scaler.scaler_experiment import FIXED_ALPHA, _CachedMF, _test_row, make_cb

SCALERS = ("standard", "categorical_onehot", "minmax")
BETAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)


def mock_topk(cb_like: ArtistCB, test: pd.DataFrame, cold: np.ndarray, k: int = 20, max_users: int = 2000,
              seed: int = 42) -> dict:
    """New-song Recall@k and percentile, split by whether the user already knows the song's artist."""
    ids = cb_like.music_info["track_id"].astype(str).to_numpy()
    position = {t: i for i, t in enumerate(ids)}
    cold_set = {str(t) for t in cold}
    df = _clean_test(test)
    df = df.loc[df["track_id"].isin(cold_set) & df["track_id"].isin(position)
                & df["user_id"].isin(cb_like.user_profiles)]
    if df.empty:
        raise ValueError("No test rows for new songs")
    answers = df.groupby("user_id")["track_id"].agg(list)
    users = np.sort(answers.index.to_numpy(dtype=str))
    if len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))

    recalls, percentiles, known_recalls, unknown_recalls = [], [], [], []
    n_answers = n_known = 0
    for user in users:
        seen = np.zeros(len(ids), dtype=bool)
        seen[[position[t] for t in cb_like.user_listened_tracks.get(user, ()) if t in position]] = True
        scores = cb_like.score_user_items(user)
        top = set(top_k_indices(scores, ids, seen, k).tolist())
        wanted = np.array([position[t] for t in answers[user]])
        hits = np.array([w in top for w in wanted], dtype=float)
        recalls.append(hits.mean())
        percentiles.append(_percentile_unseen(scores, seen)[wanted].mean())
        known = cb_like.affinity.scores(user)[wanted] > 0
        n_answers += len(wanted)
        n_known += int(known.sum())
        if known.any():
            known_recalls.append(hits[known].mean())
        if (~known).any():
            unknown_recalls.append(hits[~known].mean())
    return {
        "recall": float(np.mean(recalls)),
        "percentile": float(np.mean(percentiles)),
        "known_artist_answer_share": n_known / n_answers,
        "recall_known_artist": float(np.mean(known_recalls)) if known_recalls else None,
        "recall_unknown_artist": float(np.mean(unknown_recalls)) if unknown_recalls else None,
        "n_users": int(len(users)),
    }


def duplicate_share(music: pd.DataFrame, cold: np.ndarray, warm: set[str]) -> float:
    """Share of new songs whose title and artist also exist under a trained track_id (would flatter results)."""
    keys = {str(t): normalize_key(n, a) for t, n, a in zip(music["track_id"], music["name"], music["artist"])}
    warm_keys = {keys[t] for t in warm if keys.get(t)}
    return float(np.mean([keys.get(str(t)) in warm_keys for t in cold]))


def _report_row(cb_like, cf, test, cold, max_users, seed) -> dict:
    row = _test_row(cb_like, cf, test, cold, max_users, seed)
    topk = mock_topk(cb_like, test, cold, 20, max_users, seed)
    row["topk_recall20"], row["topk_percentile"] = topk.pop("recall"), topk.pop("percentile")
    topk.pop("n_users")
    row.update(topk)
    return row


def run_artist_experiment(music: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, scalers=SCALERS, betas=BETAS,
                          mock_cold_fraction: float = 0.10, seed: int = 42, max_users: int = 2000,
                          val_users: int = 1000, cf_kwargs: dict | None = None, n_clusters: int = 6) -> dict:
    """Choose scaler and beta on validation, then report baseline and selected settings on TEST."""
    masked, cold = mask_mock_cold(train, mock_cold_fraction, seed)
    fit_part, validation = split_history_by_user(masked, test_fraction=0.10, seed=seed)
    val_fit, val_cold = mask_mock_cold(fit_part, mock_cold_fraction, seed + 1)

    models = {name: make_cb(name, music, seed, n_clusters) for name in scalers}
    val_rows, choices = {}, {}
    for name, cb in models.items():
        cb.fit_user_profiles(val_fit)
        affinity = ArtistAffinity(cb, val_fit)
        for beta in betas:
            label = f"{name}|beta={beta}"
            print(f"[validation] {label}", flush=True)
            val_rows[label] = mock_topk(ArtistCB(cb, affinity, beta), validation, val_cold, 20, val_users, seed)
            choices[label] = (name, beta)
    best = max(val_rows, key=lambda k: (val_rows[k]["recall"], val_rows[k]["percentile"]))
    scaler, beta = choices[best]
    print(f"Selected on validation: {best}", flush=True)

    print("Training MF once on all of TRAIN...", flush=True)
    cf = _CachedMF(CollaborativeRecommender(method="als", random_state=seed, **(cf_kwargs or {})).fit(masked, music))
    baseline_cb = models.get("standard") or make_cb("standard", music, seed, n_clusters)
    rows = {}
    for label, cb, b in (("baseline", baseline_cb, 0.0), ("selected", models[scaler], beta)):
        cb.fit_user_profiles(masked)
        print(f"[test] {label}", flush=True)
        rows[label] = _report_row(ArtistCB(cb, ArtistAffinity(cb, masked), b), cf, test, cold, max_users, seed)
    return {
        "selected": {"scaler": scaler, "beta": beta},
        "selection_metric": "validation new-song Recall@20, ties by percentile",
        "validation": val_rows, "test": rows, "seed": seed, "fixed_alpha": FIXED_ALPHA,
        "n_mock_cold": int(len(cold)), "n_train_rows": int(len(masked)), "n_test_rows": int(len(test)),
        "mock_cold_duplicate_share": duplicate_share(music, cold, set(masked["track_id"].astype(str))),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add artist preference to the Stage 1 score")
    parser.add_argument("--music", default="Music Info.csv")
    parser.add_argument("--train-history", default="shared_train_history.csv")
    parser.add_argument("--test-history", default="shared_test_history.csv")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="hybrid_output/artist")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    valid = set(music["track_id"].dropna())
    train = load_history(args.train_history, valid, user_fraction=1.0)
    test = load_history(args.test_history, valid, user_fraction=1.0)
    result = run_artist_experiment(music, train, test, seed=args.seed, max_users=args.max_users)
    (out / "artist_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(result["test"]).T.round(4).T.to_string())


if __name__ == "__main__":
    main()
