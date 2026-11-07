"""Experiment: how well can each model recommend songs by artists the user has never heard?

For each user, every song by an artist they heard in TRAIN (and every song with no artist) leaves both the candidates
and the answers. This removes the advantage a random split gives artist information: artist preference cannot help,
because no remaining song is by an artist the user is known to like.

An answer is still a song the user really played, so a good discovery the user never met counts as a miss.
Popularity and random scores show what chance and "just recommend hits" achieve under the same rules.

Each feature set's new-song listener count is chosen on validation by NDCG@10 before TEST is read.

Run:
  python -m experiments.val2_unknown_artist.unknown_artist_experiment --music "data/Music Info.csv" \
      --train-history data/shared_train_history.csv --test-history data/shared_test_history.csv
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collaborative import CollaborativeRecommender
from content_based import load_history, split_history_by_user
from experiments.val1_feature_mf.feature_mf_experiment import (FEATURE_SETS, NEW_SONG_LISTENERS, FeatureMF, feature_matrix,
                                   fit_feature_regression)
from final_model import N_CLUSTERS, ArtistAffinity, _ranked_prefix, paired_bootstrap
from hybrid import BUCKETS, HybridRecommender, _clean_test, assign_buckets, mask_mock_cold, ranking_metrics
from experiments.exp2_scaler.scaler_experiment import _CachedMF, make_cb

BASELINES = (("ndcg10", "MF"), ("ndcg10", "Popularity"), ("recall20", "MF"), ("new_recall20", "CB"))


def unknown_candidates(affinity, seen: np.ndarray, user: str) -> np.ndarray:
    """True for unheard songs by a named artist the user has never heard."""
    return (affinity.codes >= 0) & (affinity.scores(user) == 0) & ~seen


def evaluate_unknown(scorers: dict, ids: np.ndarray, codes: np.ndarray, labels: np.ndarray, allowed_of, test: pd.DataFrame,
                     users, max_users: int = 2000, seed: int = 42) -> tuple[dict, dict]:
    """Rank only allowed songs for each user; answers are allowed songs from TEST. All models see the same users."""
    position = {t: i for i, t in enumerate(ids)}
    df = _clean_test(test)
    played = df.loc[df["track_id"].isin(position)].groupby("user_id")["track_id"].agg(set)
    answers = {}
    for user in sorted(set(users) & set(played.index)):
        allowed = allowed_of(user)
        tracks = {t for t in played[user] if allowed[position[t]]}
        if tracks:
            answers[user] = tracks
    chosen = np.array(sorted(answers), dtype=object)
    if len(chosen) > max_users:
        chosen = np.sort(np.random.default_rng(seed).choice(chosen, max_users, replace=False))
    label_of = dict(zip(ids, labels))

    rows, per_user = {}, {}
    for name, scorer in scorers.items():
        metrics, table, buckets = [], {}, {b: [] for b in BUCKETS}
        exposure, covered, distinct, recall20 = collections.Counter(), set(), [], []
        for user in chosen:
            ranked = _ranked_prefix(scorer(user), ids, ~allowed_of(user), 20)[:20]
            top10, head = ranked[:10], set(ids[ranked].tolist())
            relevant = answers[user]
            at10 = ranking_metrics(ids[top10].tolist(), relevant, 10)
            metrics.append(at10)
            recall20.append(len(head & relevant) / len(relevant))
            by_bucket = collections.defaultdict(set)
            for track in relevant:
                by_bucket[label_of[track]].add(track)
            for bucket, tracks in by_bucket.items():
                buckets[bucket].append(len(head & tracks) / len(tracks))
            exposure.update(labels[top10])
            covered.update(ids[top10].tolist())
            distinct.append(len(set(codes[top10].tolist())))
            new = by_bucket.get("mock_cold")
            table[user] = {"ndcg10": at10["ndcg"], "recall20": recall20[-1],
                           "new_recall20": len(head & new) / len(new) if new else np.nan}
        slots = sum(exposure.values())
        share = lambda *names: sum(exposure[n] for n in names) / slots if slots else 0.0
        mean = lambda values: float(np.mean(values)) if values else None
        rows[name] = {
            "ndcg10": mean([m["ndcg"] for m in metrics]), "recall10": mean([m["recall"] for m in metrics]),
            "precision10": mean([m["precision"] for m in metrics]), "recall20": mean(recall20),
            "new_recall20": mean(buckets["mock_cold"]), "recall20_1_4": mean(buckets["1-4"]),
            "recall20_5_24": mean(buckets["5-24"]), "recall20_25plus": mean(buckets["25+"]),
            "exposure10_new": share("mock_cold", "sample_zero"), "exposure10_1_4": share("1-4"),
            "exposure10_5_24": share("5-24"), "exposure10_25plus": share("25+"), "coverage10": len(covered) / len(ids),
            "distinct_artists": mean(distinct), "n_users": int(len(chosen)),
            "n_users_with_new_answers": len(buckets["mock_cold"]),
        }
        per_user[name] = pd.DataFrame.from_dict(table, orient="index", columns=["ndcg10", "recall20", "new_recall20"])
    return rows, per_user


def _setup(cb, cf, definition: ArtistAffinity, cold: np.ndarray, test: pd.DataFrame) -> dict:
    """What every evaluation of one trained pair of models shares."""
    plain = HybridRecommender(cb, cf)
    ids = plain.candidate_ids
    users = [u for u in _clean_test(test)["user_id"].unique() if u in cb.user_profiles and u in cf.u_map]
    return {"plain": plain, "ids": ids, "codes": definition.codes, "users": users,
            "labels": assign_buckets(ids, plain.n_listeners, {str(t) for t in cold}),
            "allowed_of": lambda user: unknown_candidates(definition, plain.seen_mask(user), user)}


def _feature_hybrid(cb, cf, ids, predicted, n_new) -> HybridRecommender:
    return HybridRecommender(cb, FeatureMF(cf, ids, predicted, n_new))


def run_unknown_artist_experiment(music: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                                  feature_sets=FEATURE_SETS, new_song_listeners=NEW_SONG_LISTENERS,
                                  mock_cold_fraction: float = 0.10, seed: int = 42, max_users: int = 2000,
                                  val_users: int = 2000, cf_kwargs: dict | None = None,
                                  n_clusters: int = N_CLUSTERS) -> dict:
    """Choose the new-song listener count on validation, then compare all models on TEST."""
    masked, cold = mask_mock_cold(train, mock_cold_fraction, seed)
    fit_part, validation = split_history_by_user(masked, test_fraction=0.10, seed=seed)
    val_fit, val_cold = mask_mock_cold(fit_part, mock_cold_fraction, seed + 1)
    make_cf = lambda rows: _CachedMF(CollaborativeRecommender(method="als", random_state=seed,
                                                              **(cf_kwargs or {})).fit(rows, music))
    cb = make_cb("standard", music, seed, n_clusters)
    ids = cb.music_info["track_id"].astype(str).to_numpy()
    evaluate = lambda setup, scorers, test_rows, limit: evaluate_unknown(
        scorers, setup["ids"], setup["codes"], setup["labels"], setup["allowed_of"], test_rows, setup["users"], limit, seed)

    print("Training on the validation part...", flush=True)
    cf = make_cf(val_fit)
    cb.fit_user_profiles(val_fit)
    setup = _setup(cb, cf, ArtistAffinity(cb, fit_part), val_cold, validation)  # artists heard before the second mask
    val_rows, selected = {}, {}
    for name in feature_sets:
        predicted, _ = fit_feature_regression(cf, ids, feature_matrix(cb, name), seed)
        scorers = {}
        for n_new in new_song_listeners:
            model = _feature_hybrid(cb, cf, ids, predicted, n_new)
            scorers[f"{name}|n={n_new}"] = lambda user, m=model: m.score_user_items(user)["S_Hybrid"]
        rows, _ = evaluate(setup, scorers, validation, val_users)
        val_rows.update(rows)
        best = max(new_song_listeners, key=lambda n: (rows[f"{name}|n={n}"]["ndcg10"],
                                                      rows[f"{name}|n={n}"]["new_recall20"] or 0.0))
        selected[name] = best
        print(f"[{name}] selected n={best}", flush=True)

    print("Training on all of TRAIN...", flush=True)
    cf = make_cf(masked)
    cb.fit_user_profiles(masked)
    setup = _setup(cb, cf, ArtistAffinity(cb, train), cold, test)  # includes the songs erased from the models
    plain = setup["plain"]
    rng = np.random.default_rng(seed)
    scorers = {"Random": lambda user: rng.random(len(ids)), "Popularity": lambda user: plain.n_listeners.astype(float),
               "CB": lambda user: plain.cb.score_user_items(user),
               "MF": lambda user: plain.cf.score_user_items(user, ids),
               "Hybrid": lambda user: plain.score_user_items(user)["S_Hybrid"]}
    r2 = {}
    for name in feature_sets:
        predicted, r2[name] = fit_feature_regression(cf, ids, feature_matrix(cb, name), seed)
        model = _feature_hybrid(cb, cf, ids, predicted, selected[name])
        scorers[f"Feature_{name}"] = lambda user, m=model: m.score_user_items(user)["S_MF"]
        scorers[f"Hybrid_{name}"] = lambda user, m=model: m.score_user_items(user)["S_Hybrid"]
    rows, per_user = evaluate(setup, scorers, test, max_users)

    intervals = {}
    for name in scorers:
        intervals[name] = {f"{metric}_vs_{base}": paired_bootstrap(per_user[name][metric], per_user[base][metric], seed=seed)
                           for metric, base in BASELINES if base != name}
    return {"selected_n_new": selected, "selection_metric": "NDCG@10 among unknown-artist songs on validation",
            "validation": val_rows, "test": rows, "confidence_intervals": intervals, "holdout_r2": r2, "seed": seed,
            "n_mock_cold": int(len(cold)), "n_train_rows": int(len(masked)), "n_test_rows": int(len(test))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare models among artists the user has never heard")
    parser.add_argument("--music", default="data/Music Info.csv")
    parser.add_argument("--train-history", default="data/shared_train_history.csv")
    parser.add_argument("--test-history", default="data/shared_test_history.csv")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/hybrid/unknown_artist")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    valid = set(music["track_id"].dropna())
    train = load_history(args.train_history, valid, user_fraction=1.0)
    test = load_history(args.test_history, valid, user_fraction=1.0)
    result = run_unknown_artist_experiment(music, train, test, seed=args.seed, max_users=args.max_users)
    (out / "unknown_artist_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(result["test"]).round(4).T.to_string())


if __name__ == "__main__":
    main()
