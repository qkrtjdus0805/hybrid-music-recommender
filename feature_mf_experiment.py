"""Experiment: predict a new song's MF vector from its music features instead of from its artist.

  q_new        = Ridge(music features -> MF song vector), trained on songs with listening history
  S_MF(u, new) = p_u . q_new, scaled to [0, 1] with the range of p_u . q_i over the user's unheard trained songs
  alpha(new)   = compute_dynamic_alpha with n = a fixed listener count (a setting), because a new song has no history

The aim is to reach songs by artists the user has not heard. Two feature sets are kept apart, because tags come
from listeners and may not exist for a truly new song: audio only, and audio with genre and tags.

Each feature set's setting is chosen on validation before TEST is read: among settings whose NDCG@10 is at least
90% of the current final model, take the highest Recall@20 on new songs by artists the user has not heard.
If none qualifies, the current final model stays; the best setting is still reported for reference.

Run:
  python feature_mf_experiment.py --music "Music Info.csv" \
      --train-history shared_train_history.csv --test-history shared_test_history.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from collaborative import CollaborativeRecommender
from content_based import AUDIO_FEATURE_COLS, load_history, split_history_by_user
from final_model import (FINAL_BETA, FINAL_CAP, FINAL_GAMMA, N_CLUSTERS, ArtistAffinity, ArtistCB, ArtistMF,
                         FinalRecommender, _ListenerCounts, evaluate_capped, paired_bootstrap)
from hybrid import mask_mock_cold
from scaler_experiment import _CachedMF, make_cb

FEATURE_SETS = ("audio", "audio_tags")
BETAS = (0.0, 0.5, 0.9)
NEW_SONG_LISTENERS = (25.0, 75.0, 225.0)  # alpha(new) = 0.625, 0.4375, 0.325
FLOOR = 0.9
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def feature_matrix(cb, name: str) -> np.ndarray:
    """Music features aligned to cb.music_info: scaled audio columns, optionally with genre and tag TF-IDF."""
    if name == "audio":
        return cb.track_features[:, :len(AUDIO_FEATURE_COLS)]
    if name == "audio_tags":
        return cb.track_features
    raise ValueError(f"Unknown feature set: {name}")


def fit_feature_regression(cf, ids: np.ndarray, features: np.ndarray, seed: int = 42) -> tuple[np.ndarray, float]:
    """Learn features -> MF vector on trained songs. Returns a predicted vector per song and the held-out R^2."""
    trained = np.flatnonzero(cf.get_item_listener_counts(ids) > 0)
    targets = cf.Q[[cf.i_map[str(ids[i])] for i in trained]]
    order = np.random.default_rng(seed).permutation(len(trained))
    n_hold = max(1, len(trained) // 10)
    hold, fit = order[:n_hold], order[n_hold:]
    trial = RidgeCV(alphas=RIDGE_ALPHAS).fit(features[trained[fit]], targets[fit])
    r2 = r2_score(targets[hold], trial.predict(features[trained[hold]]), multioutput="variance_weighted")
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit(features[trained], targets)
    return model.predict(features), float(r2)


class FeatureMF:
    """A Stage 2 model that scores new songs with feature-predicted vectors; trained songs are unchanged."""

    def __init__(self, cf, candidate_ids: np.ndarray, predicted: np.ndarray, n_new: float):
        self._cf = cf
        self.base = cf
        self._ids = np.asarray(candidate_ids, dtype=str)
        self._predicted = predicted
        counts = cf.get_item_listener_counts(self._ids)
        self._warm = counts > 0
        self._warm_q = np.array([cf.i_map[t] for t in self._ids[self._warm]], dtype=int)
        self._warm_position = {t: i for i, t in enumerate(self._ids[self._warm])}
        n_effective = counts.astype(float)
        n_effective[~self._warm] = n_new
        self._alpha = CollaborativeRecommender.compute_dynamic_alpha(_ListenerCounts(n_effective), self._ids)

    def __getattr__(self, name):
        return getattr(self._cf, name)

    def _check_order(self, track_ids) -> None:
        if not np.array_equal(np.asarray(track_ids, dtype=str), self._ids):
            raise ValueError("Candidates must keep the order given to FeatureMF")

    def compute_dynamic_alpha(self, track_ids) -> np.ndarray:
        self._check_order(track_ids)
        return self._alpha

    def score_user_items(self, user_id, candidate_track_ids) -> np.ndarray:
        self._check_order(candidate_track_ids)
        user_id = str(user_id)
        scores = self._cf.score_user_items(user_id, self._ids).astype(float).copy()
        p = self._cf.P[self._cf.u_map[user_id]]
        dots = self._cf.Q[self._warm_q] @ p
        unheard = np.ones(len(dots), dtype=bool)
        heard = [self._warm_position[t] for t in self._cf.user_listened_tracks.get(user_id, ())
                 if t in self._warm_position]
        unheard[heard] = False
        pool = dots[unheard] if unheard.any() else dots
        low, span = pool.min(), np.ptp(pool)
        new = ~self._warm
        scores[new] = np.clip((self._predicted[new] @ p - low) / span, 0.0, 1.0) if span > 1e-9 else 0.0
        return scores


def choose_feature_config(rows: dict, baseline_ndcg10: float, floor: float = FLOOR) -> tuple[str, bool]:
    """Best unknown-artist recall within the accuracy floor; if none qualifies, the best recall, marked as not met."""
    recall = lambda label: rows[label]["recall20_unknown_artist"] or 0.0
    allowed = [label for label, row in rows.items() if row["ndcg10"] >= floor * baseline_ndcg10]
    if allowed:
        return max(allowed, key=lambda k: (recall(k), rows[k]["ndcg10"])), True
    return max(rows, key=lambda k: (recall(k), rows[k]["ndcg10"])), False


def run_feature_mf_experiment(music: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, feature_sets=FEATURE_SETS,
                              betas=BETAS, new_song_listeners=NEW_SONG_LISTENERS, mock_cold_fraction: float = 0.10,
                              seed: int = 42, max_users: int = 2000, val_users: int = 2000,
                              cf_kwargs: dict | None = None, n_clusters: int = N_CLUSTERS) -> dict:
    """Choose settings on validation for each feature set, then report them and the current model on TEST."""
    masked, cold = mask_mock_cold(train, mock_cold_fraction, seed)
    fit_part, validation = split_history_by_user(masked, test_fraction=0.10, seed=seed)
    val_fit, val_cold = mask_mock_cold(fit_part, mock_cold_fraction, seed + 1)
    make_cf = lambda rows: _CachedMF(CollaborativeRecommender(method="als", random_state=seed,
                                                              **(cf_kwargs or {})).fit(rows, music))
    cb = make_cb("standard", music, seed, n_clusters)
    ids = cb.music_info["track_id"].astype(str).to_numpy()
    cap_label = str(FINAL_CAP)
    current = lambda cf, affinity: FinalRecommender(ArtistCB(cb, affinity, FINAL_BETA),
                                                    ArtistMF(cf, ids, affinity.codes, FINAL_GAMMA))

    print("Training on the validation part...", flush=True)
    cf = make_cf(val_fit)
    cb.fit_user_profiles(val_fit)
    affinity = ArtistAffinity(cb, val_fit)
    baseline = evaluate_capped(current(cf, affinity), validation, val_cold, (FINAL_CAP,), val_users, seed)[0][cap_label]
    print(f"Current final model on validation: NDCG@10={baseline['ndcg10']:.4f}", flush=True)

    val_rows, selected, choices = {}, {}, {}
    val_r2 = {}
    for feature_set in feature_sets:
        predicted, val_r2[feature_set] = fit_feature_regression(cf, ids, feature_matrix(cb, feature_set), seed)
        print(f"[{feature_set}] held-out R^2={val_r2[feature_set]:.4f}", flush=True)
        rows = {}
        for beta in betas:
            for n_new in new_song_listeners:
                label = f"{feature_set}|beta={beta}|n={n_new}"
                print(f"[validation] {label}", flush=True)
                model = FinalRecommender(ArtistCB(cb, affinity, beta), FeatureMF(cf, ids, predicted, n_new))
                rows[label] = evaluate_capped(model, validation, val_cold, (FINAL_CAP,), val_users, seed)[0][cap_label]
                choices[label] = (beta, n_new)
        best, met = choose_feature_config(rows, baseline["ndcg10"])
        selected[feature_set] = {"label": best, "beta": choices[best][0], "n_new": choices[best][1],
                                 "criterion_met": met}
        val_rows.update(rows)
        print(f"[{feature_set}] selected {best} (criterion met: {met})", flush=True)

    print("Training on all of TRAIN...", flush=True)
    cf = make_cf(masked)
    cb.fit_user_profiles(masked)
    affinity = ArtistAffinity(cb, masked)
    rows, per_user = {}, {}
    rows["final"], per_user["final"] = (x[cap_label] for x in
                                        evaluate_capped(current(cf, affinity), test, cold, (FINAL_CAP,), max_users, seed))
    intervals, r2 = {}, {}
    for feature_set in feature_sets:
        predicted, r2[feature_set] = fit_feature_regression(cf, ids, feature_matrix(cb, feature_set), seed)
        choice = selected[feature_set]
        model = FinalRecommender(ArtistCB(cb, affinity, choice["beta"]), FeatureMF(cf, ids, predicted, choice["n_new"]))
        print(f"[test] {feature_set}", flush=True)
        rows[feature_set], per_user[feature_set] = (x[cap_label] for x in
                                                    evaluate_capped(model, test, cold, (FINAL_CAP,), max_users, seed))
        intervals[feature_set] = {
            f"{metric}_vs_final": paired_bootstrap(per_user[feature_set][column], per_user["final"][column], seed=seed)
            for metric, column in (("ndcg10", "ndcg10"), ("mock_recall20", "mock_recall20"),
                                   ("unknown_recall20", "unknown_recall20"))}
    return {
        "selected": selected,
        "selection_metric": f"best unknown-artist new-song Recall@20 with NDCG@10 >= {FLOOR} x current final model",
        "validation_baseline": baseline, "validation": val_rows, "test": rows, "confidence_intervals": intervals,
        "holdout_r2": r2, "validation_holdout_r2": val_r2, "seed": seed, "cap": FINAL_CAP,
        "n_mock_cold": int(len(cold)), "n_train_rows": int(len(masked)), "n_test_rows": int(len(test)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict new songs' MF vectors from music features")
    parser.add_argument("--music", default="Music Info.csv")
    parser.add_argument("--train-history", default="shared_train_history.csv")
    parser.add_argument("--test-history", default="shared_test_history.csv")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="hybrid_output/feature_mf")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    valid = set(music["track_id"].dropna())
    train = load_history(args.train_history, valid, user_fraction=1.0)
    test = load_history(args.test_history, valid, user_fraction=1.0)
    result = run_feature_mf_experiment(music, train, test, seed=args.seed, max_users=args.max_users)
    (out / "feature_mf_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(result["test"]).round(4).to_string())


if __name__ == "__main__":
    main()
