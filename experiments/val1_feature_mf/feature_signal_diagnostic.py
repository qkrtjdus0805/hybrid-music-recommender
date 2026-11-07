"""Diagnostic: do music features rank new songs by unheard artists better than chance, and where do those songs land?

For each answer, measures its percentile among mock-cold songs (predicted vector, shuffled vectors, S_CB) and in the
whole catalogue (S_Hybrid without the artist term, n=75). 0.5 is chance. Results are not used to choose any setting.

Run:
  python -m experiments.val1_feature_mf.feature_signal_diagnostic
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from collaborative import CollaborativeRecommender
from content_based import load_history
from experiments.val1_feature_mf.feature_mf_experiment import FEATURE_SETS, FeatureMF, feature_matrix, fit_feature_regression
from final_model import ArtistAffinity, ArtistCB, FinalRecommender
from hybrid import _clean_test, mask_mock_cold
from experiments.exp2_scaler.scaler_experiment import make_cb


def percentile(values: np.ndarray, targets) -> list[float]:
    """Share of the pool ranked below each target, counting ties as half."""
    return [(values < values[t]).mean() + 0.5 * (values == values[t]).mean() for t in targets]


def main() -> None:
    music = pd.read_csv("data/Music Info.csv", encoding="utf-8-sig", dtype={"track_id": str})
    valid = set(music["track_id"].dropna())
    train = load_history("data/shared_train_history.csv", valid, user_fraction=1.0)
    test = load_history("data/shared_test_history.csv", valid, user_fraction=1.0)
    masked, cold = mask_mock_cold(train, 0.10, 42)
    cf = CollaborativeRecommender(method="als", random_state=42).fit(masked, music)
    cb = make_cb("standard", music, 42, 6)
    cb.fit_user_profiles(masked)
    affinity = ArtistAffinity(cb, masked)
    ids = cb.music_info["track_id"].astype(str).to_numpy()
    position = {t: i for i, t in enumerate(ids)}
    cold_set = {str(t) for t in cold}
    pool = np.array(sorted(position[t] for t in cold_set if t in position))
    pool_index = {p: i for i, p in enumerate(pool)}

    df = _clean_test(test)
    df = df[df["track_id"].isin(cold_set) & df["user_id"].isin(cf.u_map) & df["user_id"].isin(cb.user_profiles)]
    answers = df.groupby("user_id")["track_id"].agg(set)

    rng = np.random.default_rng(0)
    rows = {}
    for name in FEATURE_SETS:
        predicted, r2 = fit_feature_regression(cf, ids, feature_matrix(cb, name), 42)
        model = FinalRecommender(ArtistCB(cb, affinity, 0.0), FeatureMF(cf, ids, predicted, 75.0))
        shuffled = predicted[rng.permutation(len(predicted))]
        within, within_shuffled, within_cb, overall, best_rank, n_answers = [], [], [], [], [], 0
        for user in sorted(answers.index):
            heard = affinity.scores(user) > 0
            targets = [position[t] for t in answers[user] if t in position and not heard[position[t]]]
            if not targets:
                continue
            n_answers += len(targets)
            local = [pool_index[t] for t in targets]
            p = cf.P[cf.u_map[user]]
            within += percentile(predicted[pool] @ p, local)
            within_shuffled += percentile(shuffled[pool] @ p, local)
            scores = model.score_user_items(user)
            within_cb += percentile(scores["S_CB"][pool], local)
            full = scores["S_Hybrid"]
            overall += percentile(full, targets)
            best_rank.append(1 + int((full > full[targets].max()).sum()))
        rows[name] = {"holdout_r2": r2, "n_answers": n_answers, "n_users": len(best_rank),
                      "percentile_among_new_predicted": np.mean(within),
                      "percentile_among_new_shuffled": np.mean(within_shuffled),
                      "percentile_among_new_s_cb": np.mean(within_cb),
                      "percentile_catalogue_s_hybrid": np.mean(overall),
                      "median_best_rank": float(np.median(best_rank)),
                      "share_users_in_top20": float(np.mean(np.array(best_rank) <= 20)),
                      "share_users_in_top200": float(np.mean(np.array(best_rank) <= 200))}
        print(name, rows[name], flush=True)
    out = Path("outputs/hybrid/feature_mf")
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "signal_diagnostic.csv")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
