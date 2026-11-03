"""Experiment: give new songs a real S_MF by borrowing from the artist's trained songs.

  S_MF(u, new) = mean S_MF(u, j) over trained songs j by the same artist
  alpha(new)   = compute_dynamic_alpha with n = gamma * mean listeners of those songs

New songs then compete in one ranking with every other song, with no reserved slots. Songs by an artist
with no trained songs keep S_MF = 0 and alpha = 1.

Settings are chosen on validation before TEST is read: among settings whose NDCG@10 is at least 90% of
"fixed alpha + artist", take the highest new-song Recall@20. If none qualifies, take the best NDCG@10.

Run:
  python artist_mf_experiment.py --music "Music Info.csv" \
      --train-history shared_train_history.csv --test-history shared_test_history.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from artist_experiment import mock_topk
from collaborative import CollaborativeRecommender
from content_based import load_history, split_history_by_user
from final_model import ArtistAffinity, ArtistCB, ArtistMF, paired_bootstrap, per_user_metrics
from hybrid import HybridRecommender, evaluate_hybrid, mask_mock_cold
from scaler_experiment import FIXED_ALPHA, _CachedMF, make_cb

BETAS = (0.0, 0.5, 0.9)
GAMMAS = (0.0, 0.1, 0.25, 0.5, 1.0)
REFERENCE_BETA = 0.9  # the artist weight chosen in the artist experiment
ACCURACY_FLOOR = 0.9


def choose_config(rows: dict, baseline_ndcg10: float, floor: float = ACCURACY_FLOOR) -> tuple[str, bool]:
    """Best new-song recall among settings that keep enough accuracy; otherwise the most accurate setting."""
    allowed = [label for label, row in rows.items() if row["ndcg10"] >= floor * baseline_ndcg10]
    if allowed:
        return max(allowed, key=lambda k: (rows[k]["mock_recall20"], rows[k]["ndcg10"])), True
    return max(rows, key=lambda k: rows[k]["ndcg10"]), False


class _HybridScores:
    """Lets mock_topk rank by S_Hybrid while splitting answers by known and unknown artists."""

    def __init__(self, hybrid: HybridRecommender, affinity: ArtistAffinity):
        self._hybrid, self.affinity = hybrid, affinity
        self.music_info = hybrid.cb.music_info
        self.user_profiles = hybrid.cb.user_profiles
        self.user_listened_tracks = hybrid.cb.user_listened_tracks

    def score_user_items(self, user_id) -> np.ndarray:
        return self._hybrid.score_user_items(user_id)["S_Hybrid"]


def _flatten(model: dict) -> dict:
    overall, buckets, exposure = model["overall"]["@10"], model["bucket_recall"]["@20"], model["exposure"]["@10"]
    return {
        "ndcg10": overall["ndcg"], "recall10": overall["recall"], "precision10": overall["precision"],
        "mock_recall20": buckets.get("mock_cold", 0.0), "recall20_sample_zero": buckets.get("sample_zero", 0.0),
        "recall20_1_4": buckets.get("1-4", 0.0), "recall20_5_24": buckets.get("5-24", 0.0),
        "recall20_25plus": buckets.get("25+", 0.0),
        "exposure10_cold": exposure["mock_cold"] + exposure["sample_zero"], "exposure10_1_4": exposure["1-4"],
        "exposure10_5_24": exposure["5-24"], "exposure10_25plus": exposure["25+"],
        "coverage10": model["catalog_coverage"]["@10"], "mock_percentile": model["mock_cold_percentile"],
    }


def _full_row(hybrid, affinity, test, cold, max_users, seed) -> dict:
    result = evaluate_hybrid(hybrid, test, cold, max_users=max_users, seed=seed)
    row = _flatten(result["models"]["Hybrid"])
    row.update({k: result[k] for k in ("top10_cold_share", "top10_overlap_with_cb", "hybrid_effect_verifiable")})
    split = mock_topk(_HybridScores(hybrid, affinity), test, cold, 20, max_users, seed)
    row.update({k: split[k] for k in ("known_artist_answer_share", "recall_known_artist", "recall_unknown_artist")})
    row["_mf"] = _flatten(result["models"]["MF"])
    return row


def run_artist_mf_experiment(music: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, betas=BETAS, gammas=GAMMAS,
                             mock_cold_fraction: float = 0.10, seed: int = 42, max_users: int = 2000,
                             val_users: int = 500, cf_kwargs: dict | None = None, n_clusters: int = 6) -> dict:
    """Choose beta and gamma on validation, then report the chosen model and its references on TEST."""
    masked, cold = mask_mock_cold(train, mock_cold_fraction, seed)
    fit_part, validation = split_history_by_user(masked, test_fraction=0.10, seed=seed)
    val_fit, val_cold = mask_mock_cold(fit_part, mock_cold_fraction, seed + 1)
    make_cf = lambda rows: _CachedMF(CollaborativeRecommender(method="als", random_state=seed,
                                                              **(cf_kwargs or {})).fit(rows, music))
    cb = make_cb("standard", music, seed, n_clusters)
    ids = cb.music_info["track_id"].astype(str).to_numpy()
    reference_beta = REFERENCE_BETA if REFERENCE_BETA in betas else max(betas)

    print("Training MF on the validation part...", flush=True)
    cf = make_cf(val_fit)
    cb.fit_user_profiles(val_fit)
    affinity = ArtistAffinity(cb, val_fit)
    reference = HybridRecommender(ArtistCB(cb, affinity, reference_beta), cf, "fixed", FIXED_ALPHA)
    reference_result = evaluate_hybrid(reference, validation, val_cold, (10, 20), val_users, seed)
    baseline_ndcg10 = reference_result["models"]["Hybrid"]["overall"]["@10"]["ndcg"]

    val_rows, choices = {}, {}
    for beta in betas:
        for gamma in gammas:
            label = f"beta={beta}|gamma={gamma}"
            print(f"[validation] {label}", flush=True)
            model = HybridRecommender(ArtistCB(cb, affinity, beta), ArtistMF(cf, ids, affinity.codes, gamma))
            result = evaluate_hybrid(model, validation, val_cold, (10, 20), val_users, seed)
            flat = _flatten(result["models"]["Hybrid"])
            val_rows[label] = {**flat, "top10_cold_share": result["top10_cold_share"]}
            choices[label] = (beta, gamma)
    best, met = choose_config(val_rows, baseline_ndcg10)
    beta, gamma = choices[best]
    print(f"Selected on validation: {best} (accuracy floor met: {met})", flush=True)

    print("Training MF on all of TRAIN...", flush=True)
    cf = make_cf(masked)
    cb.fit_user_profiles(masked)
    affinity = ArtistAffinity(cb, masked)
    artist_cb = ArtistCB(cb, affinity, reference_beta)
    models = {
        "fixed_artist": HybridRecommender(artist_cb, cf, "fixed", FIXED_ALPHA),
        "dynamic_artist": HybridRecommender(artist_cb, cf),
        "selected": HybridRecommender(ArtistCB(cb, affinity, beta), ArtistMF(cf, ids, affinity.codes, gamma)),
    }
    rows = {}
    for name, model in models.items():
        print(f"[test] {name}", flush=True)
        rows[name] = _full_row(model, affinity, test, cold, max_users, seed)
    rows["mf"] = rows["dynamic_artist"]["_mf"]
    for name in models:
        rows[name].pop("_mf")

    print("Computing confidence intervals...", flush=True)
    per_user = {name: per_user_metrics(model, test, cold, max_users, seed) for name, model in models.items()}
    per_user["mf"] = per_user_metrics(models["dynamic_artist"], test, cold, max_users, seed, "S_MF")
    pairs = {"ndcg10_vs_mf": ("ndcg10", "selected", "mf"),
             "ndcg10_vs_fixed_artist": ("ndcg10", "selected", "fixed_artist"),
             "ndcg10_vs_dynamic_artist": ("ndcg10", "selected", "dynamic_artist"),
             "mock_recall20_vs_fixed_artist": ("mock_recall20", "selected", "fixed_artist"),
             "mock_recall20_vs_dynamic_artist": ("mock_recall20", "selected", "dynamic_artist"),
             "fixed_artist_ndcg10_vs_mf": ("ndcg10", "fixed_artist", "mf")}
    intervals = {name: paired_bootstrap(per_user[a][metric], per_user[b][metric], seed=seed)
                 for name, (metric, a, b) in pairs.items()}
    return {
        "selected": {"beta": beta, "gamma": gamma, "accuracy_floor_met": met},
        "selection_metric": f"best new-song Recall@20 with NDCG@10 >= {ACCURACY_FLOOR} x fixed alpha + artist",
        "validation_baseline_ndcg10": baseline_ndcg10, "validation": val_rows, "test": rows,
        "confidence_intervals": intervals, "seed": seed, "fixed_alpha": FIXED_ALPHA, "n_mock_cold": int(len(cold)),
        "n_train_rows": int(len(masked)), "n_test_rows": int(len(test)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate S_MF for new songs from their artist")
    parser.add_argument("--music", default="Music Info.csv")
    parser.add_argument("--train-history", default="shared_train_history.csv")
    parser.add_argument("--test-history", default="shared_test_history.csv")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="hybrid_output/artist_mf")
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETAS))
    parser.add_argument("--gammas", type=float, nargs="+", default=list(GAMMAS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    valid = set(music["track_id"].dropna())
    train = load_history(args.train_history, valid, user_fraction=1.0)
    test = load_history(args.test_history, valid, user_fraction=1.0)
    result = run_artist_mf_experiment(music, train, test, betas=tuple(args.betas), gammas=tuple(args.gammas),
                                      seed=args.seed, max_users=args.max_users)
    (out / "artist_mf_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(result["test"]).round(4).to_string())


if __name__ == "__main__":
    main()
