"""Experiment: limit how many songs one artist can place in a user's list.

The final model's scores do not change; only the list is re-ranked (cap_top_k in final_model.py). Top-20
lists use twice the Top-10 cap.

The cap is chosen on validation before TEST is read: the tightest cap whose NDCG@10 and new-song Recall@20
are both at least 90% of the uncapped list. If none qualifies, the list stays uncapped.

Run:
  python -m experiments.exp5_artist_cap.artist_cap_experiment --music "data/Music Info.csv" \
      --train-history data/shared_train_history.csv --test-history data/shared_test_history.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from content_based import load_history, split_history_by_user
from final_model import N_CLUSTERS, build_final_model, cap_top_k, evaluate_capped, paired_bootstrap  # noqa: F401
from hybrid import mask_mock_cold

CAPS = (None, 1, 2, 3, 5)
FLOOR = 0.9


def choose_cap(rows: dict, floor: float = FLOOR) -> str:
    """The tightest cap keeping both metrics at floor x the uncapped list; otherwise no cap."""
    base = rows["none"]
    for label in sorted((k for k in rows if k != "none"), key=int):
        row = rows[label]
        if row["ndcg10"] >= floor * base["ndcg10"] and row["mock_recall20"] >= floor * base["mock_recall20"]:
            return label
    return "none"


def run_cap_experiment(music: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, caps=CAPS,
                       mock_cold_fraction: float = 0.10, seed: int = 42, max_users: int = 2000, val_users: int = 1000,
                       cf_kwargs: dict | None = None, n_clusters: int = N_CLUSTERS) -> dict:
    """Choose the cap on validation, then report every cap on TEST."""
    masked, cold = mask_mock_cold(train, mock_cold_fraction, seed)
    fit_part, validation = split_history_by_user(masked, test_fraction=0.10, seed=seed)
    val_fit, val_cold = mask_mock_cold(fit_part, mock_cold_fraction, seed + 1)

    print("Training the final model on the validation part...", flush=True)
    model = build_final_model(music, val_fit, seed, cf_kwargs, n_clusters)
    val_rows, _ = evaluate_capped(model, validation, val_cold, caps, val_users, seed)
    selected = choose_cap(val_rows)
    print(f"Selected on validation: cap={selected}", flush=True)

    print("Training the final model on all of TRAIN...", flush=True)
    model = build_final_model(music, masked, seed, cf_kwargs, n_clusters)
    rows, per_user = evaluate_capped(model, test, cold, caps, max_users, seed)
    intervals = {}
    if selected != "none":
        for metric in ("ndcg10", "mock_recall20"):
            intervals[f"{metric}_vs_no_cap"] = paired_bootstrap(per_user[selected][metric], per_user["none"][metric],
                                                                seed=seed)
    return {"selected_cap": selected, "selection_metric": f"tightest cap with NDCG@10 and new-song Recall@20 >= {FLOOR} x no cap",
            "validation": val_rows, "test": rows, "confidence_intervals": intervals, "seed": seed,
            "n_mock_cold": int(len(cold)), "n_train_rows": int(len(masked)), "n_test_rows": int(len(test))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Limit songs per artist in the final model's lists")
    parser.add_argument("--music", default="data/Music Info.csv")
    parser.add_argument("--train-history", default="data/shared_train_history.csv")
    parser.add_argument("--test-history", default="data/shared_test_history.csv")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/hybrid/artist_cap")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    valid = set(music["track_id"].dropna())
    train = load_history(args.train_history, valid, user_fraction=1.0)
    test = load_history(args.test_history, valid, user_fraction=1.0)
    result = run_cap_experiment(music, train, test, seed=args.seed, max_users=args.max_users)
    (out / "artist_cap_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(result["test"]).round(4).to_string())


if __name__ == "__main__":
    main()
