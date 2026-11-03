"""Final model of Stage 3: dynamic-alpha hybrid with artist information.

  S_CB'(u, i)  = 0.1 * S_CB(u, i) + 0.9 * A(u, artist(i))
  S_Hybrid     = alpha(i) * S_CB'(u, i) + (1 - alpha(i)) * S_MF(u, i)

For a new song by an artist with trained songs:
  S_MF(u, new) = mean S_MF(u, j) over the artist's trained songs j
  alpha(new)   = compute_dynamic_alpha with n = 5 * mean listeners of those songs

Every song competes in one ranking; no slots are reserved. Lists then allow at most 5 songs per artist in a
Top-10. A song by an artist with no trained songs keeps S_MF = 0 and alpha = 1, so it can only be scored by
0.1 * S_CB. In practice new songs are found only from artists the user has already heard, and lists still
concentrate on a few artists (see README_hybrid.md).

beta and gamma were chosen on validation data in artist_mf_experiment.py, the cap in artist_cap_experiment.py.
README_hybrid.md has the results.

Run:
  python final_model.py evaluate --music "Music Info.csv" \
      --train-history shared_train_history.csv --test-history shared_test_history.csv
  python final_model.py recommend --music "Music Info.csv" --train-history shared_train_history.csv --user USER_ID
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from collaborative import CollaborativeRecommender
from content_based import HISTORY_COLS, ContentBasedRecommender, load_history
from hybrid import (BUCKETS, HybridRecommender, _clean_test, assign_buckets, evaluate_hybrid, mask_mock_cold,
                    ranking_metrics, top_k_indices)

FINAL_BETA = 0.9
FINAL_GAMMA = 5.0
FINAL_CAP = 5  # most songs by one artist in a Top-10 list
FIXED_ALPHA = 0.25  # comparison model only
N_CLUSTERS = 6


class ArtistAffinity:
    """Per-user artist preference in [0, 1], aligned to cb.music_info, learned from TRAIN only."""

    def __init__(self, cb, train: pd.DataFrame):
        music = cb.music_info
        names = music["artist"].astype("string").str.strip().str.lower()
        self.codes, _ = pd.factorize(names)  # missing artists become -1
        position = {t: i for i, t in enumerate(music["track_id"].astype(str))}
        hist = train[HISTORY_COLS].dropna().copy()
        hist[["user_id", "track_id"]] = hist[["user_id", "track_id"]].astype(str)
        hist["playcount"] = pd.to_numeric(hist["playcount"], errors="coerce")
        hist = hist.loc[(hist["playcount"] > 0) & hist["track_id"].isin(position)]
        artist = self.codes[hist["track_id"].map(position).to_numpy(dtype=int)]
        hist, artist = hist.loc[artist >= 0], artist[artist >= 0]
        users, rows = np.unique(hist["user_id"].to_numpy(dtype=str), return_inverse=True)
        self._user_row = {u: i for i, u in enumerate(users)}
        self._matrix = csr_matrix((np.log1p(hist["playcount"].to_numpy(dtype=float)), (rows, artist)),
                                  shape=(len(users), int(self.codes.max()) + 1))

    def scores(self, user_id: str) -> np.ndarray:
        """A(u, artist) for every song: the user's log-playcount on the artist over their largest such value."""
        out = np.zeros(len(self.codes))
        row = self._user_row.get(str(user_id))
        if row is None:
            return out
        values = self._matrix[row].toarray().ravel()
        known = self.codes >= 0
        out[known] = values[self.codes[known]] / values.max()
        return out


class ArtistCB:
    """A Stage 1 model whose score also counts artist preference; everything else is delegated."""

    def __init__(self, cb, affinity: ArtistAffinity, beta: float):
        self._cb, self.affinity, self.beta = cb, affinity, beta
        self.base = cb

    def __getattr__(self, name):
        return getattr(self._cb, name)

    def score_user_items(self, user_id, candidate_features=None) -> np.ndarray:
        base = self._cb.score_user_items(user_id, candidate_features)
        return (1 - self.beta) * base + self.beta * self.affinity.scores(user_id)


class _ListenerCounts:
    """Lets compute_dynamic_alpha run unchanged on replacement listener counts."""

    def __init__(self, counts: np.ndarray):
        self._counts = counts

    def get_item_listener_counts(self, _track_ids) -> np.ndarray:
        return self._counts


class ArtistMF:
    """A Stage 2 model that estimates S_MF and alpha for new songs from the same artist's trained songs."""

    def __init__(self, cf, candidate_ids: np.ndarray, artist_codes: np.ndarray, gamma: float):
        self._cf = cf
        self.base = cf
        self._ids = np.asarray(candidate_ids, dtype=str)
        self._codes = artist_codes
        counts = cf.get_item_listener_counts(self._ids)
        warm = counts > 0
        self._donors = warm & (artist_codes >= 0)
        size = int(artist_codes.max()) + 1 if len(artist_codes) else 0
        self._n_songs = np.bincount(artist_codes[self._donors], minlength=size)
        listeners = np.bincount(artist_codes[self._donors], weights=counts[self._donors], minlength=size)
        known = artist_codes >= 0
        has_donor = np.zeros(len(artist_codes), dtype=bool)
        has_donor[known] = self._n_songs[artist_codes[known]] > 0
        self._estimated = ~warm & has_donor
        n_effective = counts.astype(float)
        codes = artist_codes[self._estimated]
        n_effective[self._estimated] = gamma * listeners[codes] / self._n_songs[codes]
        self._alpha = CollaborativeRecommender.compute_dynamic_alpha(_ListenerCounts(n_effective), self._ids)

    def __getattr__(self, name):
        return getattr(self._cf, name)

    def _check_order(self, track_ids) -> None:
        if not np.array_equal(np.asarray(track_ids, dtype=str), self._ids):
            raise ValueError("Candidates must keep the order given to ArtistMF")

    def compute_dynamic_alpha(self, track_ids) -> np.ndarray:
        self._check_order(track_ids)
        return self._alpha

    def score_user_items(self, user_id, candidate_track_ids) -> np.ndarray:
        self._check_order(candidate_track_ids)
        scores = self._cf.score_user_items(user_id, self._ids).copy()
        totals = np.bincount(self._codes[self._donors], weights=scores[self._donors], minlength=len(self._n_songs))
        codes = self._codes[self._estimated]
        scores[self._estimated] = totals[codes] / self._n_songs[codes]
        return scores


CAP_PREFIX = 1000  # songs sorted first by cap_top_k; sorting the whole catalog per user dominated evaluation time


def _ranked_prefix(scores: np.ndarray, track_ids: np.ndarray, exclude: np.ndarray, size: int) -> np.ndarray:
    """The start of the full ranking (score, then id): at least `size` allowed songs, plus any tied with the last."""
    allowed = np.flatnonzero(~exclude)
    if len(allowed) > size:
        values = scores[allowed]
        allowed = allowed[values >= np.partition(values, len(values) - size)[len(values) - size]]
    return allowed[np.lexsort((track_ids[allowed], -scores[allowed]))]


def cap_top_k(scores: np.ndarray, track_ids: np.ndarray, exclude: np.ndarray, artist_codes: np.ndarray,
              cap: int | None, k: int) -> np.ndarray:
    """Best-first positions of k songs with at most cap per artist; songs with no artist are not limited.

    Skipped songs fill the list only when too few artists exist to fill it otherwise.
    """
    if cap is None:
        return top_k_indices(scores, track_ids, exclude, k)
    for size in (CAP_PREFIX, int((~exclude).sum())):  # the short list almost always fills k; else use every song
        used = collections.Counter()
        picked, skipped = [], []
        for position in _ranked_prefix(scores, track_ids, exclude, size):
            code = artist_codes[position]
            if code >= 0 and used[code] >= cap:
                skipped.append(position)
                continue
            used[code] += 1
            picked.append(position)
            if len(picked) == k:
                break
        if len(picked) == k:
            break
    picked.extend(skipped[:k - len(picked)])
    return np.array(picked, dtype=int)


class FinalRecommender(HybridRecommender):
    """The final model: S_Hybrid ranking with a limit on songs per artist."""

    def recommend_user(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        """Top-N unheard tracks; the artist limit grows with the list (5 in 10, 10 in 20)."""
        scores = self.score_user_items(user_id)
        cap = max(1, math.ceil(top_n * FINAL_CAP / 10))
        idx = cap_top_k(scores["S_Hybrid"], self.candidate_ids, self.seen_mask(user_id), self.cb.affinity.codes, cap,
                        top_n)
        return self._recommendation_table(idx, scores)


def build_final_model(music: pd.DataFrame, train: pd.DataFrame, seed: int = 42, cf_kwargs: dict | None = None,
                      n_clusters: int = N_CLUSTERS) -> FinalRecommender:
    """Train Stage 1 and Stage 2 on TRAIN and combine them with the final settings."""
    cb = ContentBasedRecommender(n_clusters=n_clusters, random_state=seed).fit(music, train)
    cf = CollaborativeRecommender(method="als", random_state=seed, **(cf_kwargs or {})).fit(train, music)
    affinity = ArtistAffinity(cb, train)
    ids = cb.music_info["track_id"].astype(str).to_numpy()
    return FinalRecommender(ArtistCB(cb, affinity, FINAL_BETA), ArtistMF(cf, ids, affinity.codes, FINAL_GAMMA))


def list_diversity(ranker, affinity: ArtistAffinity, users, k: int = 10, score_key: str = "S_Hybrid") -> dict:
    """How varied each user's own list is: distinct artists in Top-k, and the share of slots by artists they heard."""
    distinct, heard_slots, slots = [], 0, 0
    for user in users:
        idx = top_k_indices(ranker.score_user_items(user)[score_key], ranker.candidate_ids, ranker.seen_mask(user), k)
        codes = affinity.codes[idx]
        distinct.append(len(set(codes[codes >= 0].tolist())))
        heard_slots += int((affinity.scores(user)[idx] > 0).sum())
        slots += len(idx)
    return {"mean_distinct_artists": float(np.mean(distinct)), "heard_artist_share": heard_slots / slots,
            "n_users": len(distinct)}


def _sample_users(hybrid: HybridRecommender, test: pd.DataFrame, max_users: int, seed: int) -> np.ndarray:
    """The same users evaluate_hybrid samples."""
    df = _clean_test(test)
    tested = df.loc[df["track_id"].isin(hybrid.position), "user_id"].unique()
    users = np.sort(np.array([u for u in tested if u in hybrid.cb.user_profiles and u in hybrid.cf.u_map]))
    if len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
    return users


def _label(cap: int | None) -> str:
    return "none" if cap is None else str(cap)


def evaluate_capped(model, test: pd.DataFrame, cold: np.ndarray, caps=(None, FINAL_CAP), max_users: int = 2000,
                    seed: int = 42) -> tuple[dict, dict]:
    """Score each user once, then measure every cap on the same users, candidates and answers.

    Top-10 lists use the cap and Top-20 lists use twice the cap. Returns summary rows and per-user tables.
    """
    ids = model.candidate_ids
    affinity = model.cb.affinity
    codes = affinity.codes
    labels = assign_buckets(ids, model.n_listeners, {str(t) for t in cold})
    label_of = dict(zip(ids, labels))
    is_cold = model.n_listeners == 0
    df = _clean_test(test)
    answers = df.loc[df["track_id"].isin(model.position)].groupby("user_id")["track_id"].agg(set)
    users = _sample_users(model, test, max_users, seed)

    store = {_label(c): {"metrics": [], "buckets": {b: [] for b in BUCKETS}, "exposure": collections.Counter(),
                         "covered": set(), "distinct": [], "heard": 0, "slots": 0, "cold": [], "per_user": {},
                         "known": [], "unknown": [], "n_known": 0, "n_new": 0}
             for c in caps}
    for user in users:
        scores = model.score_user_items(user)["S_Hybrid"]
        seen = model.seen_mask(user)
        relevant = answers[user]
        by_bucket = collections.defaultdict(set)
        for track in relevant:
            by_bucket[label_of[track]].add(track)
        heard = affinity.scores(user) > 0
        new = by_bucket.get("mock_cold", set())
        new_known = {t for t in new if heard[model.position[t]]}
        for cap in caps:
            box = store[_label(cap)]
            top10 = cap_top_k(scores, ids, seen, codes, cap, 10)
            top20 = cap_top_k(scores, ids, seen, codes, None if cap is None else cap * 2, 20)
            metrics = ranking_metrics(ids[top10].tolist(), relevant, 10)
            box["metrics"].append(metrics)
            head = set(ids[top20].tolist())
            for bucket, tracks in by_bucket.items():
                box["buckets"][bucket].append(len(head & tracks) / len(tracks))
            box["exposure"].update(labels[top10])
            box["covered"].update(ids[top10].tolist())
            box["distinct"].append(len(set(codes[top10][codes[top10] >= 0].tolist())))
            box["heard"] += int(heard[top10].sum())
            box["slots"] += len(top10)
            box["cold"].append(float(is_cold[top10].mean()))
            box["n_new"] += len(new)
            box["n_known"] += len(new_known)
            if new_known:
                box["known"].append(len(head & new_known) / len(new_known))
            if new - new_known:
                box["unknown"].append(len(head & (new - new_known)) / len(new - new_known))
            new_unknown = new - new_known
            box["per_user"][user] = {"ndcg10": metrics["ndcg"],
                                     "mock_recall20": len(head & new) / len(new) if new else np.nan,
                                     "unknown_recall20": len(head & new_unknown) / len(new_unknown) if new_unknown
                                     else np.nan}

    rows, per_user = {}, {}
    for label, box in store.items():
        recall = lambda bucket: float(np.mean(box["buckets"][bucket])) if box["buckets"][bucket] else 0.0
        share = lambda *names: sum(box["exposure"][n] for n in names) / box["slots"]
        rows[label] = {
            "ndcg10": float(np.mean([m["ndcg"] for m in box["metrics"]])),
            "recall10": float(np.mean([m["recall"] for m in box["metrics"]])),
            "precision10": float(np.mean([m["precision"] for m in box["metrics"]])),
            "mock_recall20": recall("mock_cold"), "recall20_1_4": recall("1-4"), "recall20_5_24": recall("5-24"),
            "recall20_25plus": recall("25+"), "exposure10_cold": share("mock_cold", "sample_zero"),
            "exposure10_mock": share("mock_cold"), "exposure10_1_4": share("1-4"), "exposure10_5_24": share("5-24"),
            "exposure10_25plus": share("25+"), "top10_cold_share": float(np.mean(box["cold"])),
            "coverage10": len(box["covered"]) / len(ids), "mean_distinct_artists": float(np.mean(box["distinct"])),
            "heard_artist_share": box["heard"] / box["slots"], "n_users": int(len(users)),
            "known_artist_answer_share": box["n_known"] / box["n_new"] if box["n_new"] else None,
            "recall20_known_artist": float(np.mean(box["known"])) if box["known"] else None,
            "recall20_unknown_artist": float(np.mean(box["unknown"])) if box["unknown"] else None,
        }
        per_user[label] = pd.DataFrame.from_dict(box["per_user"], orient="index",
                                                 columns=["ndcg10", "mock_recall20", "unknown_recall20"])
    return rows, per_user


def per_user_metrics(hybrid: HybridRecommender, test: pd.DataFrame, cold: np.ndarray, max_users: int = 2000,
                     seed: int = 42, score_key: str = "S_Hybrid") -> pd.DataFrame:
    """Per-user metrics on the users evaluate_hybrid samples; new-song recall is NaN without such answers."""
    df = _clean_test(test)
    answers = df.loc[df["track_id"].isin(hybrid.position)].groupby("user_id")["track_id"].agg(set)
    cold_set = {str(t) for t in cold}
    rows = {}
    for user in _sample_users(hybrid, test, max_users, seed):
        idx = top_k_indices(hybrid.score_user_items(user)[score_key], hybrid.candidate_ids, hybrid.seen_mask(user), 20)
        ranked = hybrid.candidate_ids[idx].tolist()
        relevant = answers[user]
        metrics = ranking_metrics(ranked, relevant, 10)
        new_answers = relevant & cold_set
        rows[user] = {"ndcg10": metrics["ndcg"], "recall10": metrics["recall"],
                      "mock_recall20": len(new_answers & set(ranked)) / len(new_answers) if new_answers else np.nan}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["ndcg10", "recall10", "mock_recall20"])


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 42) -> dict:
    """95% interval for the mean per-user difference a - b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = (a - b)[~np.isnan(a) & ~np.isnan(b)]
    if len(diff) == 0:
        return {"mean_diff": None, "ci_low": None, "ci_high": None, "n": 0}
    picks = np.random.default_rng(seed).integers(0, len(diff), size=(n_boot, len(diff)))
    low, high = np.percentile(diff[picks].mean(axis=1), [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(len(diff))}


def evaluate_final(music: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, mock_cold_fraction: float = 0.10,
                   seed: int = 42, max_users: int = 2000, cf_kwargs: dict | None = None,
                   n_clusters: int = N_CLUSTERS) -> tuple[dict, FinalRecommender]:
    """Erase mock-cold tracks from TRAIN, train the final model, and evaluate it on TEST.

    `final` is the capped list. `no_cap` and `models` describe the same scores without the cap;
    in `models`, CB_artist and MF_estimated are this model's components, not plain Stage 1 and Stage 2.
    """
    masked, cold = mask_mock_cold(train, mock_cold_fraction, seed)
    model = build_final_model(music, masked, seed, cf_kwargs, n_clusters)
    full = evaluate_hybrid(model, test, cold, max_users=max_users, seed=seed)
    rows, per_user = evaluate_capped(model, test, cold, (None, FINAL_CAP), max_users, seed)
    final, no_cap = rows[_label(FINAL_CAP)], rows["none"]

    users = _sample_users(model, test, max_users, seed)
    affinity, plain_cb, plain_cf = model.cb.affinity, model.cb.base, model.cf.base
    mf = HybridRecommender(plain_cb, plain_cf)
    fixed = HybridRecommender(model.cb, plain_cf, "fixed", FIXED_ALPHA)
    final_users = per_user[_label(FINAL_CAP)]
    mf_users = per_user_metrics(mf, test, cold, max_users, seed, "S_MF")
    fixed_users = per_user_metrics(fixed, test, cold, max_users, seed)
    intervals = {
        "ndcg10_vs_mf": paired_bootstrap(final_users["ndcg10"], mf_users["ndcg10"], seed=seed),
        "ndcg10_vs_fixed_artist": paired_bootstrap(final_users["ndcg10"], fixed_users["ndcg10"], seed=seed),
        "ndcg10_vs_no_cap": paired_bootstrap(final_users["ndcg10"], per_user["none"]["ndcg10"], seed=seed),
        "mock_recall20_vs_fixed_artist": paired_bootstrap(final_users["mock_recall20"], fixed_users["mock_recall20"],
                                                          seed=seed),
        "mock_recall20_vs_no_cap": paired_bootstrap(final_users["mock_recall20"], per_user["none"]["mock_recall20"],
                                                    seed=seed),
    }
    report = {
        "final": final, "no_cap": no_cap, "confidence_intervals": intervals,
        "models": {"CB_artist": full["models"]["CB"], "MF_estimated": full["models"]["MF"],
                   "Hybrid_no_cap": full["models"]["Hybrid"]},
        "list_diversity": {
            "final": {k: final[k] for k in ("mean_distinct_artists", "heard_artist_share", "n_users")},
            "mf": list_diversity(mf, affinity, users, score_key="S_MF"),
            "fixed_artist": list_diversity(fixed, affinity, users),
        },
        "beta": FINAL_BETA, "gamma": FINAL_GAMMA, "cap": FINAL_CAP, "seed": seed, "mf_method": "als",
        "mock_cold_fraction": mock_cold_fraction, "n_mock_cold": int(len(cold)), "n_train_rows": int(len(masked)),
        "n_test_rows": int(len(test)), "n_users_dropped": full["n_users_dropped"],
        "bucket_test_rows": full["bucket_test_rows"],
    }
    model.mock_cold = cold
    return report, model


def _cf_kwargs(args) -> dict:
    options = {"n_factors": args.factors, "n_epochs": args.epochs}
    return {name: value for name, value in options.items() if value is not None}


def _evaluate_command(args, music: pd.DataFrame, train: pd.DataFrame, out: Path) -> None:
    test = load_history(args.test_history, set(music["track_id"].dropna()), user_fraction=1.0)
    report, model = evaluate_final(music, train, test, args.mock_cold_fraction, args.seed, args.max_users,
                                   _cf_kwargs(args), args.clusters)
    report.update({"music_file": str(args.music), "train_file": str(args.train_history),
                   "test_file": str(args.test_history)})
    (out / "final_evaluation_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    pd.DataFrame({"track_id": model.mock_cold}).to_csv(out / "mock_cold_tracks.csv", index=False)
    final = report["final"]
    print(f"NDCG@10={final['ndcg10']:.4f} Recall@10={final['recall10']:.4f} "
          f"new-song Recall@20={final['mock_recall20']:.4f} artists per list={final['mean_distinct_artists']:.2f}")


def _recommend_command(args, music: pd.DataFrame, train: pd.DataFrame, out: Path) -> None:
    if str(args.user) not in set(train["user_id"].astype(str)):
        raise ValueError(f"Unknown --user {args.user}: not in {args.train_history}")
    model = build_final_model(music, train, args.seed, _cf_kwargs(args), args.clusters)
    recs = model.recommend_user(str(args.user), args.top_n)
    recs.to_csv(out / "personalized_top_final.csv", index=False, encoding="utf-8-sig")
    print(recs.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 3 final model")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--music", default="Music Info.csv")
    shared.add_argument("--train-history", default="shared_train_history.csv")
    shared.add_argument("--factors", type=int, default=None)
    shared.add_argument("--epochs", type=int, default=None)
    shared.add_argument("--clusters", type=int, default=N_CLUSTERS)
    shared.add_argument("--seed", type=int, default=42)
    shared.add_argument("--output", default="hybrid_output/final")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", parents=[shared], help="Measure the model on TEST")
    evaluate.add_argument("--test-history", default="shared_test_history.csv")
    evaluate.add_argument("--mock-cold-fraction", type=float, default=0.10)
    evaluate.add_argument("--max-users", type=int, default=2000)
    evaluate.set_defaults(run=_evaluate_command)
    recommend = commands.add_parser("recommend", parents=[shared], help="Recommend for one user using all of TRAIN")
    recommend.add_argument("--user", required=True)
    recommend.add_argument("--top-n", type=int, default=10)
    recommend.set_defaults(run=_recommend_command)

    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    train = load_history(args.train_history, set(music["track_id"].dropna()), user_fraction=1.0)
    args.run(args, music, train, out)


if __name__ == "__main__":
    main()
