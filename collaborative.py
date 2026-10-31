"""Stage 2 — Collaborative Filtering (matrix factorization on log(1 + playcount)).

  2-1. r_ui = log(1 + playcount)
  2-2. Implicit ALS (default) or Biased SVD via SGD
  2-3. S_MF(u, i) in [0, 1] via per-user min-max; unseen (cold-start) songs get 0.0
  3-2. Evaluation: Precision@K, Recall@K, NDCG@K (+ RMSE for sgd)

Data loading and the train/test split are shared with Stage 1 (content_based.py).
Unit tests: test_collaborative.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from content_based import HISTORY_COLS, load_history, split_history_by_user

try:
    from numba import njit
except ImportError:  # numba is optional; SGD is just slower without it
    def njit(func):
        return func


@njit
def _sgd_epoch(u_idx, i_idx, ratings, order, mu, b_u, b_i, P, Q, lr, reg):
    sq_err = 0.0
    reg_pq = max(0.15, 3.0 * reg)
    lr_pq = 0.25 * lr
    for n in order:
        u, i = u_idx[n], i_idx[n]
        err = ratings[n] - (mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i]))
        sq_err += err * err
        b_u[u] += lr * (err - reg * b_u[u])
        b_i[i] += lr * (err - reg * b_i[i])
        p_u = P[u].copy()
        P[u] += lr_pq * (err * Q[i] - reg_pq * P[u])
        Q[i] += lr_pq * (err * p_u - reg_pq * Q[i])
    return sq_err


class CollaborativeRecommender:
    def __init__(self, method: str = "als", n_factors: int = 32, n_epochs: int = 12,
                 reg_lambda: float = 0.05, lr: float = 0.015, alpha_conf: float = 1.5,
                 random_state: int = 42):
        if method not in ("als", "sgd"):
            raise ValueError("method must be 'als' or 'sgd'")
        self.method, self.n_factors, self.n_epochs = method, n_factors, n_epochs
        self.reg, self.lr, self.alpha_conf, self.seed = reg_lambda, lr, alpha_conf, random_state
        self.music_info: pd.DataFrame | None = None
        self.mu = 0.0
        self.u_map: dict[str, int] = {}
        self.i_map: dict[str, int] = {}
        self.user_listened_tracks: dict[str, set[str]] = {}
        self.item_listener_counts: dict[str, int] = {}

    # ---------------- training ----------------
    def fit(self, train_history: pd.DataFrame, music_info_df: pd.DataFrame | None = None):
        """Train on TRAIN history only."""
        if music_info_df is not None:
            music = music_info_df.dropna(subset=["track_id"]).copy()
            music["track_id"] = music["track_id"].astype(str)
            self.music_info = music.drop_duplicates("track_id").reset_index(drop=True)

        hist = train_history[HISTORY_COLS].dropna().copy()
        hist[["user_id", "track_id"]] = hist[["user_id", "track_id"]].astype(str)
        hist["playcount"] = pd.to_numeric(hist["playcount"], errors="coerce")
        hist = hist.loc[hist["playcount"] > 0]
        if self.music_info is not None:
            hist = hist.loc[hist["track_id"].isin(set(self.music_info["track_id"]))]
        if hist.empty:
            raise ValueError("No valid training interactions")
        hist = hist.groupby(["user_id", "track_id"], as_index=False)["playcount"].sum()

        users, u_idx = np.unique(hist["user_id"], return_inverse=True)
        items, i_idx = np.unique(hist["track_id"], return_inverse=True)
        self.u_map = {u: k for k, u in enumerate(users)}
        self.i_map = {t: k for k, t in enumerate(items)}
        self.users, self.items = users, items
        ratings = np.log1p(hist["playcount"].to_numpy(dtype=np.float64))
        self.user_listened_tracks = hist.groupby("user_id")["track_id"].agg(set).to_dict()
        counts = np.bincount(i_idx, minlength=len(items))
        self.item_listener_counts = dict(zip(items.tolist(), counts.tolist()))  # n_i for Stage 3

        rng = np.random.default_rng(self.seed)
        self.b_u, self.b_i = np.zeros(len(users)), np.zeros(len(items))
        self.P = rng.normal(0, 0.1, (len(users), self.n_factors))
        self.Q = rng.normal(0, 0.1, (len(items), self.n_factors))
        if self.method == "sgd":
            self._fit_sgd(u_idx, i_idx, ratings, rng)
        else:
            self._fit_als(u_idx, i_idx, ratings)
        return self

    def _fit_sgd(self, u_idx, i_idx, ratings, rng):
        self.mu = float(ratings.mean())
        n_u, n_i = len(self.users), len(self.items)
        u_deg = np.bincount(u_idx, minlength=n_u)
        i_deg = np.bincount(i_idx, minlength=n_i)
        self.b_u = np.bincount(u_idx, weights=ratings - self.mu, minlength=n_u) / (u_deg + 10.0)
        self.b_i = np.bincount(i_idx, weights=ratings - self.mu - self.b_u[u_idx], minlength=n_i) / (i_deg + 15.0)

        k_svd = min(self.n_factors, min(n_u, n_i) - 1)
        if k_svd >= 2:
            R = csr_matrix((ratings, (u_idx, i_idx)), shape=(n_u, n_i))
            row_norm = np.sqrt(np.asarray(R.power(2).sum(axis=1)).ravel()) + 1e-9
            u_s, s_s, vt_s = svds(R.multiply(1.0 / row_norm[:, None]), k=k_svd, random_state=self.seed)
            self.P[:, :k_svd] = u_s * (s_s ** 0.7)
            self.Q[:, :k_svd] = vt_s.T * (s_s ** 0.3)

        for epoch in range(1, self.n_epochs + 1):
            order = rng.permutation(len(ratings))
            lr_t = self.lr / (1.0 + 0.25 * (epoch - 1))
            sq = _sgd_epoch(u_idx, i_idx, ratings, order, self.mu, self.b_u, self.b_i,
                            self.P, self.Q, lr_t, self.reg)
            print(f"  epoch {epoch}: train RMSE={np.sqrt(sq / len(ratings)):.4f}")

    def _fit_als(self, u_idx, i_idx, ratings):
        """Implicit ALS (Hu et al. 2008): confidence c = 1 + alpha * r, preference = 1."""
        C = csr_matrix((self.alpha_conf * ratings, (u_idx, i_idx)),
                       shape=(len(self.users), len(self.items)))
        for epoch in range(1, self.n_epochs + 1):
            self.P = self._als_step(C, self.Q)
            self.Q = self._als_step(C.T.tocsr(), self.P)
            pred = np.sum(self.P[u_idx] * self.Q[i_idx], axis=1)
            print(f"  epoch {epoch}: preference RMSE={np.sqrt(np.mean((1 - pred) ** 2)):.4f}")

    def _als_step(self, C, fixed):
        """Solve every row's factors given the other side's fixed factors."""
        k = self.n_factors
        reg = self.reg * 250.0 if self.reg < 1.0 else self.reg
        base = fixed.T @ fixed + reg * np.eye(k)
        out = np.zeros((C.shape[0], k))
        for r in range(C.shape[0]):
            s, e = C.indptr[r], C.indptr[r + 1]
            if s == e:
                continue
            F, w = fixed[C.indices[s:e]], C.data[s:e]
            out[r] = np.linalg.solve(base + (F.T * w) @ F, F.T @ (w + 1.0))
        return out

    # ---------------- scoring ----------------
    def _raw_scores(self, u: int) -> np.ndarray:
        return self.Q @ self.P[u]

    def score_user_items(self, user_id: str, candidate_track_ids=None) -> np.ndarray:
        """Stage 3 API: S_MF(u, i) in [0, 1] per candidate; 0.0 for songs unseen in TRAIN."""
        u = self.u_map.get(str(user_id))
        if u is None:
            raise KeyError(f"Unknown user: {user_id}. Check TRAIN history.")
        if candidate_track_ids is None:
            candidate_track_ids = (self.music_info["track_id"] if self.music_info is not None
                                   else self.items)
        raw = self._raw_scores(u)
        seen = [self.i_map[t] for t in self.user_listened_tracks.get(str(user_id), ()) if t in self.i_map]
        if 0 < len(seen) < len(raw):
            unseen = np.delete(raw, seen)
            lo, span = unseen.min(), np.ptp(unseen)
        else:
            lo, span = raw.min(), np.ptp(raw)
        scaled = np.clip((raw - lo) / span, 0.0, 1.0) if span > 1e-9 else np.zeros_like(raw)
        idx = np.array([self.i_map.get(str(t), -1) for t in candidate_track_ids])
        return np.where(idx >= 0, scaled[idx], 0.0)

    def get_item_listener_counts(self, candidate_track_ids) -> np.ndarray:
        return np.array([self.item_listener_counts.get(str(t), 0) for t in candidate_track_ids])

    def compute_dynamic_alpha(self, candidate_track_ids, alpha_min: float = 0.25,
                              tau: float = 25.0) -> np.ndarray:
        """alpha(i) = alpha_min + (1 - alpha_min) / (1 + n_i / tau); weight for S_CB in Stage 3."""
        n_i = self.get_item_listener_counts(candidate_track_ids)
        return alpha_min + (1 - alpha_min) / (1 + n_i / tau)

    def recommend_user(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        """Top-N unheard tracks by S_MF."""
        cand = (self.music_info if self.music_info is not None
                else pd.DataFrame({"track_id": self.items}))
        ids = cand["track_id"].astype(str)
        keep = [c for c in ["track_id", "name", "artist", "genre"] if c in cand]
        out = cand[keep].copy()
        out["n_listeners"] = self.get_item_listener_counts(ids)
        out["S_MF"] = self.score_user_items(user_id, ids)
        out = out.loc[~ids.isin(self.user_listened_tracks.get(str(user_id), set()))]
        return out.nlargest(top_n, "S_MF").reset_index(drop=True)

    # ---------------- evaluation ----------------
    def evaluate(self, test_history: pd.DataFrame, k_values=(5, 10, 20),
                 max_users: int = 500) -> dict:
        """Precision/Recall/NDCG@K on TEST songs (TRAIN songs excluded); RMSE for sgd."""
        df = test_history[HISTORY_COLS].dropna().copy()
        df[["user_id", "track_id"]] = df[["user_id", "track_id"]].astype(str)
        df["playcount"] = pd.to_numeric(df["playcount"], errors="coerce")
        df = df.loc[(df["playcount"] > 0) & df["user_id"].isin(self.u_map)
                    & df["track_id"].isin(self.i_map)]
        if df.empty:
            raise ValueError("No overlapping users/items between TRAIN and TEST")
        result = {}
        if self.method == "sgd":  # ALS outputs preferences, not log playcounts
            u, i = df["user_id"].map(self.u_map).to_numpy(), df["track_id"].map(self.i_map).to_numpy()
            pred = self.mu + self.b_u[u] + self.b_i[i] + np.sum(self.P[u] * self.Q[i], axis=1)
            err = np.log1p(df["playcount"].to_numpy()) - pred
            result["rmse"], result["mae"] = float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))

        test_items = df.groupby("user_id")["track_id"].agg(set)
        users = test_items.index.to_numpy()
        if len(users) > max_users:
            users = np.random.default_rng(self.seed).choice(users, max_users, replace=False)
        ks = sorted(k_values)
        res = {k: {"precision": [], "recall": [], "ndcg": []} for k in ks}
        for uid in users:
            scores = self._raw_scores(self.u_map[uid]).copy()
            seen = [self.i_map[t] for t in self.user_listened_tracks.get(uid, ()) if t in self.i_map]
            scores[seen] = -np.inf
            ranked = np.argsort(-scores)[:ks[-1]]
            relevant = {self.i_map[t] for t in test_items[uid]}
            for k in ks:
                hits = np.array([r in relevant for r in ranked[:k]], dtype=float)
                dcg = np.sum(hits / np.log2(np.arange(2, len(hits) + 2)))
                idcg = np.sum(1 / np.log2(np.arange(2, min(k, len(relevant)) + 2)))
                res[k]["precision"].append(hits.sum() / k)
                res[k]["recall"].append(hits.sum() / len(relevant))
                res[k]["ndcg"].append(dcg / idcg)
        result["n_users"] = len(users)
        for k in ks:
            result[f"@{k}"] = {m: float(np.mean(v)) for m, v in res[k].items()}
        return result


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2: collaborative filtering")
    p.add_argument("--music", default="Music Info.csv")
    p.add_argument("--history", default=None, help="Raw history .csv/.zip; sampled and split 80/20 (demo)")
    p.add_argument("--train-history", default=None, help="Shared TRAIN csv from team")
    p.add_argument("--test-history", default=None, help="Shared TEST csv from team")
    p.add_argument("--user-fraction", type=float, default=0.005)
    p.add_argument("--method", choices=["als", "sgd"], default="als")
    p.add_argument("--factors", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.015, help="sgd only")
    p.add_argument("--user", default=None)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--output", default="collaborative_output")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    valid = set(music["track_id"].dropna())
    if args.train_history:
        train = load_history(args.train_history, valid, user_fraction=1.0)
        test = (load_history(args.test_history, valid, user_fraction=1.0)
                if args.test_history else None)
    elif args.history:
        history = load_history(args.history, valid, user_fraction=args.user_fraction)
        train, test = split_history_by_user(history, test_fraction=0.2)
        print(f"80/20 split: TRAIN={len(train):,}, TEST={len(test):,}")
    else:
        p.error("Provide --train-history or --history")

    print(f"Training {args.method.upper()} (k={args.factors}, epochs={args.epochs})...")
    model = CollaborativeRecommender(args.method, args.factors, args.epochs, args.reg, args.lr)
    model.fit(train, music)

    if test is not None:
        metrics = model.evaluate(test)
        print(json.dumps(metrics, indent=2))
        (out / "cf_evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))

    user = str(args.user) if args.user else str(model.users[0])
    recs = model.recommend_user(user, args.top_n)
    print(f"\nTop-{args.top_n} for user {user[:10]}...")
    print(recs[[c for c in ["name", "artist", "n_listeners", "S_MF"] if c in recs]].to_string(index=False))
    recs.to_csv(out / "personalized_top10_cf.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
