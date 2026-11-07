"""Spotify 80/20 Playlist Continuation Validation (Stage 3 External Validation).

Validates the MSD-selected Final Model (final_model.py) without re-tuning on an
external dataset (data/spotify_songs.csv linked with data/Music Info.csv), comparing 4 models:
  - CB     : Stage 1 Content-Based (audio features)
  - MF     : Stage 2 Collaborative Filtering (ALS fold-in on overlapping Warm songs)
  - Hybrid : Stage 3 Proposal formula, alpha(i) = 0.25 + 0.75 / (1 + n_i / 25)
  - Final  : Stage 3 Final Artist Hybrid (beta = 0.9, gamma = 5.0, artist_cap = 5)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from collaborative import CollaborativeRecommender
from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender, load_history
from final_model import FINAL_BETA, FINAL_CAP, FINAL_GAMMA, _ListenerCounts, cap_top_k
from hybrid import (
    _scale_mf,
    fold_in_raw_scores,
    load_spotify,
    ranking_metrics,
    top_k_indices,
)

MODELS = ("CB", "MF", "Hybrid", "Final")


def make_norm_key(artist: pd.Series, name: pd.Series) -> pd.Series:
    """Create normalized key: artist.str.lower().str.strip() + ' - ' + name.str.lower().str.strip()."""
    a = artist.fillna("").astype(str).str.lower().str.strip()
    n = name.fillna("").astype(str).str.lower().str.strip()
    valid = (a != "") & (n != "") & (a != "nan") & (n != "nan")
    out = pd.Series(None, index=artist.index, dtype=object)
    out.loc[valid] = a.loc[valid] + " - " + n.loc[valid]
    return out


def link_catalogs(spotify: pd.DataFrame, music: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map overlapping Spotify tracks to data/Music Info.csv track_ids; keep Spotify IDs for Cold-Start tracks."""
    sp = spotify.copy()
    msd = music.dropna(subset=["track_id"]).sort_values("track_id").copy()

    sp["norm_key"] = make_norm_key(sp["artist"], sp["name"])
    msd["norm_key"] = make_norm_key(msd["artist"], msd["name"])

    key_to_msd: dict[str, str] = {}
    for tid, key in zip(msd["track_id"], msd["norm_key"]):
        if key is not None and not pd.isna(key):
            key_to_msd.setdefault(str(key), str(tid))

    linked = {k: key_to_msd[k] for k in sp["norm_key"].dropna().unique() if k in key_to_msd}
    sp["unified_id"] = sp["norm_key"].map(key_to_msd).fillna(sp["track_id"].astype(str))
    return sp, pd.Series(linked, dtype=object)


def split_playlists_80_20(
    spotify: pd.DataFrame,
    key_col: str = "norm_key",
    train_ratio: float = 0.80,
    min_tracks: int = 10,
    seed: int = 42,
) -> list[tuple[str, list[str], list[str]]]:
    """Split each playlist with >= min_tracks unique songs into 80% input history and 20% hidden target."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    usable = spotify.dropna(subset=[key_col])
    splits: list[tuple[str, list[str], list[str]]] = []
    for playlist_id, group_keys in usable.groupby("playlist_id", sort=True)[key_col]:
        unique_keys = np.unique(group_keys.to_numpy(dtype=str))
        if len(unique_keys) < min_tracks:
            continue
        shuffled = rng.permutation(unique_keys).tolist()
        n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))
        splits.append((str(playlist_id), shuffled[:n_train], shuffled[n_train:]))
    return splits


def evaluate_spotify_80_20(
    cb: ContentBasedRecommender | None,
    cf: CollaborativeRecommender,
    spotify: pd.DataFrame,
    music: pd.DataFrame,
    train_ratio: float = 0.80,
    min_tracks: int = 10,
    k_values: tuple[int, ...] = (5, 10, 20),
    seed: int = 42,
) -> dict:
    """Evaluate CB, MF, Hybrid (proposal formula), and Final (final_model.py) on Spotify playlists."""
    sp_linked, link = link_catalogs(spotify, music)
    splits = split_playlists_80_20(
        sp_linked, key_col="norm_key", train_ratio=train_ratio, min_tracks=min_tracks, seed=seed
    )
    if not splits:
        raise ValueError("No playlist has enough songs to evaluate")

    catalogue = (
        sp_linked.dropna(subset=["norm_key"])
        .sort_values("track_id")
        .drop_duplicates("norm_key")
        .reset_index(drop=True)
    )
    keys = catalogue["norm_key"].to_numpy(dtype=str)
    position = {k: i for i, k in enumerate(keys)}
    msd_ids = [link.get(k) for k in keys]

    # Fit audio features on candidate pool (avoids playlist_genre leakage)
    cb_sp = ContentBasedRecommender(n_clusters=min(6, len(catalogue)), random_state=seed).fit(catalogue)
    audio_only = cb_sp.track_features.copy()
    audio_only[:, len(AUDIO_FEATURE_COLS):] = 0.0

    unified_ids = np.array(
        [msd_ids[i] if msd_ids[i] is not None else str(catalogue.iloc[i]["track_id"]) for i in range(len(keys))],
        dtype=str,
    )
    is_msd_warm = np.array([t is not None for t in msd_ids], dtype=bool)
    mf_index = np.array([cf.i_map.get(t, -1) if t is not None else -1 for t in msd_ids], dtype=int)
    is_mf_warm = mf_index >= 0

    n_listeners = cf.get_item_listener_counts([t if t is not None else "" for t in msd_ids])
    alpha_dynamic = cf.compute_dynamic_alpha([t if t is not None else "" for t in msd_ids])

    # Precompute artist codes and donor statistics for Final Model (beta=0.9, gamma=5.0, cap=5)
    artist_names = catalogue["artist"].fillna("").astype(str).str.strip().str.lower().to_numpy(dtype=str)
    artist_codes, _ = pd.factorize(artist_names)
    n_artists = int(artist_codes.max()) + 1 if len(artist_codes) else 0
    known_artist = artist_codes >= 0

    donors = is_mf_warm & known_artist
    n_donor_songs = np.bincount(artist_codes[donors], minlength=n_artists)
    donor_listeners = np.bincount(artist_codes[donors], weights=n_listeners[donors], minlength=n_artists)
    has_donor = np.zeros(len(artist_codes), dtype=bool)
    has_donor[known_artist] = n_donor_songs[artist_codes[known_artist]] > 0
    estimated_mask = (~is_mf_warm) & has_donor

    n_effective = n_listeners.astype(float)
    est_codes = artist_codes[estimated_mask]
    n_effective[estimated_mask] = FINAL_GAMMA * donor_listeners[est_codes] / n_donor_songs[est_codes]
    alpha_final = CollaborativeRecommender.compute_dynamic_alpha(_ListenerCounts(n_effective), unified_ids)

    report_ks = sorted(k_values)
    max_k = max(report_ks + [10])
    store = {
        m: {
            k: {
                "hit_rate": [],
                "recall": [],
                "precision": [],
                "ndcg": [],
                "warm_recall": [],
                "cold_recall": [],
                "cold_share": [],
            }
            for k in report_ks
        }
        for m in MODELS
    }

    n_playlists_msd_input = 0
    n_playlists_mf_input = 0
    total_hidden = 0
    total_hidden_msd_warm = 0
    total_hidden_mf_warm = 0
    total_hidden_known_artist = 0

    for _, query_keys, hidden_keys in splits:
        q_idx = np.array([position[k] for k in query_keys], dtype=int)
        relevant = {k for k in hidden_keys if k in position}
        relevant_idx = np.array([position[k] for k in relevant], dtype=int)
        rel_warm = {k for k in relevant if is_mf_warm[position[k]]}
        rel_cold = relevant - rel_warm

        seen = np.zeros(len(keys), dtype=bool)
        seen[q_idx] = True

        # 1. CB score (12 audio features, playcount = 1)
        s_cb = cb_sp._cosine_01(audio_only[q_idx].mean(axis=0), audio_only)

        # 2. MF score (ALS fold-in on overlapping Warm songs)
        q_msd = [msd_ids[i] for i in q_idx if msd_ids[i] is not None]
        n_playlists_msd_input += len(q_msd) > 0
        raw_mf = fold_in_raw_scores(cf, q_msd)
        n_playlists_mf_input += raw_mf is not None
        s_mf = _scale_mf(raw_mf, mf_index, seen)

        # 3. Hybrid score (Proposal formula: dynamic alpha)
        s_hybrid = s_cb if raw_mf is None else alpha_dynamic * s_cb + (1.0 - alpha_dynamic) * s_mf

        # 4. Final Model score (final_model.py: beta=0.9, gamma=5.0, cap=5)
        q_artists = artist_codes[q_idx]
        q_artists = q_artists[q_artists >= 0]
        artist_affinity = np.zeros(len(keys), dtype=float)
        if len(q_artists) > 0:
            counts = np.bincount(q_artists, minlength=n_artists).astype(float)
            max_c = counts.max()
            if max_c > 0:
                artist_affinity[known_artist] = counts[artist_codes[known_artist]] / max_c

        total_hidden += len(relevant_idx)
        total_hidden_msd_warm += int(is_msd_warm[relevant_idx].sum())
        total_hidden_mf_warm += int(is_mf_warm[relevant_idx].sum())
        total_hidden_known_artist += int((artist_affinity[relevant_idx] > 0).sum())

        s_cb_artist = (1.0 - FINAL_BETA) * s_cb + FINAL_BETA * artist_affinity
        s_mf_est = s_mf.copy()
        if raw_mf is not None and donors.any():
            totals = np.bincount(artist_codes[donors], weights=s_mf[donors], minlength=n_artists)
            s_mf_est[estimated_mask] = totals[est_codes] / n_donor_songs[est_codes]
            s_final = alpha_final * s_cb_artist + (1.0 - alpha_final) * s_mf_est
        else:
            s_final = s_cb_artist

        for m_name, sc_vec in (("CB", s_cb), ("MF", s_mf), ("Hybrid", s_hybrid)):
            idx = top_k_indices(sc_vec, keys, seen, max_k)
            ranked_keys = keys[idx].tolist()
            for k in report_ks:
                head_idx = idx[:k]
                head_set = set(ranked_keys[:k])
                m = ranking_metrics(ranked_keys, relevant, k)
                box = store[m_name][k]
                box["hit_rate"].append(1.0 if (head_set & relevant) else 0.0)
                box["recall"].append(m["recall"])
                box["precision"].append(m["precision"])
                box["ndcg"].append(m["ndcg"])
                box["cold_share"].append(float((~is_mf_warm[head_idx]).mean()))
                if rel_warm:
                    box["warm_recall"].append(len(head_set & rel_warm) / len(rel_warm))
                if rel_cold:
                    box["cold_recall"].append(len(head_set & rel_cold) / len(rel_cold))

        for k in report_ks:
            cap = max(1, math.ceil(k * FINAL_CAP / 10))
            head_idx = cap_top_k(s_final, keys, seen, artist_codes, cap, k)
            ranked_keys = keys[head_idx].tolist()
            head_set = set(ranked_keys)
            m = ranking_metrics(ranked_keys, relevant, k)
            box = store["Final"][k]
            box["hit_rate"].append(1.0 if (head_set & relevant) else 0.0)
            box["recall"].append(m["recall"])
            box["precision"].append(m["precision"])
            box["ndcg"].append(m["ndcg"])
            box["cold_share"].append(float((~is_mf_warm[head_idx]).mean()))
            if rel_warm:
                box["warm_recall"].append(len(head_set & rel_warm) / len(rel_warm))
            if rel_cold:
                box["cold_recall"].append(len(head_set & rel_cold) / len(rel_cold))

    models_out = {}
    for m_name in MODELS:
        models_out[m_name] = {}
        for k in report_ks:
            box = store[m_name][k]
            models_out[m_name][f"@{k}"] = {
                "hit_rate": float(np.mean(box["hit_rate"])),
                "recall": float(np.mean(box["recall"])),
                "precision": float(np.mean(box["precision"])),
                "ndcg": float(np.mean(box["ndcg"])),
                "warm_recall": float(np.mean(box["warm_recall"])) if box["warm_recall"] else 0.0,
                "cold_recall": float(np.mean(box["cold_recall"])) if box["cold_recall"] else 0.0,
                "cold_share": float(np.mean(box["cold_share"])),
            }

    return {
        "train_ratio": train_ratio,
        "min_tracks": min_tracks,
        "n_playlists_evaluated": len(splits),
        "n_candidate_songs": len(keys),
        "n_msd_warm_songs": int(is_msd_warm.sum()),
        "n_mf_warm_songs": int(is_mf_warm.sum()),
        "n_cold_start_songs": int((~is_msd_warm).sum()),
        "msd_warm_rate": float(is_msd_warm.mean()),
        "mf_warm_rate": float(is_mf_warm.mean()),
        "playlists_with_msd_input_rate": n_playlists_msd_input / len(splits),
        "playlists_with_mf_input_rate": n_playlists_mf_input / len(splits),
        "hidden_targets_total": total_hidden,
        "hidden_targets_msd_warm_share": total_hidden_msd_warm / total_hidden if total_hidden else 0.0,
        "hidden_targets_mf_warm_share": total_hidden_mf_warm / total_hidden if total_hidden else 0.0,
        "hidden_targets_known_artist_share": total_hidden_known_artist / total_hidden if total_hidden else 0.0,
        "models": models_out,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate CB, MF, Hybrid, and Final on Spotify 80/20 split")
    parser.add_argument("--spotify", default="data/spotify_songs.csv.zip", help="Path to spotify_songs.csv")
    parser.add_argument("--music", default="data/Music Info.csv", help="Path to Music Info.csv")
    parser.add_argument("--train-history", default="outputs/content_based/demo_train_history.csv")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--min-tracks", type=int, default=10)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/hybrid/spotify_80_20")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    spotify = load_spotify(args.spotify)
    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    valid_ids = set(music["track_id"].dropna())
    train = load_history(args.train_history, valid_ids, user_fraction=1.0)

    cb = None
    if set(AUDIO_FEATURE_COLS).issubset(music.columns):
        cb = ContentBasedRecommender(n_clusters=6, random_state=args.seed).fit(music, train)

    cf = CollaborativeRecommender(
        method="als", n_factors=args.factors, n_epochs=args.epochs, random_state=args.seed
    ).fit(train, music)

    report = evaluate_spotify_80_20(
        cb=cb,
        cf=cf,
        spotify=spotify,
        music=music,
        train_ratio=args.train_ratio,
        min_tracks=args.min_tracks,
        seed=args.seed,
    )
    report.update({
        "spotify_file": str(args.spotify),
        "music_file": str(args.music),
        "train_file": str(args.train_history),
        "seed": args.seed,
    })

    out_file = out / "spotify_80_20_evaluation_metrics.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to {out_file}")


if __name__ == "__main__":
    main()
