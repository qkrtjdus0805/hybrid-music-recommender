"""Case Study: Our Final Model vs. Spotify vs. Tidal in 12-Dimensional Audio Space.

Feeds the 5 seed songs:
  1. Michael Jackson - Beat It (TRDDCSY128F92EA721)
  2. Nirvana - Come as You Are (TROUVHL128F426C441)
  3. Amy Winehouse - Back to Black (TRRJQES128EF35675B)
  4. Miles Davis - Blue in Green (TRPCNCR128F930958E)
  5. Led Zeppelin - Stairway to Heaven (TRUOEWJ128F9345A85)
into the Stage 3 Final Artist Hybrid Model (beta=0.9, gamma=5.0, artist_cap=5) and
compares its Top-10 output against Spotify and Tidal Top-10 lists in 12-Dimensional
Audio Space using:
  - Profile Similarity (Cohesion): mean cosine similarity to the 12D seed centroid
  - Intra-List Diversity (Serendipity): 1 - mean pairwise cosine similarity among Top-10
  - 12-Dimensional Audio Space Radar Chart (case_study_radar.png)

Run:
  python -m experiments.case_study.case_study_experiment \
      --spotify-recs data/case_study/spotify_recommended_features.csv \
      --tidal-recs data/case_study/tidal_recommended_features.csv \
      --music "data/Music Info.csv" \
      --train-history outputs/content_based/demo_train_history.csv \
      --output outputs/hybrid/case_study
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collaborative import CollaborativeRecommender
from content_based import AUDIO_FEATURE_COLS, ContentBasedRecommender, load_history
from final_model import FINAL_BETA, FINAL_CAP, FINAL_GAMMA, _ListenerCounts, cap_top_k
from hybrid import _scale_mf, fold_in_raw_scores

# The 5 seed songs fed into Our Final Model, Spotify, and Tidal
DEFAULT_SEED_TRACK_IDS = (
    "TRDDCSY128F92EA721",  # Michael Jackson - Beat It
    "TROUVHL128F426C441",  # Nirvana - Come as You Are
    "TRRJQES128EF35675B",  # Amy Winehouse - Back to Black
    "TRPCNCR128F930958E",  # Miles Davis - Blue in Green
    "TRUOEWJ128F9345A85",  # Led Zeppelin - Stairway to Heaven
)


def make_norm_key(artist: pd.Series, name: pd.Series) -> pd.Series:
    a = artist.fillna("").astype(str).str.lower().str.strip()
    n = name.fillna("").astype(str).str.lower().str.strip()
    return a + " - " + n


def resolve_seed_tracks(seeds_df: pd.DataFrame | None, music_df: pd.DataFrame) -> pd.DataFrame:
    """Match seed rows to data/Music Info.csv by track_id or normalized 'artist - name' key."""
    msd = music_df.dropna(subset=["track_id"]).drop_duplicates("track_id").copy()
    msd["norm_key"] = make_norm_key(msd["artist"], msd["name"])
    tid_map = msd.set_index("track_id")
    key_map = msd.drop_duplicates("norm_key").set_index("norm_key")

    if seeds_df is None:
        seeds_df = pd.DataFrame({"track_id": list(DEFAULT_SEED_TRACK_IDS)})

    matched_rows = []
    for _, row in seeds_df.iterrows():
        tid = str(row.get("track_id", "")).strip()
        if tid and tid in tid_map.index:
            r = tid_map.loc[tid].copy()
            r["track_id"] = tid
            matched_rows.append(r)
            continue
        if "artist" in row and "name" in row:
            k = f"{str(row['artist']).lower().strip()} - {str(row['name']).lower().strip()}"
            if k in key_map.index:
                r = key_map.loc[k].copy()
                r["track_id"] = key_map.loc[k, "track_id"]
                matched_rows.append(r)
                continue
        if set(AUDIO_FEATURE_COLS).issubset(seeds_df.columns):
            matched_rows.append(row)
        else:
            raise ValueError(f"Could not resolve seed song in data/Music Info.csv: {row.to_dict()}")
    return pd.DataFrame(matched_rows).reset_index(drop=True)


def recommend_final_from_seeds(
    cb: ContentBasedRecommender,
    cf: CollaborativeRecommender,
    seeds_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Run Stage 3 Final Artist Hybrid (beta=0.9, gamma=5.0, cap=5) for a seed song list."""
    catalogue = cb.music_info.reset_index(drop=True)
    tids = catalogue["track_id"].astype(str).to_numpy(dtype=str)
    tid_to_idx = {t: i for i, t in enumerate(tids)}

    audio_only = cb.track_features[:, :len(AUDIO_FEATURE_COLS)]
    seed_audio = cb.scaler.transform(seeds_df[AUDIO_FEATURE_COLS].to_numpy(dtype=float))
    seed_centroid = seed_audio.mean(axis=0)

    # 1. Content-Based audio similarity S_CB in [0, 1]
    s_cb = cb._cosine_01(seed_centroid, audio_only)

    # 2. ArtistAffinity from seed songs (playcount = 1 -> uniform weights)
    cat_artists = catalogue["artist"].fillna("").astype(str).str.strip().str.lower().to_numpy(dtype=str)
    artist_codes, unique_artists = pd.factorize(cat_artists)
    art_to_code = {a: i for i, a in enumerate(unique_artists)}
    n_artists = len(unique_artists)
    known_artist = artist_codes >= 0

    seed_artists = seeds_df["artist"].fillna("").astype(str).str.strip().str.lower().tolist()
    seed_codes = [art_to_code[a] for a in seed_artists if a in art_to_code]
    artist_affinity = np.zeros(len(tids), dtype=float)
    if seed_codes:
        counts = np.bincount(seed_codes, minlength=n_artists).astype(float)
        if counts.max() > 0:
            artist_affinity[known_artist] = counts[artist_codes[known_artist]] / counts.max()

    s_cb_artist = (1.0 - FINAL_BETA) * s_cb + FINAL_BETA * artist_affinity

    # 3. ALS Fold-in + ArtistMF estimation (gamma = 5.0)
    seed_tids = [str(t) for t in seeds_df.get("track_id", []) if str(t) in tid_to_idx]
    seen = np.zeros(len(tids), dtype=bool)
    for t in seed_tids:
        seen[tid_to_idx[t]] = True

    mf_index = np.array([cf.i_map.get(t, -1) for t in tids], dtype=int)
    is_mf_warm = mf_index >= 0
    n_listeners = cf.get_item_listener_counts(tids)

    donors = is_mf_warm & known_artist
    n_donor_songs = np.bincount(artist_codes[donors], minlength=n_artists)
    donor_listeners = np.bincount(artist_codes[donors], weights=n_listeners[donors], minlength=n_artists)
    has_donor = np.zeros(len(artist_codes), dtype=bool)
    has_donor[known_artist] = n_donor_songs[artist_codes[known_artist]] > 0
    estimated_mask = (~is_mf_warm) & has_donor
    est_codes = artist_codes[estimated_mask]

    n_effective = n_listeners.astype(float)
    n_effective[estimated_mask] = FINAL_GAMMA * donor_listeners[est_codes] / n_donor_songs[est_codes]
    alpha_final = CollaborativeRecommender.compute_dynamic_alpha(_ListenerCounts(n_effective), tids)

    raw_mf = fold_in_raw_scores(cf, seed_tids)
    s_mf = _scale_mf(raw_mf, mf_index, seen)
    s_mf_est = s_mf.copy()
    if raw_mf is not None and donors.any():
        totals = np.bincount(artist_codes[donors], weights=s_mf[donors], minlength=n_artists)
        s_mf_est[estimated_mask] = totals[est_codes] / n_donor_songs[est_codes]
        s_final = alpha_final * s_cb_artist + (1.0 - alpha_final) * s_mf_est
    else:
        s_final = s_cb_artist

    cap = max(1, math.ceil(top_n * FINAL_CAP / 10))
    top_idx = cap_top_k(s_final, tids, seen, artist_codes, cap, top_n)

    out = catalogue.iloc[top_idx].copy().reset_index(drop=True)
    out["n_listeners"] = n_listeners[top_idx]
    out["S_CB"] = s_cb[top_idx]
    out["ArtistAffinity"] = artist_affinity[top_idx]
    out["S_MF_est"] = s_mf_est[top_idx]
    out["S_Final"] = s_final[top_idx]
    return out


def prepare_external_recs(recs: pd.DataFrame, cb: ContentBasedRecommender, label: str) -> pd.DataFrame:
    """Fill a missing time_signature with the Music Info median (rule of hybrid.load_spotify);
    drop songs whose other audio features are unknown (e.g. not found in either dataset)."""
    recs = recs.copy()
    if "time_signature" not in recs.columns:
        recs["time_signature"] = np.nan
    recs["time_signature"] = recs["time_signature"].fillna(cb.feature_medians["time_signature"])
    known = recs[AUDIO_FEATURE_COLS].notna().all(axis=1)
    for _, row in recs.loc[~known].iterrows():
        print(f"[{label}] skipped (no audio features): {row.get('artist')} - {row.get('name')}")
    return recs.loc[known].reset_index(drop=True)


def compute_12d_metrics(
    cb: ContentBasedRecommender, seeds_df: pd.DataFrame, recs_df: pd.DataFrame
) -> dict:
    """Compute Profile Similarity (Cohesion) and Intra-List Diversity (Serendipity) in 12D audio space."""
    X_seed = cb.scaler.transform(seeds_df[AUDIO_FEATURE_COLS].to_numpy(dtype=float))
    X_rec = cb.scaler.transform(recs_df[AUDIO_FEATURE_COLS].to_numpy(dtype=float))
    seed_centroid = X_seed.mean(axis=0)

    sim_01 = ContentBasedRecommender._cosine_01(seed_centroid, X_rec)
    denom = np.linalg.norm(X_rec, axis=1) * np.linalg.norm(seed_centroid)
    sim_raw = np.divide(X_rec @ seed_centroid, denom, out=np.zeros(len(X_rec)), where=denom > 1e-12)

    norms = np.linalg.norm(X_rec, axis=1, keepdims=True)
    pair_raw = (X_rec @ X_rec.T) / np.maximum(norms @ norms.T, 1e-12)
    pair_01 = np.clip((pair_raw + 1.0) / 2.0, 0.0, 1.0)
    tri = np.triu_indices(len(X_rec), k=1)

    return {
        "profile_similarity_01": float(sim_01.mean()),
        "profile_similarity_raw": float(sim_raw.mean()),
        "intra_list_diversity_01": float(1.0 - pair_01[tri].mean()),
        "intra_list_diversity_raw": float(1.0 - pair_raw[tri].mean()),
        "mean_z_vector": {col: float(v) for col, v in zip(AUDIO_FEATURE_COLS, X_rec.mean(axis=0))},
        "mean_raw_vector": {col: float(recs_df[col].mean()) for col in AUDIO_FEATURE_COLS},
    }


def plot_radar_chart(
    cb: ContentBasedRecommender,
    seeds_df: pd.DataFrame,
    our_recs: pd.DataFrame,
    spotify_recs: pd.DataFrame,
    tidal_recs: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 12-Dimensional Audio Space Radar Chart (normalized across catalog 5th-95th percentiles)."""
    music_audio = cb.music_info[AUDIO_FEATURE_COLS]
    q05 = music_audio.quantile(0.05).to_numpy(dtype=float)
    q95 = music_audio.quantile(0.95).to_numpy(dtype=float)
    span = np.where((q95 - q05) > 1e-9, q95 - q05, 1.0)

    def to_unit(df: pd.DataFrame) -> np.ndarray:
        vals = df[AUDIO_FEATURE_COLS].mean().to_numpy(dtype=float)
        return np.clip((vals - q05) / span, 0.0, 1.0)

    angles = np.linspace(0, 2 * np.pi, len(AUDIO_FEATURE_COLS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True}, dpi=180)
    series = [
        ("Seed Songs (5 Input Tracks)", to_unit(seeds_df), "#222222", "--", 2.2, 0.06),
        ("Our Final Model (Top-10)", to_unit(our_recs), "#D95F02", "-", 2.6, 0.18),
        ("Spotify (Top-10)", to_unit(spotify_recs), "#1B9E77", "-", 2.2, 0.12),
        ("Tidal (Top-10)", to_unit(tidal_recs), "#7570B3", "-", 2.2, 0.12),
    ]
    for label, vec, color, ls, lw, fill_alpha in series:
        v = vec.tolist() + [float(vec[0])]
        ax.plot(angles, v, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=5, label=label)
        ax.fill(angles, v, color=color, alpha=fill_alpha)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AUDIO_FEATURE_COLS, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%", "50%", "75%", "95%+"], fontsize=8, color="#555555")
    ax.set_title(
        "12-Dimensional Audio Space Comparison\n(Seed Profile vs. Our Final Model vs. Spotify vs. Tidal)",
        fontsize=13,
        fontweight="bold",
        pad=24,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="12D Audio Space Case Study: Final Model vs. Spotify vs. Tidal")
    parser.add_argument("--seeds", default=None, help="Optional CSV with seed songs (defaults to the 5 seed songs)")
    parser.add_argument("--spotify-recs", default="data/case_study/spotify_recommended_features.csv")
    parser.add_argument("--tidal-recs", default="data/case_study/tidal_recommended_features.csv")
    parser.add_argument("--music", default="data/Music Info.csv")
    parser.add_argument("--train-history", default="outputs/content_based/demo_train_history.csv")
    parser.add_argument("--output", default="outputs/hybrid/case_study")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    music = pd.read_csv(args.music, encoding="utf-8-sig", dtype={"track_id": str})
    train = load_history(args.train_history, set(music["track_id"].dropna()), user_fraction=1.0)
    cb = ContentBasedRecommender(n_clusters=6, random_state=42).fit(music, train)
    cf = CollaborativeRecommender(method="als", n_factors=64, n_epochs=15, random_state=42).fit(train, music)

    raw_seeds = pd.read_csv(args.seeds, encoding="utf-8-sig") if args.seeds else None
    seeds_df = resolve_seed_tracks(raw_seeds, music)
    spotify_recs = prepare_external_recs(pd.read_csv(args.spotify_recs, encoding="utf-8-sig"), cb, "Spotify")
    tidal_recs = prepare_external_recs(pd.read_csv(args.tidal_recs, encoding="utf-8-sig"), cb, "Tidal")

    our_recs = recommend_final_from_seeds(cb, cf, seeds_df, top_n=10)
    our_recs.to_csv(out / "our_final_model_top10.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "seed_tracks": seeds_df[["track_id", "name", "artist", "year"] + AUDIO_FEATURE_COLS].to_dict(orient="records"),
        "seed_mean_raw_vector": {col: float(seeds_df[col].mean()) for col in AUDIO_FEATURE_COLS},
        "Our_Final_Model": compute_12d_metrics(cb, seeds_df, our_recs),
        "Spotify": compute_12d_metrics(cb, seeds_df, spotify_recs),
        "Tidal": compute_12d_metrics(cb, seeds_df, tidal_recs),
        "n_songs_used": {"Our_Final_Model": len(our_recs), "Spotify": len(spotify_recs), "Tidal": len(tidal_recs)},
    }
    (out / "case_study_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_radar_chart(cb, seeds_df, our_recs, spotify_recs, tidal_recs, out / "case_study_radar.png")
    print(f"Saved case study outputs to {out}")


if __name__ == "__main__":
    main()