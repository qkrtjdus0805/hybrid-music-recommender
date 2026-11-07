"""Draw the report figures from saved experiment results. Nothing is retrained.

Run: python make_figures.py --results outputs/hybrid --output outputs/hybrid/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without opening a window
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

FIGURES = ("model_comparison.png", "bucket_recall.png", "top10_composition.png", "tradeoff_curve.png")
FINAL = "최종 모델"
FINAL_GAMMA = 5.0
CURVE_BETA = "0.9"

SURFACE, INK, INK_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, QUIET_BAR = "#e1e0d9", "#c3c2b7", "#c3c2b7"
SERIES = {"MF": "#2a78d6", "고정 α + 아티스트": "#eb6834", FINAL: "#1baf7a"}  # checked by the palette validator
HISTORY_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # few listeners to many; ordinal, light to dark


def _read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _flatten(model: dict) -> dict:
    """Turn a full evaluate_hybrid model block into the flat row the figures use."""
    buckets, exposure = model["bucket_recall"]["@20"], model["exposure"]["@10"]
    return {"ndcg10": model["overall"]["@10"]["ndcg"], "mock_recall20": buckets.get("mock_cold", 0.0),
            "recall20_1_4": buckets.get("1-4", 0.0), "recall20_5_24": buckets.get("5-24", 0.0),
            "recall20_25plus": buckets.get("25+", 0.0), "exposure10_cold": exposure["mock_cold"] + exposure["sample_zero"],
            "exposure10_1_4": exposure["1-4"], "exposure10_5_24": exposure["5-24"], "exposure10_25plus": exposure["25+"],
            "coverage10": model["catalog_coverage"]["@10"]}


def build_table(root: Path) -> dict[str, dict]:
    """Test-set rows for every model, in presentation order."""
    measured = _read(root, "artist_mf/artist_mf_comparison.json")["test"]
    keys = ["ndcg10", "mock_recall20", "recall20_1_4", "recall20_5_24", "recall20_25plus", "exposure10_cold",
            "exposure10_1_4", "exposure10_5_24", "exposure10_25plus", "coverage10"]
    pick = lambda row: {k: row[k] for k in keys}
    return {
        "MF": pick(measured["mf"]),
        "제안서 동적 α": _flatten(_read(root, "rank/hybrid_evaluation_metrics.json")["models"]["Hybrid"]),
        "고정 α + 아티스트": pick(measured["fixed_artist"]),
        "동적 α + 아티스트": pick(measured["dynamic_artist"]),
        FINAL: pick(_read(root, "final/final_evaluation_metrics.json")["final"]),
    }


def curve_points(root: Path) -> list[dict]:
    """Validation results across gamma at the final beta, merged from both experiment runs."""
    points = {}
    for name in ("artist_mf/artist_mf_comparison.json", "artist_mf_gamma/artist_mf_comparison.json"):
        for label, row in _read(root, name)["validation"].items():
            beta, gamma = label.split("|")
            if beta == f"beta={CURVE_BETA}":
                value = float(gamma.split("=")[1])
                points[value] = {"gamma": value, "ndcg10": row["ndcg10"], "mock_recall20": row["mock_recall20"],
                                 "top10_cold_share": row["top10_cold_share"]}
    return [points[g] for g in sorted(points)]


def font_stack() -> list[str]:
    """Korean-capable fonts present on this computer, in order of preference."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [name for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans") if name in installed]


def _style() -> None:
    plt.rcParams.update({
        "font.family": font_stack(), "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "font.size": 11, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
    })


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _model_comparison(table: dict, path: Path) -> None:
    names = list(table)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [("ndcg10", "전체 정확도 NDCG@10"), ("mock_recall20", "모의 신곡 Recall@20")]
    for ax, (key, title) in zip(axes, panels):
        values = [table[n][key] for n in names]
        colors = [SERIES[FINAL] if n == FINAL else QUIET_BAR for n in names]
        ax.barh(names, values, height=0.45, color=colors, edgecolor=SURFACE, linewidth=2)
        for y, value in enumerate(values):
            ax.text(value + max(values) * 0.02, y, f"{value:.3f}", va="center", color=INK, fontsize=10)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_xlim(0, max(values) * 1.18)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(length=0)
    axes[0].invert_yaxis()
    fig.suptitle("모델별 전체 정확도와 모의 신곡 Recall (테스트)", x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    _save(fig, path)


def _bucket_recall(table: dict, path: Path) -> None:
    groups = [("mock_recall20", "모의 신곡"), ("recall20_1_4", "청취자 1–4명"), ("recall20_5_24", "청취자 5–24명"),
              ("recall20_25plus", "청취자 25명+")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.24
    for offset, name in enumerate(SERIES):
        positions = [i + (offset - 1) * width for i in range(len(groups))]
        values = [table[name][key] for key, _ in groups]
        ax.bar(positions, values, width=width, color=SERIES[name], edgecolor=SURFACE, linewidth=2, label=name)
        if name == FINAL:  # aqua sits under 3:1 on this surface, so its values are written out
            for x, value in zip(positions, values):
                ax.text(x, value + 0.006, f"{value:.3f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(groups)), [label for _, label in groups])
    ax.set_ylabel("Recall@20")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", ncols=3)
    ax.set_title("곡의 청취자 수 구간별 Recall@20 (테스트)", loc="left", fontsize=14, color=INK, pad=14)
    _save(fig, path)


def _top10_composition(table: dict, path: Path) -> None:
    parts = [("exposure10_cold", "이력 0인 곡"), ("exposure10_1_4", "청취자 1–4명"), ("exposure10_5_24", "청취자 5–24명"),
             ("exposure10_25plus", "청취자 25명+")]
    names = list(table)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    left = [0.0] * len(names)
    for (key, label), color in zip(parts, HISTORY_RAMP):
        values = [table[n][key] * 100 for n in names]
        ax.barh(names, values, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for y, (start, value) in enumerate(zip(left, values)):
            if value >= 7:  # only where the number fits inside the segment
                ink = INK if color == HISTORY_RAMP[0] else "#ffffff"
                ax.text(start + value / 2, y, f"{value:.0f}%", ha="center", va="center", fontsize=10, color=ink)
        left = [a + b for a, b in zip(left, values)]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Top-10에서 차지하는 비율 (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=4, borderaxespad=0.2, columnspacing=1.2)
    ax.set_title("모델별 Top-10 구성: 곡의 학습 청취자 수 기준 (테스트)", loc="left", fontsize=14, color=INK, pad=34)
    _save(fig, path)


def _tradeoff_curve(points: list[dict], path: Path) -> None:
    x = [p["top10_cold_share"] * 100 for p in points]
    y = [p["ndcg10"] for p in points]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(x, y, color=SERIES["MF"], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for point, px, py in zip(points, x, y):
        if point["gamma"] == FINAL_GAMMA:
            ax.plot(px, py, marker="o", markersize=12, color=SERIES[FINAL], markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f"선택: γ={point['gamma']:g}, NDCG@10 {py:.3f}", (px, py), xytext=(0, 14),
                        textcoords="offset points", fontsize=10, color=INK, ha="center")
        elif point["gamma"] in (0.0, 1.0):
            ax.annotate(f"γ={point['gamma']:g}", (px, py), xytext=(8, 8), textcoords="offset points", fontsize=10,
                        color=INK_SECONDARY)
    ax.set_xlabel("Top-10에서 이력 0인 곡의 비율 (%)")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, max(y) * 1.25)
    ax.set_xlim(0, max(x) * 1.1)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"γ에 따른 이력 0인 곡 비율과 NDCG@10 (검증 데이터, β={CURVE_BETA})", loc="left", fontsize=14, color=INK, pad=12)
    _save(fig, path)


def _write_table(table: dict, points: list[dict], out: Path) -> None:
    """The same numbers as the figures, for reading and for the slides' table view."""
    keys = list(next(iter(table.values())))
    with open(out / "figure_data.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *keys])
        for name, row in table.items():
            writer.writerow([name, *[f"{row[k]:.4f}" for k in keys]])
    with open(out / "tradeoff_curve_data.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw the report figures")
    parser.add_argument("--results", default="outputs/hybrid")
    parser.add_argument("--output", default="outputs/hybrid/figures")
    args = parser.parse_args(argv)
    root, out = Path(args.results), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    table, points = build_table(root), curve_points(root)
    _model_comparison(table, out / FIGURES[0])
    _bucket_recall(table, out / FIGURES[1])
    _top10_composition(table, out / FIGURES[2])
    _tradeoff_curve(points, out / FIGURES[3])
    _write_table(table, points, out)


if __name__ == "__main__":
    main()
