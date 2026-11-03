"""Checks for the report figures. No real data needed.
Run: python -m unittest test_make_figures.py -v
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from matplotlib import font_manager

from make_figures import FIGURES, build_table, curve_points, font_stack, main


def model(ndcg, mock):
    buckets = {"mock_cold": mock, "1-4": 0.1, "5-24": 0.2, "25+": 0.3}
    return {"overall": {"@10": {"ndcg": ndcg, "recall": ndcg + 0.01}}, "bucket_recall": {"@20": buckets},
            "exposure": {"@10": {"mock_cold": 0.05, "sample_zero": 0.05, "1-4": 0.2, "5-24": 0.3, "25+": 0.4}},
            "catalog_coverage": {"@10": 0.1}}


def flat(ndcg, mock):
    return {"ndcg10": ndcg, "mock_recall20": mock, "recall20_1_4": 0.1, "recall20_5_24": 0.2, "recall20_25plus": 0.3,
            "exposure10_cold": 0.1, "exposure10_1_4": 0.2, "exposure10_5_24": 0.3, "exposure10_25plus": 0.4,
            "coverage10": 0.1, "top10_cold_share": 0.1}


def val(ndcg, cold):
    return {"ndcg10": ndcg, "mock_recall20": 0.1, "top10_cold_share": cold}


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.root = root
        files = {
            "rank/hybrid_evaluation_metrics.json": {"calibration_comparison": {}, "models": {"Hybrid": model(0.02, 0.0)}},
            "artist_mf/artist_mf_comparison.json": {
                "test": {"mf": flat(0.18, 0.0), "fixed_artist": flat(0.19, 0.01), "dynamic_artist": flat(0.07, 0.18)},
                "validation": {"beta=0.9|gamma=0.0": val(0.04, 0.85), "beta=0.9|gamma=1.0": val(0.08, 0.31),
                               "beta=0.5|gamma=1.0": val(0.09, 0.39)}},
            "artist_mf_gamma/artist_mf_comparison.json": {
                "validation": {"beta=0.9|gamma=1.0": val(0.08, 0.31), "beta=0.9|gamma=5.0": val(0.09, 0.21)}},
            "final/final_evaluation_metrics.json": {"final": flat(0.12, 0.09)},
        }
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content), encoding="utf-8")

    def test_table_has_every_model(self):
        table = build_table(self.root)
        self.assertEqual(list(table), ["MF", "제안서 동적 α", "고정 α + 아티스트", "동적 α + 아티스트", "최종 모델"])
        self.assertEqual(table["최종 모델"]["ndcg10"], 0.12)
        self.assertEqual(table["MF"]["mock_recall20"], 0.0)
        self.assertAlmostEqual(table["최종 모델"]["exposure10_cold"], 0.1)

    def test_curve_merges_runs_for_the_final_beta_only(self):
        points = curve_points(self.root)
        self.assertEqual([p["gamma"] for p in points], [0.0, 1.0, 5.0])
        self.assertEqual(points[-1]["ndcg10"], 0.09)

    def test_font_stack_only_names_installed_fonts(self):
        installed = {f.name for f in font_manager.fontManager.ttflist}
        stack = font_stack()
        self.assertTrue(stack)
        self.assertTrue(set(stack) <= installed)

    def test_main_writes_every_figure_and_a_table_view(self):
        out = self.root / "figures"
        main(["--results", str(self.root), "--output", str(out)])
        for name in FIGURES:
            self.assertGreater((out / name).stat().st_size, 1000, name)
        with open(out / "figure_data.csv", encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 5)


if __name__ == "__main__":
    unittest.main()
