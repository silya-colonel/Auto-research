from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.rank_parallel_ideas import IdeaDiagnostic, rank_ideas, write_ranking_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RankParallelIdeasTests(unittest.TestCase):
    def test_passing_idea_outranks_failed_idea(self) -> None:
        ideas = [
            IdeaDiagnostic(name="CLR-YOLO", summary={"gate": {"passes_gate": False}, "low_reliability_error_fraction": 0.90}),
            IdeaDiagnostic(name="OSD-YOLO", summary={"gate": {"passes_gate": True}, "unknown_recall": 0.30}),
        ]

        ranked = rank_ideas(ideas)

        self.assertEqual(ranked[0]["name"], "OSD-YOLO")
        self.assertTrue(ranked[0]["passes_gate"])

    def test_score_and_risk_affect_ordering(self) -> None:
        ideas = [
            IdeaDiagnostic(
                name="FASD-YOLO",
                summary={"gate": {"passes_gate": True}, "usable_rate": 0.72, "risk": 0.20},
            ),
            IdeaDiagnostic(
                name="OSD-YOLO",
                summary={"gate": {"passes_gate": True}, "unknown_recall": 0.75, "risk": 0.45},
            ),
        ]

        ranked = rank_ideas(ideas)

        self.assertEqual(ranked[0]["name"], "FASD-YOLO")
        self.assertGreater(ranked[0]["ranking_score"], ranked[1]["ranking_score"])

    def test_report_contains_ranked_table_and_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            ranked = rank_ideas(
                [
                    IdeaDiagnostic(name="CLR-YOLO", summary={"gate": {"passes_gate": True}, "idea_score": 0.80}),
                    IdeaDiagnostic(name="OSD-YOLO", summary={"gate": {"passes_gate": False}, "idea_score": 0.95}),
                ]
            )

            write_ranking_report(ranked, out)
            text = out.read_text(encoding="utf-8")

            self.assertIn("| Rank | Idea | Gate | Signal | Risk | Score |", text)
            self.assertIn("Recommended next training route: CLR-YOLO", text)

    def test_cli_reads_summaries_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clr = root / "clr" / "summary.json"
            osd = root / "osd" / "summary.json"
            fasd = root / "fasd" / "summary.json"
            out = root / "ranking.md"
            for path, payload in [
                (clr, {"gate": {"passes_gate": True}, "low_reliability_error_fraction": 0.65}),
                (osd, {"gate": {"passes_gate": False}, "unknown_recall": 0.80}),
                (fasd, {"gate": {"passes_gate": True}, "usable_rate": 0.50, "risk": 0.40}),
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "rank_parallel_ideas.py"),
                    "--clr-summary",
                    str(clr),
                    "--osd-summary",
                    str(osd),
                    "--fasd-summary",
                    str(fasd),
                    "--out",
                    str(out),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("Recommended next training route:", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
