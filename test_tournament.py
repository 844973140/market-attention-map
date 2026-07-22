"""Tests for tournament evaluation and public report export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from export_report import export_report
from tournament import TournamentConfig, run_tournament


class TournamentTests(unittest.TestCase):
    def test_four_strategy_tournament_and_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tournament_report.json"
            report = run_tournament(
                TournamentConfig(
                    episodes=40,
                    memory_training_episodes=40,
                    seed=19,
                    judge_accuracy=1.0,
                    unknown_answer_rate=0.0,
                ),
                report_path=output,
            )

            self.assertEqual(len(report.leaderboard), 4)
            self.assertEqual([row.rank for row in report.leaderboard], [1, 2, 3, 4])
            self.assertEqual(
                {row.strategy for row in report.leaderboard},
                {
                    "Random Search",
                    "Fixed Budget",
                    "Adaptive Strategy",
                    "Adaptive + Memory",
                },
            )
            self.assertTrue(output.exists())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["parameters"]["episodes_per_strategy"], 40)
            self.assertEqual(len(payload["strategy_versions"]), 4)
            self.assertTrue(
                all(row["episodes"] == 40 for row in payload["leaderboard"])
            )
            self.assertTrue(
                all(
                    row["average_cost_usdc"] <= 0.01 + 1e-12
                    for row in payload["leaderboard"]
                )
            )

    def test_markdown_export_contains_selected_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "tournament_report.json"
            markdown_path = Path(directory) / "benchmark.md"
            run_tournament(
                TournamentConfig(episodes=20, memory_training_episodes=20, seed=7),
                report_path=json_path,
            )

            export_report(json_path, markdown_path)
            markdown = markdown_path.read_text(encoding="utf-8")

            self.assertIn("# AGP Agent Benchmark", markdown)
            self.assertIn("Adaptive + Memory", markdown)
            self.assertIn("## Tournament Leaderboard", markdown)
            self.assertIn("wins/USDC", markdown)


if __name__ == "__main__":
    unittest.main()
