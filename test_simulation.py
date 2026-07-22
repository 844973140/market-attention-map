"""Tests for statistical AGP simulation training."""

from __future__ import annotations

import unittest

from simulation import AGPSimulator, StrategyVariant


class AGPSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = AGPSimulator(
            total_budget_usdc=0.01,
            candidate_count=6,
            judge_accuracy=1.0,
            unknown_answer_rate=0.0,
            seed=123,
        )

    def test_one_hundred_episode_simulation_runs(self) -> None:
        report = self.simulator.run_simulation(
            episodes=100, strategy_variant=StrategyVariant.RACE
        )

        self.assertEqual(report.episodes, 100)
        self.assertGreater(report.win_rate, 0.0)
        self.assertGreater(report.average_questions, 0.0)

    def test_budget_never_exceeds_limit(self) -> None:
        for episode in range(100):
            result = self.simulator.run_episode(
                episode, StrategyVariant.AGGRESSIVE
            )
            self.assertLessEqual(result.cost_usdc, 0.01 + 1e-12)
            self.assertGreaterEqual(result.remaining_budget_usdc, -1e-12)

    def test_agent_can_complete_submission(self) -> None:
        result = self.simulator.run_episode(0, StrategyVariant.RACE)

        self.assertTrue(result.submitted)
        self.assertIsNotNone(result.submitted_answer)

    def test_benchmark_contains_all_strategies(self) -> None:
        benchmark = self.simulator.benchmark_strategies(episodes=10)

        self.assertEqual(set(benchmark.reports), set(StrategyVariant))


if __name__ == "__main__":
    unittest.main()
