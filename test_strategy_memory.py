"""Tests for persistent strategy experience and adaptive adjustments."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from adaptive_strategy import AdaptiveRaceState, AdaptiveRaceStrategy, FinishAction
from strategy_memory import (
    MemoryEnhancedAdaptiveStrategy,
    StrategyMemory,
    benchmark_memory_strategy,
)


class StrategyMemoryTests(unittest.TestCase):
    def test_data_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = StrategyMemory(path)
            memory.record_question("category", 0.52, 0.001, True)
            memory.record_decision(
                confidence=0.80,
                remaining_budget_ratio=0.75,
                candidate_count=4,
                action="continue exploration",
                successful=True,
            )

            saved_path = memory.save()

            self.assertTrue(saved_path.exists())
            payload = json.loads(saved_path.read_text("utf-8"))
            saved = payload["question_performance"]["category"]
            self.assertEqual(saved["average_gain"], 0.52)
            self.assertEqual(saved["average_cost"], 0.001)
            self.assertEqual(saved["success_rate"], 1.0)

    def test_data_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = StrategyMemory(path)
            for _ in range(10):
                memory.record_question("category", 0.52, 0.001, True)
            memory.save()

            loaded = StrategyMemory.load(path)
            performance = loaded.get_question_performance("category")

            self.assertIsNotNone(performance)
            assert performance is not None
            self.assertEqual(performance.count, 10)
            self.assertAlmostEqual(performance.average_gain, 0.52)
            self.assertAlmostEqual(performance.average_cost, 0.001)
            self.assertEqual(performance.success_rate, 1.0)

    def test_strategy_adjusts_from_history(self) -> None:
        memory = StrategyMemory()
        for _ in range(30):
            memory.record_question("adaptive_question", 0.50, 0.001, True)
            memory.record_decision(
                confidence=0.87,
                remaining_budget_ratio=0.60,
                candidate_count=2,
                action="submit",
                successful=True,
            )
        base = AdaptiveRaceStrategy(
            total_budget_usdc=0.01,
            final_guess_reserve_usdc=0.002,
            base_information_efficiency=10.0,
            base_information_gain_bits=0.03,
            base_guess_confidence=0.92,
        )
        state = AdaptiveRaceState(0.006, 0.87, 3, 2, 0.60)

        directive = MemoryEnhancedAdaptiveStrategy(base, memory).decide(state)

        self.assertEqual(directive.finish_action, FinishAction.SUBMIT)
        self.assertLess(directive.required_guess_confidence, 0.92)
        self.assertLess(directive.min_information_gain_bits, 0.03)

    def test_memory_benchmark_runs(self) -> None:
        report = benchmark_memory_strategy(
            episodes=100,
            training_episodes=100,
            seed=5,
        )

        self.assertEqual(report.episodes, 100)
        self.assertEqual(report.training_episodes, 100)
        self.assertLessEqual(report.adaptive_with_memory.max_cost_usdc, 0.01)


if __name__ == "__main__":
    unittest.main()
