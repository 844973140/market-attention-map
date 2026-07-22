"""Tests for state-aware AGP race decisions."""

from __future__ import annotations

import unittest

from adaptive_strategy import (
    AdaptiveMode,
    AdaptiveRaceState,
    AdaptiveRaceStrategy,
    FinishAction,
)
from simulation import AGPSimulator, StrategyVariant


def make_strategy() -> AdaptiveRaceStrategy:
    return AdaptiveRaceStrategy(
        total_budget_usdc=0.01,
        final_guess_reserve_usdc=0.002,
        base_information_efficiency=10.0,
        base_information_gain_bits=0.03,
        base_guess_confidence=0.92,
    )


class AdaptiveRaceStrategyTests(unittest.TestCase):
    def test_early_uncertain_race_explores(self) -> None:
        directive = make_strategy().decide(
            AdaptiveRaceState(0.01, 0.20, 0, 8, 0.05)
        )

        self.assertEqual(directive.mode, AdaptiveMode.EXPLORATION)
        self.assertEqual(directive.finish_action, FinishAction.CONTINUE)

    def test_reduced_candidate_set_optimizes(self) -> None:
        directive = make_strategy().decide(
            AdaptiveRaceState(0.007, 0.65, 2, 3, 0.45)
        )

        self.assertEqual(directive.mode, AdaptiveMode.OPTIMIZATION)
        self.assertGreater(directive.min_information_efficiency, 10.0)

    def test_high_confidence_finishes_immediately(self) -> None:
        directive = make_strategy().decide(
            AdaptiveRaceState(0.006, 0.95, 3, 2, 0.70)
        )

        self.assertEqual(directive.mode, AdaptiveMode.FINISH)
        self.assertEqual(directive.finish_action, FinishAction.SUBMIT)

    def test_likely_answer_gets_one_last_confirmation(self) -> None:
        directive = make_strategy().decide(
            AdaptiveRaceState(0.006, 0.85, 3, 2, 0.70)
        )

        self.assertEqual(directive.mode, AdaptiveMode.FINISH)
        self.assertEqual(directive.finish_action, FinishAction.LAST_CONFIRMATION)
        self.assertGreater(directive.max_question_cost_usdc, 0.0)

    def test_adaptive_simulation_benchmark_runs(self) -> None:
        benchmark = AGPSimulator(
            total_budget_usdc=0.01,
            candidate_count=6,
            judge_accuracy=1.0,
            unknown_answer_rate=0.0,
            seed=9,
        ).benchmark_adaptive_strategy(episodes=100)

        self.assertEqual(
            set(benchmark.reports),
            {StrategyVariant.RACE, StrategyVariant.ADAPTIVE},
        )
        self.assertLessEqual(
            benchmark.reports[StrategyVariant.ADAPTIVE].max_cost_usdc,
            0.01 + 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
