"""Tests for AGP race pacing and budget allocation."""

from __future__ import annotations

import unittest

from policy_engine import BudgetController, PolicyDecision, PolicyEngine
from race_strategy import BudgetAllocation, RacePhase, RaceStrategyLayer
from strategy import PlannedQuestion


def make_race() -> RaceStrategyLayer:
    return RaceStrategyLayer(
        total_budget_usdc=0.01,
        minimum_final_guess_budget_usdc=0.002,
        candidate_count=8,
        base_information_efficiency=10.0,
        base_information_gain_bits=0.03,
        base_guess_confidence=0.92,
    )


class RaceStrategyTests(unittest.TestCase):
    def test_budget_allocation_preserves_all_three_pools(self) -> None:
        allocation = BudgetAllocation.create(0.01, 0.002, 8)
        allocated = (
            allocation.exploration_budget_usdc
            + allocation.optimization_budget_usdc
            + allocation.final_guess_budget_usdc
        )
        self.assertAlmostEqual(allocated, 0.01)
        self.assertGreater(allocation.exploration_budget_usdc, 0)
        self.assertGreater(allocation.optimization_budget_usdc, 0)
        self.assertGreaterEqual(allocation.final_guess_budget_usdc, 0.002)

    def test_race_moves_from_early_to_mid_to_late(self) -> None:
        race = make_race()
        early = race.directive(confidence=0.15, questions_asked=0)
        mid = race.directive(confidence=0.60, questions_asked=2)
        late = race.directive(confidence=0.85, questions_asked=3)

        self.assertEqual(early.phase, RacePhase.EARLY_GAME)
        self.assertEqual(mid.phase, RacePhase.MID_GAME)
        self.assertEqual(late.phase, RacePhase.LATE_GAME)
        self.assertGreater(late.required_guess_confidence, 0.92)

    def test_early_spend_does_not_consume_final_budget(self) -> None:
        race = make_race()
        directive = race.directive(confidence=0.15, questions_asked=0)
        race.record_spend(0.001, directive.phase)
        updated = race.directive(confidence=0.20, questions_asked=1)

        self.assertAlmostEqual(
            updated.exploration_remaining_usdc,
            race.allocation.exploration_budget_usdc - 0.001,
        )
        self.assertEqual(
            updated.final_guess_budget_usdc,
            race.allocation.final_guess_budget_usdc,
        )

    def test_mid_game_raises_efficiency_requirement(self) -> None:
        race = make_race()
        early = race.directive(confidence=0.10, questions_asked=0)
        mid = race.directive(confidence=0.60, questions_asked=2)

        self.assertGreater(mid.min_information_efficiency, early.min_information_efficiency)

    def test_race_cost_limit_is_enforced_by_policy(self) -> None:
        race = make_race()
        directive = race.directive(confidence=0.10, questions_asked=0)
        question = PlannedQuestion(
            text="Expensive exploration question?",
            kind="category",
            yes_answers=frozenset({"checkpoint-a"}),
            signature="expensive-exploration",
            information_gain_bits=1.0,
            expected_cost_usdc=directive.max_question_cost_usdc + 0.001,
            value_per_usdc=100.0,
            expected_success_gain=0.1,
        )
        policy = PolicyEngine(BudgetController(total_budget_usdc=0.01))

        verdict = policy.review_question(
            question,
            min_information_efficiency=directive.min_information_efficiency,
            min_information_gain_bits=directive.min_information_gain_bits,
            max_allowed_cost_usdc=directive.max_question_cost_usdc,
        )

        self.assertEqual(verdict.decision, PolicyDecision.DENY)
        self.assertEqual(verdict.reason, "Active race phase cost limit exceeded")


if __name__ == "__main__":
    unittest.main()
