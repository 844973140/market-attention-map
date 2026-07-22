"""Unit tests for the local Latch-inspired policy layer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy_engine import (
    ActionAuditLog,
    BudgetController,
    PolicyDecision,
    PolicyEngine,
)
from strategy import PlannedQuestion


def make_question(
    *, gain: float, cost: float, signature: str = "question-1"
) -> PlannedQuestion:
    score = gain / cost if cost else float("inf")
    return PlannedQuestion(
        text="Is checkpoint related to category A?",
        kind="category",
        yes_answers=frozenset({"checkpoint-a"}),
        signature=signature,
        information_gain_bits=gain,
        expected_cost_usdc=cost,
        value_per_usdc=score,
        expected_success_gain=0.1,
    )


class PolicyEngineTests(unittest.TestCase):
    def test_rejects_question_over_budget(self) -> None:
        engine = PolicyEngine(
            BudgetController(total_budget_usdc=0.01, submission_reserve_usdc=0.002)
        )

        verdict = engine.review_question(make_question(gain=1.0, cost=0.009))

        self.assertEqual(verdict.decision, PolicyDecision.DENY)
        self.assertEqual(verdict.reason, "Budget risk too high")

    def test_rejects_low_value_question(self) -> None:
        engine = PolicyEngine(
            BudgetController(total_budget_usdc=0.01),
            min_information_efficiency=100.0,
            min_information_gain_bits=0.0,
        )

        verdict = engine.review_question(make_question(gain=0.1, cost=0.002))

        self.assertEqual(verdict.decision, PolicyDecision.DENY)
        self.assertEqual(verdict.information_efficiency_score, 50.0)
        self.assertIn("compared with cost", verdict.reason)

    def test_allows_high_value_question(self) -> None:
        engine = PolicyEngine(
            BudgetController(total_budget_usdc=0.01),
            min_information_efficiency=100.0,
        )

        verdict = engine.review_question(make_question(gain=0.5, cost=0.001))

        self.assertEqual(verdict.decision, PolicyDecision.ALLOW)
        self.assertEqual(verdict.information_efficiency_score, 500.0)

    def test_audit_log_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy-audit.jsonl"
            log = ActionAuditLog(path)
            engine = PolicyEngine(
                BudgetController(total_budget_usdc=0.01), audit_log=log
            )

            verdict = engine.review_question(make_question(gain=0.5, cost=0.001))

            self.assertEqual(len(log.entries), 1)
            self.assertIn("[ALLOW]", verdict.audit_entry.render())
            saved = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(saved["decision"], "ALLOW")
            self.assertEqual(saved["action_type"], "question")
            self.assertIn("timestamp", saved)

    def test_rejects_duplicate_and_low_confidence_guess(self) -> None:
        engine = PolicyEngine(BudgetController(total_budget_usdc=0.01))
        question = make_question(gain=0.5, cost=0.001)
        self.assertTrue(engine.review_question(question).allowed)
        self.assertEqual(
            engine.review_question(question).decision, PolicyDecision.DENY
        )
        self.assertEqual(
            engine.review_guess("checkpoint-a", 0.5).decision,
            PolicyDecision.DENY,
        )


if __name__ == "__main__":
    unittest.main()
