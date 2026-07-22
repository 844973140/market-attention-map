"""Tests for the AGP runtime abstraction and local mock client."""

from __future__ import annotations

import unittest

from agp_client import AGPBudgetExceededError, MockAGPClient


class MockAGPClientTests(unittest.TestCase):
    def test_question_executes_successfully(self) -> None:
        client = MockAGPClient(
            hidden_answer="checkpoint-delta",
            initial_budget_usdc=0.01,
            question_cost_usdc=0.001,
        )

        response = client.ask_question(
            'Is the hidden checkpoint exactly "checkpoint-delta"? '
            "Answer only YES or NO."
        )

        self.assertEqual(response.text, "YES")
        self.assertEqual(client.get_status().questions_asked, 1)

    def test_budget_is_deducted(self) -> None:
        client = MockAGPClient(
            hidden_answer="checkpoint-delta",
            initial_budget_usdc=0.01,
            question_cost_usdc=0.002,
        )

        client.ask_question("Any valid mock question?")

        self.assertAlmostEqual(client.get_budget(), 0.008)

    def test_question_over_budget_is_rejected(self) -> None:
        client = MockAGPClient(
            hidden_answer="checkpoint-delta",
            initial_budget_usdc=0.001,
            question_cost_usdc=0.002,
        )

        with self.assertRaises(AGPBudgetExceededError):
            client.ask_question("Too expensive?")

        self.assertAlmostEqual(client.get_budget(), 0.001)
        self.assertEqual(client.get_status().questions_asked, 0)

    def test_answer_submission_succeeds(self) -> None:
        client = MockAGPClient(
            hidden_answer="checkpoint-delta",
            initial_budget_usdc=0.01,
        )

        result = client.submit_answer("checkpoint-delta")

        self.assertTrue(result.accepted)
        self.assertEqual(result.message, "checkpoint found")
        self.assertEqual(client.get_status().state, "submitted")


if __name__ == "__main__":
    unittest.main()
