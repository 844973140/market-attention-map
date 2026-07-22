"""State-aware pacing for AGP races without changing search or policy logic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class AdaptiveMode(str, Enum):
    EXPLORATION = "EXPLORATION"
    OPTIMIZATION = "OPTIMIZATION"
    FINISH = "FINISH"


class FinishAction(str, Enum):
    CONTINUE = "CONTINUE"
    SUBMIT = "SUBMIT"
    LAST_CONFIRMATION = "LAST_CONFIRMATION"


@dataclass(frozen=True, slots=True)
class AdaptiveRaceState:
    remaining_budget: float
    current_confidence: float
    questions_used: int
    candidate_count: int
    race_progress: float

    def validate(self) -> None:
        if not math.isfinite(self.remaining_budget) or self.remaining_budget < 0:
            raise ValueError("remaining budget must be finite and non-negative")
        if not 0.0 <= self.current_confidence <= 1.0:
            raise ValueError("current confidence must be in [0, 1]")
        if self.questions_used < 0:
            raise ValueError("questions used cannot be negative")
        if self.candidate_count < 1:
            raise ValueError("candidate count must be positive")
        if not 0.0 <= self.race_progress <= 1.0:
            raise ValueError("race progress must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AdaptiveDirective:
    mode: AdaptiveMode
    finish_action: FinishAction
    min_information_efficiency: float
    min_information_gain_bits: float
    required_guess_confidence: float
    max_question_cost_usdc: float
    reason: str


class AdaptiveRaceStrategy:
    """Maps live race state to safe, cost-aware strategy constraints."""

    def __init__(
        self,
        *,
        total_budget_usdc: float,
        final_guess_reserve_usdc: float,
        base_information_efficiency: float,
        base_information_gain_bits: float,
        base_guess_confidence: float,
        expected_confirmation_cost_usdc: float = 0.0011,
    ):
        values = (
            total_budget_usdc,
            final_guess_reserve_usdc,
            base_information_efficiency,
            base_information_gain_bits,
            expected_confirmation_cost_usdc,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("adaptive strategy values must be finite and non-negative")
        if total_budget_usdc <= 0:
            raise ValueError("total budget must be positive")
        if final_guess_reserve_usdc > total_budget_usdc:
            raise ValueError("final guess reserve cannot exceed total budget")
        if not 0.0 < base_guess_confidence <= 1.0:
            raise ValueError("base guess confidence must be in (0, 1]")

        self.total_budget_usdc = total_budget_usdc
        self.final_guess_reserve_usdc = final_guess_reserve_usdc
        self.base_information_efficiency = base_information_efficiency
        self.base_information_gain_bits = base_information_gain_bits
        self.base_guess_confidence = base_guess_confidence
        self.expected_confirmation_cost_usdc = expected_confirmation_cost_usdc

    def decide(self, state: AdaptiveRaceState) -> AdaptiveDirective:
        state.validate()
        searchable_budget = max(
            0.0, state.remaining_budget - self.final_guess_reserve_usdc
        )
        if state.current_confidence >= self.base_guess_confidence:
            return self._finish(
                FinishAction.SUBMIT,
                searchable_budget,
                "confidence is sufficient; avoid over-querying",
            )

        confirmation_is_affordable = (
            searchable_budget >= self.expected_confirmation_cost_usdc
        )
        budget_is_low = searchable_budget < self.expected_confirmation_cost_usdc * 1.35
        race_is_late = state.race_progress >= 0.85

        if budget_is_low or race_is_late:
            if confirmation_is_affordable and state.current_confidence >= 0.70:
                return self._finish(
                    FinishAction.LAST_CONFIRMATION,
                    searchable_budget,
                    "one bounded confirmation is worth the remaining risk",
                )
            return self._finish(
                FinishAction.SUBMIT,
                searchable_budget,
                "budget or race time is exhausted; stop querying",
            )

        if (
            state.race_progress < 0.40
            and state.candidate_count > 3
            and state.current_confidence < 0.55
        ):
            exploration_cap = max(
                self.total_budget_usdc * 0.20, searchable_budget * 0.35
            )
            return AdaptiveDirective(
                mode=AdaptiveMode.EXPLORATION,
                finish_action=FinishAction.CONTINUE,
                min_information_efficiency=self.base_information_efficiency * 0.75,
                min_information_gain_bits=self.base_information_gain_bits * 0.50,
                required_guess_confidence=self.base_guess_confidence,
                max_question_cost_usdc=min(searchable_budget, exploration_cap),
                reason="many candidates remain; maximize information gain per cost",
            )

        if state.candidate_count <= 3 and state.current_confidence >= 0.82:
            return self._finish(
                FinishAction.LAST_CONFIRMATION,
                searchable_budget,
                "a likely answer exists; buy at most one cheap confirmation",
            )

        return AdaptiveDirective(
            mode=AdaptiveMode.OPTIMIZATION,
            finish_action=FinishAction.CONTINUE,
            min_information_efficiency=self.base_information_efficiency * 1.50,
            min_information_gain_bits=self.base_information_gain_bits * 0.50,
            required_guess_confidence=self.base_guess_confidence,
            max_question_cost_usdc=min(
                searchable_budget, self.expected_confirmation_cost_usdc * 1.25
            ),
            reason="a leading region exists; use the cheapest confidence improvement",
        )

    def _finish(
        self,
        action: FinishAction,
        searchable_budget: float,
        reason: str,
    ) -> AdaptiveDirective:
        confirmation_cap = (
            min(searchable_budget, self.expected_confirmation_cost_usdc * 1.10)
            if action is FinishAction.LAST_CONFIRMATION
            else 0.0
        )
        return AdaptiveDirective(
            mode=AdaptiveMode.FINISH,
            finish_action=action,
            min_information_efficiency=self.base_information_efficiency,
            min_information_gain_bits=self.base_information_gain_bits * 0.25,
            required_guess_confidence=self.base_guess_confidence,
            max_question_cost_usdc=confirmation_cap,
            reason=reason,
        )
