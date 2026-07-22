"""AGP race pacing and budget allocation above the search strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class RacePhase(str, Enum):
    EARLY_GAME = "early_game"
    MID_GAME = "mid_game"
    LATE_GAME = "late_game"


@dataclass(frozen=True, slots=True)
class BudgetAllocation:
    total_budget_usdc: float
    exploration_budget_usdc: float
    optimization_budget_usdc: float
    final_guess_budget_usdc: float

    @classmethod
    def create(
        cls,
        total_budget_usdc: float,
        minimum_final_guess_budget_usdc: float,
        candidate_count: int,
    ) -> "BudgetAllocation":
        if not math.isfinite(total_budget_usdc) or total_budget_usdc <= 0:
            raise ValueError("total race budget must be finite and positive")
        if (
            not math.isfinite(minimum_final_guess_budget_usdc)
            or minimum_final_guess_budget_usdc < 0
        ):
            raise ValueError("final guess budget must be finite and non-negative")
        if candidate_count < 1:
            raise ValueError("candidate count must be positive")

        final_budget = min(
            total_budget_usdc,
            max(minimum_final_guess_budget_usdc, total_budget_usdc * 0.10),
        )
        searchable_budget = total_budget_usdc - final_budget

        # More candidates justify a larger discovery allocation. The ratio stays
        # bounded so convergence always retains meaningful funds.
        complexity = min(1.0, math.log2(max(2, candidate_count)) / 6.0)
        exploration_ratio = 0.35 + 0.20 * complexity
        exploration_budget = searchable_budget * exploration_ratio
        optimization_budget = searchable_budget - exploration_budget
        return cls(
            total_budget_usdc=total_budget_usdc,
            exploration_budget_usdc=exploration_budget,
            optimization_budget_usdc=optimization_budget,
            final_guess_budget_usdc=final_budget,
        )


@dataclass(frozen=True, slots=True)
class RaceDirective:
    phase: RacePhase
    min_information_efficiency: float
    min_information_gain_bits: float
    required_guess_confidence: float
    max_question_cost_usdc: float
    exploration_remaining_usdc: float
    optimization_remaining_usdc: float
    final_guess_budget_usdc: float
    can_search: bool
    reason: str


class RaceStrategyLayer:
    """Turns race state into constraints for strategy selection and policy."""

    def __init__(
        self,
        *,
        total_budget_usdc: float,
        minimum_final_guess_budget_usdc: float,
        candidate_count: int,
        base_information_efficiency: float,
        base_information_gain_bits: float,
        base_guess_confidence: float,
    ):
        if base_information_efficiency < 0:
            raise ValueError("base information efficiency cannot be negative")
        if base_information_gain_bits < 0:
            raise ValueError("base information gain cannot be negative")
        if not 0.0 < base_guess_confidence <= 1.0:
            raise ValueError("base guess confidence must be in (0, 1]")

        self.allocation = BudgetAllocation.create(
            total_budget_usdc,
            minimum_final_guess_budget_usdc,
            candidate_count,
        )
        self.base_information_efficiency = base_information_efficiency
        self.base_information_gain_bits = base_information_gain_bits
        self.base_guess_confidence = base_guess_confidence
        self.exploration_spent_usdc = 0.0
        self.optimization_spent_usdc = 0.0
        self.current_phase = RacePhase.EARLY_GAME
        self.early_question_limit = max(
            1, min(3, math.ceil(math.log2(max(2, candidate_count)) / 2))
        )

    @property
    def exploration_remaining_usdc(self) -> float:
        return max(
            0.0,
            self.allocation.exploration_budget_usdc
            - self.exploration_spent_usdc,
        )

    @property
    def optimization_remaining_usdc(self) -> float:
        rollover = (
            self.exploration_remaining_usdc
            if self.current_phase is not RacePhase.EARLY_GAME
            else 0.0
        )
        return max(
            0.0,
            self.allocation.optimization_budget_usdc
            + rollover
            - self.optimization_spent_usdc,
        )

    def directive(self, confidence: float, questions_asked: int) -> RaceDirective:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if questions_asked < 0:
            raise ValueError("questions asked cannot be negative")

        self._advance_phase(confidence, questions_asked)
        exploration_remaining = self.exploration_remaining_usdc
        optimization_remaining = self.optimization_remaining_usdc

        if self.current_phase is RacePhase.EARLY_GAME:
            min_efficiency = self.base_information_efficiency * 0.50
            min_gain = self.base_information_gain_bits * 0.50
            required_confidence = self.base_guess_confidence
            max_question_cost = exploration_remaining
            reason = "explore broad hypotheses while protecting later budgets"
        elif self.current_phase is RacePhase.MID_GAME:
            min_efficiency = self.base_information_efficiency * 1.25
            min_gain = self.base_information_gain_bits
            required_confidence = self.base_guess_confidence
            max_question_cost = optimization_remaining
            reason = "converge using the highest information gain per USDC"
        else:
            min_efficiency = self.base_information_efficiency
            min_gain = self.base_information_gain_bits * 0.50
            required_confidence = min(
                0.99, max(self.base_guess_confidence, self.base_guess_confidence + 0.02)
            )
            # Late-game questions are capped to avoid one risky action consuming
            # the entire convergence pool.
            late_action_cap = max(
                self.allocation.total_budget_usdc * 0.15,
                optimization_remaining * 0.60,
            )
            max_question_cost = min(optimization_remaining, late_action_cap)
            reason = "verify the leading answer and reduce final-submission risk"

        return RaceDirective(
            phase=self.current_phase,
            min_information_efficiency=min_efficiency,
            min_information_gain_bits=min_gain,
            required_guess_confidence=required_confidence,
            max_question_cost_usdc=max_question_cost,
            exploration_remaining_usdc=exploration_remaining,
            optimization_remaining_usdc=optimization_remaining,
            final_guess_budget_usdc=self.allocation.final_guess_budget_usdc,
            can_search=max_question_cost > 1e-12,
            reason=reason,
        )

    def record_spend(self, actual_cost_usdc: float, phase: RacePhase) -> None:
        if not math.isfinite(actual_cost_usdc) or actual_cost_usdc < 0:
            raise ValueError("actual cost must be finite and non-negative")
        if phase is RacePhase.EARLY_GAME:
            self.exploration_spent_usdc += actual_cost_usdc
        else:
            self.optimization_spent_usdc += actual_cost_usdc

    def _advance_phase(self, confidence: float, questions_asked: int) -> None:
        if self.current_phase is RacePhase.EARLY_GAME and (
            questions_asked >= self.early_question_limit
            or confidence >= 0.55
            or self.exploration_remaining_usdc <= 1e-12
        ):
            self.current_phase = RacePhase.MID_GAME

        late_question_start = max(4, self.early_question_limit + 2)
        if self.current_phase is RacePhase.MID_GAME and (
            confidence >= 0.80
            or questions_asked >= late_question_start
            or self.optimization_remaining_usdc <= 1e-12
        ):
            self.current_phase = RacePhase.LATE_GAME
