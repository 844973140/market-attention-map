"""A lightweight, Latch-inspired policy layer for agent actions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from strategy import PlannedQuestion


class PolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


@dataclass(frozen=True, slots=True)
class AuditEntry:
    timestamp: str
    action_type: str
    question: str
    estimated_cost_usdc: float
    decision: PolicyDecision
    reason: str
    information_gain_bits: float | None = None
    information_efficiency_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data

    def render(self) -> str:
        lines = [
            f"[{self.decision.value}]",
            f"Timestamp: {self.timestamp}",
            f"Action: {self.action_type}",
        ]
        if self.question:
            lines.extend(("Question:", self.question))
        lines.extend(
            (
                f"Cost: {self.estimated_cost_usdc:.6f} USDC",
                f"Reason: {self.reason}",
            )
        )
        return "\n".join(lines)


class ActionAuditLog:
    """In-memory audit log with optional append-only JSONL persistence."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


@dataclass(slots=True)
class BudgetController:
    """Tracks the hard budget and protects the final-submission reserve."""

    total_budget_usdc: float = 0.01
    submission_reserve_usdc: float = 0.0
    spent_usdc: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.total_budget_usdc,
            self.submission_reserve_usdc,
            self.spent_usdc,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("budget values must be finite and non-negative")
        if self.submission_reserve_usdc > self.total_budget_usdc:
            raise ValueError("submission reserve cannot exceed total budget")

    @property
    def remaining_usdc(self) -> float:
        return max(0.0, self.total_budget_usdc - self.spent_usdc)

    @property
    def searchable_usdc(self) -> float:
        return max(0.0, self.remaining_usdc - self.submission_reserve_usdc)

    def can_authorize_question(self, estimated_cost_usdc: float) -> bool:
        if not math.isfinite(estimated_cost_usdc) or estimated_cost_usdc < 0:
            return False
        if self.spent_usdc > self.total_budget_usdc:
            return False
        return estimated_cost_usdc <= self.searchable_usdc + 1e-12

    def record_spend(self, actual_cost_usdc: float) -> None:
        if not math.isfinite(actual_cost_usdc) or actual_cost_usdc < 0:
            raise ValueError("actual cost must be finite and non-negative")
        self.spent_usdc += actual_cost_usdc


@dataclass(frozen=True, slots=True)
class PolicyVerdict:
    decision: PolicyDecision
    reason: str
    information_efficiency_score: float
    audit_entry: AuditEntry

    @property
    def allowed(self) -> bool:
        return self.decision is PolicyDecision.ALLOW


class PolicyEngine:
    """Reviews strategy output before any paid or irreversible action."""

    def __init__(
        self,
        budget: BudgetController | None = None,
        *,
        min_information_efficiency: float = 10.0,
        min_information_gain_bits: float = 0.03,
        min_guess_confidence: float = 0.92,
        audit_log: ActionAuditLog | None = None,
    ):
        if min_information_efficiency < 0 or not math.isfinite(
            min_information_efficiency
        ):
            raise ValueError("minimum information efficiency must be finite")
        if min_information_gain_bits < 0 or not math.isfinite(
            min_information_gain_bits
        ):
            raise ValueError("minimum information gain must be finite")
        if not 0.0 <= min_guess_confidence <= 1.0:
            raise ValueError("minimum guess confidence must be in [0, 1]")

        self.budget = budget or BudgetController()
        self.min_information_efficiency = min_information_efficiency
        self.min_information_gain_bits = min_information_gain_bits
        self.min_guess_confidence = min_guess_confidence
        self.audit_log = audit_log or ActionAuditLog()
        self.seen_question_signatures: set[str] = set()

    @staticmethod
    def information_efficiency_score(
        information_gain_bits: float, estimated_cost_usdc: float
    ) -> float:
        if estimated_cost_usdc == 0:
            return math.inf if information_gain_bits > 0 else 0.0
        return information_gain_bits / estimated_cost_usdc

    def review_question(
        self,
        question: PlannedQuestion,
        *,
        min_information_efficiency: float | None = None,
        min_information_gain_bits: float | None = None,
        max_allowed_cost_usdc: float | None = None,
        action_type: str = "question",
    ) -> PolicyVerdict:
        gain = question.information_gain_bits
        cost = question.expected_cost_usdc
        score = self.information_efficiency_score(gain, cost)
        efficiency_threshold = (
            self.min_information_efficiency
            if min_information_efficiency is None
            else min_information_efficiency
        )
        gain_threshold = (
            self.min_information_gain_bits
            if min_information_gain_bits is None
            else min_information_gain_bits
        )
        if (
            not math.isfinite(efficiency_threshold)
            or efficiency_threshold < 0
            or not math.isfinite(gain_threshold)
            or gain_threshold < 0
        ):
            raise ValueError("race policy thresholds must be finite and non-negative")
        if max_allowed_cost_usdc is not None and (
            not math.isfinite(max_allowed_cost_usdc)
            or max_allowed_cost_usdc < 0
        ):
            raise ValueError("race cost limit must be finite and non-negative")

        if (
            not question.text.strip()
            or not question.signature.strip()
            or not question.yes_answers
            or not math.isfinite(gain)
            or gain <= 0
            or not math.isfinite(cost)
            or cost < 0
        ):
            return self._verdict(
                PolicyDecision.DENY,
                "Invalid or non-informative question",
                question.text,
                cost,
                gain,
                score,
                action_type=action_type,
            )

        if question.signature in self.seen_question_signatures:
            return self._verdict(
                PolicyDecision.DENY,
                "Duplicate question",
                question.text,
                cost,
                gain,
                score,
                action_type=action_type,
            )
        # A denied action is still an attempted action, so it is remembered too.
        self.seen_question_signatures.add(question.signature)

        if not self.budget.can_authorize_question(cost):
            return self._verdict(
                PolicyDecision.DENY,
                "Budget risk too high",
                question.text,
                cost,
                gain,
                score,
                action_type=action_type,
            )
        if max_allowed_cost_usdc is not None and cost > max_allowed_cost_usdc + 1e-12:
            return self._verdict(
                PolicyDecision.DENY,
                "Active race phase cost limit exceeded",
                question.text,
                cost,
                gain,
                score,
                action_type=action_type,
            )
        if gain < gain_threshold:
            return self._verdict(
                PolicyDecision.DENY,
                "Expected information gain is below policy threshold",
                question.text,
                cost,
                gain,
                score,
                action_type=action_type,
            )
        if score < efficiency_threshold:
            return self._verdict(
                PolicyDecision.DENY,
                "Low expected information gain compared with cost",
                question.text,
                cost,
                gain,
                score,
                action_type=action_type,
            )
        return self._verdict(
            PolicyDecision.ALLOW,
            f"High information gain per unit cost (score={score:.2f})",
            question.text,
            cost,
            gain,
            score,
            action_type=action_type,
        )

    def review_guess(
        self,
        answer: str,
        confidence: float,
        *,
        required_confidence: float | None = None,
        action_type: str = "guess",
    ) -> PolicyVerdict:
        confidence_threshold = self.min_guess_confidence
        if required_confidence is not None:
            if not 0.0 <= required_confidence <= 1.0:
                raise ValueError("required confidence must be in [0, 1]")
            confidence_threshold = max(confidence_threshold, required_confidence)
        if not answer.strip() or not math.isfinite(confidence):
            return self._verdict(
                PolicyDecision.DENY,
                "Invalid guess",
                answer,
                0.0,
                None,
                0.0,
                action_type=action_type,
            )
        if self.budget.spent_usdc > self.budget.total_budget_usdc + 1e-12:
            return self._verdict(
                PolicyDecision.DENY,
                "Total budget already exceeded",
                answer,
                0.0,
                None,
                0.0,
                action_type=action_type,
            )
        if confidence < confidence_threshold:
            return self._verdict(
                PolicyDecision.DENY,
                f"Confidence below guessing threshold (required={confidence_threshold:.2%})",
                answer,
                0.0,
                None,
                0.0,
                action_type=action_type,
            )
        return self._verdict(
            PolicyDecision.ALLOW,
            "Confidence sufficient for controlled submission",
            answer,
            0.0,
            None,
            0.0,
            action_type=action_type,
        )

    def record_spend(self, actual_cost_usdc: float) -> None:
        self.budget.record_spend(actual_cost_usdc)

    def _verdict(
        self,
        decision: PolicyDecision,
        reason: str,
        question: str,
        estimated_cost_usdc: float,
        information_gain_bits: float | None,
        score: float,
        *,
        action_type: str = "question",
    ) -> PolicyVerdict:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action_type=action_type,
            question=question,
            estimated_cost_usdc=estimated_cost_usdc,
            decision=decision,
            reason=reason,
            information_gain_bits=information_gain_bits,
            information_efficiency_score=score,
        )
        self.audit_log.record(entry)
        return PolicyVerdict(decision, reason, score, entry)
