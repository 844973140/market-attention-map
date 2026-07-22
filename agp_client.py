"""Runtime boundary for AGP environments and a local mock implementation."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping


class AGPClientError(RuntimeError):
    """Base error raised by an AGP runtime adapter."""


class AGPBudgetExceededError(AGPClientError):
    """Raised when the runtime cannot fund an action."""


class AGPGameFinishedError(AGPClientError):
    """Raised when an action is attempted after final submission."""


@dataclass(frozen=True, slots=True)
class AGPStatus:
    state: str
    questions_asked: int
    remaining_budget_usdc: float
    submitted_answer: str | None = None
    accepted: bool | None = None


@dataclass(slots=True)
class AGPQuestionResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usdc: float | None = None
    raw: Mapping[str, Any] | None = None


@dataclass(slots=True)
class AGPSubmissionResponse:
    accepted: bool | None
    message: str
    cost_usdc: float | None = None
    raw: Mapping[str, Any] | None = None


class AGPClient(ABC):
    """Replace this adapter, not agent logic, for a real AGP runtime."""

    @abstractmethod
    def get_status(self) -> AGPStatus:
        """Return the current game state."""

    @abstractmethod
    def get_budget(self) -> float:
        """Return the authoritative remaining budget in USDC."""

    @abstractmethod
    def ask_question(self, question: str) -> AGPQuestionResponse:
        """Send one question to the judge and return its answer and cost."""

    @abstractmethod
    def submit_answer(self, answer: str) -> AGPSubmissionResponse:
        """Submit the final checkpoint answer."""


QuestionCost = float | Callable[[str], float]
JudgeResolver = Callable[[str], str]


class MockAGPClient(AGPClient):
    """In-memory AGP runtime with deterministic judge and budget accounting."""

    def __init__(
        self,
        *,
        hidden_answer: str,
        initial_budget_usdc: float = 0.01,
        question_cost_usdc: QuestionCost = 0.001,
        submission_cost_usdc: float = 0.0,
        hidden_category: str = "uncategorized",
        hidden_attributes: Mapping[str, Any] | None = None,
        judge_resolver: JudgeResolver | None = None,
    ):
        if not hidden_answer.strip():
            raise ValueError("hidden answer cannot be empty")
        if not math.isfinite(initial_budget_usdc) or initial_budget_usdc < 0:
            raise ValueError("initial budget must be finite and non-negative")
        if not callable(question_cost_usdc) and (
            not math.isfinite(question_cost_usdc) or question_cost_usdc < 0
        ):
            raise ValueError("question cost must be finite and non-negative")
        if not math.isfinite(submission_cost_usdc) or submission_cost_usdc < 0:
            raise ValueError("submission cost must be finite and non-negative")

        self.hidden_answer = hidden_answer.strip()
        self.hidden_category = hidden_category
        self.hidden_attributes = dict(hidden_attributes or {})
        self.initial_budget_usdc = initial_budget_usdc
        self.remaining_budget_usdc = initial_budget_usdc
        self.question_cost_usdc = question_cost_usdc
        self.submission_cost_usdc = submission_cost_usdc
        self.judge_resolver = judge_resolver
        self.questions_asked = 0
        self.submitted_answer: str | None = None
        self.accepted: bool | None = None

    def get_status(self) -> AGPStatus:
        return AGPStatus(
            state="submitted" if self.submitted_answer is not None else "running",
            questions_asked=self.questions_asked,
            remaining_budget_usdc=self.remaining_budget_usdc,
            submitted_answer=self.submitted_answer,
            accepted=self.accepted,
        )

    def get_budget(self) -> float:
        return self.remaining_budget_usdc

    def ask_question(self, question: str) -> AGPQuestionResponse:
        self._ensure_running()
        if not question.strip():
            raise ValueError("question cannot be empty")
        cost = self._question_cost(question)
        self._ensure_affordable(cost, "question")

        self.remaining_budget_usdc -= cost
        self.questions_asked += 1
        answer = (
            self.judge_resolver(question)
            if self.judge_resolver is not None
            else self._resolve_generated_question(question)
        )
        return AGPQuestionResponse(
            text=answer,
            completion_tokens=1,
            cost_usdc=cost,
            raw={
                "mock": True,
                "remaining_budget_usdc": self.remaining_budget_usdc,
            },
        )

    def submit_answer(self, answer: str) -> AGPSubmissionResponse:
        self._ensure_running()
        if not answer.strip():
            raise ValueError("answer cannot be empty")
        self._ensure_affordable(self.submission_cost_usdc, "submission")

        self.remaining_budget_usdc -= self.submission_cost_usdc
        self.submitted_answer = answer.strip()
        self.accepted = self.submitted_answer == self.hidden_answer
        message = "checkpoint found" if self.accepted else "wrong checkpoint"
        return AGPSubmissionResponse(
            accepted=self.accepted,
            message=message,
            cost_usdc=self.submission_cost_usdc,
            raw={
                "mock": True,
                "remaining_budget_usdc": self.remaining_budget_usdc,
            },
        )

    def _question_cost(self, question: str) -> float:
        value = (
            self.question_cost_usdc(question)
            if callable(self.question_cost_usdc)
            else self.question_cost_usdc
        )
        cost = float(value)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("calculated question cost must be finite and non-negative")
        return cost

    def _ensure_affordable(self, cost_usdc: float, action: str) -> None:
        if cost_usdc > self.remaining_budget_usdc + 1e-12:
            raise AGPBudgetExceededError(
                f"insufficient AGP budget for {action}: "
                f"required={cost_usdc:.6f}, "
                f"remaining={self.remaining_budget_usdc:.6f} USDC"
            )

    def _ensure_running(self) -> None:
        if self.submitted_answer is not None:
            raise AGPGameFinishedError("the AGP game already has a final submission")

    def _resolve_generated_question(self, question: str) -> str:
        """Understand the constrained yes/no templates emitted by strategy.py."""

        category_prefix = "Is the hidden checkpoint in the category "
        exact_prefix = "Is the hidden checkpoint exactly "
        list_prefix = "Is the hidden checkpoint one of the following exact candidates: "
        attribute_prefix = "Does the hidden checkpoint have attribute "

        try:
            if question.startswith(category_prefix):
                category, _ = _decode_json_at(question, len(category_prefix))
                return _yes_no(self.hidden_category == category)
            if question.startswith(exact_prefix):
                candidate, _ = _decode_json_at(question, len(exact_prefix))
                return _yes_no(self.hidden_answer == candidate)
            if question.startswith(list_prefix):
                candidates, _ = _decode_json_at(question, len(list_prefix))
                return _yes_no(self.hidden_answer in candidates)
            if question.startswith(attribute_prefix):
                key, offset = _decode_json_at(question, len(attribute_prefix))
                equal_marker = " equal to "
                equal_position = question.index(equal_marker, offset) + len(equal_marker)
                value, _ = _decode_json_at(question, equal_position)
                return _yes_no(self.hidden_attributes.get(str(key)) == value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return "UNKNOWN"
        return "UNKNOWN"


def _decode_json_at(text: str, start: int) -> tuple[Any, int]:
    remainder = text[start:]
    stripped = remainder.lstrip()
    leading_spaces = len(remainder) - len(stripped)
    value, end = json.JSONDecoder().raw_decode(stripped)
    return value, start + leading_spaces + end


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"
