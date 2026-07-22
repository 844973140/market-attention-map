"""Value-of-information strategy for finding a hidden checkpoint."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from config import AgentConfig


@dataclass(frozen=True, slots=True)
class Candidate:
    """One possible checkpoint and optional facts the judge can verify."""

    answer: str
    category: str = "uncategorized"
    prior: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlannedQuestion:
    """A yes/no question plus its predicted value and outcome partition."""

    text: str
    kind: str
    yes_answers: frozenset[str]
    signature: str
    information_gain_bits: float
    expected_cost_usdc: float
    value_per_usdc: float
    expected_success_gain: float


@dataclass(frozen=True, slots=True)
class Decision:
    action: Literal["ask", "guess"]
    reason: str
    confidence: float
    guess: str
    question: PlannedQuestion | None = None


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate for English and CJK prompts."""

    ascii_count = sum(ord(char) < 128 for char in text)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4) + non_ascii_count)


def estimate_question_cost(text: str, config: AgentConfig) -> float:
    input_tokens = estimate_tokens(text)
    return (
        config.question_base_cost_usdc
        + input_tokens * config.input_usdc_per_1k_tokens / 1000.0
        + config.max_judge_response_tokens
        * config.output_usdc_per_1k_tokens
        / 1000.0
    )


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


class InformationValueStrategy:
    """Bayesian question selection optimized for information per USDC."""

    def __init__(self, candidates: Sequence[Candidate], config: AgentConfig):
        if not candidates:
            raise ValueError("at least one checkpoint candidate is required")

        answers = [candidate.answer.strip() for candidate in candidates]
        if any(not answer for answer in answers):
            raise ValueError("candidate answers cannot be empty")
        if len(set(answers)) != len(answers):
            raise ValueError("candidate answers must be unique")
        if any(candidate.prior <= 0 for candidate in candidates):
            raise ValueError("candidate priors must be positive")

        self.config = config
        self.candidates = {
            candidate.answer.strip(): Candidate(
                answer=candidate.answer.strip(),
                category=candidate.category.strip() or "uncategorized",
                prior=candidate.prior,
                attributes=dict(candidate.attributes),
            )
            for candidate in candidates
        }
        prior_total = sum(candidate.prior for candidate in candidates)
        self.probabilities = {
            candidate.answer.strip(): candidate.prior / prior_total
            for candidate in candidates
        }
        self.asked_signatures: set[str] = set()
        self.category_questions_asked = 0
        self.unparsed_answers = 0

    @property
    def best_guess(self) -> tuple[str, float]:
        return max(self.probabilities.items(), key=lambda item: item[1])

    def ranked_candidates(self, limit: int = 5) -> list[tuple[str, float]]:
        return sorted(
            self.probabilities.items(), key=lambda item: item[1], reverse=True
        )[:limit]

    def category_probabilities(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for answer, probability in self.probabilities.items():
            category = self.candidates[answer].category
            result[category] = result.get(category, 0.0) + probability
        return result

    def decide(
        self,
        spent_usdc: float,
        questions_asked: int,
        *,
        confidence_threshold: float | None = None,
        min_information_gain_bits: float | None = None,
        min_value_per_usdc: float | None = None,
        max_question_cost_usdc: float | None = None,
    ) -> Decision:
        guess, confidence = self.best_guess
        confidence_target = (
            self.config.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        gain_threshold = (
            self.config.min_information_gain_bits
            if min_information_gain_bits is None
            else min_information_gain_bits
        )
        value_threshold = (
            self.config.min_value_per_usdc
            if min_value_per_usdc is None
            else min_value_per_usdc
        )

        if confidence >= confidence_target:
            return Decision("guess", "confidence threshold reached", confidence, guess)
        if len(self.probabilities) == 1:
            return Decision("guess", "only one candidate remains", confidence, guess)
        if questions_asked >= self.config.max_questions:
            return Decision("guess", "question limit reached", confidence, guess)

        spendable = (
            self.config.budget_usdc
            - self.config.submission_reserve_usdc
            - spent_usdc
        )
        if spendable <= 0:
            return Decision("guess", "submission reserve reached", confidence, guess)

        phase = self._current_phase()
        plans = self._category_plans() if phase == 1 else self._search_plans()
        race_cost_limit = (
            math.inf
            if max_question_cost_usdc is None
            else max(0.0, max_question_cost_usdc)
        )
        affordable = [
            plan
            for plan in plans
            if plan.expected_cost_usdc <= spendable
            and plan.expected_cost_usdc <= race_cost_limit
        ]
        if not affordable:
            return Decision(
                "guess", "no useful question fits the active race budget", confidence, guess
            )

        # A short, targeted question can beat a longer question with slightly more
        # raw information. This is the central value-per-USDC optimization.
        best = max(
            affordable,
            key=lambda plan: (
                plan.value_per_usdc,
                plan.information_gain_bits,
                plan.expected_success_gain,
            ),
        )
        if best.information_gain_bits < gain_threshold:
            return Decision("guess", "remaining information gain is too small", confidence, guess)
        if best.value_per_usdc < value_threshold:
            return Decision("guess", "remaining questions are not cost-effective", confidence, guess)
        if best.expected_success_gain < self.config.min_expected_success_gain:
            return Decision("guess", "question will not improve success enough", confidence, guess)
        return Decision("ask", f"phase {phase}: highest information value", confidence, guess, best)

    def observe(self, plan: PlannedQuestion, raw_answer: str) -> str:
        """Update the posterior using a noisy-judge likelihood model."""

        self.asked_signatures.add(plan.signature)
        if plan.kind == "category":
            self.category_questions_asked += 1

        outcome = self._parse_yes_no(raw_answer)
        if outcome is None:
            self.unparsed_answers += 1
            return "unknown"

        accuracy = self.config.judge_accuracy
        updated: dict[str, float] = {}
        for answer, prior in self.probabilities.items():
            predicted_yes = answer in plan.yes_answers
            agrees = predicted_yes == outcome
            likelihood = accuracy if agrees else 1.0 - accuracy
            updated[answer] = prior * likelihood

        total = sum(updated.values())
        if total > 0:
            self.probabilities = {
                answer: probability / total
                for answer, probability in updated.items()
            }
        return "yes" if outcome else "no"

    def _current_phase(self) -> int:
        category_probs = self.category_probabilities()
        if len(category_probs) <= 1:
            return 2
        category_confidence = max(category_probs.values())
        if (
            category_confidence < self.config.category_confidence_threshold
            and self.category_questions_asked < self.config.max_category_questions
        ):
            return 1
        return 2

    def _category_plans(self) -> list[PlannedQuestion]:
        categories: dict[str, set[str]] = {}
        for answer, candidate in self.candidates.items():
            categories.setdefault(candidate.category, set()).add(answer)

        plans: list[PlannedQuestion] = []
        for category, yes_answers in categories.items():
            text = (
                "Is the hidden checkpoint in the category "
                f"{json.dumps(category, ensure_ascii=False)}? "
                "Answer only YES or NO."
            )
            plan = self._score_question("category", text, yes_answers)
            if plan is not None:
                plans.append(plan)
        return plans

    def _search_plans(self) -> list[PlannedQuestion]:
        candidates = list(self.probabilities)
        plans: list[PlannedQuestion] = []

        # Direct confirmation is cheap and becomes attractive when one posterior
        # already dominates.
        top_answer, _ = self.best_guess
        direct_text = (
            "Is the hidden checkpoint exactly "
            f"{json.dumps(top_answer, ensure_ascii=False)}? "
            "Answer only YES or NO."
        )
        direct = self._score_question("direct", direct_text, {top_answer})
        if direct is not None:
            plans.append(direct)

        # Greedily build a probability-balanced subset: a practical binary search
        # even when candidate priors are non-uniform.
        balanced: set[str] = set()
        mass = 0.0
        for answer, probability in sorted(
            self.probabilities.items(), key=lambda item: item[1], reverse=True
        ):
            if len(balanced) >= self.config.max_candidates_in_question:
                break
            if abs((mass + probability) - 0.5) < abs(mass - 0.5):
                balanced.add(answer)
                mass += probability

        if balanced and len(balanced) < len(candidates):
            complement = set(candidates) - balanced
            listed = min(
                (balanced, complement),
                key=lambda group: sum(len(answer) for answer in group),
            )
            rendered = json.dumps(sorted(listed), ensure_ascii=False)
            text = (
                "Is the hidden checkpoint one of the following exact candidates: "
                f"{rendered}? Answer only YES or NO."
            )
            binary = self._score_question("binary", text, listed)
            if binary is not None:
                plans.append(binary)

        # Candidate metadata often produces shorter questions than enumerating a
        # list. Every non-trivial attribute split competes on the same value score.
        attribute_values: dict[str, set[str]] = {}
        raw_values: dict[tuple[str, str], Any] = {}
        for candidate in self.candidates.values():
            for key, value in candidate.attributes.items():
                encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
                attribute_values.setdefault(str(key), set()).add(encoded)
                raw_values[(str(key), encoded)] = value

        for key, encoded_values in attribute_values.items():
            for encoded in encoded_values:
                yes_answers = {
                    candidate.answer
                    for candidate in self.candidates.values()
                    if key in candidate.attributes
                    and json.dumps(
                        candidate.attributes[key], sort_keys=True, ensure_ascii=False
                    )
                    == encoded
                }
                value = raw_values[(key, encoded)]
                text = (
                    "Does the hidden checkpoint have attribute "
                    f"{json.dumps(key, ensure_ascii=False)} equal to "
                    f"{json.dumps(value, ensure_ascii=False)}? "
                    "Answer only YES or NO."
                )
                attribute = self._score_question("attribute", text, yes_answers)
                if attribute is not None:
                    plans.append(attribute)
        return plans

    def _score_question(
        self, kind: str, text: str, yes_answers: set[str]
    ) -> PlannedQuestion | None:
        all_answers = set(self.probabilities)
        if not yes_answers or yes_answers == all_answers:
            return None

        signature = kind + ":" + "|".join(sorted(yes_answers))
        if signature in self.asked_signatures:
            return None

        accuracy = self.config.judge_accuracy
        entropy_before = _entropy(list(self.probabilities.values()))
        response_stats: list[tuple[float, float, float]] = []

        for observed_yes in (True, False):
            weighted: dict[str, float] = {}
            for answer, prior in self.probabilities.items():
                predicted_yes = answer in yes_answers
                likelihood = (
                    accuracy if predicted_yes == observed_yes else 1.0 - accuracy
                )
                weighted[answer] = prior * likelihood
            response_probability = sum(weighted.values())
            if response_probability == 0:
                continue
            posterior = [value / response_probability for value in weighted.values()]
            response_stats.append(
                (
                    response_probability,
                    _entropy(posterior),
                    max(posterior),
                )
            )

        expected_entropy = sum(prob * entropy for prob, entropy, _ in response_stats)
        expected_best = sum(prob * best for prob, _, best in response_stats)
        information_gain = max(0.0, entropy_before - expected_entropy)
        current_best = max(self.probabilities.values())
        success_gain = max(0.0, expected_best - current_best)
        cost = estimate_question_cost(text, self.config)
        value = information_gain / cost if cost > 0 else math.inf
        return PlannedQuestion(
            text=text,
            kind=kind,
            yes_answers=frozenset(yes_answers),
            signature=signature,
            information_gain_bits=information_gain,
            expected_cost_usdc=cost,
            value_per_usdc=value,
            expected_success_gain=success_gain,
        )

    @staticmethod
    def _parse_yes_no(raw_answer: str) -> bool | None:
        normalized = raw_answer.strip().lower()
        normalized = re.sub(r"^[\s\"'`]+|[\s\"'`.!?。！？]+$", "", normalized)
        yes_values = {"yes", "y", "true", "是", "对", "正确", "可以"}
        no_values = {"no", "n", "false", "否", "不", "不是", "错误"}
        if normalized in yes_values:
            return True
        if normalized in no_values:
            return False
        if re.match(r"^yes\b", normalized):
            return True
        if re.match(r"^no\b", normalized):
            return False
        return None
