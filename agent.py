"""Executable AGP agent with local and HTTP judge adapters."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urljoin

try:
    import requests
except ModuleNotFoundError:  # Local simulation has no third-party dependency.
    requests = None  # type: ignore[assignment]

from agp_client import (
    AGPBudgetExceededError,
    AGPClient,
    AGPQuestionResponse as JudgeReply,
    AGPStatus,
    AGPSubmissionResponse as SubmissionResult,
    MockAGPClient,
)
from config import AgentConfig
from policy_engine import ActionAuditLog, BudgetController, PolicyEngine
from race_strategy import RaceStrategyLayer
from strategy import (
    Candidate,
    InformationValueStrategy,
    PlannedQuestion,
    estimate_question_cost,
    estimate_tokens,
)


class JudgeClient(Protocol):
    def ask(
        self, question: str, metadata: Mapping[str, Any] | None = None
    ) -> JudgeReply: ...

    def submit(self, answer: str) -> SubmissionResult: ...


class PaymentGate(Protocol):
    """Small Latch boundary: authorize before spending and settle afterward."""

    def authorize(self, amount_usdc: float, purpose: str) -> str | None: ...

    def settle(self, authorization_id: str | None, actual_usdc: float) -> None: ...


class NullPaymentGate:
    """Default for local runs or AGP accounts with built-in budget control."""

    def authorize(self, amount_usdc: float, purpose: str) -> str | None:
        del amount_usdc, purpose
        return None

    def settle(self, authorization_id: str | None, actual_usdc: float) -> None:
        del authorization_id, actual_usdc


class HttpLatchPaymentGate:
    """Generic adapter reserved for the final Latch API contract."""

    def __init__(self, config: AgentConfig):
        if requests is None:
            raise RuntimeError("remote mode requires: pip install -r requirements.txt")
        self.config = config
        self.session = requests.Session()
        if config.latch_api_key:
            self.session.headers["Authorization"] = f"Bearer {config.latch_api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def authorize(self, amount_usdc: float, purpose: str) -> str | None:
        response = self.session.post(
            _join_url(
                self.config.latch_api_base_url, self.config.latch_authorize_path
            ),
            json={"amount_usdc": round(amount_usdc, 8), "purpose": purpose},
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        data = _json_object(response)
        if data.get("approved") is False:
            raise RuntimeError(f"Latch rejected payment: {data.get('message', data)}")
        value = data.get("authorization_id") or data.get("id")
        return str(value) if value is not None else None

    def settle(self, authorization_id: str | None, actual_usdc: float) -> None:
        if authorization_id is None:
            return
        response = self.session.post(
            _join_url(self.config.latch_api_base_url, self.config.latch_settle_path),
            json={
                "authorization_id": authorization_id,
                "actual_amount_usdc": round(actual_usdc, 8),
            },
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()


class HttpAGPJudgeClient:
    """Minimal HTTP adapter; customize only this class when AGP schema changes."""

    def __init__(self, config: AgentConfig):
        if requests is None:
            raise RuntimeError("remote mode requires: pip install -r requirements.txt")
        if not config.agp_api_base_url:
            raise ValueError("AGP_API_BASE_URL is required in remote mode")
        self.config = config
        self.session = requests.Session()
        if config.agp_api_key:
            self.session.headers["Authorization"] = f"Bearer {config.agp_api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def ask(
        self, question: str, metadata: Mapping[str, Any] | None = None
    ) -> JudgeReply:
        del metadata  # Internal partitions must never be disclosed to the remote API.
        payload: dict[str, Any] = {
            "question": question,
            "max_tokens": self.config.max_judge_response_tokens,
        }
        if self.config.agp_session_id:
            payload["session_id"] = self.config.agp_session_id
        response = self.session.post(
            _join_url(self.config.agp_api_base_url, self.config.agp_question_path),
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        data = _json_object(response)
        text = _first_value(data, "answer", "reply", "message", "content")
        if text is None:
            raise RuntimeError(f"AGP response has no answer field: {data}")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return JudgeReply(
            text=str(text),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            cost_usdc=_optional_float(data.get("cost_usdc")),
            raw=data,
        )

    def submit(self, answer: str) -> SubmissionResult:
        payload: dict[str, Any] = {"answer": answer}
        if self.config.agp_session_id:
            payload["session_id"] = self.config.agp_session_id
        response = self.session.post(
            _join_url(self.config.agp_api_base_url, self.config.agp_submit_path),
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        data = _json_object(response)
        accepted_value = data.get("accepted")
        accepted = accepted_value if isinstance(accepted_value, bool) else None
        message = str(
            _first_value(data, "message", "status", "result") or "submitted"
        )
        return SubmissionResult(
            accepted=accepted,
            message=message,
            cost_usdc=_optional_float(data.get("cost_usdc")),
            raw=data,
        )


class LocalJudgeClient:
    """Deterministic offline judge used to test the full decision loop."""

    def __init__(
        self,
        hidden_answer: str,
        config: AgentConfig,
        accuracy: float = 1.0,
        seed: int = 7,
    ):
        self.hidden_answer = hidden_answer
        self.config = config
        self.accuracy = accuracy
        self.random = random.Random(seed)

    def ask(
        self, question: str, metadata: Mapping[str, Any] | None = None
    ) -> JudgeReply:
        if metadata is None or "yes_answers" not in metadata:
            raise ValueError("local judge requires the planned candidate partition")
        yes_answers = set(metadata["yes_answers"])
        answer = self.hidden_answer in yes_answers
        if self.random.random() > self.accuracy:
            answer = not answer
        text = "YES" if answer else "NO"
        prompt_tokens = estimate_tokens(question)
        completion_tokens = 1
        cost = (
            self.config.question_base_cost_usdc
            + prompt_tokens * self.config.input_usdc_per_1k_tokens / 1000.0
            + completion_tokens
            * self.config.output_usdc_per_1k_tokens
            / 1000.0
        )
        return JudgeReply(text, prompt_tokens, completion_tokens, cost)

    def submit(self, answer: str) -> SubmissionResult:
        accepted = answer == self.hidden_answer
        message = "checkpoint found" if accepted else "wrong checkpoint"
        return SubmissionResult(accepted=accepted, message=message, cost_usdc=0.0)


class JudgeClientAGPAdapter(AGPClient):
    """Compatibility bridge for the existing HTTP and local judge classes."""

    def __init__(self, judge: JudgeClient, config: AgentConfig):
        self.judge = judge
        self.config = config
        self.remaining_budget_usdc = config.budget_usdc
        self.questions_asked = 0
        self.submitted_answer: str | None = None
        self.accepted: bool | None = None
        self._question_metadata: Mapping[str, Any] | None = None

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

    def prepare_question(self, metadata: Mapping[str, Any]) -> None:
        """Pass local-only context without adding it to the AGP interface."""

        self._question_metadata = metadata

    def ask_question(self, question: str) -> JudgeReply:
        estimated_cost = estimate_question_cost(question, self.config)
        if estimated_cost > self.remaining_budget_usdc + 1e-12:
            raise AGPBudgetExceededError("legacy AGP adapter has insufficient budget")
        reply = self.judge.ask(question, self._question_metadata)
        self._question_metadata = None
        actual_cost = _question_response_cost(question, reply, self.config)
        self.remaining_budget_usdc = max(
            0.0, self.remaining_budget_usdc - actual_cost
        )
        self.questions_asked += 1
        return reply

    def submit_answer(self, answer: str) -> SubmissionResult:
        result = self.judge.submit(answer)
        cost = result.cost_usdc or 0.0
        if cost > self.remaining_budget_usdc + 1e-12:
            raise AGPBudgetExceededError("legacy AGP adapter cannot fund submission")
        self.remaining_budget_usdc -= cost
        self.submitted_answer = answer
        self.accepted = result.accepted
        return result


@dataclass(slots=True)
class BudgetLedger:
    config: AgentConfig
    spent_usdc: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    questions: int = 0

    @property
    def remaining_usdc(self) -> float:
        return max(0.0, self.config.budget_usdc - self.spent_usdc)

    def record_question(self, question: str, reply: JudgeReply) -> float:
        prompt_tokens = reply.prompt_tokens or estimate_tokens(question)
        completion_tokens = reply.completion_tokens or estimate_tokens(reply.text)
        actual_cost = reply.cost_usdc
        if actual_cost is None:
            actual_cost = (
                self.config.question_base_cost_usdc
                + prompt_tokens
                * self.config.input_usdc_per_1k_tokens
                / 1000.0
                + completion_tokens
                * self.config.output_usdc_per_1k_tokens
                / 1000.0
            )
        self.spent_usdc += actual_cost
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.questions += 1
        return actual_cost

    def record_submission(self, result: SubmissionResult) -> float:
        cost = result.cost_usdc or 0.0
        self.spent_usdc += cost
        return cost


def run_agent(
    candidates: Sequence[Candidate],
    config: AgentConfig,
    client: AGPClient | JudgeClient,
    payments: PaymentGate,
    policy: PolicyEngine | None = None,
    race: RaceStrategyLayer | None = None,
) -> SubmissionResult:
    runtime = (
        client
        if isinstance(client, AGPClient)
        else JudgeClientAGPAdapter(client, config)
    )
    runtime_status = runtime.get_status()
    runtime_budget = min(config.budget_usdc, runtime.get_budget())
    if runtime_status.state != "running":
        raise RuntimeError(f"AGP runtime is not running: {runtime_status.state}")
    if runtime_budget <= 0:
        raise AGPBudgetExceededError("AGP runtime budget is exhausted")

    strategy = InformationValueStrategy(candidates, config)
    ledger = BudgetLedger(config)
    if race is None:
        race = RaceStrategyLayer(
            total_budget_usdc=runtime_budget,
            minimum_final_guess_budget_usdc=config.submission_reserve_usdc,
            candidate_count=len(candidates),
            base_information_efficiency=config.min_value_per_usdc,
            base_information_gain_bits=config.min_information_gain_bits,
            base_guess_confidence=config.confidence_threshold,
        )
    if policy is None:
        audit_path = (
            Path(config.policy_audit_log_path)
            if config.policy_audit_log_path
            else None
        )
        policy = PolicyEngine(
            BudgetController(
                total_budget_usdc=runtime_budget,
                submission_reserve_usdc=race.allocation.final_guess_budget_usdc,
            ),
            min_information_efficiency=config.min_value_per_usdc,
            min_information_gain_bits=config.min_information_gain_bits,
            min_guess_confidence=config.confidence_threshold,
            audit_log=ActionAuditLog(audit_path),
        )

    print(
        f"Starting with {len(candidates)} candidates and "
        f"{runtime_budget:.6f} USDC. Runtime={runtime_status.state}."
    )
    print(
        "Race budget: "
        f"exploration={race.allocation.exploration_budget_usdc:.6f}, "
        f"optimization={race.allocation.optimization_budget_usdc:.6f}, "
        f"final_guess={race.allocation.final_guess_budget_usdc:.6f} USDC."
    )
    while True:
        guess, confidence = strategy.best_guess
        race_directive = race.directive(confidence, ledger.questions)
        runtime_searchable = max(
            0.0,
            runtime.get_budget() - race.allocation.final_guess_budget_usdc,
        )
        runtime_cost_limit = min(
            race_directive.max_question_cost_usdc, runtime_searchable
        )
        race_directive = replace(
            race_directive,
            max_question_cost_usdc=runtime_cost_limit,
            can_search=runtime_cost_limit > 1e-12,
        )
        decision = strategy.decide(
            ledger.spent_usdc,
            ledger.questions,
            confidence_threshold=race_directive.required_guess_confidence,
            min_information_gain_bits=race_directive.min_information_gain_bits,
            min_value_per_usdc=race_directive.min_information_efficiency,
            max_question_cost_usdc=race_directive.max_question_cost_usdc,
        )
        print(
            f"State: questions={ledger.questions}, spent={ledger.spent_usdc:.6f}, "
            f"best={guess!r}, confidence={confidence:.2%}"
        )
        print(
            f"Race [{race_directive.phase.value}]: {race_directive.reason}; "
            f"exploration_left={race_directive.exploration_remaining_usdc:.6f}, "
            f"optimization_left={race_directive.optimization_remaining_usdc:.6f}"
        )
        if decision.action == "guess":
            print(f"Stop: {decision.reason}.")
            break

        plan = decision.question
        if plan is None:  # Defensive guard for custom strategies.
            raise RuntimeError("ask decision did not include a question")
        print(
            f"Candidate question [{plan.kind}]: {plan.text}\n"
            f"  IG={plan.information_gain_bits:.3f} bits, "
            f"value={plan.value_per_usdc:.1f} bits/USDC, "
            f"estimated_cost={plan.expected_cost_usdc:.6f}"
        )
        policy_verdict = policy.review_question(
            plan,
            min_information_efficiency=race_directive.min_information_efficiency,
            min_information_gain_bits=race_directive.min_information_gain_bits,
            max_allowed_cost_usdc=race_directive.max_question_cost_usdc,
            action_type=f"question:{race_directive.phase.value}",
        )
        print(policy_verdict.audit_entry.render())
        if not policy_verdict.allowed:
            print("Policy denied the question; stopping search.")
            break

        authorization = payments.authorize(
            plan.expected_cost_usdc, f"AGP judge question {ledger.questions + 1}"
        )
        if isinstance(runtime, JudgeClientAGPAdapter):
            runtime.prepare_question(
                {"kind": plan.kind, "yes_answers": sorted(plan.yes_answers)}
            )
        try:
            reply = runtime.ask_question(plan.text)
        except AGPBudgetExceededError as error:
            payments.settle(authorization, 0.0)
            print(f"AGP Client denied the question: {error}")
            break
        actual_cost = ledger.record_question(plan.text, reply)
        policy.record_spend(actual_cost)
        race.record_spend(actual_cost, race_directive.phase)
        payments.settle(authorization, actual_cost)
        outcome = strategy.observe(plan, reply.text)
        print(
            f"AGP Judge: {reply.text!r} (parsed={outcome}, "
            f"cost={actual_cost:.6f}, runtime_budget={runtime.get_budget():.6f})"
        )

    final_answer, confidence = strategy.best_guess
    final_directive = race.directive(confidence, ledger.questions)
    guess_verdict = policy.review_guess(
        final_answer,
        confidence,
        required_confidence=final_directive.required_guess_confidence,
        action_type=f"guess:{final_directive.phase.value}",
    )
    print(guess_verdict.audit_entry.render())
    if not guess_verdict.allowed:
        result = SubmissionResult(
            accepted=None,
            message=f"submission denied by policy: {guess_verdict.reason}",
            raw={"policy_denied": True},
        )
        print(
            f"Did not submit {final_answer!r} at {confidence:.2%} confidence. "
            f"Result: {result.message}; total_spent={ledger.spent_usdc:.6f} USDC."
        )
        return result

    authorization = payments.authorize(
        min(race.allocation.final_guess_budget_usdc, ledger.remaining_usdc),
        "AGP final submission",
    )
    try:
        result = runtime.submit_answer(final_answer)
    except AGPBudgetExceededError as error:
        payments.settle(authorization, 0.0)
        result = SubmissionResult(
            accepted=None,
            message=f"submission denied by AGP runtime: {error}",
            raw={"runtime_denied": True},
        )
        print(result.message)
        return result
    submission_cost = ledger.record_submission(result)
    policy.record_spend(submission_cost)
    payments.settle(authorization, submission_cost)
    print(
        f"Submitted {final_answer!r} at {confidence:.2%} confidence. "
        f"Result: {result.message}; total_spent={ledger.spent_usdc:.6f} USDC; "
        f"tokens={ledger.prompt_tokens + ledger.completion_tokens}."
    )
    if ledger.spent_usdc > config.budget_usdc + 1e-12:
        print(
            "Warning: the remote service reported a cost above the configured budget. "
            "Update the configured price ceiling before the next run.",
            file=sys.stderr,
        )
    return result


def load_candidates(path: Path | None) -> list[Candidate]:
    if path is None:
        return _demo_candidates()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("candidate file must contain a JSON array")
    candidates: list[Candidate] = []
    for item in raw:
        if isinstance(item, str):
            candidates.append(Candidate(answer=item))
        elif isinstance(item, dict):
            if "answer" not in item:
                raise ValueError("each candidate object needs an 'answer' field")
            attributes = item.get("attributes", {})
            if not isinstance(attributes, dict):
                raise ValueError("candidate 'attributes' must be a JSON object")
            candidates.append(
                Candidate(
                    answer=str(item["answer"]),
                    category=str(item.get("category", "uncategorized")),
                    prior=float(item.get("prior", 1.0)),
                    attributes=attributes,
                )
            )
        else:
            raise ValueError("candidate entries must be strings or objects")
    return candidates


def _demo_candidates() -> list[Candidate]:
    return [
        Candidate("checkpoint-alpha", "city", attributes={"region": "north"}),
        Candidate("checkpoint-bravo", "city", attributes={"region": "south"}),
        Candidate("checkpoint-charlie", "city", attributes={"region": "west"}),
        Candidate("checkpoint-delta", "nature", attributes={"region": "north"}),
        Candidate("checkpoint-echo", "nature", attributes={"region": "south"}),
        Candidate("checkpoint-foxtrot", "nature", attributes={"region": "west"}),
        Candidate("checkpoint-golf", "historic", attributes={"region": "north"}),
        Candidate("checkpoint-hotel", "historic", attributes={"region": "south"}),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cost-aware AGP checkpoint agent")
    parser.add_argument(
        "--mode", choices=("local", "remote"), help="override AGP_MODE"
    )
    parser.add_argument(
        "--candidates", type=Path, help="JSON candidate list (required for real games)"
    )
    parser.add_argument(
        "--hidden",
        default="checkpoint-delta",
        help="hidden answer used only by local mode",
    )
    parser.add_argument("--budget", type=float, help="override AGP_BUDGET_USDC")
    parser.add_argument(
        "--confidence", type=float, help="override AGP_CONFIDENCE_THRESHOLD"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AgentConfig.from_env()
    updates: dict[str, Any] = {}
    if args.mode is not None:
        updates["mode"] = args.mode
    if args.budget is not None:
        updates["budget_usdc"] = args.budget
    if args.confidence is not None:
        updates["confidence_threshold"] = args.confidence
    if updates:
        config = replace(config, **updates)
        config.validate()

    candidates = load_candidates(args.candidates)
    answers = {candidate.answer for candidate in candidates}
    if config.mode == "local":
        if args.hidden not in answers:
            raise ValueError("--hidden must be present in the candidate list")
        hidden_candidate = next(
            candidate for candidate in candidates if candidate.answer == args.hidden
        )
        client: AGPClient | JudgeClient = MockAGPClient(
            hidden_answer=hidden_candidate.answer,
            hidden_category=hidden_candidate.category,
            hidden_attributes=hidden_candidate.attributes,
            initial_budget_usdc=config.budget_usdc,
            question_cost_usdc=lambda question: _local_question_cost(
                question, config
            ),
        )
    else:
        if args.candidates is None:
            raise ValueError("--candidates is required in remote mode")
        client = JudgeClientAGPAdapter(HttpAGPJudgeClient(config), config)

    payments: PaymentGate
    if config.latch_api_base_url:
        payments = HttpLatchPaymentGate(config)
    else:
        payments = NullPaymentGate()

    result = run_agent(candidates, config, client, payments)
    if result.raw and (
        result.raw.get("policy_denied") is True
        or result.raw.get("runtime_denied") is True
    ):
        return 3
    return 0 if result.accepted is not False else 2


def _join_url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _local_question_cost(question: str, config: AgentConfig) -> float:
    return (
        config.question_base_cost_usdc
        + estimate_tokens(question) * config.input_usdc_per_1k_tokens / 1000.0
        + config.output_usdc_per_1k_tokens / 1000.0
    )


def _question_response_cost(
    question: str, reply: JudgeReply, config: AgentConfig
) -> float:
    if reply.cost_usdc is not None:
        return reply.cost_usdc
    prompt_tokens = reply.prompt_tokens or estimate_tokens(question)
    completion_tokens = reply.completion_tokens or estimate_tokens(reply.text)
    return (
        config.question_base_cost_usdc
        + prompt_tokens * config.input_usdc_per_1k_tokens / 1000.0
        + completion_tokens * config.output_usdc_per_1k_tokens / 1000.0
    )


def _json_object(response: Any) -> dict[str, Any]:
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("API response must be a JSON object")
    return data


def _first_value(data: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
