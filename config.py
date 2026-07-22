"""Environment-based configuration for the AGP agent."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """All tunable values live here so strategy code stays deterministic."""

    # Runtime and AGP transport. Endpoint paths are intentionally configurable
    # because the final AGP contract can be wired in without changing strategy.py.
    mode: str = "local"
    agp_api_base_url: str = ""
    agp_api_key: str = ""
    agp_question_path: str = "/v1/judge/questions"
    agp_submit_path: str = "/v1/judge/submissions"
    agp_session_id: str = ""
    request_timeout_seconds: float = 30.0

    # Budget model. Prices are expressed in USDC, assuming 1 USDC ~= 1 USD.
    budget_usdc: float = 0.25
    submission_reserve_usdc: float = 0.005
    question_base_cost_usdc: float = 0.001
    input_usdc_per_1k_tokens: float = 0.001
    output_usdc_per_1k_tokens: float = 0.003
    max_judge_response_tokens: int = 16

    # Decision policy.
    confidence_threshold: float = 0.92
    category_confidence_threshold: float = 0.85
    judge_accuracy: float = 0.98
    max_questions: int = 12
    max_category_questions: int = 3
    min_information_gain_bits: float = 0.03
    min_value_per_usdc: float = 10.0
    min_expected_success_gain: float = 0.0
    max_candidates_in_question: int = 40

    # Optional Latch-compatible budget authorization service. The generic HTTP
    # adapter can later be replaced with the official SDK implementation.
    latch_api_base_url: str = ""
    latch_api_key: str = ""
    latch_authorize_path: str = "/v1/payments/authorize"
    latch_settle_path: str = "/v1/payments/settle"
    policy_audit_log_path: str = ""

    @classmethod
    def from_env(cls) -> "AgentConfig":
        config = cls(
            mode=os.getenv("AGP_MODE", "local").lower(),
            agp_api_base_url=os.getenv("AGP_API_BASE_URL", ""),
            agp_api_key=os.getenv("AGP_API_KEY", ""),
            agp_question_path=os.getenv(
                "AGP_QUESTION_PATH", "/v1/judge/questions"
            ),
            agp_submit_path=os.getenv(
                "AGP_SUBMIT_PATH", "/v1/judge/submissions"
            ),
            agp_session_id=os.getenv("AGP_SESSION_ID", ""),
            request_timeout_seconds=_env_float("AGP_TIMEOUT_SECONDS", 30.0),
            budget_usdc=_env_float("AGP_BUDGET_USDC", 0.25),
            submission_reserve_usdc=_env_float(
                "AGP_SUBMISSION_RESERVE_USDC", 0.005
            ),
            question_base_cost_usdc=_env_float(
                "AGP_QUESTION_BASE_COST_USDC", 0.001
            ),
            input_usdc_per_1k_tokens=_env_float(
                "AGP_INPUT_USDC_PER_1K_TOKENS", 0.001
            ),
            output_usdc_per_1k_tokens=_env_float(
                "AGP_OUTPUT_USDC_PER_1K_TOKENS", 0.003
            ),
            max_judge_response_tokens=_env_int(
                "AGP_MAX_JUDGE_RESPONSE_TOKENS", 16
            ),
            confidence_threshold=_env_float(
                "AGP_CONFIDENCE_THRESHOLD", 0.92
            ),
            category_confidence_threshold=_env_float(
                "AGP_CATEGORY_CONFIDENCE_THRESHOLD", 0.85
            ),
            judge_accuracy=_env_float("AGP_JUDGE_ACCURACY", 0.98),
            max_questions=_env_int("AGP_MAX_QUESTIONS", 12),
            max_category_questions=_env_int(
                "AGP_MAX_CATEGORY_QUESTIONS", 3
            ),
            min_information_gain_bits=_env_float(
                "AGP_MIN_INFORMATION_GAIN_BITS", 0.03
            ),
            min_value_per_usdc=_env_float(
                "AGP_MIN_VALUE_PER_USDC", 10.0
            ),
            min_expected_success_gain=_env_float(
                "AGP_MIN_EXPECTED_SUCCESS_GAIN", 0.0
            ),
            max_candidates_in_question=_env_int(
                "AGP_MAX_CANDIDATES_IN_QUESTION", 40
            ),
            latch_api_base_url=os.getenv("LATCH_API_BASE_URL", ""),
            latch_api_key=os.getenv("LATCH_API_KEY", ""),
            latch_authorize_path=os.getenv(
                "LATCH_AUTHORIZE_PATH", "/v1/payments/authorize"
            ),
            latch_settle_path=os.getenv(
                "LATCH_SETTLE_PATH", "/v1/payments/settle"
            ),
            policy_audit_log_path=os.getenv("AGP_POLICY_AUDIT_LOG", ""),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in {"local", "remote"}:
            raise ValueError("AGP_MODE must be 'local' or 'remote'")
        if self.budget_usdc <= 0:
            raise ValueError("AGP_BUDGET_USDC must be positive")
        if not 0.5 < self.judge_accuracy <= 1.0:
            raise ValueError("AGP_JUDGE_ACCURACY must be in (0.5, 1.0]")
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("category_confidence_threshold", self.category_confidence_threshold),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.max_questions < 0 or self.max_category_questions < 0:
            raise ValueError("question limits cannot be negative")
        if self.max_candidates_in_question < 1:
            raise ValueError("max_candidates_in_question must be positive")
