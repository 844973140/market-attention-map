"""Persistent race experience and a memory-enhanced adaptive strategy."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from adaptive_strategy import (
    AdaptiveDirective,
    AdaptiveMode,
    AdaptiveRaceState,
    AdaptiveRaceStrategy,
    FinishAction,
)
from config import AgentConfig
from simulation import (
    AGPSimulator,
    EpisodeResult,
    SimulationDirective,
    SimulationReport,
    StrategyVariant,
)


@dataclass(slots=True)
class QuestionPerformance:
    question_type: str
    count: int = 0
    total_information_gain: float = 0.0
    total_cost_usdc: float = 0.0
    successes: int = 0

    @property
    def average_gain(self) -> float:
        return self.total_information_gain / self.count if self.count else 0.0

    @property
    def average_cost(self) -> float:
        return self.total_cost_usdc / self.count if self.count else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.count if self.count else 0.0

    @property
    def information_efficiency(self) -> float:
        return self.average_gain / self.average_cost if self.average_cost > 0 else 0.0

    def record(self, information_gain: float, cost_usdc: float, successful: bool) -> None:
        if information_gain < 0 or not math.isfinite(information_gain):
            raise ValueError("information gain must be finite and non-negative")
        if cost_usdc < 0 or not math.isfinite(cost_usdc):
            raise ValueError("question cost must be finite and non-negative")
        self.count += 1
        self.total_information_gain += information_gain
        self.total_cost_usdc += cost_usdc
        self.successes += int(successful)


@dataclass(slots=True)
class DecisionPerformance:
    action: str
    count: int = 0
    successes: int = 0
    total_reward: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.count if self.count else 0.0

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.count if self.count else 0.0

    def record(self, successful: bool, reward: float) -> None:
        if not math.isfinite(reward):
            raise ValueError("decision reward must be finite")
        self.count += 1
        self.successes += int(successful)
        self.total_reward += reward


@dataclass(slots=True)
class OutcomeSummary:
    episodes: int = 0
    wins: int = 0
    total_cost_usdc: float = 0.0
    total_questions: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)

    def record(self, result: EpisodeResult) -> None:
        self.episodes += 1
        self.wins += int(result.won)
        self.total_cost_usdc += result.cost_usdc
        self.total_questions += result.questions
        if result.failure_reason:
            self.failure_reasons[result.failure_reason] = (
                self.failure_reasons.get(result.failure_reason, 0) + 1
            )


@dataclass(frozen=True, slots=True)
class BudgetRecommendation:
    exploration_budget_usdc: float
    optimization_budget_usdc: float
    final_guess_budget_usdc: float


class StrategyMemory:
    """Stores question and decision outcomes in an inspectable JSON schema."""

    VERSION = 1

    def __init__(self, path: Path | str = "memory.json"):
        self.path = Path(path)
        self.question_performance: dict[str, QuestionPerformance] = {}
        self.decision_memory: dict[str, dict[str, DecisionPerformance]] = {}
        self.outcomes = OutcomeSummary()

    def record_question(
        self,
        question_type: str,
        information_gain: float,
        cost_usdc: float,
        successful: bool,
    ) -> None:
        normalized = question_type.strip().lower()
        if not normalized:
            raise ValueError("question type cannot be empty")
        performance = self.question_performance.setdefault(
            normalized, QuestionPerformance(question_type=normalized)
        )
        performance.record(information_gain, cost_usdc, successful)

    def record_decision(
        self,
        *,
        confidence: float,
        remaining_budget_ratio: float,
        candidate_count: int,
        action: str,
        successful: bool,
        reward: float | None = None,
    ) -> None:
        state_key = self.state_key(
            confidence, remaining_budget_ratio, candidate_count
        )
        normalized_action = action.strip().lower()
        if not normalized_action:
            raise ValueError("action cannot be empty")
        state_actions = self.decision_memory.setdefault(state_key, {})
        performance = state_actions.setdefault(
            normalized_action, DecisionPerformance(action=normalized_action)
        )
        effective_reward = (
            reward if reward is not None else (1.0 if successful else 0.0)
        )
        performance.record(successful, effective_reward)

    def record_episode(
        self, result: EpisodeResult, total_budget_usdc: float = 0.01
    ) -> None:
        if total_budget_usdc <= 0 or not math.isfinite(total_budget_usdc):
            raise ValueError("total episode budget must be finite and positive")
        self.outcomes.record(result)
        average_gain = (
            result.information_gain_bits / result.questions
            if result.questions
            else 0.0
        )
        average_cost = result.cost_usdc / result.questions if result.questions else 0.0
        if result.questions:
            self.record_question(
                "adaptive_question",
                average_gain,
                average_cost,
                result.won,
            )
        action = "submit" if result.submitted else "continue"
        budget_ratio = max(
            0.0,
            min(1.0, result.remaining_budget_usdc / total_budget_usdc),
        )
        reward = int(result.won) - result.cost_usdc * 25.0
        self.record_decision(
            confidence=result.confidence,
            remaining_budget_ratio=budget_ratio,
            candidate_count=1 if result.confidence >= 0.85 else 4,
            action=action,
            successful=result.won,
            reward=reward,
        )

    def get_question_performance(
        self, question_type: str
    ) -> QuestionPerformance | None:
        return self.question_performance.get(question_type.strip().lower())

    def best_action(
        self,
        *,
        confidence: float,
        remaining_budget_ratio: float,
        candidate_count: int,
        minimum_samples: int = 5,
    ) -> str | None:
        key = self.state_key(confidence, remaining_budget_ratio, candidate_count)
        actions = self.decision_memory.get(key, {})
        eligible = [value for value in actions.values() if value.count >= minimum_samples]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda value: (value.average_reward, value.success_rate),
        ).action

    def recommended_submission_threshold(
        self,
        default_threshold: float,
        *,
        minimum_samples: int = 20,
        minimum_success_rate: float = 0.85,
    ) -> float:
        candidates: list[float] = []
        for state_key, actions in self.decision_memory.items():
            submit = actions.get("submit")
            if (
                submit is None
                or submit.count < minimum_samples
                or submit.success_rate < minimum_success_rate
            ):
                continue
            confidence_bucket = state_key.split("|")[0]
            lower_bound = {
                "confidence<50": 0.50,
                "confidence50-70": 0.50,
                "confidence70-85": 0.70,
                "confidence85-90": 0.85,
                "confidence90+": 0.90,
            }[confidence_bucket]
            # Use a small safety margin above the observed bucket boundary.
            candidates.append(lower_bound + 0.02)
        if not candidates:
            return default_threshold
        return max(0.70, min(default_threshold, min(candidates)))

    def recommended_information_gain_threshold(
        self, default_threshold: float, minimum_samples: int = 10
    ) -> float:
        records = [
            performance
            for performance in self.question_performance.values()
            if performance.count >= minimum_samples
        ]
        if not records:
            return default_threshold
        total_samples = sum(record.count for record in records)
        success_rate = sum(
            record.success_rate * record.count for record in records
        ) / total_samples
        if success_rate >= 0.85:
            return default_threshold * 0.90
        if success_rate < 0.60:
            return default_threshold * 1.20
        return default_threshold

    def recommended_budget_allocation(
        self,
        total_budget_usdc: float,
        default_final_guess_budget_usdc: float,
    ) -> BudgetRecommendation:
        final_budget = default_final_guess_budget_usdc
        budget_failures = sum(
            count
            for reason, count in self.outcomes.failure_reasons.items()
            if "budget" in reason
        )
        wrong_answers = self.outcomes.failure_reasons.get("wrong answer", 0)
        if self.outcomes.episodes >= 20 and budget_failures > wrong_answers * 2:
            final_budget *= 0.90
        elif self.outcomes.episodes >= 20 and wrong_answers > budget_failures:
            final_budget *= 1.05
        final_budget = min(total_budget_usdc, max(0.0, final_budget))
        searchable = total_budget_usdc - final_budget
        return BudgetRecommendation(
            exploration_budget_usdc=searchable * 0.40,
            optimization_budget_usdc=searchable * 0.60,
            final_guess_budget_usdc=final_budget,
        )

    @staticmethod
    def state_key(
        confidence: float, remaining_budget_ratio: float, candidate_count: int
    ) -> str:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= remaining_budget_ratio <= 1.0:
            raise ValueError("remaining budget ratio must be in [0, 1]")
        if candidate_count < 1:
            raise ValueError("candidate count must be positive")
        if confidence < 0.50:
            confidence_band = "confidence<50"
        elif confidence < 0.70:
            confidence_band = "confidence50-70"
        elif confidence < 0.85:
            confidence_band = "confidence70-85"
        elif confidence < 0.90:
            confidence_band = "confidence85-90"
        else:
            confidence_band = "confidence90+"
        if remaining_budget_ratio > 0.50:
            budget_band = "budget>50"
        elif remaining_budget_ratio > 0.25:
            budget_band = "budget25-50"
        else:
            budget_band = "budget<=25"
        candidate_band = "candidates<=3" if candidate_count <= 3 else "candidates>3"
        return f"{confidence_band}|{budget_band}|{candidate_band}"

    def save(self, path: Path | str | None = None) -> Path:
        destination = Path(path) if path is not None else self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "question_performance": {
                key: {
                    **asdict(value),
                    "average_gain": value.average_gain,
                    "average_cost": value.average_cost,
                    "success_rate": value.success_rate,
                    "information_efficiency": value.information_efficiency,
                }
                for key, value in self.question_performance.items()
            },
            "decision_memory": {
                state: {
                    action: {
                        **asdict(value),
                        "success_rate": value.success_rate,
                        "average_reward": value.average_reward,
                    }
                    for action, value in actions.items()
                }
                for state, actions in self.decision_memory.items()
            },
            "outcomes": asdict(self.outcomes),
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: Path | str = "memory.json") -> "StrategyMemory":
        source = Path(path)
        memory = cls(source)
        if not source.exists():
            return memory
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("version") != cls.VERSION:
            raise ValueError("unsupported strategy memory version")
        for key, value in payload.get("question_performance", {}).items():
            memory.question_performance[key] = QuestionPerformance(
                question_type=value["question_type"],
                count=int(value["count"]),
                total_information_gain=float(value["total_information_gain"]),
                total_cost_usdc=float(value["total_cost_usdc"]),
                successes=int(value["successes"]),
            )
        for state, actions in payload.get("decision_memory", {}).items():
            memory.decision_memory[state] = {
                action: DecisionPerformance(
                    action=value["action"],
                    count=int(value["count"]),
                    successes=int(value["successes"]),
                    total_reward=float(value["total_reward"]),
                )
                for action, value in actions.items()
            }
        memory.outcomes = OutcomeSummary(**payload.get("outcomes", {}))
        return memory


class MemoryEnhancedAdaptiveStrategy:
    """Composition wrapper that adjusts Adaptive directives from experience."""

    def __init__(
        self, base_strategy: AdaptiveRaceStrategy, memory: StrategyMemory
    ):
        self.base_strategy = base_strategy
        self.memory = memory

    def decide(self, state: AdaptiveRaceState) -> AdaptiveDirective:
        directive = self.base_strategy.decide(state)
        learned_gain_threshold = self.memory.recommended_information_gain_threshold(
            directive.min_information_gain_bits
        )
        learned_submit_threshold = self.memory.recommended_submission_threshold(
            directive.required_guess_confidence
        )
        budget_ratio = min(
            1.0,
            state.remaining_budget / self.base_strategy.total_budget_usdc,
        )
        best_action = self.memory.best_action(
            confidence=state.current_confidence,
            remaining_budget_ratio=budget_ratio,
            candidate_count=state.candidate_count,
        )

        if (
            state.current_confidence >= learned_submit_threshold
            and best_action in {None, "submit"}
        ):
            return AdaptiveDirective(
                mode=AdaptiveMode.FINISH,
                finish_action=FinishAction.SUBMIT,
                min_information_efficiency=directive.min_information_efficiency,
                min_information_gain_bits=learned_gain_threshold,
                required_guess_confidence=learned_submit_threshold,
                max_question_cost_usdc=0.0,
                reason="memory shows submission is reliable in this state",
            )

        max_cost = directive.max_question_cost_usdc
        observed = self.memory.get_question_performance("adaptive_question")
        if observed is not None and observed.count >= 10 and observed.average_cost > 0:
            max_cost = min(max_cost, observed.average_cost * 1.20)
        return replace(
            directive,
            min_information_gain_bits=learned_gain_threshold,
            required_guess_confidence=learned_submit_threshold,
            max_question_cost_usdc=max_cost,
            reason=directive.reason + "; adjusted from strategy memory",
        )


class _MemoryAwareAGPSimulator(AGPSimulator):
    def __init__(self, *, memory: StrategyMemory, **kwargs: Any):
        super().__init__(**kwargs)
        self.memory = memory

    def _directive(
        self,
        strategy_variant: StrategyVariant,
        race: Any,
        adaptive: AdaptiveRaceStrategy | None,
        search: Any,
        confidence: float,
        questions: int,
        spent_usdc: float,
        final_reserve: float,
        config: AgentConfig,
    ) -> SimulationDirective:
        if strategy_variant is not StrategyVariant.ADAPTIVE:
            return super()._directive(
                strategy_variant,
                race,
                adaptive,
                search,
                confidence,
                questions,
                spent_usdc,
                final_reserve,
                config,
            )
        if adaptive is None:
            raise RuntimeError("memory strategy requires an adaptive controller")
        highest_probability = max(search.probabilities.values())
        effective_candidates = sum(
            probability >= max(0.02, highest_probability * 0.10)
            for probability in search.probabilities.values()
        )
        searchable_total = max(1e-12, self.total_budget_usdc - final_reserve)
        progress = min(
            1.0,
            max(
                questions / max(1, config.max_questions),
                spent_usdc / searchable_total,
            ),
        )
        state = AdaptiveRaceState(
            remaining_budget=max(0.0, self.total_budget_usdc - spent_usdc),
            current_confidence=confidence,
            questions_used=questions,
            candidate_count=max(1, effective_candidates),
            race_progress=progress,
        )
        directive = MemoryEnhancedAdaptiveStrategy(adaptive, self.memory).decide(state)
        runtime_searchable = max(
            0.0, self.total_budget_usdc - final_reserve - spent_usdc
        )
        return SimulationDirective(
            phase=f"MEMORY_{directive.mode.value}",
            min_information_efficiency=directive.min_information_efficiency,
            min_information_gain_bits=directive.min_information_gain_bits,
            required_guess_confidence=directive.required_guess_confidence,
            max_question_cost_usdc=min(
                directive.max_question_cost_usdc, runtime_searchable
            ),
            finish_now=directive.finish_action is FinishAction.SUBMIT,
        )


@dataclass(frozen=True, slots=True)
class MemoryBenchmarkReport:
    episodes: int
    training_episodes: int
    learned_submission_threshold: float
    adaptive: SimulationReport
    adaptive_with_memory: SimulationReport

    def render(self) -> str:
        rows = [
            "Strategy | Win rate | Avg cost | Questions | Budget efficiency",
            "---|---:|---:|---:|---:",
        ]
        for label, report in (
            ("Adaptive Strategy", self.adaptive),
            ("Adaptive + Memory", self.adaptive_with_memory),
        ):
            rows.append(
                f"{label} | {report.win_rate:.2%} | "
                f"{report.average_cost_usdc:.6f} | "
                f"{report.average_questions:.2f} | "
                f"{report.budget_efficiency:.2f} wins/USDC"
            )
        rows.append(
            f"Learned submission threshold: {self.learned_submission_threshold:.2%}"
        )
        return "\n".join(rows)


def benchmark_memory_strategy(
    episodes: int = 10_000,
    *,
    training_episodes: int | None = None,
    memory_path: Path | str | None = None,
    seed: int = 42,
    simulator_options: Mapping[str, Any] | None = None,
) -> MemoryBenchmarkReport:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    options = dict(simulator_options or {})
    options.setdefault("seed", seed)
    base_config = options.pop("config", AgentConfig())

    baseline_simulator = AGPSimulator(config=base_config, **options)
    baseline = baseline_simulator.run_simulation(
        episodes, StrategyVariant.ADAPTIVE
    )

    memory = (
        StrategyMemory.load(memory_path)
        if memory_path is not None
        else StrategyMemory()
    )
    train_count = training_episodes or min(2000, max(200, episodes // 5))
    exploratory_config = replace(base_config, confidence_threshold=0.85)
    trainer = AGPSimulator(
        config=exploratory_config,
        seed=seed + 1_000_003,
        **{key: value for key, value in options.items() if key != "seed"},
    )
    for episode in range(train_count):
        memory.record_episode(
            trainer.run_episode(episode, StrategyVariant.ADAPTIVE),
            total_budget_usdc=float(options.get("total_budget_usdc", 0.01)),
        )
    if memory_path is not None:
        memory.save(memory_path)

    learned_threshold = memory.recommended_submission_threshold(
        base_config.confidence_threshold
    )
    total_budget = float(options.get("total_budget_usdc", 0.01))
    allocation = memory.recommended_budget_allocation(
        total_budget,
        min(base_config.submission_reserve_usdc, total_budget * 0.50),
    )
    memory_config = replace(
        base_config,
        confidence_threshold=learned_threshold,
        submission_reserve_usdc=allocation.final_guess_budget_usdc,
    )
    memory_simulator = _MemoryAwareAGPSimulator(
        memory=memory,
        config=memory_config,
        **options,
    )
    memory_report = memory_simulator.run_simulation(
        episodes, StrategyVariant.ADAPTIVE
    )
    return MemoryBenchmarkReport(
        episodes=episodes,
        training_episodes=train_count,
        learned_submission_threshold=learned_threshold,
        adaptive=baseline,
        adaptive_with_memory=memory_report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AGP strategy memory benchmark")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--training-episodes", type=int)
    parser.add_argument("--memory", type=Path, default=Path("memory.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = benchmark_memory_strategy(
        args.episodes,
        training_episodes=args.training_episodes,
        memory_path=args.memory,
        seed=args.seed,
    )
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
