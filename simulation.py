"""Statistical AGP race simulation and strategy benchmarking."""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from statistics import fmean
from typing import Mapping, Sequence

from adaptive_strategy import (
    AdaptiveRaceState,
    AdaptiveRaceStrategy,
    FinishAction,
)
from agp_client import AGPBudgetExceededError, MockAGPClient
from config import AgentConfig
from policy_engine import BudgetController, PolicyEngine
from race_strategy import RacePhase, RaceStrategyLayer
from strategy import Candidate, InformationValueStrategy, estimate_tokens


class StrategyVariant(str, Enum):
    FIXED_BUDGET = "Strategy A - Fixed Budget Exploration"
    RACE = "Strategy B - Current Race Strategy"
    AGGRESSIVE = "Strategy C - Aggressive Search"
    ADAPTIVE = "Strategy D - Adaptive Race Strategy"


@dataclass(frozen=True, slots=True)
class SimulationDirective:
    phase: str
    min_information_efficiency: float
    min_information_gain_bits: float
    required_guess_confidence: float
    max_question_cost_usdc: float
    finish_now: bool = False


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    episode: int
    strategy: StrategyVariant
    hidden_checkpoint: str
    submitted_answer: str | None
    submitted: bool
    won: bool
    cost_usdc: float
    questions: int
    confidence: float
    information_gain_bits: float
    failure_reason: str | None
    remaining_budget_usdc: float


@dataclass(frozen=True, slots=True)
class SimulationReport:
    strategy: StrategyVariant
    episodes: int
    submissions: int
    wins: int
    win_rate: float
    average_cost_usdc: float
    average_questions: float
    average_confidence: float
    average_information_gain_bits: float
    budget_efficiency: float
    max_cost_usdc: float
    failure_reasons: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.value,
            "episodes": self.episodes,
            "submissions": self.submissions,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "average_cost_usdc": self.average_cost_usdc,
            "average_questions": self.average_questions,
            "average_confidence": self.average_confidence,
            "average_information_gain_bits": self.average_information_gain_bits,
            "budget_efficiency": self.budget_efficiency,
            "max_cost_usdc": self.max_cost_usdc,
            "failure_reasons": dict(self.failure_reasons),
        }

    def render(self) -> str:
        failures = (
            ", ".join(f"{reason}={count}" for reason, count in self.failure_reasons.items())
            or "none"
        )
        return "\n".join(
            (
                self.strategy.value,
                f"episodes: {self.episodes}",
                f"win rate: {self.win_rate:.2%}",
                f"average cost: {self.average_cost_usdc:.6f} USDC",
                f"average questions: {self.average_questions:.2f}",
                f"average confidence: {self.average_confidence:.2%}",
                f"average information gain: {self.average_information_gain_bits:.3f} bits",
                f"budget efficiency: {self.budget_efficiency:.2f} wins/USDC",
                f"failure reasons: {failures}",
            )
        )


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    episodes_per_strategy: int
    reports: Mapping[StrategyVariant, SimulationReport]

    def render(self) -> str:
        rows = [
            "Strategy | Win rate | Avg cost | Avg questions | Avg confidence | Budget efficiency",
            "---|---:|---:|---:|---:|---:",
        ]
        for variant, report in self.reports.items():
            rows.append(
                f"{variant.value} | {report.win_rate:.2%} | "
                f"{report.average_cost_usdc:.6f} | "
                f"{report.average_questions:.2f} | "
                f"{report.average_confidence:.2%} | "
                f"{report.budget_efficiency:.2f} wins/USDC"
            )
        return "\n".join(rows)


class _UncertainMockAGPClient(MockAGPClient):
    """Mock runtime that can flip or withhold otherwise deterministic answers."""

    def __init__(
        self,
        *,
        judge_accuracy: float,
        unknown_answer_rate: float,
        random_source: random.Random,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        self.judge_accuracy = judge_accuracy
        self.unknown_answer_rate = unknown_answer_rate
        self.random_source = random_source

    def _resolve_generated_question(self, question: str) -> str:
        answer = super()._resolve_generated_question(question)
        if answer not in {"YES", "NO"}:
            return answer
        if self.random_source.random() < self.unknown_answer_rate:
            return "UNKNOWN"
        if self.random_source.random() > self.judge_accuracy:
            return "NO" if answer == "YES" else "YES"
        return answer


class AGPSimulator:
    """Runs repeatable AGP episodes without changing production agent logic."""

    CHECKPOINT_NAMES = (
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    )

    def __init__(
        self,
        *,
        total_budget_usdc: float = 0.01,
        candidate_count: int = 8,
        judge_accuracy: float = 0.98,
        unknown_answer_rate: float = 0.01,
        cost_variation: float = 0.04,
        seed: int = 42,
        config: AgentConfig | None = None,
    ):
        if total_budget_usdc <= 0 or not math.isfinite(total_budget_usdc):
            raise ValueError("simulation budget must be finite and positive")
        if candidate_count < 2:
            raise ValueError("simulation requires at least two candidates")
        if not 0.5 < judge_accuracy <= 1.0:
            raise ValueError("judge accuracy must be in (0.5, 1.0]")
        if not 0.0 <= unknown_answer_rate < 1.0:
            raise ValueError("unknown answer rate must be in [0, 1)")
        if not 0.0 <= cost_variation <= 0.25:
            raise ValueError("cost variation must be in [0, 0.25]")

        base_config = config or AgentConfig()
        reserve = min(base_config.submission_reserve_usdc, total_budget_usdc * 0.50)
        self.config = replace(
            base_config,
            budget_usdc=total_budget_usdc,
            submission_reserve_usdc=reserve,
            judge_accuracy=judge_accuracy,
        )
        self.config.validate()
        self.total_budget_usdc = total_budget_usdc
        self.candidate_count = candidate_count
        self.judge_accuracy = judge_accuracy
        self.unknown_answer_rate = unknown_answer_rate
        self.cost_variation = cost_variation
        self.seed = seed
        self.random_source = random.Random(seed)
        self.candidates = self.generate_candidates(candidate_count)

    @classmethod
    def generate_candidates(cls, count: int) -> list[Candidate]:
        candidates: list[Candidate] = []
        categories = ("city", "nature", "historic")
        regions = ("north", "south", "west", "east")
        for index in range(count):
            suffix = (
                cls.CHECKPOINT_NAMES[index]
                if index < len(cls.CHECKPOINT_NAMES)
                else f"candidate-{index + 1}"
            )
            candidates.append(
                Candidate(
                    answer=f"checkpoint-{suffix}",
                    category=categories[index % len(categories)],
                    attributes={
                        "region": regions[index % len(regions)],
                        "group": index % 2,
                    },
                )
            )
        return candidates

    def generate_hidden_checkpoint(
        self, random_source: random.Random | None = None
    ) -> Candidate:
        source = random_source or self.random_source
        return source.choice(self.candidates)

    def run_episode(
        self,
        episode: int = 0,
        strategy_variant: StrategyVariant = StrategyVariant.RACE,
    ) -> EpisodeResult:
        random_source = random.Random(self.seed + episode * 10_007)
        hidden = self.generate_hidden_checkpoint(random_source)
        race, final_reserve = self._build_race(strategy_variant)
        adaptive = (
            AdaptiveRaceStrategy(
                total_budget_usdc=self.total_budget_usdc,
                final_guess_reserve_usdc=final_reserve,
                base_information_efficiency=self.config.min_value_per_usdc,
                base_information_gain_bits=self.config.min_information_gain_bits,
                base_guess_confidence=self.config.confidence_threshold,
            )
            if strategy_variant is StrategyVariant.ADAPTIVE
            else None
        )
        episode_config = replace(
            self.config, submission_reserve_usdc=final_reserve
        )
        search = InformationValueStrategy(self.candidates, episode_config)
        policy = PolicyEngine(
            BudgetController(
                total_budget_usdc=self.total_budget_usdc,
                submission_reserve_usdc=final_reserve,
            ),
            min_information_efficiency=episode_config.min_value_per_usdc,
            min_information_gain_bits=episode_config.min_information_gain_bits,
            min_guess_confidence=episode_config.confidence_threshold,
        )
        client = _UncertainMockAGPClient(
            hidden_answer=hidden.answer,
            hidden_category=hidden.category,
            hidden_attributes=hidden.attributes,
            initial_budget_usdc=self.total_budget_usdc,
            question_cost_usdc=lambda question: self._sample_question_cost(
                question, random_source, episode_config
            ),
            judge_accuracy=self.judge_accuracy,
            unknown_answer_rate=self.unknown_answer_rate,
            random_source=random_source,
        )

        spent_usdc = 0.0
        questions = 0
        information_gain = 0.0
        stop_reason = "strategy stopped"

        while True:
            _, confidence = search.best_guess
            directive = self._directive(
                strategy_variant,
                race,
                adaptive,
                search,
                confidence,
                questions,
                spent_usdc,
                final_reserve,
                episode_config,
            )
            if directive.finish_now:
                stop_reason = "adaptive finish decision"
                break
            decision = search.decide(
                spent_usdc,
                questions,
                confidence_threshold=directive.required_guess_confidence,
                min_information_gain_bits=directive.min_information_gain_bits,
                min_value_per_usdc=directive.min_information_efficiency,
                max_question_cost_usdc=directive.max_question_cost_usdc,
            )
            if decision.action == "guess":
                stop_reason = decision.reason
                break

            question = decision.question
            if question is None:
                stop_reason = "strategy returned no question"
                break
            verdict = policy.review_question(
                question,
                min_information_efficiency=directive.min_information_efficiency,
                min_information_gain_bits=directive.min_information_gain_bits,
                max_allowed_cost_usdc=directive.max_question_cost_usdc,
                action_type=f"simulation:{directive.phase}",
            )
            if not verdict.allowed:
                stop_reason = f"policy denied question: {verdict.reason}"
                break

            try:
                response = client.ask_question(question.text)
            except AGPBudgetExceededError:
                stop_reason = "runtime budget rejected question"
                break

            actual_cost = response.cost_usdc or 0.0
            spent_usdc += actual_cost
            questions += 1
            policy.record_spend(actual_cost)
            if race is not None:
                race.record_spend(actual_cost, RacePhase(directive.phase))
            outcome = search.observe(question, response.text)
            if outcome != "unknown":
                information_gain += question.information_gain_bits

        final_answer, confidence = search.best_guess
        final_directive = self._directive(
            strategy_variant,
            race,
            adaptive,
            search,
            confidence,
            questions,
            spent_usdc,
            final_reserve,
            episode_config,
        )
        guess_verdict = policy.review_guess(
            final_answer,
            confidence,
            required_confidence=final_directive.required_guess_confidence,
            action_type=f"simulation-submit:{final_directive.phase}",
        )

        submitted = False
        won = False
        submitted_answer: str | None = None
        failure_reason: str | None = None
        if guess_verdict.allowed:
            try:
                submission = client.submit_answer(final_answer)
                submitted = True
                submitted_answer = final_answer
                won = submission.accepted is True
                spent_usdc = self.total_budget_usdc - client.get_budget()
                if not won:
                    failure_reason = "wrong answer"
            except AGPBudgetExceededError:
                failure_reason = "runtime budget rejected submission"
        else:
            if "Confidence below" in guess_verdict.reason:
                failure_reason = (
                    "budget exhausted before confidence threshold"
                    if "budget" in stop_reason
                    else "low confidence submission denied"
                )
            else:
                failure_reason = "policy rejected final submission"

        return EpisodeResult(
            episode=episode,
            strategy=strategy_variant,
            hidden_checkpoint=hidden.answer,
            submitted_answer=submitted_answer,
            submitted=submitted,
            won=won,
            cost_usdc=spent_usdc,
            questions=questions,
            confidence=confidence,
            information_gain_bits=information_gain,
            failure_reason=failure_reason,
            remaining_budget_usdc=client.get_budget(),
        )

    def run_simulation(
        self,
        episodes: int = 1000,
        strategy_variant: StrategyVariant = StrategyVariant.RACE,
    ) -> SimulationReport:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        results = [
            self.run_episode(episode, strategy_variant) for episode in range(episodes)
        ]
        failures = Counter(
            result.failure_reason
            for result in results
            if result.failure_reason is not None
        )
        win_rate = fmean(result.won for result in results)
        average_cost = fmean(result.cost_usdc for result in results)
        return SimulationReport(
            strategy=strategy_variant,
            episodes=episodes,
            submissions=sum(result.submitted for result in results),
            wins=sum(result.won for result in results),
            win_rate=win_rate,
            average_cost_usdc=average_cost,
            average_questions=fmean(result.questions for result in results),
            average_confidence=fmean(result.confidence for result in results),
            average_information_gain_bits=fmean(
                result.information_gain_bits for result in results
            ),
            budget_efficiency=(
                win_rate / average_cost if average_cost > 0 else math.inf
            ),
            max_cost_usdc=max(result.cost_usdc for result in results),
            failure_reasons=dict(sorted(failures.items())),
        )

    def benchmark_strategies(self, episodes: int = 1000) -> BenchmarkReport:
        reports = {
            variant: self.run_simulation(episodes, variant)
            for variant in StrategyVariant
        }
        return BenchmarkReport(episodes_per_strategy=episodes, reports=reports)

    def benchmark_adaptive_strategy(
        self, episodes: int = 10_000
    ) -> BenchmarkReport:
        variants = (StrategyVariant.RACE, StrategyVariant.ADAPTIVE)
        reports = {
            variant: self.run_simulation(episodes, variant) for variant in variants
        }
        return BenchmarkReport(episodes_per_strategy=episodes, reports=reports)

    def _build_race(
        self, strategy_variant: StrategyVariant
    ) -> tuple[RaceStrategyLayer | None, float]:
        if strategy_variant is StrategyVariant.RACE:
            race = RaceStrategyLayer(
                total_budget_usdc=self.total_budget_usdc,
                minimum_final_guess_budget_usdc=self.config.submission_reserve_usdc,
                candidate_count=self.candidate_count,
                base_information_efficiency=self.config.min_value_per_usdc,
                base_information_gain_bits=self.config.min_information_gain_bits,
                base_guess_confidence=self.config.confidence_threshold,
            )
            return race, race.allocation.final_guess_budget_usdc
        if strategy_variant is StrategyVariant.AGGRESSIVE:
            reserve = max(
                self.total_budget_usdc * 0.02,
                min(
                    self.config.submission_reserve_usdc,
                    self.total_budget_usdc * 0.10,
                ),
            )
            return None, reserve
        return None, self.config.submission_reserve_usdc

    def _directive(
        self,
        strategy_variant: StrategyVariant,
        race: RaceStrategyLayer | None,
        adaptive: AdaptiveRaceStrategy | None,
        search: InformationValueStrategy,
        confidence: float,
        questions: int,
        spent_usdc: float,
        final_reserve: float,
        config: AgentConfig,
    ) -> SimulationDirective:
        runtime_searchable = max(
            0.0, self.total_budget_usdc - final_reserve - spent_usdc
        )
        if strategy_variant is StrategyVariant.RACE:
            if race is None:
                raise RuntimeError("race strategy requires a race controller")
            directive = race.directive(confidence, questions)
            return SimulationDirective(
                phase=directive.phase.value,
                min_information_efficiency=directive.min_information_efficiency,
                min_information_gain_bits=directive.min_information_gain_bits,
                required_guess_confidence=directive.required_guess_confidence,
                max_question_cost_usdc=min(
                    directive.max_question_cost_usdc, runtime_searchable
                ),
            )
        if strategy_variant is StrategyVariant.ADAPTIVE:
            if adaptive is None:
                raise RuntimeError("adaptive strategy requires an adaptive controller")
            highest_probability = max(search.probabilities.values())
            effective_candidates = sum(
                probability >= max(0.02, highest_probability * 0.10)
                for probability in search.probabilities.values()
            )
            searchable_total = max(
                1e-12, self.total_budget_usdc - final_reserve
            )
            race_progress = min(
                1.0,
                max(
                    questions / max(1, config.max_questions),
                    spent_usdc / searchable_total,
                ),
            )
            adaptive_directive = adaptive.decide(
                AdaptiveRaceState(
                    remaining_budget=max(
                        0.0, self.total_budget_usdc - spent_usdc
                    ),
                    current_confidence=confidence,
                    questions_used=questions,
                    candidate_count=max(1, effective_candidates),
                    race_progress=race_progress,
                )
            )
            return SimulationDirective(
                phase=adaptive_directive.mode.value,
                min_information_efficiency=(
                    adaptive_directive.min_information_efficiency
                ),
                min_information_gain_bits=adaptive_directive.min_information_gain_bits,
                required_guess_confidence=(
                    adaptive_directive.required_guess_confidence
                ),
                max_question_cost_usdc=min(
                    adaptive_directive.max_question_cost_usdc,
                    runtime_searchable,
                ),
                finish_now=(
                    adaptive_directive.finish_action is FinishAction.SUBMIT
                ),
            )
        if strategy_variant is StrategyVariant.AGGRESSIVE:
            return SimulationDirective(
                phase="aggressive",
                min_information_efficiency=0.0,
                min_information_gain_bits=0.0,
                required_guess_confidence=min(
                    0.995, config.confidence_threshold + 0.05
                ),
                max_question_cost_usdc=runtime_searchable,
            )

        target_questions = max(2, math.ceil(math.log2(self.candidate_count)))
        fixed_question_cap = (
            (self.total_budget_usdc - final_reserve) / target_questions * 1.50
        )
        return SimulationDirective(
            phase="fixed_budget",
            min_information_efficiency=config.min_value_per_usdc,
            min_information_gain_bits=config.min_information_gain_bits,
            required_guess_confidence=config.confidence_threshold,
            max_question_cost_usdc=min(runtime_searchable, fixed_question_cap),
        )

    def _sample_question_cost(
        self,
        question: str,
        random_source: random.Random,
        config: AgentConfig,
    ) -> float:
        base_cost = (
            config.question_base_cost_usdc
            + estimate_tokens(question) * config.input_usdc_per_1k_tokens / 1000.0
            + config.output_usdc_per_1k_tokens / 1000.0
        )
        multiplier = random_source.uniform(
            1.0 - self.cost_variation, 1.0 + self.cost_variation
        )
        return base_cost * multiplier


def run_simulation(
    episodes: int = 1000,
    strategy_variant: StrategyVariant = StrategyVariant.RACE,
    **simulator_options: object,
) -> SimulationReport:
    """Convenience API matching the requested training entry point."""

    simulator = AGPSimulator(**simulator_options)
    return simulator.run_simulation(episodes, strategy_variant)


def benchmark_strategies(
    episodes: int = 1000, **simulator_options: object
) -> BenchmarkReport:
    simulator = AGPSimulator(**simulator_options)
    return simulator.benchmark_strategies(episodes)


def benchmark_adaptive_strategy(
    episodes: int = 10_000, **simulator_options: object
) -> BenchmarkReport:
    simulator = AGPSimulator(**simulator_options)
    return simulator.benchmark_adaptive_strategy(episodes)


def _parse_variant(value: str | StrategyVariant) -> StrategyVariant:
    if isinstance(value, StrategyVariant):
        return value
    aliases = {
        "a": StrategyVariant.FIXED_BUDGET,
        "fixed": StrategyVariant.FIXED_BUDGET,
        "b": StrategyVariant.RACE,
        "race": StrategyVariant.RACE,
        "c": StrategyVariant.AGGRESSIVE,
        "aggressive": StrategyVariant.AGGRESSIVE,
        "d": StrategyVariant.ADAPTIVE,
        "adaptive": StrategyVariant.ADAPTIVE,
    }
    try:
        return aliases[value.lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError("strategy must be A, B, C, or D") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AGP statistical simulator")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--strategy", type=_parse_variant, default=StrategyVariant.RACE)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--adaptive-benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    simulator = AGPSimulator(seed=args.seed)
    if args.adaptive_benchmark:
        print(simulator.benchmark_adaptive_strategy(args.episodes).render())
    elif args.benchmark:
        print(simulator.benchmark_strategies(args.episodes).render())
    else:
        print(simulator.run_simulation(args.episodes, args.strategy).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
