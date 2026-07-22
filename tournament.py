"""Public, repeatable tournament benchmark for AGP strategy variants."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from agp_client import AGPBudgetExceededError, MockAGPClient
from config import AgentConfig
from policy_engine import BudgetController, PolicyEngine
from simulation import AGPSimulator, SimulationReport, StrategyVariant
from strategy import (
    InformationValueStrategy,
    PlannedQuestion,
    estimate_question_cost,
    estimate_tokens,
)
from strategy_memory import benchmark_memory_strategy


TOURNAMENT_SCHEMA_VERSION = 1
STRATEGY_VERSIONS = {
    "Random Search": "random-search-v1",
    "Fixed Budget": "fixed-budget-v1",
    "Adaptive Strategy": "adaptive-v1",
    "Adaptive + Memory": "adaptive-memory-v1",
}


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    """Parameters shared by every competitor for a fair comparison."""

    episodes: int = 10_000
    memory_training_episodes: int = 2_000
    seed: int = 42
    total_budget_usdc: float = 0.01
    candidate_count: int = 8
    judge_accuracy: float = 0.98
    unknown_answer_rate: float = 0.01
    cost_variation: float = 0.04

    def validate(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if self.memory_training_episodes < 1:
            raise ValueError("memory training episodes must be positive")
        if self.total_budget_usdc <= 0 or not math.isfinite(self.total_budget_usdc):
            raise ValueError("total budget must be finite and positive")
        if self.candidate_count < 2:
            raise ValueError("candidate count must be at least two")
        if not 0.5 < self.judge_accuracy <= 1.0:
            raise ValueError("judge accuracy must be in (0.5, 1.0]")
        if not 0.0 <= self.unknown_answer_rate < 1.0:
            raise ValueError("unknown answer rate must be in [0, 1)")
        if not 0.0 <= self.cost_variation <= 0.25:
            raise ValueError("cost variation must be in [0, 0.25]")

    def simulator_options(self) -> dict[str, object]:
        return {
            "total_budget_usdc": self.total_budget_usdc,
            "candidate_count": self.candidate_count,
            "judge_accuracy": self.judge_accuracy,
            "unknown_answer_rate": self.unknown_answer_rate,
            "cost_variation": self.cost_variation,
        }

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "episodes_per_strategy": self.episodes,
            "memory_training_episodes": self.memory_training_episodes,
            "seed": self.seed,
            "total_budget_usdc": self.total_budget_usdc,
            "candidate_count": self.candidate_count,
            "judge_accuracy": self.judge_accuracy,
            "unknown_answer_rate": self.unknown_answer_rate,
            "cost_variation": self.cost_variation,
            "ranking_method": "win_rate_desc, budget_efficiency_desc, cost_asc",
        }


@dataclass(frozen=True, slots=True)
class TournamentResult:
    rank: int
    agent: str
    strategy: str
    strategy_version: str
    episodes: int
    wins: int
    win_rate: float
    average_cost_usdc: float
    average_questions: float
    budget_efficiency: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "rank": self.rank,
            "agent": self.agent,
            "strategy": self.strategy,
            "strategy_version": self.strategy_version,
            "episodes": self.episodes,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "average_cost_usdc": self.average_cost_usdc,
            "average_questions": self.average_questions,
            "budget_efficiency": self.budget_efficiency,
        }


@dataclass(frozen=True, slots=True)
class TournamentReport:
    generated_at: str
    parameters: Mapping[str, int | float | str]
    leaderboard: tuple[TournamentResult, ...]

    @property
    def winner(self) -> TournamentResult:
        return self.leaderboard[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": TOURNAMENT_SCHEMA_VERSION,
            "report_type": "AGP Tournament Evaluation",
            "generated_at": self.generated_at,
            "parameters": dict(self.parameters),
            "strategy_versions": dict(STRATEGY_VERSIONS),
            "leaderboard": [entry.to_dict() for entry in self.leaderboard],
        }

    def save(self, path: Path | str = "tournament_report.json") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output

    def render(self) -> str:
        rows = [
            "Rank | Strategy | Win rate | Average cost | Average questions | Budget efficiency",
            "---:|---|---:|---:|---:|---:",
        ]
        for result in self.leaderboard:
            rows.append(
                f"{result.rank} | {result.strategy} | {result.win_rate:.2%} | "
                f"{result.average_cost_usdc:.6f} | "
                f"{result.average_questions:.2f} | "
                f"{result.budget_efficiency:.2f} wins/USDC"
            )
        return "\n".join(rows)


@dataclass(frozen=True, slots=True)
class _RandomEpisode:
    won: bool
    cost_usdc: float
    questions: int


class AGPTournament:
    """Runs all competitors under identical simulator parameters and seeds."""

    def __init__(
        self,
        config: TournamentConfig | None = None,
        *,
        memory_path: Path | str | None = None,
    ):
        self.config = config or TournamentConfig()
        self.config.validate()
        self.memory_path = memory_path

    def run(self) -> TournamentReport:
        options = self.config.simulator_options()

        fixed_simulator = AGPSimulator(seed=self.config.seed, **options)
        fixed = fixed_simulator.run_simulation(
            self.config.episodes, StrategyVariant.FIXED_BUDGET
        )
        memory_benchmark = benchmark_memory_strategy(
            episodes=self.config.episodes,
            training_episodes=self.config.memory_training_episodes,
            memory_path=self.memory_path,
            seed=self.config.seed,
            simulator_options=options,
        )

        unranked = [
            self._run_random_search(),
            _from_simulation("Agent B", "Fixed Budget", fixed),
            _from_simulation(
                "Agent C", "Adaptive Strategy", memory_benchmark.adaptive
            ),
            _from_simulation(
                "Agent D",
                "Adaptive + Memory",
                memory_benchmark.adaptive_with_memory,
            ),
        ]
        ordered = sorted(
            unranked,
            key=lambda result: (
                -result.win_rate,
                -result.budget_efficiency,
                result.average_cost_usdc,
                result.strategy,
            ),
        )
        leaderboard = tuple(
            replace(result, rank=index)
            for index, result in enumerate(ordered, start=1)
        )
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return TournamentReport(
            generated_at=generated_at,
            parameters=self.config.to_dict(),
            leaderboard=leaderboard,
        )

    def _run_random_search(self) -> TournamentResult:
        simulator = AGPSimulator(
            seed=self.config.seed,
            **self.config.simulator_options(),
        )
        episodes = [
            self._run_random_episode(simulator, episode)
            for episode in range(self.config.episodes)
        ]
        win_rate = fmean(result.won for result in episodes)
        average_cost = fmean(result.cost_usdc for result in episodes)
        return TournamentResult(
            rank=0,
            agent="Agent A",
            strategy="Random Search",
            strategy_version=STRATEGY_VERSIONS["Random Search"],
            episodes=self.config.episodes,
            wins=sum(result.won for result in episodes),
            win_rate=win_rate,
            average_cost_usdc=average_cost,
            average_questions=fmean(result.questions for result in episodes),
            budget_efficiency=(win_rate / average_cost if average_cost else 0.0),
        )

    def _run_random_episode(
        self, simulator: AGPSimulator, episode: int
    ) -> _RandomEpisode:
        source = random.Random(self.config.seed + episode * 10_007)
        hidden = simulator.generate_hidden_checkpoint(source)
        final_reserve = simulator.config.submission_reserve_usdc
        episode_config = replace(
            simulator.config,
            budget_usdc=self.config.total_budget_usdc,
            submission_reserve_usdc=final_reserve,
        )
        search = InformationValueStrategy(simulator.candidates, episode_config)
        policy = PolicyEngine(
            BudgetController(
                total_budget_usdc=self.config.total_budget_usdc,
                submission_reserve_usdc=final_reserve,
            ),
            min_information_efficiency=episode_config.min_value_per_usdc,
            min_information_gain_bits=episode_config.min_information_gain_bits,
            min_guess_confidence=episode_config.confidence_threshold,
        )
        client = MockAGPClient(
            hidden_answer=hidden.answer,
            hidden_category=hidden.category,
            hidden_attributes=hidden.attributes,
            initial_budget_usdc=self.config.total_budget_usdc,
            question_cost_usdc=lambda question: _sample_question_cost(
                question,
                source,
                episode_config,
                self.config.cost_variation,
            ),
        )

        spent = 0.0
        questions = 0
        while questions < episode_config.max_questions:
            _, confidence = search.best_guess
            if confidence >= episode_config.confidence_threshold:
                break
            searchable = (
                self.config.total_budget_usdc - final_reserve - spent
            )
            if searchable <= 0:
                break
            question = _random_question(search, source, episode_config)
            if question is None or question.expected_cost_usdc > searchable:
                break
            verdict = policy.review_question(
                question,
                max_allowed_cost_usdc=searchable,
                action_type="tournament:random",
            )
            if not verdict.allowed:
                break
            try:
                response = client.ask_question(question.text)
            except AGPBudgetExceededError:
                break
            response.text = _add_judge_uncertainty(
                response.text,
                source,
                self.config.judge_accuracy,
                self.config.unknown_answer_rate,
            )
            actual_cost = response.cost_usdc or 0.0
            spent += actual_cost
            questions += 1
            policy.record_spend(actual_cost)
            search.observe(question, response.text)

        answer, confidence = search.best_guess
        verdict = policy.review_guess(
            answer,
            confidence,
            required_confidence=episode_config.confidence_threshold,
            action_type="tournament-submit:random",
        )
        won = False
        if verdict.allowed:
            try:
                won = client.submit_answer(answer).accepted is True
            except AGPBudgetExceededError:
                won = False
        cost = self.config.total_budget_usdc - client.get_budget()
        return _RandomEpisode(won=won, cost_usdc=cost, questions=questions)


def _from_simulation(
    agent: str, strategy: str, report: SimulationReport
) -> TournamentResult:
    return TournamentResult(
        rank=0,
        agent=agent,
        strategy=strategy,
        strategy_version=STRATEGY_VERSIONS[strategy],
        episodes=report.episodes,
        wins=report.wins,
        win_rate=report.win_rate,
        average_cost_usdc=report.average_cost_usdc,
        average_questions=report.average_questions,
        budget_efficiency=report.budget_efficiency,
    )


def _random_question(
    search: InformationValueStrategy,
    source: random.Random,
    config: AgentConfig,
) -> PlannedQuestion | None:
    answers = list(search.probabilities)
    if len(answers) < 2:
        return None

    # Random Search deliberately ignores information-efficiency ranking. It picks
    # a random non-trivial partition, while still avoiding exact repeats.
    for _ in range(24):
        group_size = source.randint(1, len(answers) - 1)
        yes_answers = frozenset(source.sample(answers, group_size))
        signature = "random:" + "|".join(sorted(yes_answers))
        if signature in search.asked_signatures:
            continue
        rendered = json.dumps(sorted(yes_answers), ensure_ascii=False)
        text = (
            "Is the hidden checkpoint one of the following exact candidates: "
            f"{rendered}? Answer only YES or NO."
        )
        gain, success_gain = _question_value(search, yes_answers)
        cost = estimate_question_cost(text, config)
        return PlannedQuestion(
            text=text,
            kind="random",
            yes_answers=yes_answers,
            signature=signature,
            information_gain_bits=gain,
            expected_cost_usdc=cost,
            value_per_usdc=gain / cost if cost else math.inf,
            expected_success_gain=success_gain,
        )
    return None


def _question_value(
    search: InformationValueStrategy, yes_answers: frozenset[str]
) -> tuple[float, float]:
    probabilities = search.probabilities
    entropy_before = _entropy(probabilities.values())
    expected_entropy = 0.0
    expected_best = 0.0
    for observed_yes in (True, False):
        weighted = {
            answer: prior
            * (
                search.config.judge_accuracy
                if ((answer in yes_answers) == observed_yes)
                else 1.0 - search.config.judge_accuracy
            )
            for answer, prior in probabilities.items()
        }
        response_probability = sum(weighted.values())
        if response_probability <= 0:
            continue
        posterior = [value / response_probability for value in weighted.values()]
        expected_entropy += response_probability * _entropy(posterior)
        expected_best += response_probability * max(posterior)
    gain = max(0.0, entropy_before - expected_entropy)
    success_gain = max(0.0, expected_best - max(probabilities.values()))
    return gain, success_gain


def _entropy(probabilities: Any) -> float:
    return -sum(value * math.log2(value) for value in probabilities if value > 0)


def _sample_question_cost(
    question: str,
    source: random.Random,
    config: AgentConfig,
    variation: float,
) -> float:
    base_cost = (
        config.question_base_cost_usdc
        + estimate_tokens(question) * config.input_usdc_per_1k_tokens / 1000.0
        + config.output_usdc_per_1k_tokens / 1000.0
    )
    return base_cost * source.uniform(1.0 - variation, 1.0 + variation)


def _add_judge_uncertainty(
    answer: str,
    source: random.Random,
    accuracy: float,
    unknown_rate: float,
) -> str:
    if answer not in {"YES", "NO"}:
        return answer
    if source.random() < unknown_rate:
        return "UNKNOWN"
    if source.random() > accuracy:
        return "NO" if answer == "YES" else "YES"
    return answer


def run_tournament(
    config: TournamentConfig | None = None,
    *,
    report_path: Path | str = "tournament_report.json",
    memory_path: Path | str | None = None,
) -> TournamentReport:
    report = AGPTournament(config, memory_path=memory_path).run()
    report.save(report_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AGP multi-strategy tournament")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--training-episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=float, default=0.01)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--judge-accuracy", type=float, default=0.98)
    parser.add_argument("--unknown-rate", type=float, default=0.01)
    parser.add_argument("--cost-variation", type=float, default=0.04)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tournament_report.json"))
    args = parser.parse_args(argv)

    config = TournamentConfig(
        episodes=args.episodes,
        memory_training_episodes=args.training_episodes,
        seed=args.seed,
        total_budget_usdc=args.budget,
        candidate_count=args.candidates,
        judge_accuracy=args.judge_accuracy,
        unknown_answer_rate=args.unknown_rate,
        cost_variation=args.cost_variation,
    )
    report = run_tournament(
        config,
        report_path=args.output,
        memory_path=args.memory,
    )
    print(report.render())
    print(f"\nReport saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
