"""Export tournament_report.json as a GitHub-friendly Markdown benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_tournament_report(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    leaderboard = payload.get("leaderboard")
    if not isinstance(leaderboard, list) or not leaderboard:
        raise ValueError("tournament report must contain a non-empty leaderboard")
    return payload


def render_markdown(
    payload: Mapping[str, Any], strategy: str = "Adaptive + Memory"
) -> str:
    leaderboard = payload.get("leaderboard")
    if not isinstance(leaderboard, list) or not leaderboard:
        raise ValueError("tournament report must contain a non-empty leaderboard")

    selected = next(
        (
            row
            for row in leaderboard
            if str(row.get("strategy", "")).casefold() == strategy.casefold()
        ),
        None,
    )
    if selected is None:
        available = ", ".join(str(row.get("strategy")) for row in leaderboard)
        raise ValueError(f"strategy {strategy!r} not found; available: {available}")

    lines = [
        "# AGP Agent Benchmark",
        "",
        "Strategy:",
        str(selected["strategy"]),
        "",
        "Episodes:",
        str(selected["episodes"]),
        "",
        "Win Rate:",
        f"{float(selected['win_rate']):.2%}",
        "",
        "Average Cost:",
        f"{float(selected['average_cost_usdc']):.6f} USDC",
        "",
        "Average Questions:",
        f"{float(selected['average_questions']):.2f}",
        "",
        "Efficiency:",
        f"{float(selected['budget_efficiency']):.2f} wins/USDC",
        "",
        "## Tournament Leaderboard",
        "",
        "| Rank | Strategy | Win Rate | Average Cost | Average Questions | Budget Efficiency |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(leaderboard, key=lambda item: int(item["rank"])):
        name = str(row["strategy"]).replace("|", "\\|")
        lines.append(
            f"| {int(row['rank'])} | {name} | "
            f"{float(row['win_rate']):.2%} | "
            f"{float(row['average_cost_usdc']):.6f} USDC | "
            f"{float(row['average_questions']):.2f} | "
            f"{float(row['budget_efficiency']):.2f} wins/USDC |"
        )

    parameters = payload.get("parameters", {})
    if isinstance(parameters, Mapping):
        lines.extend(
            [
                "",
                "## Benchmark Parameters",
                "",
                f"- Seed: `{parameters.get('seed', 'unknown')}`",
                f"- Budget per episode: `{parameters.get('total_budget_usdc', 'unknown')} USDC`",
                f"- Candidate count: `{parameters.get('candidate_count', 'unknown')}`",
                f"- Judge accuracy: `{parameters.get('judge_accuracy', 'unknown')}`",
                f"- Memory training episodes: `{parameters.get('memory_training_episodes', 'unknown')}`",
            ]
        )
    lines.extend(
        [
            "",
            "> Results are deterministic statistical simulations for the recorded seed and parameters; they are not guaranteed real-world AGP outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def export_report(
    input_path: Path | str = "tournament_report.json",
    output_path: Path | str = "AGP_AGENT_PERFORMANCE_REPORT.md",
    *,
    strategy: str = "Adaptive + Memory",
) -> Path:
    payload = load_tournament_report(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload, strategy), encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export AGP tournament Markdown")
    parser.add_argument(
        "--input", type=Path, default=Path("tournament_report.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("AGP_AGENT_PERFORMANCE_REPORT.md"),
    )
    parser.add_argument("--strategy", default="Adaptive + Memory")
    args = parser.parse_args(argv)
    output = export_report(
        args.input,
        args.output,
        strategy=args.strategy,
    )
    print(f"Markdown report saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
