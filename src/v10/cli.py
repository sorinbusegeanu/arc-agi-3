"""Command line entry points for v10 replay and offline analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .analysis import run_analysis
from .evaluation import CONDITIONING_PROFILES, run_ablation, run_optimization
from .live import run_live
from .replay import replay_trace
from .tools.memory import DEFAULT_CONTEXT_ENTRIES, default_state_dir
from .strata_client import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_STRATA_BASE_URL,
    DEFAULT_STRATA_MODEL,
    StrataClient,
)


def _parse_csv(value: str, convert):
    try:
        result = [convert(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid comma-separated value: {value}") from exc
    if not result:
        raise ValueError("comma-separated value must not be empty")
    return result


def _add_model_logging_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-request", action="store_true",
                         help="save complete outgoing model request payloads in the run directory")
    parser.add_argument("--log-response", action="store_true",
                         help="save raw model response bodies in the run directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v10", description="Offline Hydra Next replay and analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="convert a recorded JSONL trace to v10 evidence")
    replay.add_argument("--trace", required=True, type=Path)
    replay.add_argument("--run-dir", required=True, type=Path)
    replay.add_argument("--through-action", type=int, default=None,
                        help="persist only the initial state and this many action transitions")

    analyze = subparsers.add_parser("analyze", help="analyze a replay prefix through Strata")
    analyze.add_argument("--run-dir", required=True, type=Path)
    analyze.add_argument("--state-dir", type=Path, default=default_state_dir())
    analyze.add_argument(
        "--base-url", default=DEFAULT_STRATA_BASE_URL,
        help=f"OpenAI-compatible Strata endpoint base URL (default: {DEFAULT_STRATA_BASE_URL})",
    )
    analyze.add_argument(
        "--model", default=DEFAULT_STRATA_MODEL,
        help=f"deployed Qwen model identifier (default: {DEFAULT_STRATA_MODEL})",
    )
    analyze.add_argument("--through-action", type=int, default=4)
    analyze.add_argument("--conditioning", choices=CONDITIONING_PROFILES, default="memory_delta")
    analyze.add_argument("--horizon", type=int, choices=(1, 2, 3, 4), default=4)
    analyze.add_argument("--memory-context-entries", type=int, default=DEFAULT_CONTEXT_ENTRIES)
    analyze.add_argument("--include-prior-decisions", action="store_true",
                         help="include earlier structured decision summaries from this replay")
    analyze.add_argument("--timeout-seconds", type=float, default=120.0)
    analyze.add_argument("--max-output-tokens", type=int, default=1024)
    analyze.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high"),
        default=DEFAULT_REASONING_EFFORT,
    )
    analyze.add_argument("--reasoning-budget-tokens", type=int, default=None)
    analyze.add_argument("--skill", type=Path, default=None, help="override the shared v10 SKILL.md")
    _add_model_logging_options(analyze)

    ablate = subparsers.add_parser("ablate", help="compare matched offline context ablations")
    ablate.add_argument("--run-dir", required=True, type=Path)
    ablate.add_argument("--state-dir", type=Path, default=default_state_dir())
    ablate.add_argument("--through-action", type=int, default=4)
    ablate.add_argument("--horizon", type=int, choices=(1, 2, 3, 4), default=4)
    ablate.add_argument("--memory-context-entries", type=int, default=DEFAULT_CONTEXT_ENTRIES)
    ablate.add_argument("--max-output-tokens", type=int, default=1024)
    ablate.add_argument("--compare-prior-decisions", action=argparse.BooleanOptionalAction, default=True,
                        help="compare with and without prior structured decision summaries")

    optimize = subparsers.add_parser("optimize", help="sweep context, horizon and output budgets")
    optimize.add_argument("--run-dir", required=True, type=Path)
    optimize.add_argument("--state-dir", type=Path, default=default_state_dir())
    optimize.add_argument("--through-action", type=int, default=4)
    optimize.add_argument("--horizons", default="1,2,4")
    optimize.add_argument("--output-budgets", default="1024,4096")
    optimize.add_argument("--memory-limits", default="5,20")
    optimize.add_argument("--profiles", default=",".join(CONDITIONING_PROFILES))
    optimize.add_argument("--include-prior-decisions", action="store_true")
    for evaluation_parser in (ablate, optimize):
        evaluation_parser.add_argument("--base-url", default=DEFAULT_STRATA_BASE_URL)
        evaluation_parser.add_argument("--model", default=DEFAULT_STRATA_MODEL)
        evaluation_parser.add_argument("--timeout-seconds", type=float, default=120.0)
        evaluation_parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"),
                                       default=DEFAULT_REASONING_EFFORT)
        evaluation_parser.add_argument("--reasoning-budget-tokens", type=int, default=None)
        _add_model_logging_options(evaluation_parser)

    live = subparsers.add_parser("live", help="run a bounded live ARC game episode through Strata")
    live.add_argument("--game-id", required=True)
    live.add_argument(
        "--run-dir", required=True, type=Path,
        help="run output directory; existing contents are erased before the run",
    )
    live.add_argument("--state-dir", type=Path, default=default_state_dir(),
                      help="persistent v10 memory root; defaults to the user state directory")
    live.add_argument("--memory-context-entries", type=int, default=DEFAULT_CONTEXT_ENTRIES,
                      help="maximum persistent memory entries supplied to the model")
    live.add_argument(
        "--base-url", default=DEFAULT_STRATA_BASE_URL,
        help=f"OpenAI-compatible Strata endpoint base URL (default: {DEFAULT_STRATA_BASE_URL})",
    )
    live.add_argument(
        "--model", default=DEFAULT_STRATA_MODEL,
        help=f"deployed multimodal model identifier (default: {DEFAULT_STRATA_MODEL})",
    )
    live.add_argument("--env-root", default=None, help="optional ARC environment-files root")
    live.add_argument("--seed", type=int, default=0)
    live.add_argument("--max-actions", type=int, default=30)
    live.add_argument("--max-resets", type=int, default=0)
    live.add_argument("--plan-horizon", type=int, default=4)
    live.add_argument("--timeout-seconds", type=float, default=300.0)
    live.add_argument("--max-output-tokens", type=int, default=4096)
    live.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high"),
        default=DEFAULT_REASONING_EFFORT,
    )
    live.add_argument("--reasoning-budget-tokens", type=int, default=None)
    live.add_argument("--skill", type=Path, default=None, help="override the general v10 SKILL.md")
    _add_model_logging_options(live)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "replay":
            result = replay_trace(args.trace, args.run_dir, through_action=args.through_action)
        elif args.command == "analyze":
            client = StrataClient(
                base_url=args.base_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                reasoning_budget_tokens=args.reasoning_budget_tokens,
                log_request=args.log_request, log_response=args.log_response,
                log_dir=args.run_dir,
            )
            result = run_analysis(
                args.run_dir,
                client,
                through_action=args.through_action,
                skill_path=args.skill,
                state_dir=args.state_dir,
                conditioning=args.conditioning,
                horizon=args.horizon,
                memory_context_entries=args.memory_context_entries,
                include_prior_decisions=args.include_prior_decisions,
            )
        elif args.command in {"ablate", "optimize"}:
            def client_for(budget: int):
                return StrataClient(
                    base_url=args.base_url,
                    model=args.model,
                    timeout_seconds=args.timeout_seconds,
                    max_output_tokens=budget,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_budget_tokens=args.reasoning_budget_tokens,
                    log_request=args.log_request, log_response=args.log_response,
                    log_dir=args.run_dir,
                )
            if args.command == "ablate":
                result = run_ablation(
                    args.run_dir, client_for(args.max_output_tokens), through_action=args.through_action,
                    horizon=args.horizon, state_dir=args.state_dir,
                    memory_context_entries=args.memory_context_entries,
                    compare_prior_decisions=args.compare_prior_decisions,
                )
            else:
                result = run_optimization(
                    args.run_dir, client_for, through_action=args.through_action,
                    horizons=_parse_csv(args.horizons, int),
                    output_budgets=_parse_csv(args.output_budgets, int),
                    memory_limits=_parse_csv(args.memory_limits, int),
                    profiles=_parse_csv(args.profiles, str), state_dir=args.state_dir,
                    include_prior_decisions=args.include_prior_decisions,
                )
        else:
            from .arcade_env import ArcadeGameEnv

            client = StrataClient(
                base_url=args.base_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                reasoning_budget_tokens=args.reasoning_budget_tokens,
                log_request=args.log_request, log_response=args.log_response,
                log_dir=args.run_dir,
            )
            env = ArcadeGameEnv(
                args.game_id, seed=args.seed, env_root=args.env_root,
            )
            try:
                result = run_live(
                    env, args.run_dir, client, game_id=args.game_id,
                    seed=args.seed,
                    max_actions=args.max_actions, max_resets=args.max_resets,
                    plan_horizon=args.plan_horizon, skill_path=args.skill,
                    state_dir=args.state_dir,
                    memory_context_entries=args.memory_context_entries,
                )
            finally:
                env.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"v10: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
