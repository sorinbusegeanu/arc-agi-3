"""Command line entry points for v10 replay and offline analysis."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
import sys
from pathlib import Path
from typing import Sequence

from .analysis import run_analysis
from .evaluation import CONDITIONING_PROFILES, run_ablation, run_optimization
from .live import run_live
from .replay import replay_trace
from .tools.memory import (
    DEFAULT_CONTEXT_ENTRIES,
    MAX_GLOBAL_MEMORY_MARKDOWN_CHARS,
    MAX_MEMORY_MARKDOWN_CHARS,
    default_state_dir,
    reset_persistent_memory,
)
from .strata_client import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_STRATA_BASE_URL,
    DEFAULT_STRATA_MODEL,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL,
    StrataClient,
    V11_VERSION,
)
from .vllm_server import ensure_vllm_server


def _parse_csv(value: str, convert):
    try:
        result = [convert(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid comma-separated value: {value}") from exc
    if not result:
        raise ValueError("comma-separated value must not be empty")
    return result


def _parse_global_memory_max_chars(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= result <= MAX_GLOBAL_MEMORY_MARKDOWN_CHARS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_GLOBAL_MEMORY_MARKDOWN_CHARS}"
        )
    return result


def _parse_positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def _parse_game_list(value: str) -> list[str]:
    try:
        game_ids = _parse_csv(value, str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(set(game_ids)) != len(game_ids):
        raise argparse.ArgumentTypeError("game list must not contain duplicate IDs")
    if any(game_id in {".", ".."} or "/" in game_id or "\\" in game_id or "\0" in game_id
           for game_id in game_ids):
        raise argparse.ArgumentTypeError("game IDs in --game-list must be safe directory names")
    return game_ids


def _add_model_logging_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-request", action="store_true",
                         help="save complete outgoing model request payloads in the run directory")
    parser.add_argument("--log-response", action="store_true",
                         help="save raw model response bodies in the run directory")


def _add_model_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vllm", action="store_true",
        help="use the vLLM OpenAI-compatible endpoint (Strata remains the default)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help=("OpenAI-compatible endpoint URL; defaults to Strata or vLLM based on --vllm "
              f"({DEFAULT_STRATA_BASE_URL} or {DEFAULT_VLLM_BASE_URL})"),
    )
    parser.add_argument(
        "--model", default=None,
        help=("served model identifier; defaults to Strata or vLLM based on --vllm "
              f"({DEFAULT_STRATA_MODEL} or {DEFAULT_VLLM_MODEL})"),
    )
    parser.add_argument(
        "--vllm-autostart", action="store_true",
        help="with --vllm, start the selected model locally if its endpoint is not ready",
    )
    parser.add_argument(
        "--vllm-startup-timeout-seconds", type=float, default=900.0,
        help="maximum wait for an auto-started local vLLM model (default: 900 seconds)",
    )
    parser.add_argument(
        "--vllm-max-model-len", type=int, default=65536,
        help="context limit passed to an auto-started vLLM server (default: 65536)",
    )
    parser.add_argument(
        "--vllm-tool-call-parser", default=None,
        help="tool parser for auto-started vLLM (defaults to hermes; set a model-specific parser if needed)",
    )


def _selected_model_backend(args: argparse.Namespace) -> tuple[str, str, str]:
    use_vllm = bool(getattr(args, "vllm", False))
    backend = "vllm" if use_vllm else "strata"
    default_url = DEFAULT_VLLM_BASE_URL if use_vllm else DEFAULT_STRATA_BASE_URL
    default_model = DEFAULT_VLLM_MODEL if use_vllm else DEFAULT_STRATA_MODEL
    return backend, getattr(args, "base_url", None) or default_url, getattr(args, "model", None) or default_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v11", description="Offline Hydra Next replay and analysis")
    parser.add_argument("--version", action="version", version=f"v11 {V11_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="convert a recorded JSONL trace to v10 evidence")
    replay.add_argument("--trace", required=True, type=Path)
    replay.add_argument("--run-dir", required=True, type=Path)
    replay.add_argument("--through-action", type=int, default=None,
                        help="persist only the initial state and this many action transitions")

    analyze = subparsers.add_parser("analyze", help="analyze a replay prefix through the configured model backend")
    analyze.add_argument("--run-dir", required=True, type=Path)
    analyze.add_argument("--state-dir", type=Path, default=default_state_dir())
    _add_model_backend_options(analyze)
    analyze.add_argument("--through-action", type=int, default=4)
    analyze.add_argument("--conditioning", choices=CONDITIONING_PROFILES, default="pixels_actions_memory")
    analyze.add_argument("--horizon", type=int, choices=(1, 2, 3, 4), default=4)
    analyze.add_argument("--memory-context-entries", type=int, default=DEFAULT_CONTEXT_ENTRIES)
    analyze.add_argument("--include-prior-decisions", action="store_true",
                         help="include earlier structured decision summaries from this replay")
    analyze.add_argument("--timeout-seconds", type=float, default=120.0)
    analyze.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
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
    ablate.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    ablate.add_argument("--compare-prior-decisions", action=argparse.BooleanOptionalAction, default=True,
                        help="compare with and without prior structured decision summaries")

    optimize = subparsers.add_parser("optimize", help="sweep context, horizon and output budgets")
    optimize.add_argument("--run-dir", required=True, type=Path)
    optimize.add_argument("--state-dir", type=Path, default=default_state_dir())
    optimize.add_argument("--through-action", type=int, default=4)
    optimize.add_argument("--horizons", default="1,2,4")
    optimize.add_argument("--output-budgets", default=str(DEFAULT_MAX_OUTPUT_TOKENS))
    optimize.add_argument("--memory-limits", default="0,5")
    optimize.add_argument("--profiles", default=",".join(CONDITIONING_PROFILES))
    optimize.add_argument("--include-prior-decisions", action="store_true")
    for evaluation_parser in (ablate, optimize):
        _add_model_backend_options(evaluation_parser)
        evaluation_parser.add_argument("--timeout-seconds", type=float, default=120.0)
        evaluation_parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"),
                                       default=DEFAULT_REASONING_EFFORT)
        evaluation_parser.add_argument("--reasoning-budget-tokens", type=int, default=None)
        _add_model_logging_options(evaluation_parser)

    live = subparsers.add_parser("live", help="run a bounded live ARC game episode through the configured model backend")
    game_selector = live.add_mutually_exclusive_group(required=True)
    game_selector.add_argument("--game-id", help="run one game")
    game_selector.add_argument(
        "--game-list", type=_parse_game_list,
        help="comma-separated game IDs to run as a batch (for example: ez01,ez02)",
    )
    live.add_argument(
        "--parallel-games", type=_parse_positive_int, default=1,
        help="maximum number of batch games running at once (default: 1)",
    )
    live.add_argument(
        "--run-dir", required=True, type=Path,
        help="output directory (batch mode creates one cleaned subdirectory per game)",
    )
    live.add_argument("--state-dir", type=Path, default=default_state_dir(),
                      help="persistent v10 memory root; defaults to the project skills directory")
    live.add_argument(
        "--reset-memory", action="store_true",
        help="clear memory and solutions, and skip historical recovery for this run",
    )
    live.add_argument("--memory-context-entries", type=int, default=DEFAULT_CONTEXT_ENTRIES,
                      help="compatibility setting (0-5); live prompts load saved Markdown in full")
    live.add_argument("--game-memory-max-chars", type=int, default=MAX_MEMORY_MARKDOWN_CHARS,
                      help="maximum characters accepted for game MEMORY.md")
    live.add_argument("--global-memory-max-chars", type=_parse_global_memory_max_chars,
                      default=MAX_GLOBAL_MEMORY_MARKDOWN_CHARS,
                      help="maximum characters accepted for global MEMORY.md (hard limit: 4500)")
    _add_model_backend_options(live)
    live.add_argument("--env-root", default=None, help="optional ARC environment-files root")
    live.add_argument("--seed", type=int, default=0)
    live.add_argument("--max-actions", type=int, default=30)
    live.add_argument("--plan-horizon", type=int, default=8)
    live.add_argument("--discovery-tools", action=argparse.BooleanOptionalAction, default=True,
                      help="enable discovery tools during model action selection (default: enabled; use --no-discovery-tools to disable)")
    live.add_argument("--timeout-seconds", type=float, default=300.0)
    live.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    live.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high"),
        default=DEFAULT_REASONING_EFFORT,
    )
    live.add_argument("--reasoning-budget-tokens", type=int, default=None)
    live.add_argument(
        "--context-window-tokens", type=int, default=None,
        help="override the endpoint's discovered model context window for action conversation compaction",
    )
    live.add_argument("--skill", type=Path, default=None, help="override the general v10 SKILL.md")
    _add_model_logging_options(live)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        backend, base_url, model = _selected_model_backend(args)
        if getattr(args, "vllm_autostart", False):
            if backend != "vllm":
                raise ValueError("--vllm-autostart requires --vllm")
            ensure_vllm_server(
                base_url=base_url,
                model=model,
                startup_timeout_seconds=args.vllm_startup_timeout_seconds,
                max_model_len=args.vllm_max_model_len,
                tool_call_parser=args.vllm_tool_call_parser,
            )
        if args.command == "replay":
            result = replay_trace(args.trace, args.run_dir, through_action=args.through_action)
        elif args.command == "analyze":
            client = StrataClient(
                base_url=base_url,
                model=model,
                backend=backend,
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
                    base_url=base_url,
                    model=model,
                    backend=backend,
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

            if args.reset_memory:
                reset_persistent_memory(args.state_dir)
                print(f"[v11] cleared persistent memory under {args.state_dir}", file=sys.stderr, flush=True)

            def run_game(game_id: str, run_dir: Path, global_memory_lock=None):
                client = StrataClient(
                    base_url=base_url,
                    model=model,
                    backend=backend,
                    timeout_seconds=args.timeout_seconds,
                    max_output_tokens=args.max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_budget_tokens=args.reasoning_budget_tokens,
                    log_request=args.log_request, log_response=args.log_response,
                    log_dir=run_dir,
                    context_window_tokens=args.context_window_tokens,
                )
                env = ArcadeGameEnv(game_id, seed=args.seed, env_root=args.env_root)
                try:
                    return run_live(
                        env, run_dir, client, game_id=game_id,
                        seed=args.seed,
                        import_historical_routes=not args.reset_memory,
                        max_actions=args.max_actions,
                        plan_horizon=args.plan_horizon,
                        skill_path=args.skill,
                        state_dir=args.state_dir,
                        memory_context_entries=args.memory_context_entries,
                        game_memory_max_chars=args.game_memory_max_chars,
                        global_memory_max_chars=args.global_memory_max_chars,
                        discovery_tools=args.discovery_tools,
                        global_memory_lock=global_memory_lock,
                    )
                finally:
                    env.close()

            if args.game_list is None:
                result = run_game(args.game_id, args.run_dir)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0

            game_ids = args.game_list
            worker_count = min(args.parallel_games, len(game_ids))
            global_memory_lock = threading.RLock()
            results: dict[str, dict[str, object]] = {}
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="v11-game") as executor:
                futures = {
                    executor.submit(
                        run_game, game_id, args.run_dir / game_id, global_memory_lock,
                    ): game_id
                    for game_id in game_ids
                }
                for future in as_completed(futures):
                    game_id = futures[future]
                    try:
                        results[game_id] = {"game_id": game_id, "status": "completed", "result": future.result()}
                    except Exception as exc:
                        results[game_id] = {"game_id": game_id, "status": "failed", "error": str(exc)}
            batch_result = {
                "status": "completed" if all(results[game_id]["status"] == "completed" for game_id in game_ids) else "partial_failure",
                "parallel_games": worker_count,
                "games": [results[game_id] for game_id in game_ids],
            }
            print(json.dumps(batch_result, ensure_ascii=False, indent=2))
            return 0 if batch_result["status"] == "completed" else 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"v11: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
