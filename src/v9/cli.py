from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from random import Random
from typing import Any

from v9.cognition.action_selection import choose_action
from v9.curriculum import CurriculumSelection, EnvironmentSpec, resolve_curriculum_selector
from v9.environments import ARCAdapter, ChessAdapter, GymDiscreteAdapter, GymStructuredAdapter, SokobanAdapter, SudokuAdapter, SyntheticSymbolicEnvironment, make_babyai_adapter
from v9.environments.synthetic_symbolic import SyntheticSymbolicConfig
from v9.modalities.symbols import DeterministicSymbolCodec
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig, ScientificConfig
from v9.runtime.parallel_memory_coordinator import run_parallel_memory_jobs
from v9.telemetry import MetricsHTTPServer

MIX_GAMES = ("gp03", "tp02", "FrozenLake-v1", "Chess-v0", "Sudoku-v0")
RESEARCH_GAMES = ("tp01", "tp02", "gp01", "gp03", "ex01", "ex02", "lo01", "mm01", "fi01", "FrozenLake-v1", "Chess-v0", "Sudoku-v0")
ALLOCATION_ENV = {
    "allocation_lease_steps": ("ARC_AGI3_V9_ALLOCATION_LEASE_STEPS", int),
    "allocation_unsolved_weight": ("ARC_AGI3_V9_UNSOLVED_WEIGHT", float),
    "allocation_optimizing_weight": ("ARC_AGI3_V9_OPTIMIZING_WEIGHT", float),
    "allocation_stable_weight": ("ARC_AGI3_V9_STABLE_WEIGHT", float),
    "allocation_stabilization_generations": ("ARC_AGI3_V9_STABILIZATION_GENERATIONS", int),
    "allocation_max_validations_without_improvement": ("ARC_AGI3_V9_MAX_VALIDATIONS_WITHOUT_IMPROVEMENT", int),
    "allocation_optimization_validation_budget": ("ARC_AGI3_V9_OPTIMIZATION_VALIDATION_BUDGET", int),
    "allocation_min_meaningful_improvement": ("ARC_AGI3_V9_MIN_MEANINGFUL_IMPROVEMENT", int),
}


@dataclass(frozen=True, slots=True)
class ActorResult:
    actor_id: int
    game_id: str
    steps: int
    positive_boundaries: int
    negative_boundaries: int
    resets: int


def resolve_game_specs(selector: str, *, curriculum_config: str | None = None) -> tuple[EnvironmentSpec, ...]:
    normalized = selector.strip()
    curriculum = resolve_curriculum_selector(normalized, path=curriculum_config)
    if curriculum is not None:
        return curriculum.specs
    if normalized.lower() == "mix":
        games = MIX_GAMES
    elif normalized.lower() == "research_1":
        games = RESEARCH_GAMES
    else:
        games = tuple(value.strip() for value in normalized.split(",") if value.strip())
    if not games:
        raise ValueError("--games must select at least one environment")
    return tuple(EnvironmentSpec("auto", game) for game in games)


def resolve_games(selector: str, *, curriculum_config: str | None = None) -> tuple[str, ...]:
    return tuple(spec.display_name for spec in resolve_game_specs(selector, curriculum_config=curriculum_config))


def _condition_config(condition: str | None) -> tuple[bool, bool, bool]:
    value = (condition or "").upper()
    if value == "C0":
        return True, False, False
    if value == "C1":
        return False, True, False
    if value == "C2":
        return True, True, False
    if value == "C3":
        return True, True, True
    return True, True, False


def make_adapter(spec: EnvironmentSpec | str, *, seed: int, env_root: str | None, alfred_backend_factory: str | None = None):
    if isinstance(spec, str):
        spec = EnvironmentSpec("auto", spec)
    game_id = spec.game_id
    lowered = game_id.lower()
    adapter = spec.adapter.lower()
    kwargs = dict(spec.kwargs)

    if adapter == "auto":
        if game_id == "FrozenLake-v1":
            adapter = "gym_discrete"
            kwargs.setdefault("is_slippery", False)
        elif lowered in {"chess-v0", "arcagi/chess-v0"}:
            adapter = "chess"
        elif lowered in {"sudoku-v0", "arcagi/sudoku-v0"}:
            adapter = "sudoku"
        elif lowered in {"synthetic", "synthetic-symbolic"}:
            adapter = "synthetic_symbolic"
        else:
            adapter = "arc"

    if adapter == "gym_discrete":
        return GymDiscreteAdapter(game_id, seed=seed, make_kwargs=kwargs)
    if adapter in {"gym_structured", "gym_image"}:
        family = "image" if adapter == "gym_image" else "structured"
        return GymStructuredAdapter(game_id, seed=seed, make_kwargs=kwargs, observation_family=family)
    if adapter == "sokoban":
        return SokobanAdapter(environment_name=game_id, seed=seed)
    if adapter in {"minigrid", "babyai"}:
        return make_babyai_adapter(
            game_id,
            seed=seed,
            suppress_symbols=bool(spec.options.get("suppress_symbol_stream", adapter == "minigrid")),
            **kwargs,
        )
    if adapter == "chess":
        opponent = "random" if "random" in lowered else "first"
        return ChessAdapter(seed=seed, opponent=opponent)
    if adapter == "sudoku":
        clues = 36
        if "clues_" in lowered:
            try:
                clues = int(lowered.rsplit("clues_", 1)[1])
            except ValueError:
                clues = 36
        return SudokuAdapter(seed=seed, clues=clues)
    if adapter in {"synthetic_causal", "synthetic_symbolic"}:
        interaction_enabled, symbols_enabled, shuffled = _condition_config(spec.condition)
        return SyntheticSymbolicEnvironment(
            SyntheticSymbolicConfig(
                seed=seed,
                aligned=interaction_enabled,
                shuffled=shuffled,
                emit_symbols=symbols_enabled if spec.condition else adapter == "synthetic_symbolic",
                scenario=game_id,
            )
        )
    if adapter == "alfred":
        if not alfred_backend_factory:
            raise RuntimeError("ALFRED curriculum execution requires --alfred-backend-factory module:function")
        module_name, separator, attribute = alfred_backend_factory.partition(":")
        if not separator:
            raise ValueError("--alfred-backend-factory must use module:function")
        factory = getattr(importlib.import_module(module_name), attribute)
        from v9.environments import AlfredAdapter
        backend = factory(game_id=game_id, seed=seed, **kwargs)
        return AlfredAdapter(backend)
    if adapter == "arc":
        return ARCAdapter(game_id, seed=seed, env_root=env_root)
    raise ValueError(f"unsupported curriculum adapter: {spec.adapter}")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="runs/v9/continuous")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--stage-workers", type=int, default=2)
    parser.add_argument("--stage-ring-capacity", type=int, default=8192)
    parser.add_argument("--shard-ring-capacity", type=int, default=8192)
    parser.add_argument("--node-capacity-per-shard", type=int, default=250_000)
    parser.add_argument("--edge-capacity-per-shard", type=int, default=500_000)
    parser.add_argument("--action-capacity-per-shard", type=int, default=65_536)
    parser.add_argument("--snapshot-interval-seconds", type=float, default=60.0)
    parser.add_argument("--peer-interval-seconds", type=float, default=0.5)
    parser.add_argument("--no-restore", action="store_true")
    parser.add_argument("--reset-persistent-identity", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--no-peers", action="store_true")


def _runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    scientific = ScientificConfig()
    overrides = {name: value for name, value in vars(args).items() if name.startswith("allocation_") and value is not None}
    overrides["random_seeds"] = (int(getattr(args, "seed", 0)),)
    if hasattr(args, "validation_mode"):
        overrides["transfer_validation_mode"] = "learning_only" if args.no_automatic_experiments else args.validation_mode
        overrides["transfer_validation_trials_per_interval"] = args.max_transfer_experiments
        overrides["transfer_validation_time_budget_seconds"] = args.transfer_experiment_time_budget_seconds
    if overrides:
        scientific = replace(scientific, **overrides)
    return RuntimeConfig.from_path(args.root, shards=args.shards, stage_workers=args.stage_workers, stage_ring_capacity=args.stage_ring_capacity, shard_ring_capacity=args.shard_ring_capacity, node_capacity_per_shard=args.node_capacity_per_shard, edge_capacity_per_shard=args.edge_capacity_per_shard, action_capacity_per_shard=args.action_capacity_per_shard, snapshot_interval_seconds=args.snapshot_interval_seconds, peer_interval_seconds=args.peer_interval_seconds, enable_snapshots=not args.no_snapshots, restore=not args.no_restore, enable_peers=not args.no_peers, enable_lifecycle=getattr(args, "lifecycle", "on") == "on", reset_persistent_identity=args.reset_persistent_identity, scientific=scientific)


def _actor(runtime: ContinuousMemoryRuntime, spec: EnvironmentSpec, *, actor_id: int, steps: int, seed: int, env_root: str | None, epsilon: float, progress_interval: float, verbose: bool, wait: float) -> ActorResult:
    game_id = spec.display_name
    adapter = make_adapter(spec, seed=seed, env_root=env_root)
    identity = runtime.environments.register(adapter.identity())
    episode = runtime.environments.next_episode(identity)
    codec = DeterministicSymbolCodec(f"{adapter.identity().family}-raw-symbols")
    rng = Random(seed)
    positives = negatives = resets = completed = 0
    next_progress = time.monotonic() + progress_interval
    try:
        for index in range(int(steps)):
            actions = tuple(sorted(set(adapter.available_actions())))
            if not actions:
                adapter.reset()
                episode = runtime.environments.next_episode(identity)
                resets += 1
                actions = tuple(sorted(set(adapter.available_actions())))
                if not actions:
                    continue
            before = adapter.observe()
            action = choose_action(runtime.read_view, actions, rng=rng, epsilon=epsilon, target_environment_id=identity.value)
            after = adapter.step(action)
            if (spec.condition or "").upper() == "C1":
                runtime.record_symbol_stream(adapter, producer_id=actor_id, producer_sequence=index + 1, episode_id=episode, symbol_codec=codec)
            else:
                runtime.record_interaction(adapter, producer_id=actor_id, producer_sequence=index + 1, global_step=index, native_action=action, before_observation=before, after_observation=after, episode_id=episode, symbol_codec=codec)
            runtime.unified_telemetry.record_curriculum_event(step=spec.curriculum_step, environment_family=adapter.identity().family, game_scenario=spec.game_id)
            completed += 1
            boundary = adapter.boundary_event()
            positives += int(boundary.primary_valence > 0)
            negatives += int(boundary.primary_valence < 0)
            if not boundary.continuation:
                if wait:
                    time.sleep(wait)
                adapter.reset()
                episode = runtime.environments.next_episode(identity)
                resets += 1
            if verbose and time.monotonic() >= next_progress:
                print(f"v9 progress actor={actor_id} game={game_id} steps={completed}", flush=True)
                next_progress = time.monotonic() + progress_interval
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
    return ActorResult(actor_id, game_id, completed, positives, negatives, resets)


def _trajectory_rows(root: Path) -> list[dict[str, Any]]:
    from v9.runtime.snapshot import latest_snapshot
    path = latest_snapshot(root)
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    nodes = raw.get("state", {}).get("graph", {}).get("nodes", [])
    return [dict(row) for row in nodes if int(row.get("level", -1)) == 7]


def run_continuous(args: argparse.Namespace) -> int:
    if args.show_best_trajectory:
        rows = _trajectory_rows(Path(args.root))
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0 if rows else 1
    if args.save_best_trajectory:
        Path(args.save_best_trajectory).write_text(json.dumps(_trajectory_rows(Path(args.root)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if not args.games:
        raise ValueError("--games is required for a normal continuous run")
    if args.actors <= 0 or args.steps_per_game <= 0 or args.graph_check <= 0 or args.wait < 0 or args.progress_interval_seconds <= 0 or not 0 <= args.epsilon <= 1:
        raise ValueError("actors, steps-per-game, graph-check and progress interval must be positive; wait and epsilon must be valid")
    if min(args.ingest_workers, args.derivation_workers, args.ingest_queue_capacity, args.derivation_queue_capacity, args.publication_queue_capacity) <= 0:
        raise ValueError("memory worker counts and queue capacities must be positive")
    specs = resolve_game_specs(args.games, curriculum_config=args.curriculum_config)
    games = tuple(spec.display_name for spec in specs)
    runtime = ContinuousMemoryRuntime(_runtime_config(args))
    curriculum_modes = sorted({spec.validation_mode for spec in specs if spec.validation_mode})
    effective_validation_mode = ("learning_only" if args.no_automatic_experiments else (curriculum_modes[0] if len(curriculum_modes) == 1 else args.validation_mode))
    runtime.unified_telemetry.set_gauge("curriculum_validation_mode", effective_validation_mode)
    runtime.start()
    dashboard = None
    if not args.no_dashboard:
        dashboard = MetricsHTTPServer(runtime.metrics, host=args.dashboard_host, port=args.dashboard_port)
        dashboard.start()
        print(f"v9 dashboard: http://{args.dashboard_host}:{args.dashboard_port}/", flush=True)
    jobs: list[tuple[int, EnvironmentSpec, int, int]] = []
    lanes, actor_id = max(len(specs), args.actors), 1
    base_lanes, extra_lanes = divmod(lanes, len(specs))
    for game_index, spec in enumerate(specs):
        lane_count = base_lanes + int(game_index < extra_lanes)
        base_steps, extra_steps = divmod(args.steps_per_game, lane_count)
        for lane in range(lane_count):
            steps = base_steps + int(lane < extra_steps)
            if steps:
                jobs.append((actor_id, spec, steps, args.seed + actor_id * 1009))
                actor_id += 1
    print(f"v9 continuous: games={len(games)} actors={min(args.actors, len(jobs))} shards={args.shards} stage_workers={args.stage_workers} peers={'off' if args.no_peers else 'on'} lifecycle={args.lifecycle} snapshots={'off' if args.no_snapshots else 'native'} game_ids={','.join(games)}", flush=True)
    try:
        process_results = run_parallel_memory_jobs(
            runtime,
            jobs,
            actor_limit=args.actors,
            stage_workers=args.stage_workers,
            shards=args.shards,
            queue_capacity=max(args.stage_ring_capacity, args.shard_ring_capacity),
            epsilon=args.epsilon,
            env_root=args.env_root,
            alfred_backend_factory=getattr(args, "alfred_backend_factory", None),
            start_method=runtime.config.multiprocessing_start_method,
            progress_interval_seconds=args.progress_interval_seconds,
            ingest_workers=args.ingest_workers,
            derivation_workers=args.derivation_workers,
            ingest_queue_capacity=args.ingest_queue_capacity,
            derivation_queue_capacity=args.derivation_queue_capacity,
            publication_queue_capacity=args.publication_queue_capacity,
        )
        results = [
            ActorResult(
                row.actor_id,
                row.game_id,
                row.steps,
                row.positive_boundaries,
                row.negative_boundaries,
                row.resets,
            )
            for row in process_results
        ]
        runtime.wait_quiescent(args.drain_timeout)
        final = runtime.close(normal=True, timeout=args.final_save_timeout)
        metrics = runtime.metrics()
        summary = {"games": list(games), "actors": [asdict(row) for row in results], "automatic_transfer_experiments": {"mode": effective_validation_mode, "budget": runtime.config.scientific.transfer_validation_trials_per_interval, "attempted": 0, "completed": 0, "passed": 0, "blocker": "no eligible target exposes exact snapshot/restore support" if effective_validation_mode != "learning_only" else None}, "hypotheses": runtime.scientific_statuses(), "metrics": metrics, "final_snapshot": None if final is None else {**asdict(final), "path": str(final.path)}}
        target = Path(args.root) / "v9_run_summary.json"
        target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except BaseException:
        runtime.close(normal=False)
        raise
    finally:
        if dashboard is not None:
            dashboard.close()


def run_smoke(args: argparse.Namespace) -> int:
    runtime = ContinuousMemoryRuntime(_runtime_config(args))
    runtime.start()
    try:
        for index in range(args.events):
            runtime.submit(runtime.make_experience(producer_id=1, producer_sequence=index + 1, environment_instance_id=1, global_step=index, context_signature=10 + index % 3, action_id=index % 4, outcome_signature=100 + index % 5, family_signature=200 + index % 3, carrier_signature=300 + index % 7, future_option_delta=float(index % 3 - 1), changed_cells=1 + index % 12, trajectory_signature=400 + index % 9, next_context_signature=10 + (index + 1) % 3))
        runtime.wait_quiescent(args.drain_timeout)
        metrics = runtime.metrics()
        print(f"v9 smoke done events={args.events} memories={metrics['memories']} edges={metrics['edges']}", flush=True)
        runtime.close(normal=True, timeout=args.final_save_timeout)
        return 0
    except BaseException:
        runtime.close(normal=False)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arc-agi3-v9")
    for name, kind in (("allocation-lease-steps", int), ("allocation-unsolved-weight", float), ("allocation-optimizing-weight", float), ("allocation-stable-weight", float), ("allocation-stabilization-generations", int), ("allocation-max-validations-without-improvement", int), ("allocation-optimization-validation-budget", int), ("allocation-min-meaningful-improvement", int)):
        parser.add_argument(f"--{name}", type=kind, default=None)
    parser.add_argument("--allocation-plateau-priority", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    continuous = sub.add_parser("continuous-run")
    _add_runtime_arguments(continuous)
    continuous.add_argument("--games", default=None)
    continuous.add_argument("--curriculum-config", default=None)
    trajectory = continuous.add_mutually_exclusive_group()
    trajectory.add_argument("--show-best-trajectory", metavar="GAME_ID", default=None)
    trajectory.add_argument("--save-best-trajectory", metavar="FILE", default=None)
    continuous.add_argument("--steps-per-game", type=int, default=1000)
    continuous.add_argument("--actors", type=int, default=8)
    continuous.add_argument("--ingest-workers", type=int, default=4)
    continuous.add_argument("--derivation-workers", type=int, default=4)
    continuous.add_argument("--ingest-queue-capacity", type=int, default=8192)
    continuous.add_argument("--derivation-queue-capacity", type=int, default=4096)
    continuous.add_argument("--publication-queue-capacity", type=int, default=8192)
    continuous.add_argument("--seed", type=int, default=0)
    continuous.add_argument("--env-root", default=None)
    continuous.add_argument("--epsilon", type=float, default=0.10)
    continuous.add_argument("--graph-check", type=int, default=1000)
    continuous.add_argument("--wait", type=float, default=0.0)
    continuous.add_argument("--lifecycle", choices=("on", "off"), default="on")
    continuous.add_argument("--actor-timeout", type=float, default=None)
    continuous.add_argument("--progress-interval-seconds", type=float, default=60.0)
    continuous.add_argument("--dashboard-host", default="0.0.0.0")
    continuous.add_argument("--dashboard-port", type=int, default=8765)
    continuous.add_argument("--no-dashboard", action="store_true")
    continuous.add_argument("--verbose-progress", action="store_true")
    continuous.add_argument("--drain-timeout", type=float, default=300.0)
    continuous.add_argument("--final-save-timeout", type=float, default=300.0)
    continuous.add_argument("--transfer-experiment-steps", type=int, default=32)
    continuous.add_argument("--max-transfer-experiments", type=int, default=8)
    continuous.add_argument("--transfer-experiment-time-budget-seconds", type=float, default=30.0)
    continuous.add_argument("--validation-mode", choices=("learning_only", "validation_budgeted", "validation_full"), default="validation_budgeted")
    continuous.add_argument("--no-automatic-experiments", action="store_true")
    smoke = sub.add_parser("smoke")
    _add_runtime_arguments(smoke)
    smoke.add_argument("--events", type=int, default=1000)
    smoke.add_argument("--drain-timeout", type=float, default=60.0)
    smoke.add_argument("--final-save-timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name, (environment_name, converter) in ALLOCATION_ENV.items():
        if getattr(args, name) is None and os.environ.get(environment_name):
            setattr(args, name, converter(os.environ[environment_name]))
    if not args.allocation_plateau_priority and os.environ.get("ARC_AGI3_V9_PLATEAU_PRIORITY_ENABLED") == "1":
        args.allocation_plateau_priority = True
    for name, value in vars(args).items():
        if name.startswith("allocation_") and value is not None and value is not False and float(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return run_continuous(args) if args.command == "continuous-run" else run_smoke(args)
