from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from v8.actor import ActorJob, run_actor_jobs
from v8.capacity import plan_capacities
from v8.diagnostics import format_game_rate_line, format_hypothesis_line
from v8.experiments import ExperimentSummary, run_automatic_transfer_experiments
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig
from v8.snapshot import latest_complete_snapshot


_GAME_WAIT_ENV = "ARC_AGI3_GAME_WAIT_SECONDS"
_HYPOTHESIS_INITIAL_DELAY_SECONDS = 60.0


def _runtime_config(args, *, total_steps: int = 0) -> V8RuntimeConfig:
    plan = plan_capacities(
        total_steps=int(total_steps),
        shards=int(args.shards),
        root=args.root,
        restore=not args.no_restore,
        node_override=args.node_capacity_per_shard,
        edge_override=args.edge_capacity_per_shard,
        action_override=args.action_capacity_per_shard,
    )
    return V8RuntimeConfig.from_path(
        args.root,
        shards=args.shards,
        stage_workers=args.stage_workers,
        stage_ring_capacity=args.stage_ring_capacity,
        shard_ring_capacity=args.shard_ring_capacity,
        node_capacity_per_shard=plan.node_capacity_per_shard,
        edge_capacity_per_shard=plan.edge_capacity_per_shard,
        action_capacity_per_shard=plan.action_capacity_per_shard,
        snapshot_interval_seconds=args.snapshot_interval_seconds,
        enable_snapshots=not args.no_snapshots,
        restore=not args.no_restore,
        enable_peers=not args.no_peers,
        peer_interval_seconds=args.peer_interval_seconds,
    )


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="runs/v8/continuous")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--stage-workers", type=int, default=2)
    parser.add_argument("--stage-ring-capacity", type=int, default=8192)
    parser.add_argument("--shard-ring-capacity", type=int, default=8192)
    parser.add_argument("--node-capacity-per-shard", type=int, default=None)
    parser.add_argument("--edge-capacity-per-shard", type=int, default=None)
    parser.add_argument("--action-capacity-per-shard", type=int, default=None)
    parser.add_argument("--snapshot-interval-seconds", type=float, default=60.0)
    parser.add_argument("--peer-interval-seconds", type=float, default=0.5)
    parser.add_argument("--no-restore", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--no-peers", action="store_true")


def _actor_jobs(
    games: tuple[str, ...],
    *,
    actors: int,
    steps_per_game: int,
    seed: int,
    env_root: str | None,
    epsilon: float,
) -> tuple[ActorJob, ...]:
    if actors <= 0:
        raise ValueError("actors must be positive")
    jobs: list[ActorJob] = []
    actor_id = 1
    lanes = max(len(games), int(actors))
    base, extra = divmod(lanes, len(games))
    for game_index, game_id in enumerate(games):
        lane_count = max(1, base + int(game_index < extra))
        base_steps, extra_steps = divmod(int(steps_per_game), lane_count)
        for lane in range(lane_count):
            steps = base_steps + int(lane < extra_steps)
            if steps <= 0:
                continue
            jobs.append(
                ActorJob(
                    actor_id=actor_id,
                    game_id=game_id,
                    steps=steps,
                    seed=int(seed) + actor_id * 1009,
                    env_root=env_root,
                    epsilon=float(epsilon),
                )
            )
            actor_id += 1
    return tuple(jobs)


def _log(message: str) -> None:
    print(f'[{time.strftime("%H:%M")}] {message}', flush=True)


def _graph_load_line(*, snapshot_path: Path | None, restore_enabled: bool, nodes: int) -> str:
    if not restore_enabled:
        source = "empty(--no-restore)"
    elif snapshot_path is None:
        source = "empty(no-snapshot)"
    else:
        source = str(snapshot_path)
    return f"graph source={source} nodes={int(nodes)}"


def run_continuous(args) -> int:
    from v7.game_sets import resolve_game_selector

    if float(args.wait) < 0:
        raise ValueError("--wait must be non-negative")

    games = resolve_game_selector(args.games, args.env_root)
    jobs = _actor_jobs(
        games,
        actors=args.actors,
        steps_per_game=args.steps_per_game,
        seed=args.seed,
        env_root=args.env_root,
        epsilon=args.epsilon,
    )
    total_steps = sum(int(job.steps) for job in jobs)
    restore_enabled = not args.no_restore
    restore_source = latest_complete_snapshot(args.root) if restore_enabled else None
    runtime = ContinuousMemoryRuntime(_runtime_config(args, total_steps=total_steps))
    loaded_nodes = runtime.read_view.memory_count
    experiments = ExperimentSummary(0, 0, 0)
    hypothesis_reporting_started_at: float | None = None

    def accumulate_experiments(summary: ExperimentSummary) -> None:
        nonlocal experiments
        experiments = ExperimentSummary(
            experiments.attempted + summary.attempted,
            experiments.completed + summary.completed,
            experiments.passed + summary.passed,
        )

    def report_progress(rows) -> None:
        _log(format_game_rate_line(rows))
        if (
            hypothesis_reporting_started_at is not None
            and time.monotonic() - hypothesis_reporting_started_at
            >= _HYPOTHESIS_INITIAL_DELAY_SECONDS
        ):
            _log(format_hypothesis_line(runtime.scientific_statuses()))
        if args.no_automatic_experiments or args.no_peers:
            return
        remaining = max(0, int(args.max_transfer_experiments) - experiments.attempted)
        if remaining <= 0:
            return
        live = run_automatic_transfer_experiments(
            runtime,
            games=tuple(games),
            env_root=args.env_root,
            seed=args.seed + experiments.attempted * 7919,
            steps_per_trial=args.transfer_experiment_steps,
            max_trials=min(2, remaining),
        )
        accumulate_experiments(live)

    try:
        previous_wait = os.environ.get(_GAME_WAIT_ENV)
        os.environ[_GAME_WAIT_ENV] = str(float(args.wait))
        try:
            runtime.start()
            hypothesis_reporting_started_at = time.monotonic()
            print(
                f"v8 continuous: games={len(games)} actors={len(jobs)} shards={args.shards} "
                f"stage_workers={args.stage_workers} peers={'off' if args.no_peers else 'on'} "
                f"snapshots={'off' if args.no_snapshots else 'async'} wait={float(args.wait):g}s/game",
                flush=True,
            )
            _log(_graph_load_line(snapshot_path=restore_source, restore_enabled=restore_enabled, nodes=loaded_nodes))
            results = run_actor_jobs(
                runtime,
                jobs,
                timeout=args.actor_timeout,
                progress_interval_seconds=args.progress_interval_seconds,
                progress_callback=report_progress,
            )
        finally:
            if previous_wait is None:
                os.environ.pop(_GAME_WAIT_ENV, None)
            else:
                os.environ[_GAME_WAIT_ENV] = previous_wait

        runtime.wait_quiescent(timeout=args.drain_timeout)

        if not args.no_automatic_experiments and not args.no_peers:
            remaining = max(0, int(args.max_transfer_experiments) - experiments.attempted)
            if remaining > 0:
                tail = run_automatic_transfer_experiments(
                    runtime,
                    games=tuple(games),
                    env_root=args.env_root,
                    seed=args.seed + experiments.attempted * 7919,
                    steps_per_trial=args.transfer_experiment_steps,
                    max_trials=remaining,
                )
                accumulate_experiments(tail)
            runtime.wait_quiescent(timeout=args.drain_timeout)

        metrics = runtime.metrics()
        hypothesis_statuses = runtime.scientific_statuses()
        final = runtime.close(normal=True, timeout=args.final_save_timeout)
        summary = {
            "games": list(games),
            "actors": [asdict(result) for result in results],
            "automatic_transfer_experiments": asdict(experiments),
            "hypotheses": hypothesis_statuses,
            "metrics": metrics,
            "final_snapshot": None if final is None else asdict(final),
        }
        path = Path(args.root) / "v8_run_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return 0
    except KeyboardInterrupt:
        runtime.close(normal=True, timeout=args.final_save_timeout)
        raise
    except BaseException:
        runtime.close(normal=False)
        raise


def run_smoke(args) -> int:
    runtime = ContinuousMemoryRuntime(_runtime_config(args, total_steps=int(args.events)))
    try:
        runtime.start()
        for index in range(args.events):
            context = 10 + index % 3
            runtime.submit(
                runtime.make_experience(
                    producer_id=1,
                    producer_sequence=index + 1,
                    source_game_hash=1 + index % 2,
                    global_step=index,
                    context_signature=context,
                    action_id=index % 4,
                    outcome_signature=100 + index % 5,
                    family_signature=200 + index % 3,
                    carrier_signature=300 + index % 7,
                    future_option_delta=float((index % 3) - 1),
                    changed_cells=1 + index % 12,
                    trajectory_signature=400 + index % 9,
                    next_context_signature=10 + (index + 1) % 3,
                )
            )
        runtime.wait_quiescent(timeout=args.drain_timeout)
        metrics = runtime.metrics()
        print(
            f"v8 smoke done events={args.events} memories={metrics['memories']} edges={metrics['edges']}",
            flush=True,
        )
        runtime.close(normal=True, timeout=args.final_save_timeout)
        return 0
    except BaseException:
        runtime.close(normal=False)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arc-agi3-v8")
    sub = parser.add_subparsers(dest="command", required=True)

    continuous = sub.add_parser("continuous-run")
    _add_runtime_args(continuous)
    continuous.add_argument("--games", required=True)
    continuous.add_argument("--steps-per-game", type=int, default=1000)
    continuous.add_argument("--actors", type=int, default=8)
    continuous.add_argument("--seed", type=int, default=0)
    continuous.add_argument("--env-root", default=None)
    continuous.add_argument("--epsilon", type=float, default=0.10)
    continuous.add_argument(
        "--wait",
        type=float,
        default=1.0,
        help="seconds each actor waits after a complete ARC game episode (WIN/GAME_OVER)",
    )
    continuous.add_argument("--actor-timeout", type=float, default=None)
    continuous.add_argument("--progress-interval-seconds", type=float, default=60.0)
    continuous.add_argument("--drain-timeout", type=float, default=300.0)
    continuous.add_argument("--final-save-timeout", type=float, default=300.0)
    continuous.add_argument("--transfer-experiment-steps", type=int, default=32)
    continuous.add_argument("--max-transfer-experiments", type=int, default=8)
    continuous.add_argument("--no-automatic-experiments", action="store_true")

    smoke = sub.add_parser("smoke")
    _add_runtime_args(smoke)
    smoke.add_argument("--events", type=int, default=1000)
    smoke.add_argument("--drain-timeout", type=float, default=60.0)
    smoke.add_argument("--final-save-timeout", type=float, default=120.0)

    args = parser.parse_args(argv)
    if args.command == "continuous-run":
        return run_continuous(args)
    if args.command == "smoke":
        return run_smoke(args)
    return 2
