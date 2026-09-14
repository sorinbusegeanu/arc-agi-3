from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Any

from v9.hgt import train_hgt_epoch
from .lifecycle import run_lifecycle_maintenance
from .parallel_memory_coordinator_v2 import run_parallel_memory_jobs
from .transfer_validation import run_transfer_validation_interval


@dataclass(frozen=True, slots=True)
class EpochRunResult:
    epoch: int
    actors: tuple[dict[str, Any], ...]
    training: dict[str, Any]
    performance: dict[str, Any]
    metrics: dict[str, Any]


def build_epoch_jobs(specs: tuple[Any, ...], args: Any, *, epoch: int) -> list[tuple[int, Any, int, int]]:
    jobs = []
    actor_count = max(len(specs), int(args.actors))
    assigned = [specs[index % len(specs)] for index in range(actor_count)]
    lanes = {spec_index: sum(1 for index in range(actor_count) if index % len(specs) == spec_index) for spec_index in range(len(specs))}
    seen = {spec_index: 0 for spec_index in range(len(specs))}
    for actor_index, spec in enumerate(assigned):
        spec_index = actor_index % len(specs)
        lane_count = lanes[spec_index]
        lane = seen[spec_index]
        seen[spec_index] += 1
        base_steps, extra_steps = divmod(int(args.steps_per_game), lane_count)
        steps = base_steps + int(lane < extra_steps)
        if steps:
            actor_id = actor_index + 1
            seed = int(args.seed) + int(epoch) * 1_000_003 + actor_id * 1009
            jobs.append((actor_id, spec, steps, seed))
    return jobs


def _scenario_success(rows: list[Any]) -> tuple[dict[str, float], float]:
    totals: dict[str, tuple[int, int]] = {}
    for row in rows:
        successes, trials = totals.get(row.game_id, (0, 0))
        row_successes = int(getattr(row, "task_successes", 0))
        row_trials = row_successes + int(getattr(row, "task_failures", 0)) + int(getattr(row, "task_truncations", 0))
        totals[row.game_id] = (successes + row_successes, trials + row_trials)
    rates = {game_id: (successes / trials if trials else 0.0) for game_id, (successes, trials) in totals.items()}
    macro = sum(rates.values()) / len(rates) if rates else 0.0
    return rates, macro


def _game_level_metrics(rows: list[Any]) -> dict[str, Any]:
    by_game: dict[str, dict[str, int | float]] = {}
    for row in rows:
        target = by_game.setdefault(
            str(row.game_id),
            {
                "wins": 0,
                "failures": 0,
                "truncations": 0,
                "episodes": 0,
                "levels_completed": 0,
                "best_level": 0,
                "steps": 0,
            },
        )
        wins = int(getattr(row, "task_successes", 0))
        failures = int(getattr(row, "task_failures", 0))
        truncations = int(getattr(row, "task_truncations", 0))
        levels = int(getattr(row, "levels_completed", 0))
        target["wins"] += wins
        target["failures"] += failures
        target["truncations"] += truncations
        target["episodes"] += wins + failures + truncations
        target["levels_completed"] += levels
        target["best_level"] = max(int(target["best_level"]), levels)
        target["steps"] += int(getattr(row, "steps", 0))

    solved_games = sum(int(values["wins"]) > 0 for values in by_game.values())
    total_games = len(by_game)
    total_episodes = sum(int(values["episodes"]) for values in by_game.values())
    total_wins = sum(int(values["wins"]) for values in by_game.values())
    total_levels = sum(int(values["levels_completed"]) for values in by_game.values())
    return {
        "current_run_wins": total_wins / max(1, total_episodes),
        "current_run_solved_games": solved_games,
        "current_run_total_games": total_games,
        "current_run_levels_completed": total_levels,
        "current_run_best_level_by_game": {
            game_id: int(values["best_level"]) for game_id, values in sorted(by_game.items())
        },
        "current_run_game_results": by_game,
    }


def _performance_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    diagnostic = dict(metrics.get("telemetry_diagnostics", {}))
    sampled = int(diagnostic.get("sampled_steps", 0))
    ingested = int(diagnostic.get("ingested_steps", 0))
    backlog = int(diagnostic.get("sampling_backlog", max(0, sampled - ingested)))
    coordinator_actions = int(diagnostic.get("coordinator_action_requests", 0))
    return {
        "sampled_steps": sampled,
        "ingested_steps": ingested,
        "sampling_rate": float(diagnostic.get("sampling_rate", 0.0)),
        "ingestion_rate": float(diagnostic.get("ingestion_rate", 0.0)),
        "sampling_backlog": backlog,
        "ingest_queue_depth": int(diagnostic.get("ingest_queue_depth", 0)),
        "derivation_queue_depth": int(diagnostic.get("derivation_queue_depth", 0)),
        "coordinator_actions": coordinator_actions,
        "coordinator_action_requests": coordinator_actions,
        "policy_snapshot_generation": int(diagnostic.get("policy_snapshot_generation", 0)),
        "policy_snapshot_refreshes": int(diagnostic.get("policy_snapshot_refreshes", 0)),
        "optimized_sampling_path": bool(coordinator_actions == 0 and sampled == ingested and backlog == 0),
    }


def run_epochs(runtime: Any, specs: tuple[Any, ...], args: Any, *, adapter_factory: Any | None = None):
    actor_results = []
    epoch_results = []
    baseline_success: float | None = None
    transfer_attempted = transfer_completed = transfer_passed = transfer_validated = 0
    if adapter_factory is None:
        # Imported lazily to avoid the cli -> epoch_runner module cycle.
        from v9.cli import make_adapter as adapter_factory

    for epoch in range(1, int(args.epochs) + 1):
        jobs = build_epoch_jobs(specs, args, epoch=epoch)
        print(f"{time.strftime('[%H:%M]')} epoch {epoch}/{args.epochs} sampling start actors={len(jobs)}", flush=True)
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
            actor_view_refresh_steps=args.actor_view_refresh_steps,
            actor_view_refresh_ms=args.actor_view_refresh_ms,
        )
        actor_results.extend(process_results)
        runtime.wait_quiescent(args.drain_timeout)
        runtime.flush_deferred_memory_updates()

        # Current-epoch behavioral evidence must be visible before model validation
        # and before any memory is hidden or physically compacted.
        scenario_success, behavioral_success = _scenario_success(process_results)
        if baseline_success is None:
            baseline_success = behavioral_success
        behavioral_gain = behavioral_success - baseline_success
        runtime.set_telemetry_gauge("behavioral_success_rate", behavioral_success)
        runtime.set_telemetry_gauge("behavioral_success_gain", behavioral_gain)
        runtime.set_telemetry_gauge("successful_scenarios", sum(rate > 0.0 for rate in scenario_success.values()))
        game_level = _game_level_metrics(process_results)
        runtime.set_telemetry_gauge("current_run_wins", float(game_level["current_run_wins"]))
        runtime.set_telemetry_gauge("current_run_solved_games", int(game_level["current_run_solved_games"]))
        runtime.set_telemetry_gauge("current_run_total_games", int(game_level["current_run_total_games"]))
        runtime.set_telemetry_gauge("current_run_levels_completed", int(game_level["current_run_levels_completed"]))
        runtime.set_telemetry_gauge(
            "current_run_best_level_by_game",
            json.dumps(game_level["current_run_best_level_by_game"], sort_keys=True),
        )

        # M4 -> M5 is a causal gate. Drive only matched held-out trials from an
        # exactly restorable target state; failed trials remain failed evidence.
        transfer = run_transfer_validation_interval(
            runtime,
            specs,
            args,
            epoch=epoch,
            adapter_factory=adapter_factory,
        )
        transfer_attempted += int(transfer.attempted)
        transfer_completed += int(transfer.completed)
        transfer_passed += int(transfer.passed)
        transfer_validated += int(transfer.validated_concepts)
        runtime.set_telemetry_gauge("transfer_experiments_attempted", transfer_attempted)
        runtime.set_telemetry_gauge("transfer_experiments_completed", transfer_completed)
        runtime.set_telemetry_gauge("transfer_experiments_passed", transfer_passed)
        runtime.set_telemetry_gauge("transfer_concepts_validated", transfer_validated)
        runtime.set_telemetry_gauge("transfer_experiment_blocker", transfer.blocker or "")

        requested_training_steps = int(args.hgt_training_epochs)
        effective_training_steps = (
            requested_training_steps
            if requested_training_steps > 1
            else int(runtime.config.scientific.hgt_gradient_accumulation)
        )
        training = train_hgt_epoch(
            runtime,
            epoch=epoch,
            training_epochs=effective_training_steps,
            learning_rate=args.hgt_learning_rate,
            root=args.root,
        )
        print(
            f"{time.strftime('[%H:%M]')} epoch {epoch}/{args.epochs} training "
            f"status={training.status} model={training.model_version} "
            f"train_loss={training.training_loss:.4f} val_loss={training.validation_loss:.4f} "
            f"examples={training.examples} steps={training.training_steps}",
            flush=True,
        )

        # Compaction is evidence-gated by the just-computed behavioral and held-out
        # HGT metrics. Newly dormant memories affect the next epoch, not the model
        # evaluation used to decide whether forgetting is safe this epoch.
        lifecycle_result = run_lifecycle_maintenance(runtime)
        for key, value in lifecycle_result.items():
            runtime.set_telemetry_gauge(f"lifecycle_{key}", value)

        if runtime.config.enable_snapshots:
            runtime.snapshot()
        full_metrics = getattr(runtime, "full_metrics", runtime.metrics)
        metrics = full_metrics()
        performance = _performance_summary(metrics)
        epoch_results.append(
            EpochRunResult(
                epoch=epoch,
                actors=tuple(asdict(row) for row in process_results),
                training={
                    **asdict(training),
                    "behavioral_success_rate": behavioral_success,
                    "behavioral_success_gain": behavioral_gain,
                    "scenario_success_rate": scenario_success,
                    "game_level_metrics": game_level,
                    "transfer_validation": asdict(transfer),
                },
                performance=performance,
                metrics=metrics,
            )
        )
    return actor_results, epoch_results
