from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

from v9.hgt import train_hgt_epoch

from .parallel_memory_coordinator import run_parallel_memory_jobs


@dataclass(frozen=True, slots=True)
class EpochRunResult:
    epoch: int
    actors: tuple[dict[str, Any], ...]
    training: dict[str, Any]
    metrics: dict[str, Any]


def build_epoch_jobs(specs: tuple[Any, ...], args: Any, *, epoch: int) -> list[tuple[int, Any, int, int]]:
    jobs = []
    actor_count = max(len(specs), int(args.actors))
    assigned = [specs[index % len(specs)] for index in range(actor_count)]
    lanes = {
        spec_index: sum(1 for index in range(actor_count) if index % len(specs) == spec_index)
        for spec_index in range(len(specs))
    }
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
        success, episodes = totals.get(row.game_id, (0, 0))
        totals[row.game_id] = (
            success + int(row.positive_boundaries),
            episodes + int(row.episode_boundaries),
        )
    rates = {
        game_id: (success / episodes if episodes else 0.0)
        for game_id, (success, episodes) in totals.items()
    }
    macro = sum(rates.values()) / len(rates) if rates else 0.0
    return rates, macro


def run_epochs(runtime: Any, specs: tuple[Any, ...], args: Any):
    actor_results = []
    epoch_results = []
    baseline_success: float | None = None
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
        scenario_success, behavioral_success = _scenario_success(process_results)
        if baseline_success is None:
            baseline_success = behavioral_success
        behavioral_gain = behavioral_success - baseline_success
        runtime.set_telemetry_gauge("behavioral_success_rate", behavioral_success)
        runtime.set_telemetry_gauge("behavioral_success_gain", behavioral_gain)
        runtime.set_telemetry_gauge("successful_scenarios", sum(rate > 0.0 for rate in scenario_success.values()))

        print(f"{time.strftime('[%H:%M]')} epoch {epoch}/{args.epochs} training start", flush=True)
        training = train_hgt_epoch(
            runtime,
            epoch=epoch,
            training_epochs=args.hgt_training_epochs,
            learning_rate=args.hgt_learning_rate,
            root=args.root,
        )
        print(
            f"{time.strftime('[%H:%M]')} epoch {epoch}/{args.epochs} training status={training.status} "
            f"model={training.model_version} train_loss={training.training_loss:.4f} "
            f"val_loss={training.validation_loss:.4f} examples={training.examples}",
            flush=True,
        )
        if runtime.config.enable_snapshots:
            runtime.snapshot()
        epoch_results.append(
            EpochRunResult(
                epoch=epoch,
                actors=tuple(asdict(row) for row in process_results),
                training={
                    **asdict(training),
                    "behavioral_success_rate": behavioral_success,
                    "behavioral_success_gain": behavioral_gain,
                    "scenario_success_rate": scenario_success,
                },
                metrics=runtime.metrics(),
            )
        )
    return actor_results, epoch_results
