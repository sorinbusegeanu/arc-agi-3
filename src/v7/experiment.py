from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter

from v7.environment.ablation import MatureSamplingJob, ablation_names
from v7.environment.online_sampling import sample_job as online_sample_job
from v7.environment.parallel_sampling import ParallelSamplingPool
from v7.environment.runner import ArcGameRunResult
from v7.evaluation.cognition_metrics import (
    CognitionMetricsAccumulator,
    cross_game_transfer_counts,
)
from v7.game_sets import resolve_game_selector
from v7.hypotheses import evaluate_hypothesis_suite
from v7.memory.evidence_lifecycle import TransferTrialRecord
from v7.memory.evidence_store import EvidenceRecord
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId
from v7.parallel import ParallelExecutionConfig
from v7.runtime import V7Runtime, V7RuntimeConfig


@dataclass(frozen=True, slots=True)
class V7ExperimentConfig:
    games: tuple[str, ...]
    steps_per_game: int = 1000
    epochs: int = 1
    seed: int = 0
    env_root: str | None = None
    commit_every: int = 1000
    epsilon: float = 0.10
    workers: int = 1
    initial_workers: int | None = None
    derivation_workers: int = 4
    report_workers: int = 4
    max_tasks_per_child: int | None = None
    ram_ramp_threshold_percent: float = 85.0
    initial_worker_ramp_delay_seconds: float = 20.0
    per_worker_ramp_delay_seconds: float = 5.0
    ablation_mask: int = 0

    def __post_init__(self) -> None:
        if not self.games:
            raise ValueError("at least one game is required")
        if self.steps_per_game < 1 or self.epochs < 1 or self.commit_every < 1:
            raise ValueError("steps_per_game, epochs and commit_every must be positive")
        if self.workers <= 0 or self.derivation_workers <= 0 or self.report_workers <= 0:
            raise ValueError("worker counts must be positive")
        if int(self.ablation_mask) < 0:
            raise ValueError("ablation_mask must be non-negative")


@dataclass(frozen=True, slots=True)
class V7ExperimentResult:
    epochs: int
    games: int
    total_steps: int
    final_generation: int
    final_memories: int
    wins: int
    failures: int
    levels_completed: int
    runs: tuple[ArcGameRunResult, ...]
    execution_metrics: dict[str, object] = field(default_factory=dict)
    cognition_metrics: dict[str, object] = field(default_factory=dict)


def _log(epoch: int, message: str) -> None:
    print(f'[{time.strftime("%H:%M")} E{epoch + 1:04d}] {message}', flush=True)


def _append_hypothesis_log(root: Path, epoch: int, hypotheses: dict[str, dict[str, object]]) -> None:
    timestamp = time.strftime("%H:%M")
    lines = [
        f'[{timestamp} E{epoch + 1:04d}] {hypothesis_id} blockers: {"; ".join(payload["blockers"])}\n'
        for hypothesis_id, payload in sorted(hypotheses.items())
        if payload["final_decision"] != "VALID"
    ]
    if lines:
        with (root / "hypotheis.log").open("a", encoding="utf-8") as stream:
            stream.writelines(lines)


def _representative(actions: list[int]) -> int | None:
    if not actions:
        return None
    counts = Counter(int(value) for value in actions)
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def _sampling_shards(
    *,
    games: tuple[str, ...],
    epoch: int,
    steps_per_game: int,
    workers: int,
    seed: int,
    env_root: str | None,
    epsilon: float,
    ablation_mask: int,
    first_job_index: int,
    epoch_count: int,
) -> tuple[MatureSamplingJob, ...]:
    """Build balanced independent sampling lanes without multiplying step budget."""
    game_count = len(games)
    if game_count <= 0:
        return ()
    max_useful_jobs = game_count * int(steps_per_game)
    target_jobs = max(game_count, min(int(workers), max_useful_jobs))
    base_shards, extra_shards = divmod(target_jobs, game_count)
    shard_counts = tuple(
        base_shards + int(game_index < extra_shards)
        for game_index in range(game_count)
    )
    seed_stride = max(1, int(epoch_count) * game_count)
    jobs: list[MatureSamplingJob] = []
    next_job_index = int(first_job_index)
    for game_index, game_id in enumerate(games):
        shard_count = int(shard_counts[game_index])
        base_steps, extra_steps = divmod(int(steps_per_game), shard_count)
        game_ordinal = int(epoch) * game_count + game_index
        game_step_base = game_ordinal * int(steps_per_game)
        consumed = 0
        for shard_index in range(shard_count):
            shard_steps = base_steps + int(shard_index < extra_steps)
            jobs.append(
                MatureSamplingJob(
                    job_index=next_job_index,
                    epoch=int(epoch),
                    game_id=game_id,
                    steps=shard_steps,
                    seed=int(seed) + game_ordinal + shard_index * seed_stride,
                    global_step_offset=game_step_base + consumed,
                    env_root=env_root,
                    epsilon=float(epsilon),
                    ablation_mask=int(ablation_mask),
                )
            )
            consumed += shard_steps
            next_job_index += 1
    return tuple(jobs)


def _namespace_sampling_trajectories(sampled, jobs) -> tuple:
    """Give parallel lanes stable trajectory identities across epochs."""
    lane_by_job = {
        int(job.job_index): lane_index
        for lane_index, job in enumerate(jobs)
    }
    namespaced = []
    for batch in sampled:
        lane_index = lane_by_job[int(batch.job_index)]
        prefix = f"lane_{lane_index:04d}/"
        trajectories = tuple(
            replace(
                trajectory,
                level_key=prefix + str(trajectory.level_key),
            )
            for trajectory in batch.trajectories
        )
        namespaced.append(replace(batch, trajectories=trajectories))
    return tuple(namespaced)


def _append_aggregated_game_results(
    results: list[ArcGameRunResult],
    *,
    sampled,
    games: tuple[str, ...],
    generation: int,
    memories: int,
) -> None:
    grouped = defaultdict(list)
    for batch in sampled:
        grouped[str(batch.game_id)].append(batch)
    for game_id in games:
        batches = grouped.get(str(game_id), ())
        results.append(
            ArcGameRunResult(
                game_id=game_id,
                steps=sum(int(batch.steps) for batch in batches),
                generation=int(generation),
                memories=int(memories),
                wins=sum(int(batch.wins) for batch in batches),
                failures=sum(int(batch.failures) for batch in batches),
                levels_completed=sum(int(batch.levels_completed) for batch in batches),
                resets=sum(int(batch.resets) for batch in batches),
            )
        )


def _epoch_game_rates(sampled) -> tuple[float, float]:
    """Return shard-invariant percentages of games won and games advancing a level."""
    grouped = defaultdict(list)
    for batch in sampled:
        grouped[str(batch.game_id)].append(batch)
    games_played = len(grouped)
    if games_played == 0:
        return 0.0, 0.0
    games_won = sum(
        any(int(batch.wins) > 0 for batch in batches)
        for batches in grouped.values()
    )
    games_with_level = sum(
        any(int(batch.levels_completed) > 0 for batch in batches)
        for batches in grouped.values()
    )
    return (
        100.0 * games_won / games_played,
        100.0 * games_with_level / games_played,
    )


def _write_trajectory_evidence(runtime: V7Runtime, sampled) -> int:
    """Persist trajectories and credit terminal transfer to memories used along them."""
    generation_id = int(runtime.writer.mutable_generation_id)
    records: list[EvidenceRecord] = []
    transfer_records: list[TransferTrialRecord] = []
    for batch in sampled:
        actions: list[int] = []
        contexts: list[int] = []
        used_memory_ids: set[int] = set()
        usage_counts: Counter[int] = Counter()
        future_sum = 0.0
        raw_option_sum = 0.0
        level_index = 0
        terminal_index = 0
        trajectory_keys = tuple(
            str(trajectory.level_key)
            for trajectory in getattr(batch, "trajectories", ()) or ()
        )
        for row in batch.evidence:
            actions.append(int(row.action_id))
            context_values = tuple(
                int(value)
                for value in getattr(row, "context_signatures", ()) or ()
            )
            contexts.append(
                context_values[-2]
                if len(context_values) >= 2
                else int(row.context_signature)
            )
            for raw_memory_id in (
                *(getattr(row, "decision_role_ids", ()) or ()),
                *(getattr(row, "decision_concept_ids", ()) or ()),
            ):
                memory_id = int(raw_memory_id)
                used_memory_ids.add(memory_id)
                usage_counts[memory_id] += 1
            future_sum += float(row.future_option_delta)
            raw_option_sum += float(getattr(row, "raw_action_option_delta", 0.0) or 0.0)
            polarity = int(getattr(row, "terminal_polarity", 0) or 0)
            if polarity == 0:
                continue
            level_key = (
                trajectory_keys[terminal_index]
                if terminal_index < len(trajectory_keys)
                else f"level_{level_index:04d}"
            )
            terminal_index += 1
            records.append(
                EvidenceRecord(
                    memory_id=None,
                    evidence_type=int(EvidenceType.TRAJECTORY),
                    generation_id=generation_id,
                    source_game=batch.game_id,
                    source_context=level_key,
                    source_global_step=row.source_global_step,
                    payload={
                        "epoch": int(batch.epoch),
                        "level_key": level_key,
                        "steps_to_success": len(actions),
                        "future_option_sum": future_sum,
                        "raw_action_option_sum": raw_option_sum,
                        "future_option_per_action": future_sum / max(1, len(actions)),
                        "representative_action": _representative(actions),
                        "action_sequence": list(actions[:256]),
                        "context_sequence": list(contexts[:256]),
                        "success": polarity > 0,
                    },
                )
            )
            for raw_memory_id in sorted(used_memory_ids):
                memory_id = MemoryId(raw_memory_id)
                source_games = tuple(
                    game
                    for game in runtime.lifecycle_evidence.provenance_source_games(memory_id)
                    if game != batch.game_id
                )
                if not source_games:
                    continue
                if runtime.lifecycle_evidence.transfer_trial_exists(
                    memory_id,
                    target_game=batch.game_id,
                    source_global_step=row.source_global_step,
                ):
                    continue
                transfer_records.append(
                    TransferTrialRecord(
                        memory_id=memory_id,
                        generation_id=generation_id,
                        source_game=source_games[0],
                        target_game=batch.game_id,
                        success=polarity > 0,
                        score=1.0 if polarity > 0 else 0.0,
                        payload={
                            "source_games": list(source_games),
                            "source_game_count": len(source_games),
                            "target_context": level_key,
                            "source_global_step": row.source_global_step,
                            "future_option_delta": future_sum,
                            "raw_action_option_delta": raw_option_sum,
                            "attribution": "trajectory_usage",
                            "usage_count": int(usage_counts[raw_memory_id]),
                        },
                    )
                )
            if polarity > 0:
                level_index += 1
            actions.clear()
            contexts.clear()
            used_memory_ids.clear()
            usage_counts.clear()
            future_sum = 0.0
            raw_option_sum = 0.0
    evidence_rows = runtime.evidence.append_evidence_batch(records)
    transfer_rows = runtime.lifecycle_evidence.append_transfer_trials(transfer_records)
    return evidence_rows + transfer_rows


def _write_cognition_metrics(
    root: Path,
    *,
    epoch: int | None,
    accumulator: CognitionMetricsAccumulator,
    runtime: V7Runtime,
    ablation_mask: int,
) -> dict[str, object]:
    trials, successes = cross_game_transfer_counts(
        runtime.lifecycle_evidence.connection
    )
    payload = accumulator.snapshot(
        transfer_trials=trials,
        transfer_successes=successes,
    ).as_dict()
    payload["ablation_mask"] = int(ablation_mask)
    payload["ablations"] = list(ablation_names(ablation_mask))
    if epoch is None:
        path = root / "cognition_metrics.json"
    else:
        directory = root / "reports" / f"epoch_{epoch + 1:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "cognition_metrics.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def run_experiment(root: str | Path, config: V7ExperimentConfig) -> V7ExperimentResult:
    """Run parallel games with immutable shared memory and worker-local online learning."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    parallel_config = ParallelExecutionConfig(
        workers=config.workers,
        initial_workers=config.initial_workers,
        derivation_workers=config.derivation_workers,
        report_workers=config.report_workers,
        max_tasks_per_child=config.max_tasks_per_child,
        ram_ramp_threshold_percent=config.ram_ramp_threshold_percent,
        initial_worker_ramp_delay_seconds=config.initial_worker_ramp_delay_seconds,
        per_worker_ramp_delay_seconds=config.per_worker_ramp_delay_seconds,
    )
    active_ablations = ablation_names(config.ablation_mask)
    ablation_text = ",".join(active_ablations) if active_ablations else "none"
    print(
        f"v7 experiment: games={len(config.games)} epochs={config.epochs} steps/game={config.steps_per_game} "
        f"workers={config.workers} initial_workers={parallel_config.resolved_initial_workers} "
        f"derivation_workers={config.derivation_workers} report_workers={config.report_workers} "
        f"ablations={ablation_text}",
        flush=True,
    )
    runtime = V7Runtime(
        V7RuntimeConfig.from_path(
            root_path,
            restore=True,
            derivation_workers=config.derivation_workers,
            max_tasks_per_child=config.max_tasks_per_child,
        )
    )
    results: list[ArcGameRunResult] = []
    global_job_index = 0
    final_generation = int(runtime.writer.published_view.generation_id)
    final_memories = len(runtime.writer.published_view.nodes)
    metrics: dict[str, object] = {}
    cognition_metrics: dict[str, object] = {}
    cognition = CognitionMetricsAccumulator()
    running_win_rate = 0.0
    running_level_rate = 0.0
    try:
        with ParallelSamplingPool(
            directory=root_path / "segments",
            config=parallel_config,
            worker_fn=online_sample_job,
        ) as sampling_pool:
            for epoch in range(config.epochs):
                epoch_started = perf_counter()
                record = runtime.publisher.current_record
                if record is None:
                    raise RuntimeError("v7 runtime has no published generation")
                _log(
                    epoch,
                    f"epoch starting generation={int(record.generation_id)} memories={final_memories}",
                )
                jobs = _sampling_shards(
                    games=config.games,
                    epoch=epoch,
                    steps_per_game=config.steps_per_game,
                    workers=config.workers,
                    seed=config.seed,
                    env_root=config.env_root,
                    epsilon=config.epsilon,
                    ablation_mask=config.ablation_mask,
                    first_job_index=global_job_index,
                    epoch_count=config.epochs,
                )
                global_job_index += len(jobs)

                sampling_started = perf_counter()
                sampled = sampling_pool.run_wave(handle=record.handle, jobs=jobs)
                sampled = _namespace_sampling_trajectories(sampled, jobs)
                cognition.observe_epoch(epoch, sampled)
                epoch_win_rate, epoch_level_rate = _epoch_game_rates(sampled)
                running_win_rate += epoch_win_rate
                running_level_rate += epoch_level_rate
                completed_epochs = epoch + 1
                avg_win_rate = running_win_rate / completed_epochs
                avg_level_rate = running_level_rate / completed_epochs
                _log(
                    epoch,
                    f"sampling done seconds={perf_counter() - sampling_started:.2f}",
                )

                ingestion_phase_started = perf_counter()
                commit_count = 0
                buffer = []
                for batch in sampled:
                    buffer.extend(batch.evidence)
                    while len(buffer) >= config.commit_every:
                        chunk = tuple(buffer[: config.commit_every])
                        del buffer[: config.commit_every]
                        ingestion_started = perf_counter()
                        runtime.observe_batch(chunk)
                        sampling_pool.metrics.canonical_ingestion_seconds += (
                            perf_counter() - ingestion_started
                        )
                        commit_started = perf_counter()
                        committed = runtime.commit(
                            run_lifecycle=False,
                            derive_hierarchy=False,
                        )
                        sampling_pool.metrics.generation_commit_seconds += (
                            perf_counter() - commit_started
                        )
                        final_generation = int(committed.state.generation_id)
                        final_memories = len(committed.view.nodes)
                        commit_count += 1
                if buffer:
                    ingestion_started = perf_counter()
                    runtime.observe_batch(tuple(buffer))
                    sampling_pool.metrics.canonical_ingestion_seconds += (
                        perf_counter() - ingestion_started
                    )
                    commit_started = perf_counter()
                    committed = runtime.commit(
                        run_lifecycle=False,
                        derive_hierarchy=False,
                    )
                    sampling_pool.metrics.generation_commit_seconds += (
                        perf_counter() - commit_started
                    )
                    final_generation = int(committed.state.generation_id)
                    final_memories = len(committed.view.nodes)
                    commit_count += 1

                _write_trajectory_evidence(runtime, sampled)
                finalize_started = perf_counter()
                committed = runtime.commit(run_lifecycle=True, derive_hierarchy=True)
                sampling_pool.metrics.generation_commit_seconds += (
                    perf_counter() - finalize_started
                )
                final_generation = int(committed.state.generation_id)
                final_memories = len(committed.view.nodes)
                _log(
                    epoch,
                    f"ingestion done commits={commit_count} generation={final_generation} memories={final_memories} "
                    f"seconds={perf_counter() - ingestion_phase_started:.2f}",
                )

                _append_aggregated_game_results(
                    results,
                    sampled=sampled,
                    games=config.games,
                    generation=final_generation,
                    memories=final_memories,
                )

                hypotheses = evaluate_hypothesis_suite(
                    runtime,
                    epoch=epoch,
                    output_root=root_path,
                    workers=config.report_workers,
                )
                cognition_metrics = _write_cognition_metrics(
                    root_path,
                    epoch=epoch,
                    accumulator=cognition,
                    runtime=runtime,
                    ablation_mask=config.ablation_mask,
                )
                hypothesis_summary = " ".join(
                    f'{hypothesis_id}={payload["final_decision"]}'
                    for hypothesis_id, payload in sorted(hypotheses.items())
                )
                _log(epoch, f"hypotheses {hypothesis_summary}")
                _append_hypothesis_log(root_path, epoch, hypotheses)
                _log(
                    epoch,
                    f"cognition solved_games={cognition_metrics['ever_solved_game_count']}/{len(config.games)} "
                    f"repeat_rate={cognition_metrics['repeat_solution_rate']} "
                    f"retention_rate={cognition_metrics['solution_retention_rate']}",
                )
                _log(
                    epoch,
                    f"epoch done generation={final_generation} memories={final_memories} "
                    f"win_rate={epoch_win_rate:.1f}% level_rate={epoch_level_rate:.1f}% "
                    f"avg_win_rate={avg_win_rate:.1f}% avg_level_rate={avg_level_rate:.1f}% "
                    f"seconds={perf_counter() - epoch_started:.2f}",
                )

            cognition_metrics = _write_cognition_metrics(
                root_path,
                epoch=None,
                accumulator=cognition,
                runtime=runtime,
                ablation_mask=config.ablation_mask,
            )
            metrics = sampling_pool.metrics.as_dict()
            metrics.update(
                {
                    "derivation_workers": config.derivation_workers,
                    "report_workers": config.report_workers,
                    "nested_pool_oversubscription": False,
                    "generation_model": "immutable_wave_single_writer_with_local_online_overlay",
                    "ablation_mask": int(config.ablation_mask),
                    "ablations": list(active_ablations),
                }
            )
            (root_path / "parallel_runtime_metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    finally:
        runtime.close()

    summary = V7ExperimentResult(
        epochs=config.epochs,
        games=len(config.games),
        total_steps=sum(item.steps for item in results),
        final_generation=final_generation,
        final_memories=final_memories,
        wins=sum(item.wins for item in results),
        failures=sum(item.failures for item in results),
        levels_completed=sum(item.levels_completed for item in results),
        runs=tuple(results),
        execution_metrics=metrics,
        cognition_metrics=cognition_metrics,
    )
    (root_path / "experiment_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def resolve_games(selector: str, env_root: str | None = None) -> tuple[str, ...]:
    return resolve_game_selector(selector, env_root)
