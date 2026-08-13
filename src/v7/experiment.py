from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

from v7.environment.parallel_sampling import ParallelSamplingPool, SamplingJob
from v7.environment.runner import ArcGameRunResult
from v7.game_sets import resolve_game_selector
from v7.hypotheses import evaluate_hypothesis_suite
from v7.memory.evidence_store import EvidenceRecord
from v7.memory.evidence_types import EvidenceType
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

    def __post_init__(self) -> None:
        if not self.games:
            raise ValueError('at least one game is required')
        if self.steps_per_game < 1 or self.epochs < 1 or self.commit_every < 1:
            raise ValueError('steps_per_game, epochs and commit_every must be positive')
        if self.workers <= 0 or self.derivation_workers <= 0 or self.report_workers <= 0:
            raise ValueError('worker counts must be positive')


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


def _log(epoch: int, message: str) -> None:
    print(f'[{time.strftime("%H:%M")} E{epoch + 1:04d}] {message}', flush=True)


def _write_trajectory_evidence(runtime: V7Runtime, sampled) -> int:
    generation_id = int(runtime.writer.mutable_generation_id)
    records = []
    for batch in sampled:
        for row in batch.trajectories:
            records.append(EvidenceRecord(
                memory_id=None,
                evidence_type=int(EvidenceType.TRAJECTORY),
                generation_id=generation_id,
                source_game=row.game_id,
                source_context=row.level_key,
                source_global_step=row.source_global_step,
                payload={
                    'epoch': int(row.epoch),
                    'level_key': row.level_key,
                    'steps_to_success': int(row.steps_to_success),
                    'future_option_sum': float(row.future_option_sum),
                    'future_option_per_action': float(row.future_option_sum) / max(1, int(row.steps_to_success)),
                    'representative_action': row.representative_action,
                    'success': bool(row.success),
                },
            ))
    return runtime.evidence.append_evidence_batch(records)


def run_experiment(root: str | Path, config: V7ExperimentConfig) -> V7ExperimentResult:
    """Run deterministic parallel sampling waves against immutable v7 generations."""
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
    print(
        f'v7 experiment: games={len(config.games)} epochs={config.epochs} steps/game={config.steps_per_game} '
        f'workers={config.workers} initial_workers={parallel_config.resolved_initial_workers} '
        f'derivation_workers={config.derivation_workers} report_workers={config.report_workers}',
        flush=True,
    )
    runtime = V7Runtime(V7RuntimeConfig.from_path(
        root_path,
        restore=True,
        derivation_workers=config.derivation_workers,
        max_tasks_per_child=config.max_tasks_per_child,
    ))
    results: list[ArcGameRunResult] = []
    global_job_index = 0
    final_generation = int(runtime.writer.published_view.generation_id)
    final_memories = len(runtime.writer.published_view.nodes)
    metrics: dict[str, object] = {}
    try:
        with ParallelSamplingPool(directory=root_path / 'segments', config=parallel_config) as sampling_pool:
            for epoch in range(config.epochs):
                epoch_started = perf_counter()
                record = runtime.publisher.current_record
                if record is None:
                    raise RuntimeError('v7 runtime has no published generation')
                _log(epoch, f'epoch starting generation={int(record.generation_id)} memories={final_memories}')
                jobs: list[SamplingJob] = []
                for game_index, game_id in enumerate(config.games):
                    ordinal = epoch * len(config.games) + game_index
                    jobs.append(SamplingJob(
                        job_index=global_job_index,
                        epoch=epoch,
                        game_id=game_id,
                        steps=config.steps_per_game,
                        seed=config.seed + ordinal,
                        global_step_offset=ordinal * config.steps_per_game,
                        env_root=config.env_root,
                        epsilon=config.epsilon,
                    ))
                    global_job_index += 1

                sampling_started = perf_counter()
                sampled = sampling_pool.run_wave(handle=record.handle, jobs=jobs)
                _log(
                    epoch,
                    f'sampling done jobs={len(sampled)} levels={sum(batch.levels_completed for batch in sampled)} '
                    f'wins={sum(batch.wins for batch in sampled)} failures={sum(batch.failures for batch in sampled)} '
                    f'seconds={perf_counter() - sampling_started:.2f}',
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
                        sampling_pool.metrics.canonical_ingestion_seconds += perf_counter() - ingestion_started
                        commit_started = perf_counter()
                        committed = runtime.commit(run_lifecycle=False, derive_hierarchy=False)
                        sampling_pool.metrics.generation_commit_seconds += perf_counter() - commit_started
                        final_generation = int(committed.state.generation_id)
                        final_memories = len(committed.view.nodes)
                        commit_count += 1
                if buffer:
                    ingestion_started = perf_counter()
                    runtime.observe_batch(tuple(buffer))
                    sampling_pool.metrics.canonical_ingestion_seconds += perf_counter() - ingestion_started
                    commit_started = perf_counter()
                    committed = runtime.commit(run_lifecycle=False, derive_hierarchy=False)
                    sampling_pool.metrics.generation_commit_seconds += perf_counter() - commit_started
                    final_generation = int(committed.state.generation_id)
                    final_memories = len(committed.view.nodes)
                    commit_count += 1
                    buffer.clear()

                _write_trajectory_evidence(runtime, sampled)
                finalize_started = perf_counter()
                committed = runtime.commit(run_lifecycle=True, derive_hierarchy=True)
                sampling_pool.metrics.generation_commit_seconds += perf_counter() - finalize_started
                final_generation = int(committed.state.generation_id)
                final_memories = len(committed.view.nodes)
                _log(
                    epoch,
                    f'ingestion done commits={commit_count} generation={final_generation} memories={final_memories} '
                    f'seconds={perf_counter() - ingestion_phase_started:.2f}',
                )

                for batch in sampled:
                    results.append(ArcGameRunResult(
                        game_id=batch.game_id,
                        steps=batch.steps,
                        generation=final_generation,
                        memories=final_memories,
                        wins=batch.wins,
                        failures=batch.failures,
                        levels_completed=batch.levels_completed,
                        resets=batch.resets,
                    ))

                hypotheses = evaluate_hypothesis_suite(
                    runtime,
                    epoch=epoch,
                    output_root=root_path,
                    workers=config.report_workers,
                )
                summary = ' '.join(
                    f'{hypothesis_id}={payload["final_decision"]}'
                    for hypothesis_id, payload in sorted(hypotheses.items())
                )
                _log(epoch, f'hypotheses {summary}')
                _log(
                    epoch,
                    f'epoch done generation={final_generation} memories={final_memories} '
                    f'levels={sum(batch.levels_completed for batch in sampled)} wins={sum(batch.wins for batch in sampled)} '
                    f'failures={sum(batch.failures for batch in sampled)} seconds={perf_counter() - epoch_started:.2f}',
                )

            metrics = sampling_pool.metrics.as_dict()
            metrics.update({
                'derivation_workers': config.derivation_workers,
                'report_workers': config.report_workers,
                'nested_pool_oversubscription': False,
                'generation_model': 'immutable_wave_single_writer',
            })
            (root_path / 'parallel_runtime_metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8')
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
    )
    (root_path / 'experiment_summary.json').write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding='utf-8')
    print(
        f'v7 experiment complete: epochs={summary.epochs} games={summary.games} total_steps={summary.total_steps} '
        f'generation={summary.final_generation} memories={summary.final_memories} '
        f'levels={summary.levels_completed} wins={summary.wins} failures={summary.failures}',
        flush=True,
    )
    return summary


def resolve_games(selector: str, env_root: str | None = None) -> tuple[str, ...]:
    return resolve_game_selector(selector, env_root)
