from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

from v7.environment.parallel_sampling import ParallelSamplingPool, SamplingJob
from v7.environment.runner import ArcGameRunResult
from v7.game_sets import resolve_game_selector
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
    try:
        with ParallelSamplingPool(directory=root_path / 'segments', config=parallel_config) as sampling_pool:
            for epoch in range(config.epochs):
                record = runtime.publisher.current_record
                if record is None:
                    raise RuntimeError('v7 runtime has no published generation')
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
                sampled = sampling_pool.run_wave(handle=record.handle, jobs=jobs)

                since_commit = 0
                for batch in sampled:
                    ingestion_started = perf_counter()
                    for row in batch.evidence:
                        runtime.observe(row)
                        since_commit += 1
                        if since_commit >= config.commit_every:
                            sampling_pool.metrics.canonical_ingestion_seconds += perf_counter() - ingestion_started
                            commit_started = perf_counter()
                            committed = runtime.commit()
                            sampling_pool.metrics.generation_commit_seconds += perf_counter() - commit_started
                            final_generation = int(committed.state.generation_id)
                            final_memories = len(committed.view.nodes)
                            since_commit = 0
                            ingestion_started = perf_counter()
                    sampling_pool.metrics.canonical_ingestion_seconds += perf_counter() - ingestion_started
                if since_commit:
                    commit_started = perf_counter()
                    committed = runtime.commit()
                    sampling_pool.metrics.generation_commit_seconds += perf_counter() - commit_started
                    final_generation = int(committed.state.generation_id)
                    final_memories = len(committed.view.nodes)
                    since_commit = 0

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
    return summary


def resolve_games(selector: str, env_root: str | None = None) -> tuple[str, ...]:
    return resolve_game_selector(selector, env_root)
