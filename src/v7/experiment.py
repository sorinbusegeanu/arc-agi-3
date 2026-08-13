from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

from v7.environment.parallel_sampling import ParallelSamplingPool, SamplingJob
from v7.environment.runner import ArcGameRunResult
from v7.game_sets import resolve_game_selector
from v7.memory.ids import MemoryLevel
from v7.memory.reporting import StrictHypothesisReporter
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


def _epoch_hypothesis_rows(runtime: V7Runtime) -> dict[str, dict[str, object]]:
    """Build conservative v7 hypothesis rows from evidence already represented in the runtime.

    These rows never upgrade a hypothesis to VALID from proxy evidence alone. Where
    v7 has only structural emergence but not the full hypothesis-specific measurement,
    the decision remains PARTIALLY_VALID or INSUFFICIENT_EVIDENCE.
    """
    view = runtime.writer.published_view
    level_counts = {
        level: sum(1 for node in view.nodes.values() if node.level == level)
        for level in MemoryLevel
    }
    supported = {
        level: sum(1 for node in view.nodes.values() if node.level == level and int(node.support_count) >= 2)
        for level in MemoryLevel
    }
    pe_count = sum(1 for score in view.scores.values() if float(score.prediction_error) > 0.0)
    fo_count = sum(1 for score in view.scores.values() if float(score.future_option_delta) != 0.0)

    rows: dict[str, dict[str, object]] = {}

    def row(hypothesis_id: str, decision: str, measurement: object, evidence_rows: int, *, proxy_only: bool = False) -> None:
        rows[hypothesis_id] = {
            'raw_decision': decision,
            'evidence': {
                'evidence_rows': int(evidence_rows),
                'measurement': measurement,
                'proxy_only': bool(proxy_only),
            },
        }

    row('H01', 'VALID' if supported[MemoryLevel.M1] > 0 else ('PARTIALLY_VALID' if level_counts[MemoryLevel.M1] > 0 else 'INSUFFICIENT_EVIDENCE'), {'m1': level_counts[MemoryLevel.M1], 'supported_m1': supported[MemoryLevel.M1]}, level_counts[MemoryLevel.M1])
    row('H02', 'PARTIALLY_VALID' if pe_count > 0 else 'INSUFFICIENT_EVIDENCE', {'prediction_error_memories': pe_count}, pe_count, proxy_only=True)
    row('H03', 'VALID' if supported[MemoryLevel.M2] > 0 else ('PARTIALLY_VALID' if level_counts[MemoryLevel.M2] > 0 else 'INSUFFICIENT_EVIDENCE'), {'m2': level_counts[MemoryLevel.M2], 'supported_m2': supported[MemoryLevel.M2]}, level_counts[MemoryLevel.M2])
    row('H04', 'INSUFFICIENT_EVIDENCE', {'carrier_specific_evaluator': False}, 0)
    row('H05', 'VALID' if supported[MemoryLevel.M3] > 0 else ('PARTIALLY_VALID' if level_counts[MemoryLevel.M3] > 0 else 'INSUFFICIENT_EVIDENCE'), {'m3': level_counts[MemoryLevel.M3], 'supported_m3': supported[MemoryLevel.M3]}, level_counts[MemoryLevel.M3])
    row('H06', 'INSUFFICIENT_EVIDENCE', {'role_transfer_specific_evaluator': False}, 0)
    row('H07', 'PARTIALLY_VALID' if level_counts[MemoryLevel.M4] > 0 else 'INSUFFICIENT_EVIDENCE', {'m4': level_counts[MemoryLevel.M4]}, level_counts[MemoryLevel.M4], proxy_only=True)
    row('H08', 'PARTIALLY_VALID' if level_counts[MemoryLevel.M5] > 0 else 'INSUFFICIENT_EVIDENCE', {'m5': level_counts[MemoryLevel.M5]}, level_counts[MemoryLevel.M5], proxy_only=True)
    row('H09', 'PARTIALLY_VALID' if fo_count > 0 else 'INSUFFICIENT_EVIDENCE', {'future_option_memories': fo_count}, fo_count, proxy_only=True)
    row('H10', 'INSUFFICIENT_EVIDENCE', {'future_option_attention_specific_evaluator': False}, 0)
    row('H11', 'INSUFFICIENT_EVIDENCE', {'future_option_transfer_specific_evaluator': False}, 0)
    row('H12', 'PARTIALLY_VALID' if level_counts[MemoryLevel.M6] > 0 else 'INSUFFICIENT_EVIDENCE', {'m6': level_counts[MemoryLevel.M6]}, level_counts[MemoryLevel.M6], proxy_only=True)
    return rows


def _log_hypothesis_summary(epoch: int, runtime: V7Runtime, workers: int) -> None:
    reports = StrictHypothesisReporter().evaluate_suite(_epoch_hypothesis_rows(runtime), workers=workers)
    summary = ' '.join(f'{hypothesis_id}={report.final_decision}' for hypothesis_id, report in reports.items())
    _log(epoch, f'hypotheses {summary}')


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
                _log(epoch, f'sampling starting jobs={len(config.games)} workers={config.workers} initial_workers={parallel_config.resolved_initial_workers}')
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

                since_commit = 0
                evidence_rows = sum(len(batch.evidence) for batch in sampled)
                _log(epoch, f'ingestion starting evidence_rows={evidence_rows} commit_every={config.commit_every}')
                ingestion_phase_started = perf_counter()
                commit_count = 0
                for batch in sampled:
                    ingestion_started = perf_counter()
                    for evidence in batch.evidence:
                        runtime.observe(evidence)
                        since_commit += 1
                        if since_commit >= config.commit_every:
                            sampling_pool.metrics.canonical_ingestion_seconds += perf_counter() - ingestion_started
                            commit_started = perf_counter()
                            committed = runtime.commit()
                            sampling_pool.metrics.generation_commit_seconds += perf_counter() - commit_started
                            final_generation = int(committed.state.generation_id)
                            final_memories = len(committed.view.nodes)
                            commit_count += 1
                            since_commit = 0
                            ingestion_started = perf_counter()
                    sampling_pool.metrics.canonical_ingestion_seconds += perf_counter() - ingestion_started
                if since_commit:
                    commit_started = perf_counter()
                    committed = runtime.commit()
                    sampling_pool.metrics.generation_commit_seconds += perf_counter() - commit_started
                    final_generation = int(committed.state.generation_id)
                    final_memories = len(committed.view.nodes)
                    commit_count += 1
                    since_commit = 0
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
                _log_hypothesis_summary(epoch, runtime, config.report_workers)
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
