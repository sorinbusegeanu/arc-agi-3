from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Callable, Iterable

from v7.derivation.scientific import EpisodeEvidence
from v7.environment.arc_adapter import ArcGridEnvironment
from v7.environment.encoding import SupportedPredictionTracker, grid_signature, transition_signature
from v7.memory.read_view import MemoryReadView
from v7.memory.scoring import VectorizedActionScorer
from v7.memory.transport.base import ReadViewHandle
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport
from v7.parallel import AdaptiveConcurrencyController, ParallelExecutionConfig, ParallelRuntimeMetrics

_WORKER_VIEW: MemoryReadView | None = None
_WORKER_GENERATION = -1


@dataclass(frozen=True, slots=True)
class SamplingJob:
    job_index: int
    epoch: int
    game_id: str
    steps: int
    seed: int
    global_step_offset: int
    env_root: str | None = None
    epsilon: float = 0.10


@dataclass(frozen=True, slots=True)
class SamplingBatchResult:
    job_index: int
    epoch: int
    game_id: str
    seed: int
    steps: int
    wins: int
    failures: int
    levels_completed: int
    resets: int
    evidence: tuple[EpisodeEvidence, ...]
    worker_seconds: float
    mmap_reattach_count: int
    mmap_reattach_seconds: float


def _attach_generation(directory: str, handle: ReadViewHandle) -> tuple[MemoryReadView, int, float]:
    global _WORKER_VIEW, _WORKER_GENERATION
    generation = int(handle.generation_id)
    if _WORKER_VIEW is not None and _WORKER_GENERATION == generation:
        return _WORKER_VIEW, 0, 0.0
    started = perf_counter()
    _WORKER_VIEW = SegmentedMmapReadViewTransport(directory).attach(handle)
    _WORKER_GENERATION = generation
    return _WORKER_VIEW, 1, perf_counter() - started


def _choose_action(view: MemoryReadView, actions: Iterable[int], rng: Random, epsilon: float) -> int:
    ordered = tuple(sorted(set(int(action) for action in actions)))
    if not ordered:
        raise ValueError('environment returned no available actions')
    batch = VectorizedActionScorer().score(view.packed_cognition, ordered)
    unseen = [
        int(action)
        for action, count in zip(batch.action_ids, batch.evidence_counts, strict=True)
        if int(count) == 0
    ]
    if unseen:
        return unseen[rng.randrange(len(unseen))]
    if rng.random() < float(epsilon):
        return ordered[rng.randrange(len(ordered))]
    best = batch.best_action()
    return ordered[0] if best is None else int(best)


def sample_job(directory: str, handle: ReadViewHandle, job: SamplingJob) -> SamplingBatchResult:
    started = perf_counter()
    view, reattach_count, reattach_seconds = _attach_generation(directory, handle)
    env = ArcGridEnvironment(game_id=job.game_id, seed=job.seed, env_root=job.env_root)
    rng = Random(job.seed)
    predictor = SupportedPredictionTracker()
    evidence_rows: list[EpisodeEvidence] = []
    wins = failures = levels_completed = 0
    for local_step in range(1, job.steps + 1):
        before = env.observe()
        before_actions = env.available_actions()
        action = _choose_action(view, before_actions, rng, job.epsilon)
        context = grid_signature(before)
        after = env.step(action)
        outcome = transition_signature(before, after)
        prediction_error = predictor.prediction_error(context, action, outcome)
        predictor.observe(context, action, outcome)
        after_actions = env.available_actions()
        evidence_rows.append(EpisodeEvidence(
            context_signature=context,
            action_id=action,
            outcome_signature=outcome,
            success=env.last_outcome_polarity != 'negative',
            prediction_error=prediction_error,
            future_option_delta=float(len(set(after_actions)) - len(set(before_actions))),
            source_game=job.game_id,
            source_context=str(context),
            source_global_step=job.global_step_offset + local_step,
        ))
        wins += int(env.last_outcome_state == 'WIN')
        failures += int(env.last_outcome_state == 'GAME_OVER')
        levels_completed += int(bool(env.level_completed_event))
    return SamplingBatchResult(
        job_index=job.job_index,
        epoch=job.epoch,
        game_id=job.game_id,
        seed=job.seed,
        steps=job.steps,
        wins=wins,
        failures=failures,
        levels_completed=levels_completed,
        resets=env.reset_count,
        evidence=tuple(evidence_rows),
        worker_seconds=perf_counter() - started,
        mmap_reattach_count=reattach_count,
        mmap_reattach_seconds=reattach_seconds,
    )


WorkerFunction = Callable[[str, ReadViewHandle, SamplingJob], SamplingBatchResult]
ProgressCallback = Callable[[SamplingBatchResult, int, int], None]


class ParallelSamplingPool:
    """Persistent ARC sampling pool with bounded in-flight work and RAM-aware ramp."""

    def __init__(
        self,
        *,
        directory: str | Path,
        config: ParallelExecutionConfig,
        worker_fn: WorkerFunction = sample_job,
        memory_probe: Callable[[], float] | None = None,
    ) -> None:
        self.directory = str(directory)
        self.config = config
        self.worker_fn = worker_fn
        self.metrics = ParallelRuntimeMetrics(config.workers, config.resolved_initial_workers)
        self._memory_probe = memory_probe
        self._pool: ProcessPoolExecutor | None = None
        if config.workers > 1:
            kwargs: dict[str, int] = {}
            if config.max_tasks_per_child is not None and int(config.max_tasks_per_child) > 0:
                kwargs['max_tasks_per_child'] = int(config.max_tasks_per_child)
            self._pool = ProcessPoolExecutor(max_workers=config.workers, **kwargs)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None

    def __enter__(self) -> 'ParallelSamplingPool':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run_wave(
        self,
        *,
        handle: ReadViewHandle,
        jobs: Iterable[SamplingJob],
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[SamplingBatchResult, ...]:
        ordered_jobs = tuple(sorted(jobs, key=lambda job: job.job_index))
        if not ordered_jobs:
            return ()
        total = len(ordered_jobs)
        started = perf_counter()
        if self._pool is None:
            outputs_list: list[SamplingBatchResult] = []
            for job in ordered_jobs:
                result = self.worker_fn(self.directory, handle, job)
                outputs_list.append(result)
                if progress_callback is not None:
                    progress_callback(result, len(outputs_list), total)
            outputs = tuple(outputs_list)
            self._record_results(outputs, perf_counter() - started, peak=1)
            return outputs

        controller = AdaptiveConcurrencyController(
            self.config,
            **({} if self._memory_probe is None else {'memory_probe': self._memory_probe}),
        )
        pending_index = 0
        futures: dict[Future[SamplingBatchResult], SamplingJob] = {}
        outputs: list[SamplingBatchResult] = []
        peak = 0
        while pending_index < len(ordered_jobs) or futures:
            active = controller.maybe_ramp(self.metrics)
            capacity = active * self.config.max_in_flight_per_worker
            while pending_index < len(ordered_jobs) and len(futures) < capacity:
                job = ordered_jobs[pending_index]
                pending_index += 1
                future = self._pool.submit(self.worker_fn, self.directory, handle, job)
                futures[future] = job
                self.metrics.jobs_submitted += 1
                self.metrics.in_flight_peak = max(self.metrics.in_flight_peak, len(futures))
            peak = max(peak, min(active, len(futures)))
            if not futures:
                continue
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                try:
                    result = future.result()
                    outputs.append(result)
                    self.metrics.jobs_completed += 1
                    if progress_callback is not None:
                        progress_callback(result, len(outputs), total)
                except Exception:
                    self.metrics.jobs_failed += 1
                    raise
        outputs.sort(key=lambda item: item.job_index)
        self._record_results(tuple(outputs), perf_counter() - started, peak=peak)
        return tuple(outputs)

    def _record_results(self, outputs: tuple[SamplingBatchResult, ...], wall_seconds: float, *, peak: int) -> None:
        if self._pool is None:
            self.metrics.jobs_submitted += len(outputs)
            self.metrics.jobs_completed += len(outputs)
            self.metrics.in_flight_peak = max(self.metrics.in_flight_peak, 1 if outputs else 0)
        self.metrics.peak_active_workers = max(self.metrics.peak_active_workers, peak)
        self.metrics.sampling_wall_seconds += wall_seconds
        for result in outputs:
            self.metrics.worker_seconds += result.worker_seconds
            self.metrics.steps += result.steps
            self.metrics.evidence_batches += 1
            self.metrics.evidence_rows += len(result.evidence)
            self.metrics.max_evidence_batch_rows = max(self.metrics.max_evidence_batch_rows, len(result.evidence))
            self.metrics.mmap_reattach_count += result.mmap_reattach_count
            self.metrics.mmap_reattach_seconds += result.mmap_reattach_seconds
