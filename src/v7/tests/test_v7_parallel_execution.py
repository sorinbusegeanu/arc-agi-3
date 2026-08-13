from __future__ import annotations

from pathlib import Path

from v7.derivation.scientific import EpisodeEvidence
from v7.environment.parallel_sampling import ParallelSamplingPool, SamplingBatchResult, SamplingJob
from v7.memory.reporting import StrictHypothesisReporter
from v7.memory.transport.base import ReadViewHandle
from v7.parallel import AdaptiveConcurrencyController, ParallelExecutionConfig, ParallelRuntimeMetrics


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_ram_aware_ramp_is_bounded_and_records_events():
    clock = _Clock()
    config = ParallelExecutionConfig(
        workers=4,
        initial_workers=2,
        ram_ramp_threshold_percent=80.0,
        initial_worker_ramp_delay_seconds=10.0,
        per_worker_ramp_delay_seconds=5.0,
    )
    metrics = ParallelRuntimeMetrics(4, 2)
    ram = [50.0]
    controller = AdaptiveConcurrencyController(config, memory_probe=lambda: ram[0], clock=clock)
    assert controller.maybe_ramp(metrics) == 2
    clock.value = 10.0
    assert controller.maybe_ramp(metrics) == 3
    ram[0] = 90.0
    clock.value = 15.0
    assert controller.maybe_ramp(metrics) == 3
    assert metrics.ram_ramp_blocked_count == 1
    ram[0] = 40.0
    clock.value = 20.0
    assert controller.maybe_ramp(metrics) == 4
    assert [event['workers'] for event in metrics.ramp_events] == [3, 4]


def _fake_worker(directory: str, handle: ReadViewHandle, job: SamplingJob) -> SamplingBatchResult:
    del directory, handle
    evidence = tuple(
        EpisodeEvidence(
            context_signature=100 + job.job_index,
            action_id=1,
            outcome_signature=200 + step,
            success=True,
            source_game=job.game_id,
            source_global_step=job.global_step_offset + step,
        )
        for step in range(1, job.steps + 1)
    )
    return SamplingBatchResult(
        job_index=job.job_index,
        epoch=job.epoch,
        game_id=job.game_id,
        seed=job.seed,
        steps=job.steps,
        wins=0,
        failures=0,
        levels_completed=0,
        resets=0,
        evidence=evidence,
        worker_seconds=0.01,
        mmap_reattach_count=0,
        mmap_reattach_seconds=0.0,
    )


def test_inline_sampling_pool_preserves_job_order_and_metrics(tmp_path: Path):
    config = ParallelExecutionConfig(workers=1, initial_workers=1)
    jobs = (
        SamplingJob(2, 0, 'g2', 2, 2, 20),
        SamplingJob(1, 0, 'g1', 2, 1, 10),
    )
    with ParallelSamplingPool(directory=tmp_path, config=config, worker_fn=_fake_worker) as pool:
        result = pool.run_wave(handle=ReadViewHandle(0, 'unused'), jobs=jobs)
        assert [row.job_index for row in result] == [1, 2]
        assert pool.metrics.jobs_completed == 2
        assert pool.metrics.steps == 4
        assert pool.metrics.evidence_rows == 4
        assert pool.metrics.peak_active_workers == 1


def test_parallel_reporting_matches_serial_and_is_stably_ordered():
    rows = {
        'H12': {'raw_decision': 'VALID', 'evidence': {'evidence_rows': 1, 'measurement': 1.0}},
        'H01': {'raw_decision': 'INVALID', 'evidence': {'evidence_rows': 2, 'measurement': 0.0}},
    }
    reporter = StrictHypothesisReporter()
    serial = reporter.evaluate_suite(rows, workers=1)
    parallel = reporter.evaluate_suite(rows, workers=4)
    assert tuple(parallel) == tuple(sorted(parallel))
    assert {key: value.final_decision for key, value in parallel.items()} == {
        key: value.final_decision for key, value in serial.items()
    }
