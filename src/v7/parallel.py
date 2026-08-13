from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable
import json


def ram_used_percent() -> float:
    try:
        values: dict[str, int] = {}
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            key, raw = line.split(':', 1)
            values[key] = int(raw.strip().split()[0])
        total = values['MemTotal']
        available = values.get('MemAvailable', values.get('MemFree', 0))
        return max(0.0, min(100.0, 100.0 * (1.0 - available / total))) if total else 0.0
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return 0.0


@dataclass(frozen=True, slots=True)
class ParallelExecutionConfig:
    workers: int = 1
    initial_workers: int | None = None
    derivation_workers: int = 4
    report_workers: int = 4
    max_tasks_per_child: int | None = None
    ram_ramp_threshold_percent: float = 85.0
    initial_worker_ramp_delay_seconds: float = 20.0
    per_worker_ramp_delay_seconds: float = 5.0
    max_in_flight_per_worker: int = 1

    def __post_init__(self) -> None:
        if self.workers <= 0 or self.derivation_workers <= 0 or self.report_workers <= 0:
            raise ValueError('worker counts must be positive')
        initial = self.workers if self.initial_workers is None else int(self.initial_workers)
        if initial <= 0 or initial > self.workers:
            raise ValueError('initial_workers must be in [1, workers]')
        if not 0.0 < self.ram_ramp_threshold_percent <= 100.0:
            raise ValueError('ram_ramp_threshold_percent must be in (0, 100]')
        if self.initial_worker_ramp_delay_seconds < 0 or self.per_worker_ramp_delay_seconds < 0:
            raise ValueError('worker ramp delays must be non-negative')
        if self.max_in_flight_per_worker <= 0:
            raise ValueError('max_in_flight_per_worker must be positive')

    @property
    def resolved_initial_workers(self) -> int:
        return self.workers if self.initial_workers is None else int(self.initial_workers)


@dataclass(slots=True)
class ParallelRuntimeMetrics:
    configured_workers: int
    initial_workers: int
    peak_active_workers: int = 0
    jobs_submitted: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    ramp_events: list[dict[str, float | int | str]] = field(default_factory=list)
    ram_ramp_blocked_count: int = 0
    in_flight_peak: int = 0
    sampling_wall_seconds: float = 0.0
    worker_seconds: float = 0.0
    steps: int = 0
    evidence_batches: int = 0
    evidence_rows: int = 0
    max_evidence_batch_rows: int = 0
    canonical_ingestion_seconds: float = 0.0
    generation_commit_seconds: float = 0.0
    mmap_reattach_count: int = 0
    mmap_reattach_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data['steps_per_second'] = 0.0 if self.sampling_wall_seconds <= 0 else self.steps / self.sampling_wall_seconds
        denom = self.sampling_wall_seconds * max(1, self.peak_active_workers)
        data['worker_utilization'] = 0.0 if denom <= 0 else min(1.0, self.worker_seconds / denom)
        data['mean_evidence_batch_rows'] = 0.0 if self.evidence_batches <= 0 else self.evidence_rows / self.evidence_batches
        return data

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding='utf-8')


class AdaptiveConcurrencyController:
    def __init__(self, config: ParallelExecutionConfig, *, memory_probe: Callable[[], float] = ram_used_percent, clock: Callable[[], float] = perf_counter) -> None:
        self.config = config
        self.memory_probe = memory_probe
        self.clock = clock
        self.current = config.resolved_initial_workers
        self.started = clock()
        self.last_ramp = self.started
        self._first_ramp_done = False

    def maybe_ramp(self, metrics: ParallelRuntimeMetrics) -> int:
        if self.current >= self.config.workers:
            return self.current
        now = self.clock()
        required_delay = self.config.per_worker_ramp_delay_seconds if self._first_ramp_done else self.config.initial_worker_ramp_delay_seconds
        if now - self.last_ramp < required_delay:
            return self.current
        ram = float(self.memory_probe())
        if ram >= self.config.ram_ramp_threshold_percent:
            metrics.ram_ramp_blocked_count += 1
            self.last_ramp = now
            return self.current
        self.current += 1
        self.last_ramp = now
        self._first_ramp_done = True
        metrics.ramp_events.append({'seconds': now - self.started, 'workers': self.current, 'ram_used_percent': ram})
        return self.current
