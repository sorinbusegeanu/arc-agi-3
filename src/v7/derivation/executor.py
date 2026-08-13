from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from v7.derivation.batches import DerivedMutationBatch
from v7.derivation.vectorized import VectorizedDerivationEngine
from v7.derivation.workers import DerivationTask, DerivationTaskResult, DerivationTaskPlanner
from v7.memory.transport.base import ReadViewHandle
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport

_VIEW = None
_VIEW_GENERATION = -1
_ENGINE: VectorizedDerivationEngine | None = None
_TRANSPORT_DIRECTORY: str | None = None


def _init_worker(directory: str) -> None:
    global _ENGINE, _TRANSPORT_DIRECTORY
    _TRANSPORT_DIRECTORY = directory
    _ENGINE = VectorizedDerivationEngine()


def _ensure_view(handle: ReadViewHandle):
    global _VIEW, _VIEW_GENERATION
    if _TRANSPORT_DIRECTORY is None:
        raise RuntimeError('derivation worker transport is not initialized')
    if _VIEW is None or _VIEW_GENERATION != int(handle.generation_id):
        _VIEW = SegmentedMmapReadViewTransport(_TRANSPORT_DIRECTORY).attach(handle)
        _VIEW_GENERATION = int(handle.generation_id)
    return _VIEW


def _run_task(payload) -> DerivationTaskResult[DerivedMutationBatch]:
    handle, task, kernel = payload
    if _ENGINE is None:
        raise RuntimeError('derivation worker is not initialized')
    view = _ensure_view(handle)
    if int(task.generation_id) != int(view.generation_id):
        raise ValueError('derivation task targets a different generation')
    return _ENGINE.run(view, task, kernel)


@dataclass(frozen=True, slots=True)
class ParallelDerivationConfig:
    workers: int = 4
    max_tasks_per_child: int | None = None
    chunk_size: int = 256

    def __post_init__(self) -> None:
        if self.workers <= 0:
            raise ValueError('workers must be positive')
        if self.chunk_size <= 0:
            raise ValueError('chunk_size must be positive')


class ParallelDerivationExecutor:
    """Persistent process executor over compact dirty-range task descriptors."""

    def __init__(self, *, directory: str | Path, config: ParallelDerivationConfig | None = None) -> None:
        self.directory = str(directory)
        self.config = config or ParallelDerivationConfig()
        self._pool: ProcessPoolExecutor | None = None

    def start(self) -> None:
        if self._pool is not None or self.config.workers <= 1:
            return
        kwargs: dict[str, int] = {}
        if self.config.max_tasks_per_child is not None and int(self.config.max_tasks_per_child) > 0:
            kwargs['max_tasks_per_child'] = int(self.config.max_tasks_per_child)
        self._pool = ProcessPoolExecutor(max_workers=self.config.workers, initializer=_init_worker, initargs=(self.directory,), **kwargs)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None

    def __enter__(self) -> 'ParallelDerivationExecutor':
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run(self, *, handle: ReadViewHandle, tasks: Iterable[DerivationTask], kernel: Callable) -> tuple[DerivationTaskResult[DerivedMutationBatch], ...]:
        ordered_tasks = tuple(tasks)
        if not ordered_tasks:
            return ()
        if self.config.workers <= 1:
            view = SegmentedMmapReadViewTransport(self.directory).attach(handle)
            engine = VectorizedDerivationEngine()
            return tuple(engine.run(view, task, kernel) for task in ordered_tasks)
        self.start()
        if self._pool is None:
            raise RuntimeError('derivation pool failed to start')
        payloads = tuple((handle, task, kernel) for task in ordered_tasks)
        return tuple(self._pool.map(_run_task, payloads))

    def run_dirty_plan(self, *, handle: ReadViewHandle, plan, kernel: Callable):
        tasks = DerivationTaskPlanner(chunk_size=self.config.chunk_size).build(plan, generation_id=handle.generation_id)
        return self.run(handle=handle, tasks=tasks, kernel=kernel)
