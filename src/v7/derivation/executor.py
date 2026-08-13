from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from v7.derivation.batches import DerivedMutationBatch
from v7.derivation.vectorized import VectorizedDerivationEngine
from v7.derivation.workers import DerivationTask, DerivationTaskResult
from v7.memory.transport.base import ReadViewHandle
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport

_KERNEL: Callable | None = None
_VIEW = None
_ENGINE: VectorizedDerivationEngine | None = None


def _init_worker(directory: str, handle: ReadViewHandle, kernel: Callable) -> None:
    global _KERNEL, _VIEW, _ENGINE
    _KERNEL = kernel
    _VIEW = SegmentedMmapReadViewTransport(directory).attach(handle)
    _ENGINE = VectorizedDerivationEngine()


def _run_task(task: DerivationTask) -> DerivationTaskResult[DerivedMutationBatch]:
    if _KERNEL is None or _VIEW is None or _ENGINE is None:
        raise RuntimeError("derivation worker is not initialized")
    return _ENGINE.run(_VIEW, task, _KERNEL)


@dataclass(frozen=True, slots=True)
class ParallelDerivationConfig:
    workers: int = 4
    max_tasks_per_child: int | None = None

    def __post_init__(self) -> None:
        if self.workers <= 0:
            raise ValueError("workers must be positive")


class ParallelDerivationExecutor:
    """Bounded process executor attaching one immutable mmap generation per worker."""

    def __init__(self, *, directory: str | Path, config: ParallelDerivationConfig | None = None) -> None:
        self.directory = str(directory)
        self.config = config or ParallelDerivationConfig()

    def run(
        self,
        *,
        handle: ReadViewHandle,
        tasks: Iterable[DerivationTask],
        kernel: Callable,
    ) -> tuple[DerivationTaskResult[DerivedMutationBatch], ...]:
        ordered_tasks = tuple(tasks)
        if not ordered_tasks:
            return ()
        kwargs = {}
        if self.config.max_tasks_per_child is not None:
            kwargs["max_tasks_per_child"] = int(self.config.max_tasks_per_child)
        with ProcessPoolExecutor(
            max_workers=self.config.workers,
            initializer=_init_worker,
            initargs=(self.directory, handle, kernel),
            **kwargs,
        ) as pool:
            results = tuple(pool.map(_run_task, ordered_tasks))
        return results
