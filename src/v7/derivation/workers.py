from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

from v7.derivation.dependencies import DirtyDerivationPlan
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.read_view import MemoryReadView
from v7.memory.transport.base import ReadViewHandle, ReadViewTransport

T = TypeVar("T")
DerivationKernel = Callable[[MemoryReadView, MemoryLevel, tuple[MemoryId, ...]], T]

_DERIVATION_LEVELS = (
    MemoryLevel.M2,
    MemoryLevel.M3,
    MemoryLevel.M4,
    MemoryLevel.M5,
    MemoryLevel.M6,
)


@dataclass(frozen=True, slots=True)
class DerivationTask:
    generation_id: GenerationId
    level: MemoryLevel
    memory_ids: tuple[MemoryId, ...]

    def __post_init__(self) -> None:
        if self.level not in _DERIVATION_LEVELS:
            raise ValueError("derivation tasks must target M2-M6")
        if not self.memory_ids:
            raise ValueError("derivation task cannot be empty")
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("derivation task memory_ids must be unique")


@dataclass(frozen=True, slots=True)
class DerivationTaskResult(Generic[T]):
    task: DerivationTask
    output: T


class DerivationTaskPlanner:
    """Convert one dirty dependency plan into deterministic bounded work chunks."""

    def __init__(self, *, chunk_size: int = 256) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)

    def build(
        self,
        plan: DirtyDerivationPlan,
        *,
        generation_id: GenerationId,
    ) -> tuple[DerivationTask, ...]:
        tasks: list[DerivationTask] = []
        for level in _DERIVATION_LEVELS:
            memory_ids = plan.ids_for_level(level)
            for start in range(0, len(memory_ids), self.chunk_size):
                chunk = memory_ids[start : start + self.chunk_size]
                if chunk:
                    tasks.append(
                        DerivationTask(
                            generation_id=generation_id,
                            level=level,
                            memory_ids=chunk,
                        )
                    )
        return tuple(tasks)


class DerivationWorker:
    """Worker-side execution boundary over one attached immutable generation.

    A process can construct one worker, attach the generation once, then execute many
    compact task descriptors containing only level and MemoryIds. No mutable writer
    state, graph dictionaries, role rows or profile caches cross this boundary.
    """

    def __init__(
        self,
        *,
        transport: ReadViewTransport,
        handle: ReadViewHandle,
    ) -> None:
        self._view = transport.attach(handle)
        if self._view.generation_id != handle.generation_id:
            raise ValueError("attached derivation generation mismatch")

    @property
    def generation_id(self) -> GenerationId:
        return self._view.generation_id

    @property
    def read_view(self) -> MemoryReadView:
        return self._view

    def run(self, task: DerivationTask, kernel: DerivationKernel[T]) -> DerivationTaskResult[T]:
        if task.generation_id != self._view.generation_id:
            raise ValueError("derivation task targets a different generation")
        output = kernel(self._view, task.level, task.memory_ids)
        return DerivationTaskResult(task=task, output=output)

    def run_many(
        self,
        tasks: Iterable[DerivationTask],
        kernel: DerivationKernel[T],
    ) -> tuple[DerivationTaskResult[T], ...]:
        return tuple(self.run(task, kernel) for task in tasks)
