from __future__ import annotations

from v7.derivation.batches import DerivedMutationBatch
from v7.derivation.executor import ParallelDerivationConfig, ParallelDerivationExecutor
from v7.derivation.workers import DerivationTask
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.models import MemoryNode
from v7.memory.read_view import MemoryReadView
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport


def _identity_kernel(batch):
    return DerivedMutationBatch(
        generation_id=batch.generation_id,
        source_level=batch.level,
        source_ids=tuple(int_id for int_id in map(type(batch.memory_ids[0]).__call__, batch.memory_ids)) if batch.memory_ids.size else (),
    )


def _kernel(batch):
    from v7.memory.ids import MemoryId
    return DerivedMutationBatch(
        generation_id=batch.generation_id,
        source_level=batch.level,
        source_ids=tuple(MemoryId(int(value)) for value in batch.memory_ids),
    )


def test_parallel_executor_attaches_generation_and_returns_in_task_order(tmp_path) -> None:
    ids = MemoryIdAllocator()
    a, b = ids.allocate_many(2)
    view = MemoryReadView.freeze(
        generation_id=GenerationId(1),
        nodes={
            a: MemoryNode(a, MemoryLevel.M2, 20, GenerationId(1), GenerationId(1), support_count=2),
            b: MemoryNode(b, MemoryLevel.M2, 20, GenerationId(1), GenerationId(1), support_count=3),
        },
        scores={},
        adjacency={},
    )
    transport = SegmentedMmapReadViewTransport(tmp_path)
    handle = transport.publish(view)
    tasks = (
        DerivationTask(GenerationId(1), MemoryLevel.M2, (a,)),
        DerivationTask(GenerationId(1), MemoryLevel.M2, (b,)),
    )
    results = ParallelDerivationExecutor(
        directory=tmp_path,
        config=ParallelDerivationConfig(workers=1),
    ).run(handle=handle, tasks=tasks, kernel=_kernel)
    assert tuple(result.task for result in results) == tasks
    assert tuple(result.output.source_ids for result in results) == ((a,), (b,))
