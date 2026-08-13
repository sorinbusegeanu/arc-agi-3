from __future__ import annotations

import pytest

from v7.derivation.dependencies import DependencyMutation, MemoryDependencyGraph
from v7.derivation.workers import DerivationTask, DerivationTaskPlanner, DerivationWorker
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.models import MemoryNode
from v7.memory.read_view import MemoryReadView
from v7.memory.transport.local import LocalReadViewTransport


def _dirty_plan_and_view():
    ids = MemoryIdAllocator()
    m1, m2a, m2b, m3, m4, m5, m6, clean_m3 = ids.allocate_many(8)
    graph = MemoryDependencyGraph()
    graph.apply_dependency_batch(
        [
            DependencyMutation(m1, MemoryLevel.M1, m2a, MemoryLevel.M2),
            DependencyMutation(m1, MemoryLevel.M1, m2b, MemoryLevel.M2),
            DependencyMutation(m2a, MemoryLevel.M2, m3, MemoryLevel.M3),
            DependencyMutation(m3, MemoryLevel.M3, m4, MemoryLevel.M4),
            DependencyMutation(m4, MemoryLevel.M4, m5, MemoryLevel.M5),
            DependencyMutation(m5, MemoryLevel.M5, m6, MemoryLevel.M6),
        ]
    )
    graph.register_node(clean_m3, MemoryLevel.M3)
    graph.mark_dirty([m1])
    plan = graph.snapshot_plan()
    all_nodes = (m1, m2a, m2b, m3, m4, m5, m6, clean_m3)
    nodes = {
        memory_id: MemoryNode(
            memory_id,
            MemoryLevel(index if index <= 6 else 3),
            100 + index,
            GenerationId(1),
            GenerationId(1),
            support_count=1,
        )
        for index, memory_id in enumerate(all_nodes, start=1)
    }
    view = MemoryReadView.freeze(
        generation_id=GenerationId(1),
        nodes=nodes,
        scores={},
        adjacency={},
    )
    return plan, view, (m1, m2a, m2b, m3, m4, m5, m6, clean_m3)


def test_task_planner_consumes_only_dirty_m2_to_m6_and_chunks_deterministically() -> None:
    plan, _, ids = _dirty_plan_and_view()
    _, m2a, m2b, m3, m4, m5, m6, clean_m3 = ids
    tasks = DerivationTaskPlanner(chunk_size=1).build(
        plan,
        generation_id=GenerationId(1),
    )

    assert [(task.level, task.memory_ids) for task in tasks] == [
        (MemoryLevel.M2, (m2a,)),
        (MemoryLevel.M2, (m2b,)),
        (MemoryLevel.M3, (m3,)),
        (MemoryLevel.M4, (m4,)),
        (MemoryLevel.M5, (m5,)),
        (MemoryLevel.M6, (m6,)),
    ]
    assert all(clean_m3 not in task.memory_ids for task in tasks)


def test_worker_attaches_generation_once_and_runs_compact_dirty_ids_only() -> None:
    plan, view, ids = _dirty_plan_and_view()
    _, m2a, m2b, _, _, _, _, _ = ids
    transport = LocalReadViewTransport()
    handle = transport.publish(view)
    worker = DerivationWorker(transport=transport, handle=handle)
    tasks = DerivationTaskPlanner(chunk_size=8).build(
        plan,
        generation_id=GenerationId(1),
    )

    def kernel(read_view, level, memory_ids):
        assert read_view is worker.read_view
        assert read_view.compact_arena.generation_id == GenerationId(1)
        return (
            level,
            tuple(read_view.compact_arena.nodes.get(memory_id).support_count for memory_id in memory_ids),
        )

    results = worker.run_many(tasks, kernel)
    assert results[0].task.memory_ids == (m2a, m2b)
    assert results[0].output == (MemoryLevel.M2, (1, 1))
    assert len(results) == 5


def test_worker_rejects_task_from_different_generation() -> None:
    _, view, ids = _dirty_plan_and_view()
    _, m2a, *_ = ids
    transport = LocalReadViewTransport()
    handle = transport.publish(view)
    worker = DerivationWorker(transport=transport, handle=handle)
    task = DerivationTask(
        generation_id=GenerationId(2),
        level=MemoryLevel.M2,
        memory_ids=(m2a,),
    )

    with pytest.raises(ValueError, match="different generation"):
        worker.run(task, lambda read_view, level, memory_ids: None)


def test_task_contract_rejects_non_derivation_levels_and_duplicates() -> None:
    memory_id = MemoryIdAllocator().allocate()
    with pytest.raises(ValueError, match="M2-M6"):
        DerivationTask(GenerationId(1), MemoryLevel.M1, (memory_id,))
    with pytest.raises(ValueError, match="unique"):
        DerivationTask(GenerationId(1), MemoryLevel.M2, (memory_id, memory_id))


def test_task_planner_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        DerivationTaskPlanner(chunk_size=0)
