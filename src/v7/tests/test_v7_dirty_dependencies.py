from __future__ import annotations

import pytest

from v7.derivation.dependencies import DependencyMutation, MemoryDependencyGraph
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.models import NodeMutation
from v7.memory.writer import CanonicalMemoryWriter


def test_dependency_graph_propagates_only_registered_upstream_neighborhood() -> None:
    ids = MemoryIdAllocator()
    m1_a, m1_b, m2_a, m3_a, m4_a = ids.allocate_many(5)
    graph = MemoryDependencyGraph()
    graph.apply_dependency_batch(
        [
            DependencyMutation(m1_a, MemoryLevel.M1, m2_a, MemoryLevel.M2),
            DependencyMutation(m2_a, MemoryLevel.M2, m3_a, MemoryLevel.M3),
            DependencyMutation(m3_a, MemoryLevel.M3, m4_a, MemoryLevel.M4),
        ]
    )
    graph.register_node(m1_b, MemoryLevel.M1)

    assert graph.mark_dirty([m1_a]) == 4
    plan = graph.snapshot_plan()

    assert plan.ids_for_level(MemoryLevel.M1) == (m1_a,)
    assert plan.ids_for_level(MemoryLevel.M2) == (m2_a,)
    assert plan.ids_for_level(MemoryLevel.M3) == (m3_a,)
    assert plan.ids_for_level(MemoryLevel.M4) == (m4_a,)
    assert m1_b not in plan.ids_for_level(MemoryLevel.M1)
    assert plan.total_count == 4


def test_dependency_graph_rejects_downward_or_same_level_edges() -> None:
    source_id, target_id = MemoryIdAllocator().allocate_many(2)

    with pytest.raises(ValueError, match="higher memory level"):
        DependencyMutation(source_id, MemoryLevel.M3, target_id, MemoryLevel.M3)
    with pytest.raises(ValueError, match="higher memory level"):
        DependencyMutation(source_id, MemoryLevel.M4, target_id, MemoryLevel.M2)


def test_dependency_registration_is_deduplicated_and_level_identity_is_stable() -> None:
    source_id, target_id = MemoryIdAllocator().allocate_many(2)
    graph = MemoryDependencyGraph()
    mutation = DependencyMutation(source_id, MemoryLevel.M1, target_id, MemoryLevel.M2)

    assert graph.apply_dependency_batch([mutation, mutation]) == 1
    graph.register_node(source_id, MemoryLevel.M1)
    with pytest.raises(ValueError, match="level is immutable"):
        graph.register_node(source_id, MemoryLevel.M2)


def test_consume_plan_clears_only_derivation_dirty_state() -> None:
    source_id, target_id = MemoryIdAllocator().allocate_many(2)
    graph = MemoryDependencyGraph()
    graph.apply_dependency_batch(
        [DependencyMutation(source_id, MemoryLevel.M1, target_id, MemoryLevel.M2)]
    )
    graph.mark_dirty([source_id])

    consumed = graph.consume_plan()

    assert consumed.total_count == 2
    assert graph.dirty_count == 0
    assert graph.snapshot_plan().total_count == 0


def test_writer_node_mutation_marks_registered_dependents_dirty() -> None:
    ids = MemoryIdAllocator()
    m1, m2, m3 = ids.allocate_many(3)
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [
            NodeMutation(m1, MemoryLevel.M1, 10, support_delta=1),
            NodeMutation(m2, MemoryLevel.M2, 20, support_delta=1),
            NodeMutation(m3, MemoryLevel.M3, 30, support_delta=1),
        ]
    )
    writer.consume_dirty_derivation_plan()
    writer.apply_dependency_batch(
        [
            DependencyMutation(m1, MemoryLevel.M1, m2, MemoryLevel.M2),
            DependencyMutation(m2, MemoryLevel.M2, m3, MemoryLevel.M3),
        ]
    )

    writer.apply_mutation_batch([NodeMutation(m1, MemoryLevel.M1, 10, support_delta=1)])
    plan = writer.dirty_derivation_plan()

    assert plan.ids_for_level(MemoryLevel.M1) == (m1,)
    assert plan.ids_for_level(MemoryLevel.M2) == (m2,)
    assert plan.ids_for_level(MemoryLevel.M3) == (m3,)
    assert plan.total_count == 3


def test_writer_unrelated_branch_is_not_dirtied() -> None:
    ids = MemoryIdAllocator()
    m1_a, m1_b, m2_a, m2_b = ids.allocate_many(4)
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [
            NodeMutation(m1_a, MemoryLevel.M1, 10, support_delta=1),
            NodeMutation(m1_b, MemoryLevel.M1, 10, support_delta=1),
            NodeMutation(m2_a, MemoryLevel.M2, 20, support_delta=1),
            NodeMutation(m2_b, MemoryLevel.M2, 20, support_delta=1),
        ]
    )
    writer.consume_dirty_derivation_plan()
    writer.apply_dependency_batch(
        [
            DependencyMutation(m1_a, MemoryLevel.M1, m2_a, MemoryLevel.M2),
            DependencyMutation(m1_b, MemoryLevel.M1, m2_b, MemoryLevel.M2),
        ]
    )

    writer.apply_mutation_batch([NodeMutation(m1_a, MemoryLevel.M1, 10, support_delta=1)])
    plan = writer.dirty_derivation_plan()

    assert plan.ids_for_level(MemoryLevel.M1) == (m1_a,)
    assert plan.ids_for_level(MemoryLevel.M2) == (m2_a,)


def test_dependency_plan_cannot_be_consumed_while_generation_is_prepared() -> None:
    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch([NodeMutation(memory_id, MemoryLevel.M1, 10, support_delta=1)])
    writer.prepare_generation()

    with pytest.raises(RuntimeError, match="generation is prepared"):
        writer.consume_dirty_derivation_plan()
