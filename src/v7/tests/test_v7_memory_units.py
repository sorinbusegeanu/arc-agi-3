from __future__ import annotations

import pytest

from v7.memory.ids import MemoryIdAllocator, MemoryLevel, validate_memory_id
from v7.memory.models import EdgeMutation, NodeMutation
from v7.memory.writer import CanonicalMemoryWriter


def test_memory_id_validation_rejects_zero_negative_and_overflow() -> None:
    assert int(validate_memory_id(1)) == 1
    with pytest.raises(ValueError):
        validate_memory_id(0)
    with pytest.raises(ValueError):
        validate_memory_id(-1)
    with pytest.raises(ValueError):
        validate_memory_id(1 << 64)


def test_memory_id_allocator_is_monotonic_and_deterministic() -> None:
    allocator = MemoryIdAllocator(next_value=7)
    assert tuple(map(int, allocator.allocate_many(3))) == (7, 8, 9)
    assert int(allocator.allocate()) == 10


def test_published_read_view_isolated_from_later_writer_mutations() -> None:
    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [NodeMutation(memory_id, MemoryLevel.M1, 10, support_delta=1)]
    )
    _, first_view, _ = writer.commit_generation()

    writer.apply_mutation_batch(
        [NodeMutation(memory_id, MemoryLevel.M1, 10, support_delta=4)]
    )
    _, second_view, _ = writer.commit_generation()

    assert first_view.nodes[memory_id].support_count == 1
    assert second_view.nodes[memory_id].support_count == 5
    with pytest.raises(TypeError):
        first_view.nodes[memory_id] = second_view.nodes[memory_id]  # type: ignore[index]


def test_node_batch_validation_is_atomic() -> None:
    allocator = MemoryIdAllocator()
    good_id, bad_id = allocator.allocate_many(2)
    writer = CanonicalMemoryWriter()

    with pytest.raises(ValueError):
        writer.apply_mutation_batch(
            [
                NodeMutation(good_id, MemoryLevel.M1, 10, support_delta=1),
                NodeMutation(bad_id, MemoryLevel.M1, 10, support_delta=-1),
            ]
        )

    _, view, delta = writer.commit_generation()
    assert good_id not in view.nodes
    assert bad_id not in view.nodes
    assert delta.mutation_count == 0


def test_existing_node_identity_is_immutable() -> None:
    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [NodeMutation(memory_id, MemoryLevel.M1, 10, support_delta=1)]
    )

    with pytest.raises(ValueError):
        writer.apply_mutation_batch(
            [NodeMutation(memory_id, MemoryLevel.M2, 10, support_delta=1)]
        )
    with pytest.raises(ValueError):
        writer.apply_mutation_batch(
            [NodeMutation(memory_id, MemoryLevel.M1, 11, support_delta=1)]
        )


def test_edge_batch_validation_is_atomic() -> None:
    allocator = MemoryIdAllocator()
    source_a, target_a, source_b, target_b = allocator.allocate_many(4)
    writer = CanonicalMemoryWriter()
    writer.apply_edge_batch([EdgeMutation(source_a, 3, target_a, support_delta=2)])

    with pytest.raises(ValueError):
        writer.apply_edge_batch(
            [
                EdgeMutation(source_b, 3, target_b, support_delta=1),
                EdgeMutation(source_a, 3, target_a, support_delta=-3),
            ]
        )

    _, view, delta = writer.commit_generation()
    assert view.neighbors([source_a], 3) == ((target_a,),)
    assert view.neighbors([source_b], 3) == ((),)
    assert len(delta.edges) == 1
    assert delta.edges[0].support_count == 2
