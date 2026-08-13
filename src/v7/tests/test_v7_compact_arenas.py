from __future__ import annotations

import pytest

from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.models import MemoryNode, MemoryScore
from v7.memory.read_view import MemoryReadView


def _view() -> MemoryReadView:
    ids = MemoryIdAllocator()
    first_id, second_id, third_id = ids.allocate_many(3)
    nodes = {
        first_id: MemoryNode(first_id, MemoryLevel.M1, 11, GenerationId(1), GenerationId(2), support_count=3),
        second_id: MemoryNode(second_id, MemoryLevel.M3, 31, GenerationId(1), GenerationId(2), status_flags=4, support_count=7),
        third_id: MemoryNode(third_id, MemoryLevel.M4, 41, GenerationId(2), GenerationId(2), support_count=1),
    }
    scores = {
        first_id: MemoryScore(first_id, significance=0.5, future_option_delta=2.0),
        second_id: MemoryScore(second_id, transfer_prior=0.75, explanatory_potential=0.25),
    }
    return MemoryReadView.freeze(
        generation_id=GenerationId(2),
        nodes=nodes,
        scores=scores,
        adjacency={
            (first_id, 7): (third_id, second_id, second_id),
            (first_id, 3): (second_id,),
            (second_id, 7): (third_id,),
        },
    )


def test_read_view_builds_numeric_columnar_arenas() -> None:
    view = _view()
    arena = view.compact_arena

    assert arena.generation_id == GenerationId(2)
    assert isinstance(arena.nodes.memory_ids, memoryview)
    assert isinstance(arena.scores.significance, memoryview)
    assert arena.nodes.memory_ids.format == "Q"
    assert arena.nodes.levels.format == "B"
    assert arena.scores.significance.format == "d"
    assert arena.nodes.memory_ids.readonly
    assert arena.scores.significance.readonly
    assert arena.adjacency.targets.readonly
    assert arena.nodes.count == 3
    assert arena.scores.count == 2
    assert arena.payload_bytes > 0


def test_compact_numeric_buffers_reject_mutation() -> None:
    view = _view()
    with pytest.raises(TypeError):
        view.compact_arena.nodes.memory_ids[0] = 999
    with pytest.raises(TypeError):
        view.compact_arena.adjacency.targets[0] = 999


def test_columnar_node_and_score_lookup_preserves_semantics() -> None:
    view = _view()
    memory_ids = tuple(view.nodes)

    assert view.get_nodes(memory_ids) == tuple(view.nodes[memory_id] for memory_id in memory_ids)
    assert view.get_scores(memory_ids) == tuple(view.scores.get(memory_id) for memory_id in memory_ids)


def test_packed_adjacency_is_sorted_deduplicated_and_exact() -> None:
    view = _view()
    first_id, second_id, third_id = tuple(view.nodes)
    adjacency = view.compact_arena.adjacency

    assert adjacency.edge_group_count == 3
    assert adjacency.target_count == 4
    assert adjacency.neighbors(first_id, 7) == (second_id, third_id)
    assert adjacency.neighbors(first_id, 3) == (second_id,)
    assert adjacency.neighbors(second_id, 7) == (third_id,)
    assert adjacency.neighbors(third_id, 7) == ()
    assert view.neighbors([first_id, second_id], 7) == (
        (second_id, third_id),
        (third_id,),
    )


def test_compact_arena_is_generation_immutable() -> None:
    view = _view()
    first_id = next(iter(view.nodes))
    before = view.compact_arena.nodes.get(first_id)

    mutable_copy = dict(view.nodes)
    mutable_copy[first_id] = MemoryNode(
        first_id,
        MemoryLevel.M1,
        11,
        GenerationId(1),
        GenerationId(3),
        support_count=999,
    )

    assert view.compact_arena.nodes.get(first_id) == before
    assert view.get_nodes([first_id])[0].support_count == 3
