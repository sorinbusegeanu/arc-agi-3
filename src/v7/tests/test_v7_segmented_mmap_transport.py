from __future__ import annotations

from v7.memory.arenas.mapped import MappedCompactMemoryArena
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.indexes.cognition import (
    ActionAggregate,
    CognitionIndexes,
)
from v7.memory.models import MemoryNode, MemoryScore
from v7.memory.read_view import MemoryReadView
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport


def _view() -> tuple[MemoryReadView, tuple]:
    ids = MemoryIdAllocator()
    first_id, second_id, role_id, concept_id = ids.allocate_many(4)
    indexes = CognitionIndexes.freeze(
        contingency_by_context_action={(10, 2): (first_id,)},
        role_by_context_action_family={(10, 2, second_id): (role_id,)},
        role_by_context_action={(10, 2): (role_id,)},
        concepts_by_role={role_id: (concept_id,)},
        action_aggregates={2: ActionAggregate(future_option_sum=3.0, future_option_count=2, failure_count=1)},
    )
    view = MemoryReadView.freeze(
        generation_id=GenerationId(4),
        nodes={
            first_id: MemoryNode(first_id, MemoryLevel.M1, 11, GenerationId(1), GenerationId(4), support_count=5),
            second_id: MemoryNode(second_id, MemoryLevel.M2, 21, GenerationId(2), GenerationId(4), support_count=3),
            role_id: MemoryNode(role_id, MemoryLevel.M3, 31, GenerationId(3), GenerationId(4), support_count=2),
            concept_id: MemoryNode(concept_id, MemoryLevel.M4, 41, GenerationId(4), GenerationId(4), support_count=1),
        },
        scores={first_id: MemoryScore(first_id, significance=0.5, future_option_delta=1.25)},
        adjacency={(first_id, 7): (second_id, role_id)},
        cognition_indexes=indexes,
    )
    return view, (first_id, second_id, role_id, concept_id)


def test_segmented_transport_attaches_direct_mmap_numeric_arena(tmp_path) -> None:
    view, ids = _view()
    first_id, second_id, role_id, concept_id = ids
    transport = SegmentedMmapReadViewTransport(tmp_path)
    handle = transport.publish(view)
    attached = transport.attach(handle)

    assert isinstance(attached.compact_arena, MappedCompactMemoryArena)
    assert isinstance(attached.compact_arena.nodes.memory_ids, memoryview)
    assert isinstance(attached.compact_arena.scores.significance, memoryview)
    assert attached.get_nodes([first_id, second_id]) == view.get_nodes([first_id, second_id])
    assert attached.get_scores([first_id]) == view.get_scores([first_id])
    assert attached.neighbors([first_id], 7) == ((second_id, role_id),)
    score_input = attached.score_inputs(
        context_signature=10,
        action_ids=[2],
        family_ids_by_action={2: second_id},
    )[0]
    assert score_input.contingency_ids == (first_id,)
    assert score_input.role_ids == (role_id,)
    assert score_input.concept_ids == (concept_id,)
    assert score_input.aggregate.future_option_mean == 1.5


def test_segmented_transport_can_attach_from_second_transport_instance(tmp_path) -> None:
    view, ids = _view()
    first_id = ids[0]
    publisher = SegmentedMmapReadViewTransport(tmp_path)
    handle = publisher.publish(view)

    reader = SegmentedMmapReadViewTransport(tmp_path)
    attached = reader.attach(handle)
    assert attached.generation_id == GenerationId(4)
    assert attached.get_nodes([first_id])[0].support_count == 5


def test_segmented_transport_detects_numeric_segment_tampering(tmp_path) -> None:
    view, _ = _view()
    transport = SegmentedMmapReadViewTransport(tmp_path)
    handle = transport.publish(view)
    node_path = tmp_path / "generation-4.nodes"
    payload = bytearray(node_path.read_bytes())
    payload[0] ^= 1
    node_path.write_bytes(payload)

    try:
        transport.attach(handle)
    except ValueError as exc:
        assert "segment digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered segment was accepted")
