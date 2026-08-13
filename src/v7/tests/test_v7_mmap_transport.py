from __future__ import annotations

import pytest

from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.indexes.cognition import (
    ActionAggregateDelta,
    ContingencyIndexMutation,
    RoleConceptIndexMutation,
    RoleIndexMutation,
)
from v7.memory.models import EdgeMutation, NodeMutation, ScoreMutation
from v7.memory.transport.mmap import MmapReadViewTransport
from v7.memory.writer import CanonicalMemoryWriter


def _build_view():
    ids = MemoryIdAllocator()
    contingency_id, family_id, role_id, concept_id = ids.allocate_many(4)
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [
            NodeMutation(contingency_id, MemoryLevel.M1, 10, support_delta=2),
            NodeMutation(family_id, MemoryLevel.M2, 20, support_delta=1),
            NodeMutation(role_id, MemoryLevel.M3, 30, support_delta=1),
            NodeMutation(concept_id, MemoryLevel.M4, 40, support_delta=1),
        ]
    )
    writer.apply_score_batch(
        [ScoreMutation(contingency_id, significance=0.5, future_option_delta=1.25)]
    )
    writer.apply_edge_batch([EdgeMutation(contingency_id, 7, family_id, support_delta=1)])
    writer.apply_contingency_index_batch([ContingencyIndexMutation(11, 2, contingency_id)])
    writer.apply_role_index_batch([RoleIndexMutation(11, 2, role_id, family_id=family_id)])
    writer.apply_role_concept_index_batch([RoleConceptIndexMutation(role_id, concept_id)])
    writer.apply_action_aggregate_batch(
        [
            ActionAggregateDelta(
                action_id=2,
                future_option_sum_delta=3.0,
                future_option_count_delta=2,
                positive_count_delta=1,
                failure_count_delta=1,
            )
        ]
    )
    _, view, _ = writer.commit_generation()
    return contingency_id, family_id, role_id, concept_id, view


def test_mmap_transport_round_trips_full_read_view(tmp_path) -> None:
    contingency_id, family_id, role_id, concept_id, view = _build_view()
    publisher_transport = MmapReadViewTransport(tmp_path)
    handle = publisher_transport.publish(view)

    reader_transport = MmapReadViewTransport(tmp_path)
    attached = reader_transport.attach(handle)

    assert attached is not view
    assert attached.generation_id == view.generation_id
    assert attached.nodes[contingency_id].support_count == 2
    assert attached.scores[contingency_id].future_option_delta == 1.25
    assert attached.neighbors([contingency_id], 7) == ((family_id,),)
    score_input = attached.score_inputs(
        context_signature=11,
        action_ids=[2],
        family_ids_by_action={2: family_id},
    )[0]
    assert score_input.contingency_ids == (contingency_id,)
    assert score_input.role_ids == (role_id,)
    assert score_input.concept_ids == (concept_id,)
    assert score_input.aggregate.future_option_mean == 1.5
    assert score_input.aggregate.failure_count == 1


def test_mmap_publish_is_content_addressed_and_idempotent(tmp_path) -> None:
    *_, view = _build_view()
    transport = MmapReadViewTransport(tmp_path)

    first = transport.publish(view)
    second = transport.publish(view)

    assert first == second
    assert transport.retained_generations == (1,)


def test_mmap_release_removes_generation_file(tmp_path) -> None:
    *_, view = _build_view()
    transport = MmapReadViewTransport(tmp_path)
    handle = transport.publish(view)

    transport.release(handle)

    assert transport.retained_generations == ()
    with pytest.raises(KeyError):
        transport.attach(handle)


def test_mmap_attach_rejects_tampered_payload(tmp_path) -> None:
    *_, view = _build_view()
    transport = MmapReadViewTransport(tmp_path)
    handle = transport.publish(view)
    filename = handle.transport_key.split(":", 1)[0]
    path = tmp_path / filename
    payload = path.read_bytes()
    path.write_bytes(payload + b"corrupt")

    with pytest.raises(ValueError, match="digest mismatch"):
        transport.attach(handle)
