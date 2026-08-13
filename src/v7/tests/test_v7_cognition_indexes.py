from __future__ import annotations

import pytest

from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.indexes.cognition import (
    ActionAggregateDelta,
    CognitionIndexBuilder,
    ContingencyIndexMutation,
    RoleConceptIndexMutation,
    RoleIndexMutation,
)
from v7.memory.models import NodeMutation
from v7.memory.writer import CanonicalMemoryWriter


def test_index_builder_deduplicates_batches_and_freezes_sorted_ids() -> None:
    allocator = MemoryIdAllocator()
    contingency_a, contingency_b = allocator.allocate_many(2)
    builder = CognitionIndexBuilder()

    applied = builder.apply_contingency_batch(
        [
            ContingencyIndexMutation(100, 2, contingency_b),
            ContingencyIndexMutation(100, 2, contingency_a),
            ContingencyIndexMutation(100, 2, contingency_a),
        ]
    )
    indexes = builder.freeze()

    assert applied == 2
    assert indexes.contingency_by_context_action[(100, 2)] == (
        contingency_a,
        contingency_b,
    )
    with pytest.raises(TypeError):
        indexes.contingency_by_context_action[(100, 2)] = ()  # type: ignore[index]


def test_action_aggregate_batch_coalesces_to_o1_read_summary() -> None:
    builder = CognitionIndexBuilder()
    applied = builder.apply_action_aggregate_batch(
        [
            ActionAggregateDelta(
                action_id=4,
                future_option_sum_delta=2.0,
                future_option_count_delta=1,
                positive_count_delta=1,
            ),
            ActionAggregateDelta(
                action_id=4,
                future_option_sum_delta=4.0,
                future_option_count_delta=2,
                negative_count_delta=1,
                failure_count_delta=1,
            ),
        ]
    )
    aggregate = builder.freeze().action_aggregates[4]

    assert applied == 1
    assert aggregate.future_option_sum == 6.0
    assert aggregate.future_option_count == 3
    assert aggregate.future_option_mean == 2.0
    assert aggregate.positive_count == 1
    assert aggregate.negative_count == 1
    assert aggregate.failure_count == 1


def test_action_aggregate_rejects_negative_counts_without_partial_update() -> None:
    builder = CognitionIndexBuilder()
    builder.apply_action_aggregate_batch(
        [ActionAggregateDelta(action_id=1, future_option_count_delta=2)]
    )

    with pytest.raises(ValueError):
        builder.apply_action_aggregate_batch(
            [
                ActionAggregateDelta(action_id=2, positive_count_delta=1),
                ActionAggregateDelta(action_id=1, future_option_count_delta=-3),
            ]
        )

    indexes = builder.freeze()
    assert indexes.action_aggregates[1].future_option_count == 2
    assert 2 not in indexes.action_aggregates


def test_batch_score_inputs_prefers_exact_family_roles_then_expands_concepts_once() -> None:
    allocator = MemoryIdAllocator()
    contingency_id, family_id, exact_role, fallback_role, concept_a, concept_b = allocator.allocate_many(6)
    builder = CognitionIndexBuilder()
    builder.apply_contingency_batch(
        [ContingencyIndexMutation(500, 7, contingency_id)]
    )
    builder.apply_role_batch(
        [
            RoleIndexMutation(500, 7, exact_role, family_id=family_id),
            RoleIndexMutation(500, 7, fallback_role),
        ]
    )
    builder.apply_role_concept_batch(
        [
            RoleConceptIndexMutation(exact_role, concept_a),
            RoleConceptIndexMutation(exact_role, concept_b),
            RoleConceptIndexMutation(exact_role, concept_a),
        ]
    )
    builder.apply_action_aggregate_batch(
        [
            ActionAggregateDelta(
                action_id=7,
                future_option_sum_delta=3.0,
                future_option_count_delta=2,
                positive_count_delta=2,
            )
        ]
    )

    rows = builder.freeze().score_inputs(
        context_signature=500,
        action_ids=[7],
        family_ids_by_action={7: family_id},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.contingency_ids == (contingency_id,)
    assert row.role_ids == (exact_role,)
    assert row.concept_ids == (concept_a, concept_b)
    assert row.aggregate.future_option_mean == 1.5


def test_batch_score_inputs_falls_back_when_exact_family_has_no_candidates() -> None:
    allocator = MemoryIdAllocator()
    unknown_family, fallback_role = allocator.allocate_many(2)
    builder = CognitionIndexBuilder()
    builder.apply_role_batch([RoleIndexMutation(20, 3, fallback_role)])

    row = builder.freeze().score_inputs(
        context_signature=20,
        action_ids=[3],
        family_ids_by_action={3: unknown_family},
    )[0]

    assert row.role_ids == (fallback_role,)


def test_batch_score_inputs_enforces_role_and_concept_bounds() -> None:
    allocator = MemoryIdAllocator()
    role_ids = allocator.allocate_many(5)
    concept_ids = allocator.allocate_many(10)
    builder = CognitionIndexBuilder()
    builder.apply_role_batch(
        [RoleIndexMutation(1, 1, role_id) for role_id in role_ids]
    )
    builder.apply_role_concept_batch(
        [
            RoleConceptIndexMutation(role_ids[index % len(role_ids)], concept_id)
            for index, concept_id in enumerate(concept_ids)
        ]
    )

    row = builder.freeze().score_inputs(
        context_signature=1,
        action_ids=[1],
        role_limit=2,
        concept_limit=3,
    )[0]

    assert len(row.role_ids) == 2
    assert len(row.concept_ids) <= 3


def test_writer_publishes_cognition_indexes_only_at_generation_commit() -> None:
    allocator = MemoryIdAllocator()
    contingency_id, role_id, concept_id = allocator.allocate_many(3)
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [
            NodeMutation(contingency_id, MemoryLevel.M1, 10, support_delta=1),
            NodeMutation(role_id, MemoryLevel.M3, 30, support_delta=1),
            NodeMutation(concept_id, MemoryLevel.M4, 40, support_delta=1),
        ]
    )
    writer.apply_contingency_index_batch(
        [ContingencyIndexMutation(9, 2, contingency_id)]
    )
    writer.apply_role_index_batch([RoleIndexMutation(9, 2, role_id)])
    writer.apply_role_concept_index_batch(
        [RoleConceptIndexMutation(role_id, concept_id)]
    )

    assert writer.published_view.score_inputs(
        context_signature=9,
        action_ids=[2],
    )[0].role_ids == ()

    _, published, _ = writer.commit_generation()
    row = published.score_inputs(context_signature=9, action_ids=[2])[0]

    assert row.contingency_ids == (contingency_id,)
    assert row.role_ids == (role_id,)
    assert row.concept_ids == (concept_id,)
