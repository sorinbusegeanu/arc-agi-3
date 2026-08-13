from __future__ import annotations

import numpy as np

from v7.derivation.batches import DerivedMutationBatch, DeterministicDerivedBatchMerger
from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import EpisodeEvidence, ScientificDerivationKernels
from v7.derivation.workers import DerivationTask, DerivationTaskResult
from v7.memory.canonical import CanonicalMemoryKey
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.indexes.cognition import ActionAggregate, CognitionIndexes
from v7.memory.models import EdgeMutation, NodeMutation, ScoreMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.scoring import VectorizedActionScorer
from v7.memory.writer import CanonicalMemoryWriter


def test_scientific_pipeline_builds_m1_to_m6_and_reuses_canonical_m1() -> None:
    writer = CanonicalMemoryWriter()
    pipeline = MemoryLearningPipeline(writer)

    e1 = EpisodeEvidence(10, 2, 100, True, prediction_error=0.4, future_option_delta=1.0)
    e2 = EpisodeEvidence(11, 2, 101, True, prediction_error=0.2, future_option_delta=0.5)
    m1a = pipeline.observe_episode(e1)
    assert pipeline.observe_episode(e1) == m1a
    m1b = pipeline.observe_episode(e2)

    m2 = pipeline.derive_m2(action_id=2, member_ids=(m1a, m1b), outcome_class=7)
    r1 = pipeline.derive_m3(family_id=m2, context_class=10, action_id=2, member_ids=(m1a,))
    r2 = pipeline.derive_m3(family_id=m2, context_class=11, action_id=2, member_ids=(m1b,))
    c1 = pipeline.derive_m4(role_ids=(r1, r2), relation_signature=70)
    c2 = pipeline.derive_m4(role_ids=(r1, r2), relation_signature=71)
    m5 = pipeline.derive_m5(concept_ids=(c1, c2), transition_signature=80)
    m6 = pipeline.derive_m6(world_model_ids=(m5,), action_signature=90, efficiency_gain=0.25)

    _, view, _ = writer.commit_generation()
    assert [view.nodes[mid].level for mid in (m1a, m2, r1, c1, m5, m6)] == [
        MemoryLevel.M1, MemoryLevel.M2, MemoryLevel.M3, MemoryLevel.M4, MemoryLevel.M5, MemoryLevel.M6
    ]
    assert view.nodes[m1a].support_count == 2
    score_input = view.score_inputs(context_signature=10, action_ids=(2,), family_ids_by_action={2: m2})[0]
    assert m1a in score_input.contingency_ids
    assert r1 in score_input.role_ids
    assert c1 in score_input.concept_ids


def test_canonical_resolution_is_deterministic_for_parallel_duplicate_candidates() -> None:
    ids = MemoryIdAllocator()
    source_a, source_b = ids.allocate_many(2)
    candidate_a = ScientificDerivationKernels.m2_family(action_id=1, member_ids=(source_a,), outcome_class=5)
    candidate_b = ScientificDerivationKernels.m2_family(action_id=2, member_ids=(source_b,), outcome_class=6)

    def merged(reverse: bool):
        writer = CanonicalMemoryWriter()
        writer.apply_mutation_batch((
            NodeMutation(source_a, MemoryLevel.M1, 10, 1),
            NodeMutation(source_b, MemoryLevel.M1, 10, 1),
        ))
        t1 = DerivationTask(GenerationId(1), MemoryLevel.M2, (source_a,))
        t2 = DerivationTask(GenerationId(1), MemoryLevel.M2, (source_b,))
        results = [
            DerivationTaskResult(t1, DerivedMutationBatch(GenerationId(1), MemoryLevel.M2, (source_a,), semantic_candidates=(candidate_a,))),
            DerivationTaskResult(t2, DerivedMutationBatch(GenerationId(1), MemoryLevel.M2, (source_b,), semantic_candidates=(candidate_b,))),
        ]
        if reverse:
            results.reverse()
        stats = DeterministicDerivedBatchMerger().apply(results, writer=writer)
        return writer.canonical_memory_id(candidate_a.key), writer.canonical_memory_id(candidate_b.key), stats

    forward = merged(False)
    reverse = merged(True)
    assert forward[:2] == reverse[:2]
    assert forward[2].candidates == 2


def test_vectorized_action_scoring_uses_packed_aggregates() -> None:
    indexes = CognitionIndexes.freeze(
        contingency_by_context_action={}, role_by_context_action_family={}, role_by_context_action={}, concepts_by_role={},
        action_aggregates={
            1: ActionAggregate(future_option_sum=4.0, future_option_count=2, positive_count=4, failure_count=0),
            2: ActionAggregate(future_option_sum=0.0, future_option_count=1, positive_count=1, negative_count=2, failure_count=2, contradiction_count=1),
        },
    )
    view = MemoryReadView.freeze(generation_id=GenerationId(1), nodes={}, scores={}, adjacency={}, cognition_indexes=indexes)
    result = VectorizedActionScorer().score(view.packed_cognition, (1, 2))
    assert result.action_ids.dtype == np.int64
    assert result.scores.flags.writeable is False
    assert result.best_action() == 1


def test_incremental_generation_reuses_unchanged_arena_sections() -> None:
    ids = MemoryIdAllocator()
    a, b = ids.allocate_many(2)
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch((NodeMutation(a, MemoryLevel.M1, 10, 1), NodeMutation(b, MemoryLevel.M2, 20, 1)))
    writer.apply_score_batch((ScoreMutation(a, significance=0.1),))
    writer.apply_edge_batch((EdgeMutation(a, 7, b, 1),))
    _, view1, _ = writer.commit_generation()

    writer.apply_score_batch((ScoreMutation(a, significance=0.9),))
    _, view2, _ = writer.commit_generation()
    assert view2.compact_arena.nodes is view1.compact_arena.nodes
    assert view2.compact_arena.adjacency is view1.compact_arena.adjacency
    assert view2.compact_arena.scores is not view1.compact_arena.scores
    assert view2.packed_cognition is view1.packed_cognition


def test_scientific_kernels_reject_under_supported_higher_order_candidates() -> None:
    one = MemoryIdAllocator().allocate()
    try:
        ScientificDerivationKernels.m4_concept(role_ids=(one,), relation_signature=1)
    except ValueError as exc:
        assert "two roles" in str(exc)
    else:
        raise AssertionError("single-role concept was accepted")

    key = CanonicalMemoryKey(MemoryLevel.M2, 200, (1, 2, 3))
    assert key.level == MemoryLevel.M2
