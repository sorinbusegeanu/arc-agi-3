from __future__ import annotations

from v7.developmental_v707 import (
    CarrierPersistenceRuntime,
    ContextRefinementRuntime,
    MemoryCompressionRuntime,
    developmental_memory_fitness,
    infer_developmental_outcome,
    promotion_assessment,
    trajectory_efficiency,
)
from v7.derivation.scientific import TYPE_CONCEPT, TYPE_CONTINGENCY
from v7.memory.canonical import CanonicalCandidateMutation, CanonicalMemoryKey
from v7.memory.evidence_lifecycle import (
    ContradictionRecord,
    EvidenceLifecycleStore,
    ProvenanceRecord,
)
from v7.memory.evidence_store import EvidenceRecord, EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.gate_validation import EmpiricalGateValidator
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.models import MemoryNode, MemoryScore, NodeMutation
from v7.memory.state import CognitiveState, GateId, GateValidationState
from v7.memory.writer import CanonicalMemoryWriter


def test_terminal_label_does_not_define_developmental_significance() -> None:
    left = infer_developmental_outcome(
        future_option_delta=-2.0,
        continuation_delta=-1.0,
        environment_terminal_type="WIN",
    )
    right = infer_developmental_outcome(
        future_option_delta=-2.0,
        continuation_delta=-1.0,
        environment_terminal_type="GAME_OVER",
    )
    assert left.learned_significance == right.learned_significance
    assert left.developmental_polarity == right.developmental_polarity == -1


def test_promotion_score_prefers_explanatory_transfer_over_frequency() -> None:
    frequent = MemoryNode(MemoryId(1), MemoryLevel.M4, TYPE_CONCEPT, 1, 1, support_count=30)
    explanatory = MemoryNode(MemoryId(2), MemoryLevel.M4, TYPE_CONCEPT, 1, 1, support_count=4)
    frequent_score = MemoryScore(
        MemoryId(1), significance=.40, learning_value=.30, transfer_prior=.10, explanatory_potential=.10
    )
    explanatory_score = MemoryScore(
        MemoryId(2), significance=.65, learning_value=.55, transfer_prior=.90, explanatory_potential=.90
    )
    assert promotion_assessment(explanatory, explanatory_score).score > promotion_assessment(
        frequent, frequent_score
    ).score


def test_promotion_score_cannot_validate_without_empirical_trials() -> None:
    writer = CanonicalMemoryWriter(gate_candidates=True)
    key = CanonicalMemoryKey(MemoryLevel.M4, TYPE_CONCEPT, (11, 12))
    writer.apply_canonical_candidate_batch(
        (
            CanonicalCandidateMutation(
                key,
                support_delta=6,
                significance=.9,
                learning_value=.9,
                transfer_prior=.9,
                explanatory_potential=.9,
            ),
        )
    )
    view = writer.prepare_generation()[1]
    decision = EmpiricalGateValidator().evaluate(view, gate_summaries={})[0]
    assert decision.probe_eligible
    assert not decision.validated
    assert decision.next_validation_state != GateValidationState.VALIDATED


def test_signed_future_options_change_developmental_fitness() -> None:
    node = MemoryNode(MemoryId(1), MemoryLevel.M4, TYPE_CONCEPT, 1, 1, support_count=4)
    base = dict(
        significance=.6,
        prediction_error=.3,
        learning_value=.6,
        transfer_prior=.6,
        explanatory_potential=.6,
    )
    positive = MemoryScore(MemoryId(1), future_option_delta=3.0, **base)
    negative = MemoryScore(MemoryId(1), future_option_delta=-3.0, **base)
    assert developmental_memory_fitness(node, positive, stage="TRANSFER") > developmental_memory_fitness(
        node, negative, stage="TRANSFER"
    )


def test_developmental_weights_change_residency_pressure_not_validation_state() -> None:
    node = MemoryNode(
        MemoryId(1),
        MemoryLevel.M5,
        500,
        1,
        1,
        support_count=5,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(GateValidationState.VALIDATED),
        gate_id=int(GateId.G45),
    )
    score = MemoryScore(
        MemoryId(1),
        significance=.3,
        prediction_error=.2,
        learning_value=.4,
        transfer_prior=.7,
        explanatory_potential=.8,
        future_option_delta=2.0,
    )
    control = developmental_memory_fitness(node, score, empirical_transfer=.8, stage="CONTROL")
    planning = developmental_memory_fitness(node, score, empirical_transfer=.8, stage="PLANNING")
    assert control != planning
    assert node.validation_state == int(GateValidationState.VALIDATED)


def test_contradiction_refines_context_before_rejecting_parent(tmp_path) -> None:
    writer = CanonicalMemoryWriter(gate_candidates=True)
    key = CanonicalMemoryKey(MemoryLevel.M1, TYPE_CONTINGENCY, (10, 1, 100))
    parent_id = writer.apply_canonical_candidate_batch(
        (CanonicalCandidateMutation(key, support_delta=6, learning_value=.5),)
    )[key]
    node = writer._nodes[parent_id]
    writer.apply_mutation_batch(
        (
            NodeMutation(
                parent_id,
                node.level,
                node.type_id,
                cognitive_state=int(CognitiveState.ACTIVE),
                validation_state=int(GateValidationState.VALIDATED),
                gate_id=int(GateId.G01),
            ),
        )
    )
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    evidence = EvidenceStore(tmp_path / "evidence.sqlite")
    lifecycle.append_contradictions(
        (
            ContradictionRecord(parent_id, 1, 1.0, source_game="g", source_global_step=5),
            ContradictionRecord(parent_id, 1, 1.0, source_game="g", source_global_step=6),
        )
    )
    rows = []
    step = 1
    for context, outcome in ((20, 100), (20, 100), (20, 100), (30, 200), (30, 200), (30, 200)):
        rows.append(
            EvidenceRecord(
                parent_id,
                int(EvidenceType.EPISODE),
                1,
                {
                    "context_signatures": [context],
                    "action_id": 1,
                    "outcome_signature": outcome,
                },
                source_game="g",
                source_context=str(context),
                source_global_step=step,
            )
        )
        step += 1
    evidence.append_evidence_batch(rows)
    decisions = ContextRefinementRuntime(lifecycle, evidence).run(
        writer.published_view, writer=writer
    )
    accepted = [item for item in decisions if item.accepted]
    assert accepted
    assert accepted[0].prediction_gain > 0
    assert accepted[0].child_memory_id is not None
    assert writer._nodes[parent_id].cognitive_state == int(CognitiveState.PROBE_ONLY)
    assert writer._nodes[parent_id].validation_state == int(GateValidationState.VALIDATED)
    evidence.close()
    lifecycle.close()


def test_failed_context_refinement_leaves_parent_active(tmp_path) -> None:
    writer = CanonicalMemoryWriter(gate_candidates=False)
    key = CanonicalMemoryKey(MemoryLevel.M1, TYPE_CONTINGENCY, (10, 1, 100))
    parent_id = writer.apply_canonical_candidate_batch((CanonicalCandidateMutation(key, support_delta=4),))[key]
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    evidence = EvidenceStore(tmp_path / "evidence.sqlite")
    lifecycle.append_contradictions(
        (
            ContradictionRecord(parent_id, 1, .8),
            ContradictionRecord(parent_id, 1, .8),
        )
    )
    evidence.append_evidence_batch(
        EvidenceRecord(
            parent_id,
            int(EvidenceType.EPISODE),
            1,
            {"context_signatures": [20], "action_id": 1, "outcome_signature": 100},
            source_global_step=i,
        )
        for i in range(1, 5)
    )
    ContextRefinementRuntime(lifecycle, evidence).run(writer.published_view, writer=writer)
    assert writer._nodes[parent_id].cognitive_state == int(CognitiveState.ACTIVE)
    evidence.close()
    lifecycle.close()


def test_validated_abstraction_replaces_redundant_parent_and_keeps_provenance(tmp_path) -> None:
    writer = CanonicalMemoryWriter(gate_candidates=True)
    parent_key = CanonicalMemoryKey(MemoryLevel.M1, TYPE_CONTINGENCY, (10, 1, 100))
    parent_id = writer.apply_canonical_candidate_batch((CanonicalCandidateMutation(parent_key, support_delta=4),))[parent_key]
    parent = writer._nodes[parent_id]
    writer.apply_mutation_batch(
        (NodeMutation(parent_id, parent.level, parent.type_id, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(GateValidationState.VALIDATED), gate_id=int(GateId.G01)),)
    )
    concept_key = CanonicalMemoryKey(MemoryLevel.M4, TYPE_CONCEPT, (77, 88))
    concept_id = writer.apply_canonical_candidate_batch(
        (CanonicalCandidateMutation(concept_key, support_delta=4, explanatory_potential=.9, transfer_prior=.9),)
    )[concept_key]
    concept = writer._nodes[concept_id]
    writer.apply_mutation_batch(
        (NodeMutation(concept_id, concept.level, concept.type_id, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(GateValidationState.VALIDATED), gate_id=int(GateId.G34)),)
    )
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    lifecycle.append_provenance(
        (
            ProvenanceRecord(parent_id, 1, source_context="ctx"),
            ProvenanceRecord(concept_id, 1, parent_memory_id=parent_id),
            ProvenanceRecord(concept_id, 1, source_context="ctx"),
        )
    )
    runtime = MemoryCompressionRuntime(lifecycle)
    first = runtime.run(writer=writer)
    assert any(item.parent_memory_id == parent_id for item in first)
    assert writer._nodes[parent_id].cognitive_state == int(CognitiveState.PROBE_ONLY)
    assert runtime.is_provenance_only(parent_id, concept_id)
    with lifecycle.connection:
        lifecycle.connection.execute(
            "UPDATE memory_compression_replacements SET generation_id=-2 WHERE parent_memory_id=?",
            (int(parent_id),),
        )
    runtime.run(writer=writer)
    assert writer._nodes[parent_id].cognitive_state == int(CognitiveState.RETIRED)
    tombstone = lifecycle.connection.execute(
        "SELECT replacement_memory_id FROM memory_tombstones WHERE memory_id=?",
        (int(parent_id),),
    ).fetchone()
    assert tombstone == (int(concept_id),)
    assert lifecycle.provenance_parents(concept_id) == (parent_id,)
    lifecycle.close()


def test_unique_coverage_prevents_unsafe_compression(tmp_path) -> None:
    writer = CanonicalMemoryWriter(gate_candidates=False)
    parent_key = CanonicalMemoryKey(MemoryLevel.M1, TYPE_CONTINGENCY, (10, 1, 100))
    parent_id = writer.apply_canonical_candidate_batch((CanonicalCandidateMutation(parent_key, support_delta=4),))[parent_key]
    concept_key = CanonicalMemoryKey(MemoryLevel.M4, TYPE_CONCEPT, (77, 88))
    concept_id = writer.apply_canonical_candidate_batch((CanonicalCandidateMutation(concept_key, support_delta=4),))[concept_key]
    concept = writer._nodes[concept_id]
    writer.apply_mutation_batch(
        (NodeMutation(concept_id, concept.level, concept.type_id, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(GateValidationState.VALIDATED), gate_id=int(GateId.G34)),)
    )
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    lifecycle.append_provenance(
        (
            ProvenanceRecord(parent_id, 1, source_context="unique"),
            ProvenanceRecord(concept_id, 1, parent_memory_id=parent_id),
            ProvenanceRecord(concept_id, 1, source_context="covered"),
        )
    )
    decisions = MemoryCompressionRuntime(lifecycle, unique_coverage_tolerance=0.10).run(writer=writer)
    assert decisions == ()
    assert writer._nodes[parent_id].cognitive_state == int(CognitiveState.ACTIVE)
    lifecycle.close()


def test_m6_efficiency_compares_equivalent_outcomes_not_raw_length() -> None:
    short = trajectory_efficiency(
        actions=(1, 2), contexts=(10, 11), future_option_sum=4.0, raw_action_option_sum=0.0
    )
    long = trajectory_efficiency(
        actions=(1, 2, 3, 4), contexts=(10, 11, 12, 13), future_option_sum=4.0, raw_action_option_sum=0.0
    )
    harmful_short = trajectory_efficiency(
        actions=(1,), contexts=(10,), future_option_sum=-4.0, raw_action_option_sum=0.0
    )
    assert short.equivalent_group == long.equivalent_group
    assert short.efficiency > long.efficiency
    assert harmful_short.efficiency < long.efficiency


def test_carrier_persistence_survives_appearance_signature_change(tmp_path) -> None:
    lifecycle = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    evidence = EvidenceStore(tmp_path / "evidence.sqlite")
    payloads = (
        (1, 11, 7, [1], [2]),
        (2, 22, 7, [2], [3]),
        (3, 11, 8, [3], [4]),
        (4, 22, 8, [4], [5]),
    )
    evidence.append_evidence_batch(
        EvidenceRecord(
            None,
            int(EvidenceType.EPISODE),
            1,
            {
                "carrier_signature": carrier,
                "outcome_signature": outcome,
                "context_signatures": contexts,
                "next_context_signatures": next_contexts,
            },
            source_game="g",
            source_global_step=step,
        )
        for step, carrier, outcome, contexts, next_contexts in payloads
    )
    runtime = CarrierPersistenceRuntime(lifecycle, evidence)
    links = runtime.run(writer=CanonicalMemoryWriter())
    assert links and links[0].support_count >= 2
    assert runtime.canonical_carrier_signature(11) == runtime.canonical_carrier_signature(22)
    evidence.close()
    lifecycle.close()
