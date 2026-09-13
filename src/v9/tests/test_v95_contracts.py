from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from v9.cognition.action_selection import GroundedActionSignal, action_scores
from v9.cognition.developmental_stage import DevelopmentalStage, DevelopmentalStageTracker, StageEvidence
from v9.cognition.grounding import GroundingMaturity
from v9.cognition.similarity import StructuralCandidateIndex
from v9.cognition.transfer import TransferTrustState
from v9.memory import (
    DerivationProvenance,
    M0Episode,
    M1GroundedContingency,
    M1NormalizedRelation,
    M2TransformationFamily,
    M3FunctionalRole,
    M4Concept,
    M5ConsequenceStructure,
    M6Outcome,
    M7Strategy,
    MemoryLevel,
    MemoryType,
    MemoryUid,
)
from v9.memory.identity import ContextScopeId, EpisodeId, EventUid, LineageUid
from v9.memory.model import CanonicalNode, CognitiveState, ExperienceEvent
from v9.memory.m1_grounded import GroundedRelation
from v9.memory.m1_normalized import NormalizedChannel
from v9.modalities.contract import InteractionEvent, TimelineIdentity, WORLD_MODALITY
from v9.mutation import ContextRegistry, EffectiveStateResolver, LineageContextOverlay, LineageStore, MutationKind, MutationProposal, MutationWrite, ProposalClass, ReadSet
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig, ScientificConfig
from v9.runtime.lifecycle import LifecycleRegistry
from v9.runtime.publication import CanonicalGraph
from v9.research.experiments import run_matched_transfer_trial
from v9.memory.residency import PayloadStore, ResidencyState


def _m0(sequence: int, *, environment: int = 7) -> M0Episode:
    event_id = EventUid.from_producer(1, sequence)
    experience = ExperienceEvent(event_id, sequence, 1, sequence, environment, sequence, 3, 2, 4, next_context_signature=5)
    identity = TimelineIdentity(event_id, sequence, 1, sequence, environment, EpisodeId(1), WORLD_MODALITY)
    return M0Episode.from_event(InteractionEvent(identity, experience), context_signature=3, payload_digest=99)


def test_m0_through_m7_identities_exclude_mutable_state_and_parent_connectivity() -> None:
    event = _m0(1)
    alternate_provenance = _m0(1, environment=99)
    assert event.uid == alternate_provenance.uid

    grounded_a = M1GroundedContingency.build(GroundedRelation.ACTION_CONDITIONED, (_m0(1),))
    grounded_b = M1GroundedContingency.build(GroundedRelation.ACTION_CONDITIONED, (_m0(2),))
    assert grounded_a.uid == grounded_b.uid
    normalized_a = M1NormalizedRelation.build("REL", NormalizedChannel.WORLD, (grounded_a,))
    normalized_b = M1NormalizedRelation.build("REL", NormalizedChannel.WORLD, (grounded_b,))
    assert normalized_a.uid == normalized_b.uid

    family_a = M2TransformationFamily.form((normalized_a, normalized_b))
    family_b = replace(family_a, recurrence=99, compression_benefit=88.0)
    assert family_a.uid == family_b.uid
    role_a = M3FunctionalRole.form((family_a,), consequence_signature=7)
    role_b = M3FunctionalRole.form((family_b,), consequence_signature=7)
    assert role_a.uid == role_b.uid
    concept_a = M4Concept.candidate((role_a,), compression_benefit=1.0, explanatory_reach=2, transfer_prior=0.1, formation_scope=(7,))
    concept_b = M4Concept.candidate((role_b,), compression_benefit=4.0, explanatory_reach=9, transfer_prior=0.9, formation_scope=(8,))
    assert concept_a.uid == concept_b.uid
    consequence_a = M5ConsequenceStructure.form((concept_a,), (11,))
    consequence_b = M5ConsequenceStructure.form((concept_b.with_validation((9,)),), (11,))
    assert consequence_a.uid == consequence_b.uid
    outcome_a = M6Outcome.form((consequence_a,), diameter_bound=0)
    outcome_b = M6Outcome.form((consequence_b,), diameter_bound=0)
    assert outcome_a.uid == outcome_b.uid
    strategy_a = M7Strategy.form(outcome_a, target_environment_id=9, native_actions=(2,), successes=1, trials=1, primary_valence_sum=1, realized_cost_sum=3)
    strategy_b = M7Strategy.form(outcome_b, target_environment_id=9, native_actions=(2,), successes=7, trials=9, primary_valence_sum=-2, realized_cost_sum=30)
    assert strategy_a.uid == strategy_b.uid


def test_context_and_lineage_have_one_effective_state_authority() -> None:
    node = CanonicalNode.build(MemoryLevel.M4, MemoryType.CONCEPT, (1, 2), 1)
    evidence = MemoryUid.derive("evidence", 1)
    contexts = ContextRegistry(limit=2)
    context = contexts.form((4, 8), evidence_refs=(evidence,), formation_watermark=2, support=2, improvement=0.25)
    lineages = LineageStore()
    lineage = LineageUid(3)
    lineages.put_overlay(LineageContextOverlay(node.uid, lineage, context.scope_id, lifecycle=CognitiveState.PROBATION))
    state = EffectiveStateResolver(contexts, lineages).resolve(node, lineage_uid=lineage, context_scope_id=context.scope_id)
    assert state.context == context
    assert state.lifecycle is CognitiveState.PROBATION
    assert state.usable


def test_stateful_mutation_requires_read_set_and_nested_read_view_is_immutable() -> None:
    graph = CanonicalGraph(1)
    node = CanonicalNode.build(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (1,), 1)
    with pytest.raises(ValueError, match="read set"):
        MutationProposal.build(MutationKind.UPDATE_VALIDATION, target_partitions=(0,), read_set=ReadSet.build((), maximum_size=1), evidence_refs=(), causal_watermark=1, writes=(MutationWrite(node=node, payload={}),), proposal_class=ProposalClass.STATEFUL)
    additive = MutationProposal.build(MutationKind.UPSERT_NODE, target_partitions=(0,), read_set=ReadSet.build((), maximum_size=1), evidence_refs=(), causal_watermark=1, writes=(MutationWrite(node=node, payload={"nested": {"values": [1, 2]}}),))
    graph.publish(additive)
    with pytest.raises(TypeError):
        graph.read_view().payloads[node.uid]["nested"]["new"] = 3


def test_structural_candidate_index_retrieval_is_bounded() -> None:
    index = StructuralCandidateIndex(bucket_capacity=3, bucket_scan_limit=1)
    nodes = [CanonicalNode.build(MemoryLevel.M2, MemoryType.FAMILY, (1 + 256 * value,), value) for value in range(10)]
    for node in nodes:
        index.add(node)
    keys = StructuralCandidateIndex.keys(nodes[0])
    assert len(index.retrieve(keys, limit=2)) <= 2
    assert all(len(bucket) <= 3 for bucket in index.buckets.values())


def test_learning_only_never_promotes_and_target_scoped_negative_transfer_restarts(tmp_path: Path) -> None:
    scientific = ScientificConfig(transfer_validation_mode="learning_only")
    config = RuntimeConfig.from_path(tmp_path, restore=False, scientific=scientific)
    runtime = ContinuousMemoryRuntime(config)
    for index in range(2):
        runtime.submit(runtime.make_experience(producer_id=1, producer_sequence=index + 1, environment_instance_id=7, global_step=index, context_signature=3, action_id=2, outcome_signature=4, family_signature=5))
    concept = next(iter(runtime._m4))
    runtime.record_transfer_validation(concept, target_environment_id=9, target_native_action=4, enabled_metric=0.0, ablated_metric=1.0, context_scope_id=2)
    runtime.record_transfer_validation(concept, target_environment_id=9, target_native_action=4, enabled_metric=0.0, ablated_metric=1.0, context_scope_id=2)
    runtime.record_transfer_validation(concept, target_environment_id=10, target_native_action=4, enabled_metric=0.0, ablated_metric=1.0, context_scope_id=2)
    runtime.record_transfer_validation(concept, target_environment_id=11, target_native_action=4, enabled_metric=1.0, ablated_metric=0.0, context_scope_id=2)
    runtime.record_transfer_validation(concept, target_environment_id=12, target_native_action=4, enabled_metric=1.0, ablated_metric=0.0, context_scope_id=2)
    assert runtime.metrics()["memory_levels"]["M5"] == 0
    assert runtime.transfer_trust.records[(concept, 9, 2)].state is TransferTrustState.FAILED
    assert runtime.transfer_trust.records[(concept, 10, 2)].state is TransferTrustState.PROBATION
    runtime.close()
    restored = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, scientific=scientific))
    assert restored.transfer_trust.records[(concept, 9, 2)].state is TransferTrustState.FAILED
    assert restored.transfer_trust.records[(concept, 10, 2)].state is TransferTrustState.PROBATION


def test_stage_isf_replay_and_restart_are_causally_persisted(tmp_path: Path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    for index in range(2):
        runtime.submit(runtime.make_experience(producer_id=1, producer_sequence=index + 1, environment_instance_id=7, global_step=index, context_signature=3, action_id=2, outcome_signature=4, family_signature=5, changed_cells=2))
    before_m0 = runtime.metrics()["memory_levels"]["M0"]
    replay = runtime.replay_once()
    assert replay.processed <= runtime.config.scientific.replay_candidates_per_interval
    assert runtime.metrics()["memory_levels"]["M0"] == before_m0
    assert runtime.isf.decisions[-1].evidence_availability_watermark <= runtime.isf.decisions[-1].decision_watermark
    stage = runtime.stage_tracker.stage
    runtime.close()
    restored = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path))
    assert restored.stage_tracker.stage is stage
    assert restored.replay.processed == replay.processed
    assert len(restored.isf.decisions) == 2


def test_grounded_action_influence_requires_g4_validated_higher_memory() -> None:
    graph = CanonicalGraph(1)
    signals = (
        GroundedActionSignal(2, 10.0, GroundingMaturity.G3, MemoryLevel.M4, True, 7, 7),
        GroundedActionSignal(3, 5.0, GroundingMaturity.G4, MemoryLevel.M3, True, 7, 7),
        GroundedActionSignal(4, 3.0, GroundingMaturity.G4, MemoryLevel.M4, True, 7, 7),
    )
    scores = action_scores(graph.read_view(), (2, 3, 4), grounded_signals=signals, target_environment_id=7)
    assert scores == {2: 0.0, 3: 0.0, 4: 3.0}
    cross_environment = (
        GroundedActionSignal(2, 4.0, GroundingMaturity.G4, MemoryLevel.M4, True, 6, 7),
        GroundedActionSignal(3, 5.0, GroundingMaturity.G5, MemoryLevel.M4, True, 6, 7),
    )
    assert action_scores(graph.read_view(), (2, 3), grounded_signals=cross_environment, target_environment_id=7) == {2: 0.0, 3: 5.0}


def test_lifecycle_retirement_preserves_authoritative_dependencies() -> None:
    uid = MemoryUid.derive("memory", 1)
    registry = LifecycleRegistry()
    registry.observe(uid, support_delta=0, relevant_opportunity=True, watermark=1)
    pending = registry.retire_if_exhausted(uid, required_opportunities=1, watermark=2, has_authoritative_dependency=True)
    assert pending.state is CognitiveState.RETIRE_PENDING


def test_lifecycle_reactivation_and_payload_pressure_preserve_provenance() -> None:
    uid = MemoryUid.derive("memory", 2)
    registry = LifecycleRegistry()
    registry.observe(uid, support_delta=0, relevant_opportunity=True, watermark=1)
    retired = registry.retire_if_exhausted(uid, required_opportunities=1, watermark=2)
    assert retired.state is CognitiveState.RETIRED
    assert registry.reactivate(uid, support_delta=1, watermark=3).state is CognitiveState.REACTIVATED
    payloads = PayloadStore(byte_budget=3)
    first = payloads.put(b"abc", uid)
    second = payloads.put(b"de", uid)
    assert payloads.states[first.payload_uid] is ResidencyState.RETIRED
    assert first.payload_uid in payloads.provenance
    assert payloads.states[second.payload_uid] is ResidencyState.HOT


def test_stage_seven_requires_demonstrated_replanning_and_efficiency() -> None:
    tracker = DevelopmentalStageTracker()
    evidence = StageEvidence(stable_contingencies=1, structural_abstractions=1, held_out_transfer_successes=1, mature_consequences=1, outcome_equivalences=1, learned_preferences=1, alternative_strategies=1)
    for watermark in range(6):
        tracker.close_interval(evidence, evidence_watermark=watermark)
    assert tracker.stage is DevelopmentalStage.ALTERNATIVE_STRATEGIES
    tracker.close_interval(evidence, evidence_watermark=7)
    assert tracker.stage is DevelopmentalStage.ALTERNATIVE_STRATEGIES
    tracker.close_interval(replace(evidence, demonstrated_replans=1, efficient_replans=1), evidence_watermark=8)
    assert tracker.stage is DevelopmentalStage.DEMONSTRATED_REPLANNING


def test_generalized_transfer_trial_uses_exact_same_captured_target_state() -> None:
    class Environment:
        def __init__(self) -> None:
            self.value = 5

        def capture_state(self):
            return self.value

        def restore_state(self, state):
            self.value = state

        def available_actions(self):
            return (0, 1)

        def step(self, action):
            self.value += 1 if action else -1
            return self.value

    environment = Environment()
    result = run_matched_transfer_trial(environment, enabled_policy=lambda _o, _a, _s: 1, ablated_policy=lambda _o, _a, _s: 0, metric=lambda env: env.value, horizon=2)
    assert result.matched
    assert result.effect == 4.0
    assert environment.value == 5
