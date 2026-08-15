from __future__ import annotations

import sqlite3

from v7.memory.canonical import CanonicalCandidateMutation, CanonicalMemoryKey
from v7.memory.evidence_lifecycle import (
    EvidenceLifecycleStore,
    GateTrialRecord,
    ProvenanceRecord,
)
from v7.memory.gate_validation import EmpiricalGateValidator, GateTrialSummary
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import RoleConceptIndexMutation, RoleIndexMutation
from v7.memory.lifecycle import MemoryLifecycleController
from v7.memory.models import MemoryNode, NodeMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.schema import ensure_v7_schema
from v7.memory.state import CognitiveState, GateId, GateValidationState, gate_for_identity
from v7.memory.writer import CanonicalMemoryWriter


def _view(*nodes: MemoryNode) -> MemoryReadView:
    return MemoryReadView.freeze(
        generation_id=1,
        nodes={node.memory_id: node for node in nodes},
        scores={},
        adjacency={},
    )


def test_gate_mapping_covers_reusable_hierarchy() -> None:
    assert gate_for_identity(MemoryLevel.M1, 100) == GateId.G01
    assert gate_for_identity(MemoryLevel.M2, 200) == GateId.G12
    assert gate_for_identity(MemoryLevel.M3, 302) == GateId.G23C
    assert gate_for_identity(MemoryLevel.M3, 300) == GateId.G23R
    assert gate_for_identity(MemoryLevel.M4, 400) == GateId.G34
    assert gate_for_identity(MemoryLevel.M5, 500) == GateId.G45
    assert gate_for_identity(MemoryLevel.M6, 600) == GateId.G56


def test_new_canonical_candidate_is_probe_only_until_gate_validation() -> None:
    writer = CanonicalMemoryWriter()
    role_id = MemoryId(100)
    writer.apply_mutation_batch(
        (NodeMutation(role_id, MemoryLevel.M3, 300, support_delta=4),)
    )
    writer.apply_role_index_batch((RoleIndexMutation(11, 2, role_id, None),))
    key = CanonicalMemoryKey(MemoryLevel.M4, 400, (99,))
    candidate = CanonicalCandidateMutation(
        key=key,
        support_delta=3,
        parents=(role_id,),
        transfer_prior=0.8,
    )
    concept_id = writer.apply_canonical_candidate_batch((candidate,))[key]
    writer.apply_role_concept_index_batch(
        (RoleConceptIndexMutation(role_id, concept_id),)
    )
    _state, view, _delta = writer.commit_generation()

    node = view.nodes[concept_id]
    assert node.gate_id == int(GateId.G34)
    assert node.validation_state == int(GateValidationState.PROBE_ELIGIBLE)
    assert node.cognitive_state == int(CognitiveState.PROBE_ONLY)
    normal = view.score_inputs(context_signature=11, action_ids=(2,))[0]
    probe = view.probe_score_inputs(context_signature=11, action_ids=(2,))[0]
    assert concept_id not in normal.concept_ids
    assert concept_id in probe.concept_ids


def test_quarantined_memory_is_absent_even_from_probe_index() -> None:
    role = MemoryNode(MemoryId(1), MemoryLevel.M3, 300, 1, 1, support_count=4)
    concept = MemoryNode(
        MemoryId(2),
        MemoryLevel.M4,
        400,
        1,
        1,
        support_count=4,
        cognitive_state=int(CognitiveState.QUARANTINED),
        validation_state=int(GateValidationState.REJECTED),
        gate_id=int(GateId.G34),
    )
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        (
            NodeMutation(role.memory_id, role.level, role.type_id, support_delta=4),
            NodeMutation(
                concept.memory_id,
                concept.level,
                concept.type_id,
                support_delta=4,
                cognitive_state=concept.cognitive_state,
                validation_state=concept.validation_state,
                gate_id=concept.gate_id,
            ),
        )
    )
    writer.apply_role_index_batch((RoleIndexMutation(7, 3, role.memory_id, None),))
    writer.apply_role_concept_index_batch(
        (RoleConceptIndexMutation(role.memory_id, concept.memory_id),)
    )
    _state, view, _delta = writer.commit_generation()
    row = view.probe_score_inputs(context_signature=7, action_ids=(3,))[0]
    assert concept.memory_id not in row.concept_ids


def test_frozen_candidate_scope_cannot_be_rewritten_by_later_provenance(tmp_path) -> None:
    store = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        memory_id = MemoryId(42)
        store.append_provenance(
            (ProvenanceRecord(memory_id, 1, source_game="game_a", source_context="ctx_a"),)
        )
        scope = store.freeze_candidate_scope(memory_id, 1)
        assert scope.provenance_games == ("game_a",)
        store.append_provenance(
            (ProvenanceRecord(memory_id, 3, source_game="game_b", source_context="ctx_b"),)
        )
        store.append_gate_trials(
            (
                GateTrialRecord(
                    memory_id,
                    4,
                    GateId.G34,
                    1,
                    target_game="game_a",
                    target_context="target-a",
                    contribution=0.2,
                    causal_gain=0.2,
                    intervention_type="decision_score_ablation",
                    paired_trial_id="a",
                ),
                GateTrialRecord(
                    memory_id,
                    4,
                    GateId.G34,
                    1,
                    target_game="game_b",
                    target_context="target-b",
                    contribution=0.2,
                    causal_gain=0.2,
                    intervention_type="decision_score_ablation",
                    paired_trial_id="b",
                ),
            )
        )
        summary = store.gate_trial_summary((memory_id,))[memory_id]
        assert summary.trials == 1
        assert summary.successes == 1
        assert summary.mean_causal_gain == 0.2
    finally:
        store.close()


def test_terminal_success_without_counterfactual_gain_cannot_validate(tmp_path) -> None:
    store = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        memory_id = MemoryId(7)
        store.append_provenance(
            (ProvenanceRecord(memory_id, 1, source_game="source"),)
        )
        store.freeze_candidate_scope(memory_id, 1)
        store.append_gate_trials(
            (
                GateTrialRecord(
                    memory_id,
                    2,
                    GateId.G34,
                    1,
                    target_game="target",
                    target_context="x",
                    contribution=0.0,
                    causal_gain=0.0,
                    terminal_gain=1.0,
                    intervention_type="decision_score_ablation",
                    paired_trial_id="zero-gain",
                ),
            )
        )
        assert store.gate_trial_summary((memory_id,)) == {}
    finally:
        store.close()


def test_positive_and_negative_gate_evidence_activate_or_quarantine() -> None:
    memory_id = MemoryId(9)
    node = MemoryNode(
        memory_id,
        MemoryLevel.M4,
        400,
        1,
        1,
        support_count=4,
        cognitive_state=int(CognitiveState.PROBE_ONLY),
        validation_state=int(GateValidationState.PROBE_ELIGIBLE),
        gate_id=int(GateId.G34),
    )
    view = _view(node)
    validator = EmpiricalGateValidator()
    positive = validator.evaluate(
        view,
        gate_summaries={
            memory_id: GateTrialSummary(
                trials=4,
                successes=4,
                independent_targets=2,
                mean_causal_gain=0.20,
            )
        },
    )[0]
    assert positive.trusted
    assert positive.next_cognitive_state == CognitiveState.ACTIVE

    negative = validator.evaluate(
        view,
        gate_summaries={
            memory_id: GateTrialSummary(
                trials=2,
                successes=0,
                independent_targets=1,
                mean_causal_gain=-0.10,
            )
        },
    )[0]
    assert negative.rejected
    assert negative.next_cognitive_state == CognitiveState.QUARANTINED


def test_invalid_parent_blocks_higher_gate_even_with_positive_trials() -> None:
    memory_id = MemoryId(11)
    node = MemoryNode(
        memory_id,
        MemoryLevel.M5,
        500,
        1,
        1,
        support_count=5,
        cognitive_state=int(CognitiveState.PROBE_ONLY),
        validation_state=int(GateValidationState.PROBE_ELIGIBLE),
        gate_id=int(GateId.G45),
    )
    decision = EmpiricalGateValidator().evaluate(
        _view(node),
        gate_summaries={
            memory_id: GateTrialSummary(
                trials=5,
                successes=5,
                independent_targets=2,
                mean_causal_gain=0.30,
            )
        },
        parent_validity={memory_id: False},
    )[0]
    assert not decision.validated
    assert not decision.probe_eligible
    assert decision.next_cognitive_state == CognitiveState.QUARANTINED


def test_lifecycle_uses_persistent_windows_and_can_reactivate_valid_memory() -> None:
    memory_id = MemoryId(15)
    active = MemoryNode(
        memory_id,
        MemoryLevel.M4,
        400,
        1,
        1,
        support_count=4,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(GateValidationState.VALIDATED),
        gate_id=int(GateId.G34),
    )
    controller = MemoryLifecycleController()
    probe = controller.evaluate(
        _view(active), lifecycle_windows={memory_id: (2, 0, 0)}
    )[0]
    assert probe.next_cognitive_state == int(CognitiveState.PROBE_ONLY)
    quarantine = controller.evaluate(
        _view(active), lifecycle_windows={memory_id: (4, 0, 0)}
    )[0]
    assert quarantine.next_cognitive_state == int(CognitiveState.QUARANTINED)

    quarantined = MemoryNode(
        memory_id,
        MemoryLevel.M4,
        400,
        1,
        2,
        support_count=4,
        cognitive_state=int(CognitiveState.QUARANTINED),
        validation_state=int(GateValidationState.VALIDATED),
        gate_id=int(GateId.G34),
    )
    recovered = controller.evaluate(
        _view(quarantined), lifecycle_windows={memory_id: (0, 0, 2)}
    )[0]
    assert recovered.next_cognitive_state == int(CognitiveState.ACTIVE)


def test_schema_v1_migrates_to_v2(tmp_path) -> None:
    path = tmp_path / "state.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE v7_schema_meta(schema_version INTEGER NOT NULL)")
        connection.execute("INSERT INTO v7_schema_meta VALUES (1)")
        connection.execute(
            """
            CREATE TABLE memory_nodes(
                memory_id INTEGER PRIMARY KEY,
                level_id INTEGER NOT NULL,
                type_id INTEGER NOT NULL,
                created_generation INTEGER NOT NULL,
                updated_generation INTEGER NOT NULL,
                status_flags INTEGER NOT NULL DEFAULT 0,
                support_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.commit()
        ensure_v7_schema(connection)
        version = connection.execute("SELECT schema_version FROM v7_schema_meta").fetchone()[0]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_nodes)")}
        assert version == 2
        assert {"cognitive_state", "validation_state", "gate_id"} <= columns
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='gate_trials'"
        ).fetchone()
    finally:
        connection.close()
