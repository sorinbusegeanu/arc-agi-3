from __future__ import annotations

from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.development import DevelopmentalLifecycleRuntime
from v7.memory.evidence_lifecycle import (
    EvidenceLifecycleStore,
    ProvenanceRecord,
    TransferTrialRecord,
)
from v7.memory.evidence_store import EvidenceStore
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.lifecycle import LifecyclePolicy, MemoryLifecycleController
from v7.memory.models import MemoryNode, MemoryScore, NodeMutation, ScoreMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.writer import CanonicalMemoryWriter


def _concept_view(memory_id, *, support=3, flags=0):
    return MemoryReadView.freeze(
        generation_id=GenerationId(1),
        nodes={
            memory_id: MemoryNode(
                memory_id,
                MemoryLevel.M4,
                400,
                GenerationId(1),
                GenerationId(1),
                status_flags=flags,
                support_count=support,
            )
        },
        scores={
            memory_id: MemoryScore(
                memory_id,
                significance=0.8,
                transfer_prior=0.9,
                explanatory_potential=0.9,
            )
        },
        adjacency={},
    )


def test_m4_candidate_is_not_transfer_validated_without_trials() -> None:
    memory_id = MemoryIdAllocator().allocate()
    view = _concept_view(memory_id)
    from v7.memory.concept_validation import EmpiricalConceptValidator

    result = EmpiricalConceptValidator().evaluate(view, transfer_summary={})[0]
    assert result.candidate is True
    assert result.validated is False
    assert result.empirical_transfer is None
    assert result.next_flags & int(ConceptValidationStatus.CANDIDATE)
    assert not result.next_flags & int(ConceptValidationStatus.TRANSFER_VALIDATED)


def test_successful_unseen_transfer_validates_concept(tmp_path) -> None:
    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter(initial_generation=1)
    writer.apply_mutation_batch(
        (NodeMutation(memory_id, MemoryLevel.M4, 400, support_delta=3),)
    )
    writer.apply_score_batch(
        (
            ScoreMutation(
                memory_id,
                significance=0.8,
                transfer_prior=0.9,
                explanatory_potential=0.9,
            ),
        )
    )
    _, view, _ = writer.commit_generation()

    ledger = EvidenceLifecycleStore(tmp_path / "ledger.sqlite")
    events = EvidenceStore(tmp_path / "events.sqlite")
    try:
        ledger.append_provenance(
            (
                ProvenanceRecord(
                    memory_id=memory_id,
                    generation_id=1,
                    source_game="source",
                ),
            )
        )
        ledger.append_transfer_trials(
            (
                TransferTrialRecord(
                    memory_id,
                    2,
                    "source",
                    "heldout-a",
                    True,
                    0.9,
                    {"source_global_step": 1, "attribution": "trajectory_usage"},
                ),
                TransferTrialRecord(
                    memory_id,
                    2,
                    "source",
                    "heldout-b",
                    True,
                    0.8,
                    {"source_global_step": 2, "attribution": "trajectory_usage"},
                ),
            )
        )
        runtime = DevelopmentalLifecycleRuntime(
            evidence_lifecycle=ledger,
            evidence_store=events,
            lifecycle=MemoryLifecycleController(
                LifecyclePolicy(promote_threshold=0.2, retain_threshold=0.1)
            ),
        )
        result = runtime.run(view, writer=writer)
        assert len(result.concepts) == 1
        concept = result.concepts[0]
        assert concept.transfer_trials == 2
        assert concept.empirical_transfer == 1.0
        assert concept.validated is True
        assert result.concept_mutations == 1
        _, updated, _ = writer.commit_generation()
        assert updated.nodes[memory_id].status_flags & int(
            ConceptValidationStatus.TRANSFER_VALIDATED
        )
        assert view.scores[memory_id].transfer_prior == 0.9
        validation_rows = events.connection.execute(
            "SELECT payload_json FROM evidence_records WHERE evidence_type=1004"
        ).fetchall()
        assert validation_rows
        assert '"heldout_validation":true' in validation_rows[0][0]
    finally:
        events.close()
        ledger.close()


def test_failed_heldout_transfer_rejects_previously_validated_concept() -> None:
    memory_id = MemoryIdAllocator().allocate()
    flags = int(
        ConceptValidationStatus.CANDIDATE
        | ConceptValidationStatus.TRANSFER_VALIDATED
    )
    view = _concept_view(memory_id, flags=flags)
    from v7.memory.concept_validation import EmpiricalConceptValidator

    decision = EmpiricalConceptValidator().evaluate(
        view,
        transfer_summary={memory_id: (2, 0, 0.1)},
    )[0]
    assert decision.validated is False
    assert decision.rejected is True
    assert decision.next_flags & int(ConceptValidationStatus.TRANSFER_REJECTED)
    assert not decision.next_flags & int(
        ConceptValidationStatus.TRANSFER_VALIDATED
    )
