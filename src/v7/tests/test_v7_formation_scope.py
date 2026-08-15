from __future__ import annotations

from v7.memory.evidence_lifecycle import (
    EvidenceLifecycleStore,
    ProvenanceRecord,
    TransferTrialRecord,
)
from v7.memory.ids import MemoryId


def test_heldout_scope_is_frozen_at_concept_creation_generation(tmp_path) -> None:
    ledger = EvidenceLifecycleStore(tmp_path / "ledger.sqlite")
    concept = MemoryId(10)
    parent = MemoryId(11)
    try:
        ledger.append_provenance(
            (
                ProvenanceRecord(concept, 2, parent_memory_id=parent),
                ProvenanceRecord(parent, 1, source_game="source"),
                # This provenance appears only after the concept was created.
                ProvenanceRecord(parent, 5, source_game="heldout"),
            )
        )
        ledger.append_transfer_trials(
            (
                TransferTrialRecord(
                    concept,
                    4,
                    "source",
                    "heldout",
                    True,
                    1.0,
                    {
                        "source_global_step": 1,
                        "attribution": "trajectory_usage",
                    },
                ),
            )
        )
        current = ledger.heldout_transfer_summary((concept,))
        assert concept not in current
        frozen = ledger.heldout_transfer_summary(
            (concept,),
            formation_generations={concept: 2},
        )
        assert frozen[concept][:2] == (1, 1)
        assert ledger.provenance_source_games_at(concept, 2) == ("source",)
    finally:
        ledger.close()
