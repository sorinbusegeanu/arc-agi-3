from __future__ import annotations

from v7.memory.evidence_lifecycle import EvidenceLifecycleStore, GateTrialRecord
from v7.memory.evidence_store import EvidenceStore
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.lifecycle_runtime import MemoryLifecycleRuntime
from v7.memory.models import NodeMutation
from v7.memory.state import CognitiveState, GateId
from v7.memory.writer import CanonicalMemoryWriter


def test_persistent_low_utility_retires_and_tombstones_memory(tmp_path) -> None:
    writer = CanonicalMemoryWriter()
    memory_id = MemoryId(1)
    writer.apply_mutation_batch(
        (NodeMutation(memory_id, MemoryLevel.M1, 100, support_delta=0),)
    )
    _state, view, _delta = writer.commit_generation()

    lifecycle_store = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    evidence_store = EvidenceStore(tmp_path / "evidence.sqlite")
    try:
        for generation in range(1, 8):
            lifecycle_store.update_lifecycle_window(
                memory_id,
                generation_id=generation,
                utility=0.0,
                harm=False,
            )
        lifecycle_store.freeze_candidate_scope(memory_id, 1)
        lifecycle_store.append_gate_trials(
            (
                GateTrialRecord(
                    memory_id=memory_id,
                    generation_id=8,
                    gate_id=GateId.G01,
                    candidate_generation=1,
                    target_game="fresh_target",
                    target_context="fresh_context",
                    participated=True,
                    contribution=1.0,
                    causal_gain=0.0,
                    intervention_type="heldout_prediction_ablation",
                    paired_trial_id="retirement-fresh-evidence",
                ),
            )
        )
        decisions, _stats = MemoryLifecycleRuntime(
            evidence_store=evidence_store,
            evidence_lifecycle=lifecycle_store,
        ).run(view, writer=writer)
        decision = next(item for item in decisions if item.memory_id == memory_id)
        assert decision.next_cognitive_state == int(CognitiveState.RETIRED)
        assert decision.retired
        tombstone = lifecycle_store.connection.execute(
            "SELECT retired_generation,reason,provenance_pointer FROM memory_tombstones WHERE memory_id=?",
            (int(memory_id),),
        ).fetchone()
        assert tombstone is not None
        assert tombstone[1] == "persistent_low_utility"
        assert tombstone[2] == "memory:1"
    finally:
        evidence_store.close()
        lifecycle_store.close()
