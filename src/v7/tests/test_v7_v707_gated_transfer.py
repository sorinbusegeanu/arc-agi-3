from __future__ import annotations

from v7.memory.evidence_lifecycle import EvidenceLifecycleStore, TransferTrialRecord
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.lifecycle_runtime import MemoryLifecycleRuntime
from v7.memory.models import NodeMutation
from v7.memory.state import CognitiveState, GateId, GateValidationState
from v7.memory.writer import CanonicalMemoryWriter


def test_explicit_gate_ignores_legacy_associative_transfer_for_lifecycle(tmp_path) -> None:
    writer = CanonicalMemoryWriter()
    memory_id = MemoryId(71)
    writer.apply_mutation_batch(
        (
            NodeMutation(
                memory_id,
                MemoryLevel.M4,
                400,
                support_delta=3,
                cognitive_state=int(CognitiveState.PROBE_ONLY),
                validation_state=int(GateValidationState.PROBE_ELIGIBLE),
                gate_id=int(GateId.G34),
            ),
        )
    )
    _, view, _ = writer.commit_generation()

    ledger = EvidenceLifecycleStore(tmp_path / "ledger.sqlite")
    try:
        ledger.append_transfer_trials(
            (
                TransferTrialRecord(
                    memory_id=memory_id,
                    generation_id=2,
                    source_game="source",
                    target_game="target",
                    success=True,
                    score=1.0,
                ),
            )
        )
        decisions, stats = MemoryLifecycleRuntime(
            evidence_lifecycle=ledger
        ).run(view, writer=writer)
        decision = next(item for item in decisions if item.memory_id == memory_id)
        assert decision.empirical_transfer == 0.0
        assert stats.transfer_signals == 0
    finally:
        ledger.close()
