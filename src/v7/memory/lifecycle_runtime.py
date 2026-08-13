from __future__ import annotations

from dataclasses import dataclass

from v7.memory.evidence_lifecycle import EvidenceLifecycleStore
from v7.memory.evidence_store import EvidenceRecord, EvidenceStore
from v7.memory.lifecycle import LifecycleDecision, MemoryLifecycleController
from v7.memory.read_view import MemoryReadView
from v7.memory.writer import CanonicalMemoryWriter

EVIDENCE_PROMOTION = 1001
EVIDENCE_DEMOTION = 1002
EVIDENCE_REPLAY = 1003


@dataclass(frozen=True, slots=True)
class LifecycleRunStats:
    evaluated: int
    promoted: int
    demoted: int
    replay_queued: int
    evidence_records: int
    transfer_signals: int
    contradiction_signals: int


class MemoryLifecycleRuntime:
    """Apply lifecycle decisions and append their historical evidence."""

    def __init__(
        self,
        controller: MemoryLifecycleController | None = None,
        evidence_store: EvidenceStore | None = None,
        evidence_lifecycle: EvidenceLifecycleStore | None = None,
    ) -> None:
        self.controller = controller or MemoryLifecycleController()
        self.evidence_store = evidence_store
        self.evidence_lifecycle = evidence_lifecycle

    def run(self, view: MemoryReadView, *, writer: CanonicalMemoryWriter) -> tuple[tuple[LifecycleDecision, ...], LifecycleRunStats]:
        empirical_transfer = {}
        contradiction_severity = {}
        if self.evidence_lifecycle is not None:
            transfer_summary = self.evidence_lifecycle.transfer_summary(view.nodes.keys())
            empirical_transfer = {
                memory_id: successes / total if total > 0 else 0.0
                for memory_id, (total, successes, _mean_score) in transfer_summary.items()
            }
            contradiction_summary = self.evidence_lifecycle.contradiction_summary(view.nodes.keys())
            contradiction_severity = {
                memory_id: max_severity
                for memory_id, (_total, max_severity) in contradiction_summary.items()
            }
        decisions = self.controller.apply(
            view,
            writer=writer,
            empirical_transfer=empirical_transfer,
            contradiction_severity=contradiction_severity,
        )
        records: list[EvidenceRecord] = []
        generation_id = int(writer.mutable_generation_id)
        for decision in decisions:
            common = {
                "fitness": decision.fitness,
                "previous_flags": decision.previous_flags,
                "next_flags": decision.next_flags,
                "empirical_transfer": decision.empirical_transfer,
                "contradiction_severity": decision.contradiction_severity,
            }
            if decision.promote:
                records.append(EvidenceRecord(decision.memory_id, EVIDENCE_PROMOTION, generation_id, common))
            if decision.demote:
                records.append(EvidenceRecord(decision.memory_id, EVIDENCE_DEMOTION, generation_id, common))
            if decision.replay:
                records.append(EvidenceRecord(decision.memory_id, EVIDENCE_REPLAY, generation_id, common))
        written = self.evidence_store.append_evidence_batch(records) if self.evidence_store is not None else 0
        stats = LifecycleRunStats(
            evaluated=len(decisions),
            promoted=sum(1 for item in decisions if item.promote),
            demoted=sum(1 for item in decisions if item.demote),
            replay_queued=sum(1 for item in decisions if item.replay),
            evidence_records=written,
            transfer_signals=len(empirical_transfer),
            contradiction_signals=len(contradiction_severity),
        )
        return decisions, stats
