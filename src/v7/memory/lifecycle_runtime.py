from __future__ import annotations

from dataclasses import dataclass

from v7.memory.evidence_lifecycle import (
    EvidenceLifecycleStore,
    MemoryTombstoneRecord,
)
from v7.memory.evidence_store import EvidenceRecord, EvidenceStore
from v7.memory.lifecycle import LifecycleDecision, MemoryLifecycleController
from v7.memory.read_view import MemoryReadView
from v7.memory.state import CognitiveState
from v7.memory.status import memory_cognitive_state
from v7.memory.writer import CanonicalMemoryWriter

EVIDENCE_PROMOTION = 1001
EVIDENCE_DEMOTION = 1002
EVIDENCE_REPLAY = 1003
EVIDENCE_RETIREMENT = 1006


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
    """Apply persistent selective-forgetting decisions and append evidence."""

    def __init__(
        self,
        controller: MemoryLifecycleController | None = None,
        evidence_store: EvidenceStore | None = None,
        evidence_lifecycle: EvidenceLifecycleStore | None = None,
    ) -> None:
        self.controller = controller or MemoryLifecycleController()
        self.evidence_store = evidence_store
        self.evidence_lifecycle = evidence_lifecycle

    @staticmethod
    def _has_live_dependents(
        writer: CanonicalMemoryWriter,
        view: MemoryReadView,
        memory_id,
    ) -> bool:
        graph = getattr(writer, "_dependencies", None)
        if graph is None:
            return False
        dependents = getattr(graph, "_upstream_to_dependents", {}).get(memory_id, ())
        for dependent_id in dependents:
            state = memory_cognitive_state(view.nodes.get(dependent_id))
            if state is not None and state != CognitiveState.RETIRED:
                return True
        return False

    def run(
        self,
        view: MemoryReadView,
        *,
        writer: CanonicalMemoryWriter,
    ) -> tuple[tuple[LifecycleDecision, ...], LifecycleRunStats]:
        empirical_transfer = {}
        contradiction_severity = {}
        lifecycle_windows = {}
        gate_summaries = {}
        if self.evidence_lifecycle is not None:
            memory_ids = tuple(view.nodes.keys())
            gate_summaries = self.evidence_lifecycle.gate_trial_summary(memory_ids)
            legacy_transfer = self.evidence_lifecycle.transfer_summary(memory_ids)
            for memory_id in memory_ids:
                gate = gate_summaries.get(memory_id)
                if gate is not None and gate.trials > 0:
                    empirical_transfer[memory_id] = gate.success_rate
                elif memory_id in legacy_transfer:
                    total, successes, _mean_score = legacy_transfer[memory_id]
                    empirical_transfer[memory_id] = (
                        successes / total if total > 0 else 0.0
                    )
            contradiction_summary = self.evidence_lifecycle.contradiction_summary(memory_ids)
            contradiction_severity = {
                memory_id: max_severity
                for memory_id, (_total, max_severity) in contradiction_summary.items()
            }
            generation_id = int(writer.mutable_generation_id)
            for memory_id, node in view.nodes.items():
                empirical = float(empirical_transfer.get(memory_id, 0.0))
                fitness = self.controller.fitness(
                    node,
                    view.scores.get(memory_id),
                    empirical_transfer=empirical,
                )
                gate = gate_summaries.get(memory_id)
                causal = 0.0 if gate is None else float(gate.mean_causal_gain)
                contradiction = float(contradiction_severity.get(memory_id, 0.0))
                harm = causal < 0.0 or contradiction >= self.controller.policy.replay_contradiction_severity
                utility = max(0.0, min(1.0, 0.70 * fitness + 0.30 * max(0.0, causal)))
                window = self.evidence_lifecycle.update_lifecycle_window(
                    memory_id,
                    generation_id=generation_id,
                    utility=utility,
                    harm=harm,
                    low_threshold=self.controller.policy.retain_threshold,
                    positive_threshold=self.controller.policy.promote_threshold,
                )
                low_windows = int(window.consecutive_low_windows)
                harm_windows = int(window.consecutive_harm_windows)
                if self._has_live_dependents(writer, view, memory_id):
                    low_windows = min(
                        low_windows,
                        self.controller.policy.retire_after_low_windows - 1,
                    )
                    harm_windows = min(
                        harm_windows,
                        self.controller.policy.retire_after_harm_windows - 1,
                    )
                lifecycle_windows[memory_id] = (
                    low_windows,
                    harm_windows,
                    int(window.consecutive_positive_windows),
                )
        decisions = self.controller.apply(
            view,
            writer=writer,
            empirical_transfer=empirical_transfer,
            contradiction_severity=contradiction_severity,
            lifecycle_windows=(
                lifecycle_windows if self.evidence_lifecycle is not None else None
            ),
        )
        records: list[EvidenceRecord] = []
        generation_id = int(writer.mutable_generation_id)
        registry = getattr(writer, "_canonical_registry", None)
        for decision in decisions:
            window = lifecycle_windows.get(decision.memory_id, (0, 0, 0))
            common = {
                "fitness": decision.fitness,
                "previous_flags": decision.previous_flags,
                "next_flags": decision.next_flags,
                "empirical_transfer": decision.empirical_transfer,
                "contradiction_severity": decision.contradiction_severity,
                "previous_cognitive_state": decision.previous_cognitive_state,
                "next_cognitive_state": decision.next_cognitive_state,
                "low_windows": int(window[0]),
                "harm_windows": int(window[1]),
                "positive_windows": int(window[2]),
            }
            if decision.promote:
                records.append(
                    EvidenceRecord(
                        decision.memory_id,
                        EVIDENCE_PROMOTION,
                        generation_id,
                        common,
                    )
                )
            if decision.demote:
                records.append(
                    EvidenceRecord(
                        decision.memory_id,
                        EVIDENCE_DEMOTION,
                        generation_id,
                        common,
                    )
                )
            if decision.replay:
                records.append(
                    EvidenceRecord(
                        decision.memory_id,
                        EVIDENCE_REPLAY,
                        generation_id,
                        common,
                    )
                )
            if decision.retired:
                node = view.nodes[decision.memory_id]
                reason = (
                    "persistent_harm"
                    if int(window[1]) >= self.controller.policy.retire_after_harm_windows
                    else "persistent_low_utility"
                )
                records.append(
                    EvidenceRecord(
                        decision.memory_id,
                        EVIDENCE_RETIREMENT,
                        generation_id,
                        {**common, "reason": reason},
                    )
                )
                if self.evidence_lifecycle is not None:
                    canonical_key = None
                    if registry is not None:
                        key = registry.key_for(decision.memory_id)
                        if key is not None:
                            canonical_key = (
                                f"M{int(key.level)}:{int(key.type_id)}:"
                                + ",".join(str(int(value)) for value in key.parts)
                            )
                    self.evidence_lifecycle.append_tombstone(
                        MemoryTombstoneRecord(
                            memory_id=decision.memory_id,
                            level_id=int(node.level),
                            type_id=int(node.type_id),
                            retired_generation=generation_id,
                            reason=reason,
                            canonical_key=canonical_key,
                            provenance_pointer=f"memory:{int(decision.memory_id)}",
                        )
                    )
        written = (
            self.evidence_store.append_evidence_batch(records)
            if self.evidence_store is not None
            else 0
        )
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
