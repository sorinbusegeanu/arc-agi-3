from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass

from v7.derivation.scientific import TYPE_CARRIER, TYPE_FAMILY
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore, GateTrialRecord
from v7.memory.evidence_store import EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.planning import planning_context
from v7.memory.state import GateId
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class StructuralGateStats:
    family_trials: int = 0
    carrier_trials: int = 0

    @property
    def total(self) -> int:
        return int(self.family_trials) + int(self.carrier_trials)


class StructuralGateRuntime:
    """Create held-out evidence for gates that are not direct action memories.

    G12 is an invariance/compression gate: a family must recur after formation
    in independent contexts/actions. G23C is predictive: the carrier's
    pre-formation outcome distribution must improve on the action-only
    baseline in held-out episodes.
    """

    def __init__(
        self,
        *,
        evidence_store: EvidenceStore,
        lifecycle_store: EvidenceLifecycleStore,
    ) -> None:
        self.evidence_store = evidence_store
        self.lifecycle_store = lifecycle_store

    def run(self, writer: CanonicalMemoryWriter) -> StructuralGateStats:
        episodes = self._episodes()
        if not episodes:
            return StructuralGateStats()
        nodes = getattr(writer, "_nodes")
        registry = getattr(writer, "_canonical_registry")
        generation_id = int(writer.mutable_generation_id)
        family_records: list[GateTrialRecord] = []
        carrier_records: list[GateTrialRecord] = []

        for memory_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
            if node.level == MemoryLevel.M2 and int(node.type_id) == TYPE_FAMILY:
                key = registry.key_for(memory_id)
                if key is None or not key.parts:
                    continue
                outcome = int(key.parts[0])
                scope = self.lifecycle_store.freeze_candidate_scope(
                    memory_id,
                    int(node.created_generation),
                )
                for row in episodes:
                    if int(row["generation_id"]) <= int(scope.candidate_generation):
                        continue
                    if int(row.get("outcome_signature") or 0) != outcome:
                        continue
                    target_context = self._target_context(row)
                    pair_id = (
                        f"g12:{int(memory_id)}:{row.get('source_game')}:"
                        f"{row.get('source_global_step')}:{target_context}"
                    )
                    family_records.append(
                        GateTrialRecord(
                            memory_id=memory_id,
                            generation_id=generation_id,
                            gate_id=GateId.G12,
                            candidate_generation=scope.candidate_generation,
                            target_game=row.get("source_game"),
                            target_context=target_context,
                            participated=True,
                            contribution=0.25,
                            causal_gain=0.25,
                            prediction_gain=0.25,
                            intervention_type="heldout_family_invariance",
                            paired_trial_id=pair_id,
                            payload={
                                "action_id": int(row.get("action_id") or 0),
                                "outcome_signature": outcome,
                                "evidence_generation": int(row["generation_id"]),
                            },
                        )
                    )

            elif node.level == MemoryLevel.M3 and int(node.type_id) == TYPE_CARRIER:
                key = registry.key_for(memory_id)
                if key is None or not key.parts:
                    continue
                carrier = int(key.parts[0])
                scope = self.lifecycle_store.freeze_candidate_scope(
                    memory_id,
                    int(node.created_generation),
                )
                formation = [
                    row
                    for row in episodes
                    if int(row["generation_id"]) <= int(scope.candidate_generation)
                ]
                carrier_counts: dict[int, Counter[int]] = defaultdict(Counter)
                baseline_counts: dict[int, Counter[int]] = defaultdict(Counter)
                for row in formation:
                    action = int(row.get("action_id") or 0)
                    outcome = int(row.get("outcome_signature") or 0)
                    baseline_counts[action][outcome] += 1
                    if int(row.get("carrier_signature") or -1) == carrier:
                        carrier_counts[action][outcome] += 1
                for row in episodes:
                    if int(row["generation_id"]) <= int(scope.candidate_generation):
                        continue
                    if int(row.get("carrier_signature") or -1) != carrier:
                        continue
                    action = int(row.get("action_id") or 0)
                    outcome = int(row.get("outcome_signature") or 0)
                    local = carrier_counts.get(action, Counter())
                    baseline = baseline_counts.get(action, Counter())
                    local_total = sum(local.values())
                    baseline_total = sum(baseline.values())
                    if local_total <= 0 or baseline_total <= 0:
                        continue
                    carrier_probability = local[outcome] / local_total
                    baseline_probability = baseline[outcome] / baseline_total
                    lift = float(carrier_probability - baseline_probability)
                    if lift == 0.0:
                        continue
                    target_context = self._target_context(row)
                    pair_id = (
                        f"g23c:{int(memory_id)}:{row.get('source_game')}:"
                        f"{row.get('source_global_step')}:{target_context}"
                    )
                    carrier_records.append(
                        GateTrialRecord(
                            memory_id=memory_id,
                            generation_id=generation_id,
                            gate_id=GateId.G23C,
                            candidate_generation=scope.candidate_generation,
                            target_game=row.get("source_game"),
                            target_context=target_context,
                            participated=True,
                            contribution=lift,
                            causal_gain=lift,
                            prediction_gain=lift,
                            intervention_type="heldout_carrier_prediction_ablation",
                            paired_trial_id=pair_id,
                            payload={
                                "action_id": action,
                                "outcome_signature": outcome,
                                "carrier_probability": carrier_probability,
                                "baseline_probability": baseline_probability,
                                "evidence_generation": int(row["generation_id"]),
                            },
                        )
                    )

        family_count = self.lifecycle_store.append_gate_trials(family_records)
        carrier_count = self.lifecycle_store.append_gate_trials(carrier_records)
        return StructuralGateStats(family_count, carrier_count)

    @staticmethod
    def _target_context(row: dict[str, object]) -> str:
        contexts = tuple(int(value) for value in row.get("context_signatures", ()) or ())
        fallback = int(row.get("context_signature") or 0)
        return str(planning_context(contexts, fallback=fallback))

    def _episodes(self) -> list[dict[str, object]]:
        rows = self.evidence_store.connection.execute(
            "SELECT source_game,source_context,source_global_step,payload_json,generation_id "
            "FROM evidence_records WHERE evidence_type=? ORDER BY evidence_id",
            (int(EvidenceType.EPISODE),),
        ).fetchall()
        result: list[dict[str, object]] = []
        for game, context, step, payload_json, generation in rows:
            try:
                payload = json.loads(str(payload_json or "{}"))
            except (TypeError, json.JSONDecodeError):
                payload = {}
            payload.update(
                {
                    "source_game": game,
                    "source_context": context,
                    "source_global_step": step,
                    "generation_id": int(generation),
                }
            )
            result.append(payload)
        return result


__all__ = ["StructuralGateRuntime", "StructuralGateStats"]
