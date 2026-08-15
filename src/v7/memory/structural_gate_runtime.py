from __future__ import annotations

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
        self._last_evidence_id = 0
        self._known_family_ids: set[int] = set()
        self._known_carrier_ids: set[int] = set()
        self._carrier_baselines: dict[
            int,
            tuple[dict[int, Counter[int]], dict[int, Counter[int]]],
        ] = {}

    def run(self, writer: CanonicalMemoryWriter) -> StructuralGateStats:
        nodes = getattr(writer, "_nodes")
        registry = getattr(writer, "_canonical_registry")
        candidates = tuple(
            (memory_id, node)
            for memory_id, node in sorted(nodes.items(), key=lambda item: int(item[0]))
            if (
                node.level == MemoryLevel.M2 and int(node.type_id) == TYPE_FAMILY
            )
            or (
                node.level == MemoryLevel.M3 and int(node.type_id) == TYPE_CARRIER
            )
        )
        has_new_candidates = any(
            (
                node.level == MemoryLevel.M2
                and int(memory_id) not in self._known_family_ids
            )
            or (
                node.level == MemoryLevel.M3
                and int(memory_id) not in self._known_carrier_ids
            )
            for memory_id, node in candidates
        )
        new_episodes = self._episodes(after_evidence_id=self._last_evidence_id)
        if has_new_candidates:
            all_episodes = self._episodes()
        else:
            all_episodes = new_episodes
        if not all_episodes and not candidates:
            return StructuralGateStats()
        generation_id = int(writer.mutable_generation_id)
        family_records: list[GateTrialRecord] = []
        carrier_records: list[GateTrialRecord] = []

        for memory_id, node in candidates:
            if node.level == MemoryLevel.M2 and int(node.type_id) == TYPE_FAMILY:
                key = registry.key_for(memory_id)
                if key is None or not key.parts:
                    continue
                candidate_id = int(memory_id)
                first_run = candidate_id not in self._known_family_ids
                self._known_family_ids.add(candidate_id)
                outcome = int(key.parts[0])
                scope = self.lifecycle_store.freeze_candidate_scope(
                    memory_id,
                    int(node.created_generation),
                )
                for row in all_episodes if first_run else new_episodes:
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
                candidate_id = int(memory_id)
                first_run = candidate_id not in self._known_carrier_ids
                self._known_carrier_ids.add(candidate_id)
                carrier = int(key.parts[0])
                scope = self.lifecycle_store.freeze_candidate_scope(
                    memory_id,
                    int(node.created_generation),
                )
                baseline = self._carrier_baselines.get(candidate_id)
                if baseline is None:
                    carrier_counts = defaultdict(Counter)
                    baseline_counts = defaultdict(Counter)
                    baseline = (carrier_counts, baseline_counts)
                    self._carrier_baselines[candidate_id] = baseline
                    formation_rows = all_episodes
                else:
                    carrier_counts, baseline_counts = baseline
                    formation_rows = new_episodes
                for row in formation_rows:
                    if int(row["generation_id"]) > int(scope.candidate_generation):
                        continue
                    action = int(row.get("action_id") or 0)
                    outcome = int(row.get("outcome_signature") or 0)
                    baseline_counts[action][outcome] += 1
                    if int(row.get("carrier_signature") or -1) == carrier:
                        carrier_counts[action][outcome] += 1
                for row in all_episodes if first_run else new_episodes:
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
        if new_episodes:
            self._last_evidence_id = max(
                int(row.get("evidence_id") or 0) for row in new_episodes
            )
        return StructuralGateStats(family_count, carrier_count)

    @staticmethod
    def _target_context(row: dict[str, object]) -> str:
        contexts = tuple(int(value) for value in row.get("context_signatures", ()) or ())
        fallback = int(row.get("context_signature") or 0)
        return str(planning_context(contexts, fallback=fallback))

    def _episodes(self, *, after_evidence_id: int = 0) -> list[dict[str, object]]:
        return self.evidence_store.load_evidence(
            int(EvidenceType.EPISODE),
            after_evidence_id=after_evidence_id,
        )


__all__ = ["StructuralGateRuntime", "StructuralGateStats"]
