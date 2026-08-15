from __future__ import annotations

import math
from collections import defaultdict
from hashlib import blake2b

import v7.developmental_v707 as d
from v7.derivation.scientific import ScientificDerivationKernels, TYPE_STRATEGY
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.models import MemoryScore, NodeMutation, ScoreMutation
from v7.memory.state import CognitiveState, GateId, GateValidationState
from v7.memory.status import memory_cognitive_state, memory_is_active, memory_validation_state

_HARDENED = False


def _sequence_key(actions, contexts) -> int:
    digest = blake2b(digest_size=8)
    digest.update(b"strategy-sequence-v2")
    digest.update(str(tuple(int(v) for v in actions)).encode("ascii"))
    digest.update(str(tuple(int(v) for v in contexts)).encode("ascii"))
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def _promotion_assessment(node, score, *, threshold: float = 0.020):
    score = score or MemoryScore(memory_id=node.memory_id)
    support_prior = 1.0 - math.exp(-max(0, int(node.support_count)) / 3.0)
    isf = d._clamp01(
        0.25 * max(0.0, float(score.significance))
        + 0.20 * max(0.0, float(score.prediction_error))
        + 0.25 * max(0.0, float(score.learning_value))
        + 0.15 * max(0.0, float(score.transfer_prior))
        + 0.15 * max(0.0, float(score.explanatory_potential))
    )
    reach = d._clamp01(max(float(score.explanatory_potential), 0.35 * support_prior))
    transfer = d._clamp01(max(float(score.transfer_prior), 0.25 * support_prior))
    value = float(isf * reach * transfer)
    return d.PromotionAssessment(
        node.memory_id, isf, reach, transfer, value, value >= float(threshold)
    )


def _compression_run(self, *, writer):
    nodes = getattr(writer, "_nodes")
    scores = getattr(writer, "_scores")
    generation = int(writer.mutable_generation_id)
    parent_map = defaultdict(list)
    context_map = defaultdict(set)
    for memory_id, parent_id, source_context in self.lifecycle_store.connection.execute(
        "SELECT memory_id,parent_memory_id,source_context FROM provenance_records"
    ).fetchall():
        if parent_id is not None:
            parent_map[int(memory_id)].append(MemoryId(int(parent_id)))
        if source_context is not None and str(source_context):
            context_map[int(memory_id)].add(str(source_context))
    context_cache: dict[int, set[str]] = {}

    def contexts(memory_id: MemoryId) -> set[str]:
        raw_id = int(memory_id)
        cached = context_cache.get(raw_id)
        if cached is not None:
            return cached
        direct = set(context_map.get(raw_id, ()))
        cached = direct or self._contexts(memory_id)
        context_cache[raw_id] = cached
        return cached

    replacement_rows = {
        int(parent_id): (int(first_generation), int(state))
        for parent_id, first_generation, state in self.lifecycle_store.connection.execute(
            "SELECT parent_memory_id,generation_id,replacement_state "
            "FROM memory_compression_replacements"
        ).fetchall()
    }
    pending_replacements = {
        int(row[0])
        for row in self.lifecycle_store.connection.execute(
            "SELECT DISTINCT replacement_memory_id "
            "FROM memory_compression_replacements WHERE replacement_state<3"
        ).fetchall()
    }
    last_generation = int(getattr(self, "_last_generation", -1))
    decisions = []
    mutations = []
    upsert_rows = []
    tombstones = []
    for replacement_id, replacement in sorted(nodes.items(), key=lambda item: int(item[0])):
        if (
            last_generation >= 0
            and int(replacement.updated_generation) <= last_generation
            and int(replacement_id) not in pending_replacements
        ):
            continue
        if int(replacement.level) < int(MemoryLevel.M2) or not memory_is_active(replacement):
            continue
        validation = memory_validation_state(replacement)
        if int(getattr(replacement, "gate_id", GateId.NONE)) != int(GateId.NONE):
            if validation not in {GateValidationState.VALIDATED, GateValidationState.TRUSTED}:
                continue
        for parent_id in parent_map.get(int(replacement_id), ()):
            parent = nodes.get(parent_id)
            if parent is None or int(parent.level) >= int(replacement.level):
                continue
            parent_contexts = contexts(parent_id)
            replacement_contexts = contexts(replacement_id)
            unique = (
                0.0
                if not parent_contexts
                else len(parent_contexts - replacement_contexts)
                / max(1, len(parent_contexts))
            )
            if unique > self.unique_coverage_tolerance:
                continue
            parent_score = scores.get(parent_id, MemoryScore(parent_id))
            replacement_score = scores.get(replacement_id, MemoryScore(replacement_id))
            parent_utility = max(
                max(0.0, float(parent_score.significance)),
                max(0.0, float(parent_score.learning_value)),
                max(0.0, float(parent_score.transfer_prior)),
                max(0.0, float(parent_score.explanatory_potential)),
                abs(float(parent_score.future_option_delta)),
            )
            replacement_utility = max(
                max(0.0, float(replacement_score.significance)),
                max(0.0, float(replacement_score.learning_value)),
                max(0.0, float(replacement_score.transfer_prior)),
                max(0.0, float(replacement_score.explanatory_potential)),
                abs(float(replacement_score.future_option_delta)),
            )
            if memory_is_active(parent) and parent_utility > 0.25:
                if (
                    validation != GateValidationState.TRUSTED
                    or replacement_utility + 0.05 < parent_utility
                ):
                    continue
            existing = replacement_rows.get(int(parent_id))
            current = memory_cognitive_state(parent) or CognitiveState.ACTIVE
            if existing is None:
                first_generation = generation
                phase = 1
                next_state = CognitiveState.PROBE_ONLY
            else:
                first_generation, phase = int(existing[0]), int(existing[1])
                age = max(0, generation - first_generation)
                if age >= 2:
                    phase = 3
                    next_state = CognitiveState.RETIRED
                elif age >= 1:
                    phase = 2
                    next_state = CognitiveState.QUARANTINED
                else:
                    phase = max(1, phase)
                    next_state = CognitiveState.PROBE_ONLY
            if current != next_state:
                mutations.append(
                    NodeMutation(
                        parent_id,
                        parent.level,
                        parent.type_id,
                        cognitive_state=int(next_state),
                    )
                )
            replacement_rows[int(parent_id)] = (int(first_generation), int(phase))
            upsert_rows.append(
                (
                    int(parent_id), int(replacement_id), int(first_generation), float(unique),
                    int(phase), "redundant_under_validated_abstraction",
                )
            )
            if next_state == CognitiveState.RETIRED:
                from v7.memory.evidence_lifecycle import MemoryTombstoneRecord
                tombstones.append(
                    MemoryTombstoneRecord(
                        memory_id=parent_id,
                        level_id=int(parent.level),
                        type_id=int(parent.type_id),
                        retired_generation=generation,
                        reason="compressed_redundant_parent",
                        replacement_memory_id=replacement_id,
                        provenance_pointer=f"memory:{int(parent_id)}",
                    )
                )
            decisions.append(d.CompressionDecision(parent_id, replacement_id, unique, next_state))
    if mutations:
        writer.apply_mutation_batch(mutations)
    if upsert_rows:
        with self.lifecycle_store.connection:
            self.lifecycle_store.connection.executemany(
                    "INSERT INTO memory_compression_replacements("
                    "parent_memory_id,replacement_memory_id,generation_id,unique_coverage_score,"
                    "replacement_state,provenance_only,reason) VALUES (?,?,?,?,?,1,?) "
                    "ON CONFLICT(parent_memory_id) DO UPDATE SET "
                    "replacement_memory_id=excluded.replacement_memory_id,"
                    "unique_coverage_score=excluded.unique_coverage_score,"
                    "replacement_state=excluded.replacement_state,provenance_only=1",
                    upsert_rows,
                )
    if tombstones:
        self.lifecycle_store.append_tombstones(tombstones)
    self._last_generation = generation
    return tuple(decisions)


def _canonical_carrier_map(self) -> dict[int, int]:
    rows = self.lifecycle_store.connection.execute(
        "SELECT carrier_a,carrier_b FROM carrier_persistence_links "
        "WHERE support_count>=2"
    ).fetchall()
    parents: dict[int, int] = {}

    def find(value: int) -> int:
        parents.setdefault(value, value)
        root = value
        while parents[root] != root:
            root = parents[root]
        while parents[value] != value:
            parent = parents[value]
            parents[value] = root
            value = parent
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for raw_left, raw_right in rows:
        union(int(raw_left), int(raw_right))
    groups = defaultdict(list)
    for value in parents:
        groups[find(value)].append(value)
    mapping: dict[int, int] = {}
    for values in groups.values():
        ordered = tuple(sorted(values))
        digest = blake2b(digest_size=8)
        digest.update(b"persistent-carrier-v707")
        digest.update(str(ordered).encode("ascii"))
        canonical = int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)
        mapping.update((value, canonical) for value in ordered)
    return mapping


def _canonical_carrier_signature(self, raw_signature: int) -> int:
    value = int(raw_signature)
    return self.canonical_carrier_map().get(value, value)


def _efficiency_run(self, *, writer):
    rows = self.evidence_store.load_evidence(
        int(d.EvidenceType.TRAJECTORY),
        after_evidence_id=int(getattr(self, "_last_evidence_id", 0)),
    )
    generation = int(writer.mutable_generation_id)
    metrics = []
    trial_rows = []
    for payload in rows:
        game = payload.get("source_game")
        context = payload.get("source_context")
        actions = tuple(int(v) for v in payload.get("action_sequence", ()) or ())
        contexts = tuple(int(v) for v in payload.get("context_sequence", ()) or ())
        if not actions:
            continue
        metric = d.trajectory_efficiency(
            actions=actions,
            contexts=contexts,
            future_option_sum=float(payload.get("future_option_sum") or 0.0),
            raw_action_option_sum=float(payload.get("raw_action_option_sum") or 0.0),
        )
        metrics.append((game, str(payload.get("level_key") or context or ""), metric))
        trial_rows.append(
            (
                generation, game, str(payload.get("level_key") or context or ""),
                int(metric.action_signature), float(metric.outcome_quality),
                float(metric.interaction_cost), float(metric.efficiency),
                int(metric.equivalent_group),
            )
        )
    if trial_rows:
        with self.lifecycle_store.connection:
            self.lifecycle_store.connection.executemany(
                "INSERT OR IGNORE INTO trajectory_efficiency_trials("
                "generation_id,source_game,level_key,action_signature,outcome_quality,"
                "interaction_cost,efficiency,equivalent_group) VALUES (?,?,?,?,?,?,?,?)",
                trial_rows,
            )
    if not metrics:
        if rows:
            self._last_evidence_id = max(
                int(row.get("evidence_id") or 0) for row in rows
            )
        return ()
    by_group = defaultdict(list)
    for _game, _level, metric in metrics:
        by_group[metric.equivalent_group].append(metric)
    for group, group_metrics in by_group.items():
        max_quality = max(
            float(self._max_quality_by_group.get(group, float("-inf"))),
            max(item.outcome_quality for item in group_metrics),
        )
        self._max_quality_by_group[group] = max_quality
        for item in group_metrics:
            if item.outcome_quality < max_quality - 0.50:
                continue
            value = d._clamp01(0.5 + 0.5 * math.tanh(item.efficiency))
            self._best_by_signature[item.action_signature] = max(
                self._best_by_signature.get(item.action_signature, 0.0), value
            )
    best_by_signature = self._best_by_signature
    registry = getattr(writer, "_canonical_registry")
    nodes = getattr(writer, "_nodes")
    active_models = tuple(
        sorted(
            (mid for mid, node in nodes.items() if node.level == MemoryLevel.M5 and memory_is_active(node)),
            key=int,
        )
    )[:64]
    if active_models:
        existing = {
            int(key.parts[-1])
            for mid, node in tuple(nodes.items())
            if node.level == MemoryLevel.M6 and int(node.type_id) == TYPE_STRATEGY
            for key in (registry.key_for(mid),)
            if key is not None and key.parts
        }
        for signature, value in sorted(best_by_signature.items()):
            if signature in existing or value <= 0.50:
                continue
            candidate = ScientificDerivationKernels.m6_strategy(
                world_model_ids=active_models,
                action_signature=int(signature),
                efficiency_gain=float(value),
            )
            writer.apply_canonical_candidate_batch((candidate,))
    mutations = []
    for memory_id, node in tuple(nodes.items()):
        if node.level != MemoryLevel.M6 or int(node.type_id) != TYPE_STRATEGY:
            continue
        key = registry.key_for(memory_id)
        if key is None or not key.parts:
            continue
        value = best_by_signature.get(int(key.parts[-1]))
        if value is None:
            continue
        current = getattr(writer, "_scores").get(memory_id, MemoryScore(memory_id))
        mutations.append(
            ScoreMutation(
                memory_id=memory_id,
                significance=max(float(current.significance), value),
                learning_value=max(float(current.learning_value), value),
                future_option_delta=max(float(current.future_option_delta), 2.0 * value - 1.0),
            )
        )
    if mutations:
        writer.apply_score_batch(mutations)
    self._last_evidence_id = max(
        int(row.get("evidence_id") or 0) for row in rows
    )
    return tuple(metric for _game, _level, metric in metrics)


def harden_v707_extensions() -> None:
    global _HARDENED
    if _HARDENED:
        return
    _HARDENED = True
    from v7.derivation.online_runtime import OnlineHierarchyBuilder
    from v7.memory.lifecycle import MemoryLifecycleController

    d.PROMOTION_THRESHOLD = 0.020
    d.promotion_assessment = _promotion_assessment
    d._sequence_key = _sequence_key
    d.MemoryCompressionRuntime.run = _compression_run
    d.CarrierPersistenceRuntime.canonical_carrier_map = _canonical_carrier_map
    d.CarrierPersistenceRuntime.canonical_carrier_signature = _canonical_carrier_signature
    d.TrajectoryEfficiencyRuntime.run = _efficiency_run

    current_load = OnlineHierarchyBuilder._load

    def _carrier_aware_load(self, evidence_type):
        rows = current_load(self, evidence_type)
        if int(evidence_type) == int(d.EvidenceType.EPISODE):
            runtime = getattr(self, "_v707_carrier_runtime", None)
            if runtime is None:
                runtime = d.CarrierPersistenceRuntime(
                    self.lifecycle_store,
                    self.evidence_store,
                )
                self._v707_carrier_runtime = runtime
            carrier_map = runtime.canonical_carrier_map()
            for row in rows:
                raw = row.get("carrier_signature")
                if raw is not None:
                    value = int(raw)
                    row["carrier_signature"] = carrier_map.get(value, value)
        return rows

    OnlineHierarchyBuilder._load = _carrier_aware_load

    original = getattr(MemoryLifecycleController, "_v707_original_fitness", None)
    current_fitness = MemoryLifecycleController.fitness
    if original is not None:
        def _stage_only_fitness(self, node, score, *, empirical_transfer=0.0):
            stage = getattr(self, "_v707_stage", None)
            if stage is None:
                return original(self, node, score, empirical_transfer=empirical_transfer)
            return d.developmental_memory_fitness(
                node, score, empirical_transfer=empirical_transfer, stage=str(stage)
            )

        MemoryLifecycleController.fitness = _stage_only_fitness
        MemoryLifecycleController._v707_hardened_prior_fitness = current_fitness


__all__ = ["harden_v707_extensions"]
