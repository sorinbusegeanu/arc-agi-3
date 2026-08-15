from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import blake2b
from typing import Iterable, Mapping

from v7.derivation.scientific import TYPE_CONTINGENCY, TYPE_STRATEGY
from v7.memory.canonical import CanonicalCandidateMutation, CanonicalMemoryKey
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore, MemoryTombstoneRecord
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import ContingencyIndexMutation
from v7.memory.models import MemoryNode, MemoryScore, NodeMutation, ScoreMutation
from v7.memory.state import CognitiveState, GateId, GateValidationState, gate_for_identity, is_gate_validated
from v7.memory.status import memory_cognitive_state, memory_is_active, memory_validation_state


# ---------------------------------------------------------------------------
# Learned developmental outcome: environment terminal labels are diagnostics,
# never primitive developmental meaning.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DevelopmentalOutcome:
    future_option_delta: float
    continuation_delta: float
    transition_reversibility: float
    learned_significance: float
    developmental_polarity: int
    environment_terminal_type: str | None = None


def _signed_unit(value: float) -> float:
    return math.tanh(float(value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def infer_developmental_outcome(
    *,
    future_option_delta: float,
    continuation_delta: float = 0.0,
    transition_reversibility: float = 0.0,
    environment_terminal_type: str | None = None,
) -> DevelopmentalOutcome:
    """Infer significance only from interaction consequences.

    ``environment_terminal_type`` is intentionally excluded from the score. It
    remains available solely for benchmark/report diagnostics.
    """
    future = _signed_unit(float(future_option_delta))
    continuation = _signed_unit(float(continuation_delta))
    reversibility = _signed_unit(float(transition_reversibility))
    signed = 0.65 * future + 0.25 * continuation + 0.10 * reversibility
    significance = _clamp01(0.5 + 0.5 * signed)
    polarity = 1 if signed > 0.05 else -1 if signed < -0.05 else 0
    return DevelopmentalOutcome(
        future_option_delta=float(future_option_delta),
        continuation_delta=float(continuation_delta),
        transition_reversibility=float(transition_reversibility),
        learned_significance=significance,
        developmental_polarity=polarity,
        environment_terminal_type=environment_terminal_type,
    )


def developmental_outcome_for_evidence(evidence) -> DevelopmentalOutcome:
    raw_delta = float(getattr(evidence, "raw_action_option_delta", 0.0) or 0.0)
    observable = bool(getattr(evidence, "future_option_observable", True))
    continuation_delta = 0.0 if observable else -1.0
    # A zero raw delta on an ordinary transition means preservation, not harm.
    return infer_developmental_outcome(
        future_option_delta=raw_delta,
        continuation_delta=continuation_delta,
        environment_terminal_type=str(
            getattr(evidence, "terminal_polarity", 0)
        ),
    )


# ---------------------------------------------------------------------------
# Paper candidate promotion: ISF × explanatory reach × prospective transfer.
# It allocates testing only; it can never establish scientific validation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromotionAssessment:
    memory_id: MemoryId
    interaction_significance: float
    explanatory_reach: float
    transfer_prior: float
    score: float
    probe_worthy: bool


PROMOTION_THRESHOLD = 0.025


def promotion_assessment(
    node: MemoryNode,
    score: MemoryScore | None,
    *,
    threshold: float = PROMOTION_THRESHOLD,
) -> PromotionAssessment:
    score = score or MemoryScore(memory_id=node.memory_id)
    support_prior = 1.0 - math.exp(-max(0, int(node.support_count)) / 3.0)
    # ISF is an operational bounded estimate using only evidence available at
    # formation/testing time. No TP_emp term is used here.
    isf = _clamp01(
        0.25 * max(0.0, float(score.significance))
        + 0.20 * max(0.0, float(score.prediction_error))
        + 0.25 * max(0.0, float(score.learning_value))
        + 0.15 * max(0.0, float(score.transfer_prior))
        + 0.15 * max(0.0, float(score.explanatory_potential))
    )
    # Lower levels naturally begin before explicit explanatory estimates exist;
    # recurrence supplies only a bounded prospective reach prior, not validation.
    reach = _clamp01(
        max(float(score.explanatory_potential), 0.35 * support_prior)
    )
    transfer = _clamp01(max(float(score.transfer_prior), 0.25 * support_prior))
    value = float(isf * reach * transfer)
    return PromotionAssessment(
        memory_id=node.memory_id,
        interaction_significance=isf,
        explanatory_reach=reach,
        transfer_prior=transfer,
        score=value,
        probe_worthy=value >= float(threshold),
    )


# ---------------------------------------------------------------------------
# Developmental W(t): scientific validation remains invariant; only attention,
# replay and memory residency use these stage-dependent weights.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DevelopmentalWeights:
    significance: float
    prediction_error: float
    learning_value: float
    transfer_prior: float
    empirical_transfer: float
    explanatory: float
    future_option: float
    support: float


_STAGE_WEIGHTS: Mapping[str, DevelopmentalWeights] = {
    "CONTROL": DevelopmentalWeights(.23, .27, .27, .04, .02, .07, .08, .02),
    "CONTINGENCY": DevelopmentalWeights(.16, .24, .27, .08, .03, .08, .11, .03),
    "ABSTRACTION": DevelopmentalWeights(.11, .13, .18, .17, .05, .21, .12, .03),
    "TRANSFER": DevelopmentalWeights(.08, .08, .12, .18, .16, .22, .13, .03),
    "PLANNING": DevelopmentalWeights(.07, .07, .10, .12, .17, .18, .25, .04),
    "STRATEGY": DevelopmentalWeights(.05, .05, .08, .10, .18, .17, .30, .07),
}


def developmental_weights(stage: str) -> DevelopmentalWeights:
    return _STAGE_WEIGHTS.get(str(stage).upper(), _STAGE_WEIGHTS["CONTROL"])


def developmental_memory_fitness(
    node: MemoryNode,
    score: MemoryScore | None,
    *,
    empirical_transfer: float = 0.0,
    stage: str = "CONTROL",
) -> float:
    score = score or MemoryScore(memory_id=node.memory_id)
    w = developmental_weights(stage)
    support = min(1.0, math.log1p(max(0, int(node.support_count))) / math.log1p(8.0))
    future = _signed_unit(float(score.future_option_delta))
    value = (
        w.significance * max(0.0, float(score.significance))
        + w.prediction_error * max(0.0, float(score.prediction_error))
        + w.learning_value * max(0.0, float(score.learning_value))
        + w.transfer_prior * max(0.0, float(score.transfer_prior))
        + w.empirical_transfer * _clamp01(empirical_transfer)
        + w.explanatory * max(0.0, float(score.explanatory_potential))
        + w.future_option * future
        + w.support * support
    )
    return max(0.0, float(value))


# ---------------------------------------------------------------------------
# Persistent v7.0.7 evidence tables. They are deliberately isolated from the
# canonical node store and can be reconstructed without changing MemoryId.
# ---------------------------------------------------------------------------


_V707_DDL = """
CREATE TABLE IF NOT EXISTS context_refinement_trials (
    refinement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_memory_id INTEGER NOT NULL,
    child_memory_id INTEGER,
    generation_id INTEGER NOT NULL,
    candidate_context_signature INTEGER NOT NULL,
    partition_feature TEXT NOT NULL,
    positive_support INTEGER NOT NULL,
    contradiction_support INTEGER NOT NULL,
    baseline_accuracy REAL NOT NULL,
    refined_accuracy REAL NOT NULL,
    prediction_gain REAL NOT NULL,
    causal_gain REAL NOT NULL,
    accepted INTEGER NOT NULL CHECK(accepted IN (0,1)),
    UNIQUE(parent_memory_id, candidate_context_signature, generation_id)
);
CREATE INDEX IF NOT EXISTS idx_context_refinement_parent
ON context_refinement_trials(parent_memory_id, generation_id, accepted);

CREATE TABLE IF NOT EXISTS memory_compression_replacements (
    parent_memory_id INTEGER PRIMARY KEY,
    replacement_memory_id INTEGER NOT NULL,
    generation_id INTEGER NOT NULL,
    unique_coverage_score REAL NOT NULL,
    replacement_state INTEGER NOT NULL DEFAULT 1,
    provenance_only INTEGER NOT NULL DEFAULT 1 CHECK(provenance_only IN (0,1)),
    reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_compression_replacement
ON memory_compression_replacements(replacement_memory_id, provenance_only);

CREATE TABLE IF NOT EXISTS trajectory_efficiency_trials (
    efficiency_trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL,
    source_game TEXT,
    level_key TEXT,
    action_signature INTEGER NOT NULL,
    outcome_quality REAL NOT NULL,
    interaction_cost REAL NOT NULL,
    efficiency REAL NOT NULL,
    equivalent_group INTEGER NOT NULL,
    UNIQUE(generation_id, source_game, level_key, action_signature)
);

CREATE TABLE IF NOT EXISTS carrier_persistence_links (
    carrier_a INTEGER NOT NULL,
    carrier_b INTEGER NOT NULL,
    support_count INTEGER NOT NULL DEFAULT 1,
    first_generation INTEGER NOT NULL,
    last_generation INTEGER NOT NULL,
    predictive_gain REAL NOT NULL DEFAULT 0,
    PRIMARY KEY(carrier_a, carrier_b)
);
"""


def ensure_v707_schema(store: EvidenceLifecycleStore) -> None:
    with store.connection:
        store.connection.executescript(_V707_DDL)


# ---------------------------------------------------------------------------
# Contradiction-driven context expansion.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContextRefinementDecision:
    parent_memory_id: MemoryId
    candidate_context_signature: int
    baseline_accuracy: float
    refined_accuracy: float
    prediction_gain: float
    positive_support: int
    contradiction_support: int
    accepted: bool
    child_memory_id: MemoryId | None = None


class ContextRefinementRuntime:
    def __init__(
        self,
        lifecycle_store: EvidenceLifecycleStore,
        evidence_store,
        *,
        minimum_contradictions: int = 2,
        minimum_partition_support: int = 2,
        minimum_prediction_gain: float = 0.10,
    ) -> None:
        self.lifecycle_store = lifecycle_store
        self.evidence_store = evidence_store
        self.minimum_contradictions = max(1, int(minimum_contradictions))
        self.minimum_partition_support = max(2, int(minimum_partition_support))
        self.minimum_prediction_gain = max(0.0, float(minimum_prediction_gain))
        self._last_contradiction_id = 0
        ensure_v707_schema(lifecycle_store)

    def _m1_ancestors(self, memory_id: MemoryId, nodes) -> tuple[MemoryId, ...]:
        result: set[MemoryId] = set()
        frontier = [memory_id]
        seen: set[MemoryId] = set()
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            node = nodes.get(current)
            if node is not None and node.level == MemoryLevel.M1:
                result.add(current)
                continue
            frontier.extend(self.lifecycle_store.provenance_parents(current))
        return tuple(sorted(result, key=int))

    def _episode_rows(self) -> list[dict[str, object]]:
        return self.evidence_store.load_evidence(int(EvidenceType.EPISODE))

    @staticmethod
    def _partition_gain(rows: list[dict[str, object]], outcome: int, context: int) -> tuple[float, float, int, int]:
        if not rows:
            return 0.0, 0.0, 0, 0
        labels = [int(row.get("outcome_signature") or 0) == int(outcome) for row in rows]
        positives = sum(labels)
        negatives = len(labels) - positives
        baseline = max(positives, negatives) / len(labels)
        inside = []
        outside = []
        for row, label in zip(rows, labels, strict=True):
            contexts = tuple(int(v) for v in row.get("context_signatures", ()) or ())
            (inside if int(context) in contexts else outside).append(label)
        if len(inside) < 2 or not outside:
            return baseline, baseline, positives, negatives
        correct = 0
        for group in (inside, outside):
            p = sum(group)
            correct += max(p, len(group) - p)
        refined = correct / len(labels)
        return baseline, refined, positives, negatives

    def run(self, view, *, writer) -> tuple[ContextRefinementDecision, ...]:
        nodes = getattr(writer, "_nodes")
        registry = getattr(writer, "_canonical_registry")
        new_contradictions = self.lifecycle_store.connection.execute(
            "SELECT contradiction_id,memory_id FROM contradiction_records "
            "WHERE severity>0 AND contradiction_id>? ORDER BY contradiction_id",
            (int(self._last_contradiction_id),),
        ).fetchall()
        if not new_contradictions:
            return ()
        next_watermark = int(new_contradictions[-1][0])
        driver_ids = tuple(
            sorted({int(memory_id) for _row_id, memory_id in new_contradictions})
        )
        contradiction_rows: list[tuple[int, int]] = []
        for ids in self.lifecycle_store._memory_id_chunks(
            MemoryId(memory_id) for memory_id in driver_ids
        ):
            placeholders = ",".join("?" for _ in ids)
            contradiction_rows.extend(
                self.lifecycle_store.connection.execute(
                    "SELECT memory_id,COUNT(*) FROM contradiction_records "
                    f"WHERE severity>0 AND memory_id IN ({placeholders}) "
                    "GROUP BY memory_id HAVING COUNT(*)>=?",
                    (*ids, self.minimum_contradictions),
                ).fetchall()
            )
        if not contradiction_rows:
            self._last_contradiction_id = next_watermark
            return ()
        episodes = self._episode_rows()
        generation = int(writer.mutable_generation_id)
        decisions: list[ContextRefinementDecision] = []
        for raw_driver, _count in contradiction_rows:
            driver_id = MemoryId(int(raw_driver))
            for parent_id in self._m1_ancestors(driver_id, nodes):
                parent = nodes.get(parent_id)
                key = registry.key_for(parent_id)
                if parent is None or key is None or len(key.parts) < 3:
                    continue
                original_context, action, outcome = map(int, key.parts[:3])
                relevant = [
                    row for row in episodes
                    if int(row.get("action_id") or 0) == action
                ]
                if len(relevant) < self.minimum_partition_support * 2:
                    continue
                candidates = Counter(
                    int(context)
                    for row in relevant
                    for context in (row.get("context_signatures", ()) or ())
                    if int(context) != original_context
                )
                best = None
                for context, support in candidates.items():
                    if support < self.minimum_partition_support:
                        continue
                    baseline, refined, positives, negatives = self._partition_gain(
                        relevant, outcome, context
                    )
                    gain = refined - baseline
                    candidate = (gain, support, -context, context, baseline, refined, positives, negatives)
                    if best is None or candidate > best:
                        best = candidate
                if best is None:
                    continue
                gain, support, _neg_context, context, baseline, refined, positives, negatives = best
                accepted = (
                    gain >= self.minimum_prediction_gain
                    and positives > 0
                    and negatives > 0
                    and support >= self.minimum_partition_support
                )
                child_id = None
                if accepted:
                    child_key = CanonicalMemoryKey(
                        MemoryLevel.M1,
                        TYPE_CONTINGENCY,
                        (int(context), action, outcome),
                    )
                    mutation = CanonicalCandidateMutation(
                        key=child_key,
                        support_delta=max(1, int(support)),
                        significance=max(0.0, min(1.0, refined)),
                        learning_value=max(0.0, min(1.0, gain)),
                        transfer_prior=max(0.0, min(1.0, support / max(1, len(relevant)))),
                        explanatory_potential=max(0.0, min(1.0, refined)),
                    )
                    child_id = writer.apply_canonical_candidate_batch((mutation,))[child_key]
                    writer.apply_contingency_index_batch(
                        (ContingencyIndexMutation(int(context), action, child_id),)
                    )
                    current_parent = nodes.get(parent_id)
                    if current_parent is not None and memory_is_active(current_parent):
                        writer.apply_mutation_batch(
                            (
                                NodeMutation(
                                    parent_id,
                                    current_parent.level,
                                    current_parent.type_id,
                                    cognitive_state=int(CognitiveState.PROBE_ONLY),
                                ),
                            )
                        )
                decision = ContextRefinementDecision(
                    parent_id,
                    int(context),
                    float(baseline),
                    float(refined),
                    float(gain),
                    int(positives),
                    int(negatives),
                    bool(accepted),
                    child_id,
                )
                decisions.append(decision)
                with self.lifecycle_store.connection:
                    self.lifecycle_store.connection.execute(
                        "INSERT OR IGNORE INTO context_refinement_trials("
                        "parent_memory_id,child_memory_id,generation_id,candidate_context_signature,"
                        "partition_feature,positive_support,contradiction_support,baseline_accuracy,"
                        "refined_accuracy,prediction_gain,causal_gain,accepted) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            int(parent_id),
                            None if child_id is None else int(child_id),
                            generation,
                            int(context),
                            "context_membership",
                            int(positives),
                            int(negatives),
                            float(baseline),
                            float(refined),
                            float(gain),
                            float(gain),
                            1 if accepted else 0,
                        ),
                    )
        self._last_contradiction_id = next_watermark
        return tuple(decisions)


# ---------------------------------------------------------------------------
# Compression/replacement: validated higher abstractions may make lower payload
# cognitively redundant while preserving durable provenance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompressionDecision:
    parent_memory_id: MemoryId
    replacement_memory_id: MemoryId
    unique_coverage_score: float
    next_cognitive_state: CognitiveState


class MemoryCompressionRuntime:
    def __init__(
        self,
        lifecycle_store: EvidenceLifecycleStore,
        *,
        unique_coverage_tolerance: float = 0.10,
    ) -> None:
        self.lifecycle_store = lifecycle_store
        self.unique_coverage_tolerance = max(0.0, float(unique_coverage_tolerance))
        self._last_generation = -1
        ensure_v707_schema(lifecycle_store)

    def _contexts(self, memory_id: MemoryId) -> set[str]:
        rows = self.lifecycle_store.connection.execute(
            "SELECT DISTINCT source_context FROM provenance_records WHERE memory_id=? "
            "AND source_context IS NOT NULL AND source_context<>''",
            (int(memory_id),),
        ).fetchall()
        direct = {str(row[0]) for row in rows}
        if direct:
            return direct
        return set(self.lifecycle_store.provenance_source_contexts_at(memory_id, 1 << 60))

    def _unique_coverage(self, parent_id: MemoryId, replacement_id: MemoryId) -> float:
        parent = self._contexts(parent_id)
        if not parent:
            return 0.0
        replacement = self._contexts(replacement_id)
        return len(parent - replacement) / max(1, len(parent))

    def is_provenance_only(self, parent_id: MemoryId, replacement_id: MemoryId | None = None) -> bool:
        if replacement_id is None:
            row = self.lifecycle_store.connection.execute(
                "SELECT 1 FROM memory_compression_replacements WHERE parent_memory_id=? AND provenance_only=1",
                (int(parent_id),),
            ).fetchone()
        else:
            row = self.lifecycle_store.connection.execute(
                "SELECT 1 FROM memory_compression_replacements WHERE parent_memory_id=? "
                "AND replacement_memory_id=? AND provenance_only=1",
                (int(parent_id), int(replacement_id)),
            ).fetchone()
        return row is not None

    def run(self, *, writer) -> tuple[CompressionDecision, ...]:
        nodes = getattr(writer, "_nodes")
        generation = int(writer.mutable_generation_id)
        decisions: list[CompressionDecision] = []
        for replacement_id, replacement in sorted(nodes.items(), key=lambda item: int(item[0])):
            if int(replacement.level) < int(MemoryLevel.M2) or not memory_is_active(replacement):
                continue
            validation = memory_validation_state(replacement)
            if int(getattr(replacement, "gate_id", GateId.NONE)) != int(GateId.NONE):
                if validation is None or not is_gate_validated(validation):
                    continue
            for parent_id in self.lifecycle_store.provenance_parents(replacement_id):
                parent = nodes.get(parent_id)
                if parent is None or int(parent.level) >= int(replacement.level):
                    continue
                unique = self._unique_coverage(parent_id, replacement_id)
                if unique > self.unique_coverage_tolerance:
                    continue
                existing = self.lifecycle_store.connection.execute(
                    "SELECT generation_id,replacement_state FROM memory_compression_replacements "
                    "WHERE parent_memory_id=?",
                    (int(parent_id),),
                ).fetchone()
                current = memory_cognitive_state(parent) or CognitiveState.ACTIVE
                if existing is None:
                    next_state = CognitiveState.PROBE_ONLY
                    first_generation = generation
                    phase = 1
                else:
                    first_generation, phase = int(existing[0]), int(existing[1])
                    age = max(0, generation - first_generation)
                    if age >= 2:
                        next_state = CognitiveState.RETIRED
                        phase = 3
                    elif age >= 1:
                        next_state = CognitiveState.QUARANTINED
                        phase = 2
                    else:
                        next_state = CognitiveState.PROBE_ONLY
                        phase = max(1, phase)
                if current != next_state:
                    writer.apply_mutation_batch(
                        (
                            NodeMutation(
                                parent_id,
                                parent.level,
                                parent.type_id,
                                cognitive_state=int(next_state),
                            ),
                        )
                    )
                with self.lifecycle_store.connection:
                    self.lifecycle_store.connection.execute(
                        "INSERT INTO memory_compression_replacements("
                        "parent_memory_id,replacement_memory_id,generation_id,unique_coverage_score,"
                        "replacement_state,provenance_only,reason) VALUES (?,?,?,?,?,1,?) "
                        "ON CONFLICT(parent_memory_id) DO UPDATE SET "
                        "replacement_memory_id=excluded.replacement_memory_id,"
                        "unique_coverage_score=excluded.unique_coverage_score,"
                        "replacement_state=excluded.replacement_state,provenance_only=1",
                        (
                            int(parent_id),
                            int(replacement_id),
                            int(first_generation),
                            float(unique),
                            int(phase),
                            "redundant_under_validated_abstraction",
                        ),
                    )
                if next_state == CognitiveState.RETIRED:
                    self.lifecycle_store.append_tombstone(
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
                decisions.append(
                    CompressionDecision(parent_id, replacement_id, unique, next_state)
                )
        return tuple(decisions)


# ---------------------------------------------------------------------------
# M6 efficiency among future-option-equivalent trajectories.
# ---------------------------------------------------------------------------


def _sequence_key(actions: Iterable[int], contexts: Iterable[int]) -> int:
    digest = blake2b(digest_size=8)
    digest.update(b"v707-trajectory-sequence")
    digest.update(str(tuple(int(v) for v in actions)).encode("ascii"))
    digest.update(str(tuple(int(v) for v in contexts)).encode("ascii"))
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class TrajectoryEfficiency:
    action_signature: int
    outcome_quality: float
    interaction_cost: float
    efficiency: float
    equivalent_group: int


def trajectory_efficiency(
    *,
    actions: Iterable[int],
    contexts: Iterable[int],
    future_option_sum: float,
    raw_action_option_sum: float = 0.0,
    risk: float = 0.0,
    uncertainty: float = 0.0,
    outcome_band: float = 0.50,
) -> TrajectoryEfficiency:
    actions_t = tuple(int(v) for v in actions)
    contexts_t = tuple(int(v) for v in contexts)
    steps = max(1, len(actions_t))
    repeated_pairs = steps - len(set(zip(contexts_t[:steps], actions_t, strict=False)))
    repeated_states = max(0, len(contexts_t) - len(set(contexts_t)))
    loop_ratio = repeated_pairs / steps
    repeated_state_ratio = repeated_states / max(1, len(contexts_t))
    cost = (
        float(steps)
        * (1.0 + 0.75 * loop_ratio + 0.50 * repeated_state_ratio)
        + 2.0 * max(0.0, float(risk))
        + 1.5 * max(0.0, float(uncertainty))
    )
    quality = float(future_option_sum) + 0.5 * float(raw_action_option_sum)
    # Future-option destruction is a cost, not an opportunity to look efficient.
    if quality < 0.0:
        cost += abs(quality)
    efficiency = quality / max(1.0, cost)
    band = max(1e-6, float(outcome_band))
    group = int(round(quality / band))
    return TrajectoryEfficiency(
        _sequence_key(actions_t, contexts_t), quality, cost, efficiency, group
    )


class TrajectoryEfficiencyRuntime:
    def __init__(self, lifecycle_store: EvidenceLifecycleStore, evidence_store) -> None:
        self.lifecycle_store = lifecycle_store
        self.evidence_store = evidence_store
        self._last_evidence_id = 0
        self._best_by_signature: dict[int, float] = {}
        self._max_quality_by_group: dict[int, float] = {}
        ensure_v707_schema(lifecycle_store)

    def run(self, *, writer) -> tuple[TrajectoryEfficiency, ...]:
        rows = self.evidence_store.connection.execute(
            "SELECT source_game,source_context,payload_json FROM evidence_records "
            "WHERE evidence_type=? ORDER BY evidence_id",
            (int(EvidenceType.TRAJECTORY),),
        ).fetchall()
        generation = int(writer.mutable_generation_id)
        metrics: list[tuple[str | None, str | None, TrajectoryEfficiency]] = []
        for game, context, payload_json in rows:
            try:
                payload = json.loads(str(payload_json or "{}"))
            except (TypeError, json.JSONDecodeError):
                payload = {}
            actions = tuple(int(v) for v in payload.get("action_sequence", ()) or ())
            contexts = tuple(int(v) for v in payload.get("context_sequence", ()) or ())
            if not actions:
                continue
            metric = trajectory_efficiency(
                actions=actions,
                contexts=contexts,
                future_option_sum=float(payload.get("future_option_sum") or 0.0),
                raw_action_option_sum=float(payload.get("raw_action_option_sum") or 0.0),
            )
            metrics.append((game, str(payload.get("level_key") or context or ""), metric))
            with self.lifecycle_store.connection:
                self.lifecycle_store.connection.execute(
                    "INSERT OR IGNORE INTO trajectory_efficiency_trials("
                    "generation_id,source_game,level_key,action_signature,outcome_quality,"
                    "interaction_cost,efficiency,equivalent_group) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        generation,
                        game,
                        str(payload.get("level_key") or context or ""),
                        int(metric.action_signature),
                        float(metric.outcome_quality),
                        float(metric.interaction_cost),
                        float(metric.efficiency),
                        int(metric.equivalent_group),
                    ),
                )
        if not metrics:
            return ()
        # Rank only inside comparable future-option outcome groups.
        best_by_signature: dict[int, float] = {}
        by_group: dict[int, list[TrajectoryEfficiency]] = defaultdict(list)
        for _game, _level, metric in metrics:
            by_group[metric.equivalent_group].append(metric)
        for group_metrics in by_group.values():
            max_quality = max(item.outcome_quality for item in group_metrics)
            comparable = [
                item for item in group_metrics
                if item.outcome_quality >= max_quality - 0.50
            ]
            for item in comparable:
                normalized = _clamp01(0.5 + 0.5 * math.tanh(item.efficiency))
                best_by_signature[item.action_signature] = max(
                    best_by_signature.get(item.action_signature, 0.0), normalized
                )
        registry = getattr(writer, "_canonical_registry")
        nodes = getattr(writer, "_nodes")
        mutations: list[ScoreMutation] = []
        for memory_id, node in nodes.items():
            if node.level != MemoryLevel.M6 or int(node.type_id) != TYPE_STRATEGY:
                continue
            key = registry.key_for(memory_id)
            if key is None or not key.parts:
                continue
            signature = int(key.parts[-1])
            value = best_by_signature.get(signature)
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
        return tuple(metric for _game, _level, metric in metrics)


# ---------------------------------------------------------------------------
# Carrier persistence across appearance-changing transformations.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CarrierPersistenceLink:
    carrier_a: int
    carrier_b: int
    support_count: int
    predictive_gain: float


class CarrierPersistenceRuntime:
    def __init__(self, lifecycle_store: EvidenceLifecycleStore, evidence_store) -> None:
        self.lifecycle_store = lifecycle_store
        self.evidence_store = evidence_store
        self._last_evidence_id = 0
        self._last_by_game: dict[
            str,
            tuple[int, int, dict[str, object], int],
        ] = {}
        self._counts: Counter[tuple[int, int]] = Counter()
        self._gains: defaultdict[tuple[int, int], float] = defaultdict(float)
        self._first: dict[tuple[int, int], int] = {}
        self._last: dict[tuple[int, int], int] = {}
        ensure_v707_schema(lifecycle_store)

    def run(self, *, writer) -> tuple[CarrierPersistenceLink, ...]:
        rows = self.evidence_store.load_evidence(
            int(EvidenceType.EPISODE),
            after_evidence_id=self._last_evidence_id,
        )
        if not rows:
            return ()
        by_game: dict[
            str,
            list[tuple[int, int, dict[str, object], int]],
        ] = defaultdict(list)
        for payload in rows:
            game = str(payload.get("source_game") or "")
            by_game[game].append(
                (
                    int(payload.get("evidence_id") or 0),
                    int(payload.get("source_global_step") or -1),
                    payload,
                    int(payload.get("generation_id") or 0),
                )
            )
        if self._last_evidence_id:
            out_of_order = any(
                game in self._last_by_game
                and min(item[1] for item in game_rows)
                <= self._last_by_game[game][1]
                for game, game_rows in by_game.items()
            )
            if out_of_order:
                self._last_evidence_id = 0
                self._last_by_game.clear()
                self._counts.clear()
                self._gains.clear()
                self._first.clear()
                self._last.clear()
                return self.run(writer=writer)
        changed: set[tuple[int, int]] = set()
        for game_rows in by_game.values():
            game = str(game_rows[0][2].get("source_game") or "")
            prior = self._last_by_game.get(game)
            if prior is not None:
                game_rows.append(prior)
            game_rows.sort(key=lambda row: (row[1], row[0]))
            for left, right in zip(game_rows, game_rows[1:]):
                _left_id, left_step, a, gen_a = left
                _right_id, right_step, b, gen_b = right
                if right_step != left_step + 1:
                    continue
                ca = a.get("carrier_signature")
                cb = b.get("carrier_signature")
                if ca is None or cb is None or int(ca) == int(cb):
                    continue
                # Continuity is evidence-supported: same functional family or
                # immediately chained context, rather than raw appearance.
                same_family = int(a.get("outcome_signature") or 0) == int(b.get("outcome_signature") or 0)
                next_contexts = set(int(v) for v in a.get("next_context_signatures", ()) or ())
                current_contexts = set(int(v) for v in b.get("context_signatures", ()) or ())
                continuous = bool(next_contexts & current_contexts)
                if not (same_family or continuous):
                    continue
                key = tuple(sorted((int(ca), int(cb))))
                self._counts[key] += 1
                self._gains[key] += 0.5 * float(same_family) + 0.5 * float(continuous)
                self._first.setdefault(key, min(gen_a, gen_b))
                self._last[key] = max(gen_a, gen_b)
                changed.add(key)
            self._last_by_game[game] = max(
                game_rows,
                key=lambda row: (row[1], row[0]),
            )
        self._last_evidence_id = max(
            self._last_evidence_id,
            max(int(row.get("evidence_id") or 0) for row in rows),
        )
        links: list[CarrierPersistenceLink] = []
        for key in sorted(changed):
            count = int(self._counts[key])
            if count < 2:
                continue
            gain = self._gains[key] / count
            with self.lifecycle_store.connection:
                self.lifecycle_store.connection.execute(
                    "INSERT INTO carrier_persistence_links("
                    "carrier_a,carrier_b,support_count,first_generation,last_generation,predictive_gain) "
                    "VALUES (?,?,?,?,?,?) ON CONFLICT(carrier_a,carrier_b) DO UPDATE SET "
                    "support_count=MAX(carrier_persistence_links.support_count,excluded.support_count),"
                    "last_generation=MAX(carrier_persistence_links.last_generation,excluded.last_generation),"
                    "predictive_gain=MAX(carrier_persistence_links.predictive_gain,excluded.predictive_gain)",
                    (
                        key[0],
                        key[1],
                        count,
                        self._first[key],
                        self._last[key],
                        gain,
                    ),
                )
            links.append(CarrierPersistenceLink(key[0], key[1], count, gain))
        return tuple(links)

    def canonical_carrier_signature(self, raw_signature: int) -> int:
        rows = self.lifecycle_store.connection.execute(
            "SELECT carrier_a,carrier_b FROM carrier_persistence_links WHERE carrier_a=? OR carrier_b=?",
            (int(raw_signature), int(raw_signature)),
        ).fetchall()
        group = {int(raw_signature)}
        changed = True
        while changed:
            changed = False
            for a, b in rows:
                if int(a) in group or int(b) in group:
                    before = len(group)
                    group.update((int(a), int(b)))
                    changed |= len(group) != before
        if len(group) == 1:
            return int(raw_signature)
        digest = blake2b(digest_size=8)
        digest.update(b"persistent-carrier-v707")
        digest.update(str(tuple(sorted(group))).encode("ascii"))
        return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


# ---------------------------------------------------------------------------
# Integration hooks. They preserve v7.0.6 state machines and compatibility
# flags, while replacing the remaining heuristic developmental pressures.
# ---------------------------------------------------------------------------


_INSTALLED = False


def install_v707_extensions() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from v7.derivation.online_runtime import OnlineHierarchyBuilder
    from v7.derivation.pipeline import MemoryLearningPipeline
    from v7.memory.development import DevelopmentalLifecycleRuntime
    from v7.memory.developmental_policy import profile_for_view
    from v7.memory.gate_validation import EmpiricalGateValidator
    from v7.memory.lifecycle import MemoryLifecycleController
    from v7.memory.status import memory_is_derivation_eligible
    from v7.runtime import V7Runtime

    # 1) Label-independent M1 significance/future-option learning.
    original_candidate = MemoryLearningPipeline._m1_candidate

    @staticmethod
    def _v707_m1_candidate(evidence, context_signature):
        outcome = developmental_outcome_for_evidence(evidence)
        key = CanonicalMemoryKey(
            MemoryLevel.M1,
            TYPE_CONTINGENCY,
            (
                int(context_signature),
                int(evidence.action_id),
                int(evidence.outcome_signature),
            ),
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=1,
            significance=float(outcome.learned_significance),
            prediction_error=max(0.0, float(evidence.prediction_error)),
            learning_value=max(0.0, float(evidence.prediction_error)),
            future_option_delta=float(outcome.future_option_delta),
        )

    MemoryLearningPipeline._m1_candidate = _v707_m1_candidate
    MemoryLearningPipeline._v707_original_m1_candidate = original_candidate

    original_pipeline_observe = MemoryLearningPipeline.observe_batch

    def _v707_pipeline_observe(self, rows):
        episodes = tuple(rows)
        result = original_pipeline_observe(self, episodes)
        # Persist the learned polarity independently from benchmark labels.
        if self.evidence_store is not None:
            for evidence in episodes:
                if evidence.source_global_step is None:
                    continue
                outcome = developmental_outcome_for_evidence(evidence)
                matches = self.evidence_store.connection.execute(
                    "SELECT evidence_id,payload_json FROM evidence_records WHERE evidence_type=? "
                    "AND source_game IS ? AND source_global_step=? ORDER BY evidence_id DESC LIMIT 1",
                    (int(EvidenceType.EPISODE), evidence.source_game, int(evidence.source_global_step)),
                ).fetchall()
                for evidence_id, payload_json in matches:
                    try:
                        payload = json.loads(str(payload_json or "{}"))
                    except (TypeError, json.JSONDecodeError):
                        payload = {}
                    payload["environment_terminal_polarity"] = int(getattr(evidence, "terminal_polarity", 0) or 0)
                    payload["developmental_polarity"] = int(outcome.developmental_polarity)
                    payload["developmental_significance"] = float(outcome.learned_significance)
                    payload["developmental_future_option_delta"] = float(outcome.future_option_delta)
                    with self.evidence_store.connection:
                        self.evidence_store.connection.execute(
                            "UPDATE evidence_records SET payload_json=? WHERE evidence_id=?",
                            (json.dumps(payload, separators=(",", ":"), sort_keys=True), int(evidence_id)),
                        )
        return result

    MemoryLearningPipeline.observe_batch = _v707_pipeline_observe

    # 2) All higher derivation consumes learned polarity, while raw labels remain
    # in the evidence ledger for benchmark/reporting compatibility.
    original_load = OnlineHierarchyBuilder._load

    def _v707_load(self, evidence_type):
        rows = original_load(self, evidence_type)
        if int(evidence_type) == int(EvidenceType.EPISODE):
            for row in rows:
                if "developmental_polarity" in row:
                    row["terminal_polarity"] = int(row["developmental_polarity"])
                if "developmental_future_option_delta" in row:
                    row["future_option_delta"] = float(row["developmental_future_option_delta"])
        return rows

    OnlineHierarchyBuilder._load = _v707_load

    # 3) Gate allocation uses PromotionScore only before trials exist. Scientific
    # validation still requires genuine held-out gate evidence.
    original_gate_evaluate = EmpiricalGateValidator.evaluate

    def _v707_gate_evaluate(self, view, *, gate_summaries, memory_ids=None, parent_validity=None):
        adjusted = dict(gate_summaries)
        ids = tuple(memory_ids if memory_ids is not None else view.nodes.keys())
        for memory_id in ids:
            node = view.nodes.get(memory_id)
            summary = adjusted.get(memory_id)
            if node is None or summary is None:
                continue
            gate = gate_for_identity(node.level, node.type_id)
            if gate in {GateId.G34, GateId.G45, GateId.G56} and summary.trials > 0:
                effective = (
                    float(summary.mean_causal_gain)
                    + 0.30 * float(summary.future_option_gain)
                    + (0.20 * float(summary.efficiency_gain) if gate == GateId.G56 else 0.0)
                )
                adjusted[memory_id] = replace(summary, mean_causal_gain=effective)
        decisions = original_gate_evaluate(
            self,
            view,
            gate_summaries=adjusted,
            memory_ids=memory_ids,
            parent_validity=parent_validity,
        )
        filtered = []
        for decision in decisions:
            node = view.nodes.get(decision.memory_id)
            summary = adjusted.get(decision.memory_id)
            if node is None:
                filtered.append(decision)
                continue
            assessment = promotion_assessment(node, view.scores.get(decision.memory_id))
            # Existing empirical trials are never discarded because a prospective
            # estimate was low: prospective scores allocate tests, not truth.
            has_trials = summary is not None and int(summary.trials) > 0
            if (
                not has_trials
                and decision.next_validation_state
                not in {GateValidationState.VALIDATED, GateValidationState.TRUSTED}
                and not assessment.probe_worthy
            ):
                decision = replace(
                    decision,
                    probe_eligible=False,
                    tested=False,
                    validated=False,
                    rejected=False,
                    trusted=False,
                    next_validation_state=GateValidationState.STRUCTURAL_CANDIDATE,
                    next_cognitive_state=CognitiveState.PROBE_ONLY,
                )
            filtered.append(decision)
        return tuple(filtered)

    EmpiricalGateValidator.evaluate = _v707_gate_evaluate

    # 4) Developmental W(t) + signed future-option pressure.
    original_fitness = MemoryLifecycleController.fitness

    def _v707_fitness(self, node, score, *, empirical_transfer=0.0):
        stage = str(getattr(self, "_v707_stage", "CONTROL"))
        return developmental_memory_fitness(
            node,
            score,
            empirical_transfer=empirical_transfer,
            stage=stage,
        )

    MemoryLifecycleController.fitness = _v707_fitness
    MemoryLifecycleController._v707_original_fitness = original_fitness

    # 5) Runtime causal credit also ignores benchmark terminal labels.
    original_utility = V7Runtime._observed_decision_utility

    @staticmethod
    def _v707_observed_decision_utility(evidence):
        outcome = developmental_outcome_for_evidence(evidence)
        signed = 2.0 * float(outcome.learned_significance) - 1.0
        future = _signed_unit(float(outcome.future_option_delta))
        return signed, 0.0, future

    V7Runtime._observed_decision_utility = _v707_observed_decision_utility
    V7Runtime._v707_original_observed_decision_utility = original_utility

    # 6) Specialized Hydra heads share the same writer/evidence substrate.
    original_development_run = DevelopmentalLifecycleRuntime.run

    def _v707_development_run(self, view, *, writer):
        ensure_v707_schema(self.evidence_lifecycle)
        profile = profile_for_view(view)
        self.lifecycle_runtime.controller._v707_stage = profile.stage.name
        result = original_development_run(self, view, writer=writer)
        context_runtime = getattr(self, "_v707_context_runtime", None)
        if context_runtime is None:
            context_runtime = ContextRefinementRuntime(
                self.evidence_lifecycle,
                self.evidence_store,
            )
            self._v707_context_runtime = context_runtime
        context_runtime.run(view, writer=writer)
        compression_runtime = getattr(self, "_v707_compression_runtime", None)
        if compression_runtime is None:
            compression_runtime = MemoryCompressionRuntime(self.evidence_lifecycle)
            self._v707_compression_runtime = compression_runtime
        compression_runtime.run(writer=writer)
        return result

    DevelopmentalLifecycleRuntime.run = _v707_development_run

    original_hierarchy_derive = OnlineHierarchyBuilder.derive

    def _v707_hierarchy_derive(self):
        result = original_hierarchy_derive(self)
        efficiency_runtime = getattr(self, "_v707_efficiency_runtime", None)
        if efficiency_runtime is None:
            efficiency_runtime = TrajectoryEfficiencyRuntime(
                self.lifecycle_store,
                self.evidence_store,
            )
            self._v707_efficiency_runtime = efficiency_runtime
        efficiency_runtime.run(writer=self.writer)
        carrier_runtime = getattr(self, "_v707_carrier_runtime", None)
        if carrier_runtime is None:
            carrier_runtime = CarrierPersistenceRuntime(
                self.lifecycle_store,
                self.evidence_store,
            )
            self._v707_carrier_runtime = carrier_runtime
        carrier_runtime.run(writer=self.writer)
        return result

    OnlineHierarchyBuilder.derive = _v707_hierarchy_derive

    # 7) Compression converts runtime dependency into provenance-only. A retired
    # compressed parent therefore remains scientific provenance without blocking
    # the validated abstraction that replaced it.
    original_dependency = DevelopmentalLifecycleRuntime._dependency_satisfied

    def _v707_dependency_satisfied(self, view, memory_id, visiting):
        if memory_id in visiting:
            return True
        visiting = set(visiting)
        visiting.add(memory_id)
        for parent_id in self.evidence_lifecycle.provenance_parents(memory_id):
            row = self.evidence_lifecycle.connection.execute(
                "SELECT replacement_memory_id FROM memory_compression_replacements "
                "WHERE parent_memory_id=? AND provenance_only=1",
                (int(parent_id),),
            ).fetchone()
            if row is not None and int(row[0]) == int(memory_id):
                continue
            parent = view.nodes.get(parent_id)
            if parent is None:
                continue
            parent_gate = gate_for_identity(parent.level, parent.type_id)
            if parent_gate != GateId.NONE and not memory_is_derivation_eligible(parent):
                return False
            if not _v707_dependency_satisfied(self, view, parent_id, visiting):
                return False
        return True

    DevelopmentalLifecycleRuntime._dependency_satisfied = _v707_dependency_satisfied
    DevelopmentalLifecycleRuntime._v707_original_dependency_satisfied = original_dependency


__all__ = [
    "CarrierPersistenceLink",
    "CarrierPersistenceRuntime",
    "CompressionDecision",
    "ContextRefinementDecision",
    "ContextRefinementRuntime",
    "DevelopmentalOutcome",
    "DevelopmentalWeights",
    "MemoryCompressionRuntime",
    "PromotionAssessment",
    "TrajectoryEfficiency",
    "TrajectoryEfficiencyRuntime",
    "developmental_memory_fitness",
    "developmental_outcome_for_evidence",
    "developmental_weights",
    "ensure_v707_schema",
    "infer_developmental_outcome",
    "install_v707_extensions",
    "promotion_assessment",
    "trajectory_efficiency",
]
