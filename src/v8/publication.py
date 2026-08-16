from __future__ import annotations

import math
import time
from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable

from v8.arena import (
    ArenaDescriptor,
    EdgeRecord,
    NodeRecord,
    SharedActionArena,
    SharedEdgeArena,
    SharedNodeArena,
)
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    signed_u64,
    stable_u64,
)


@dataclass(frozen=True, slots=True)
class ShardReadDescriptor:
    nodes: ArenaDescriptor
    edges: ArenaDescriptor
    actions: ArenaDescriptor


@dataclass(frozen=True, slots=True)
class ActionScore:
    action_id: int
    support_count: int
    score: float
    evidence_shards: int


@dataclass(frozen=True, slots=True)
class PlannedAction:
    action_id: int
    outcome_uid: MemoryUid
    strategy_uid: MemoryUid
    score: float
    preference_influenced: bool = False


@dataclass(frozen=True, slots=True)
class _StrategyRow:
    action_id: int
    outcome_uid: MemoryUid
    strategy_uid: MemoryUid
    support: int
    reliability: float
    mean_cost: float
    context_bucket: int
    probationary: bool = False
    transferable: bool = False


_LINEAGE_RELATIONS = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
}


class LiveReadView:
    """Bounded-staleness live view over independently published RAM shards."""

    def __init__(self, descriptors: Iterable[ShardReadDescriptor]) -> None:
        self.descriptors = tuple(descriptors)
        self._nodes = tuple(SharedNodeArena.attach(d.nodes) for d in self.descriptors)
        self._edges = tuple(SharedEdgeArena.attach(d.edges) for d in self.descriptors)
        self._actions = tuple(SharedActionArena.attach(d.actions) for d in self.descriptors)
        self._record_cache: dict[int, tuple[tuple[object, ...], int]] = {}
        self._strategy_version: tuple[int, ...] = ()
        self._strategy_by_context: dict[int, list[_StrategyRow]] = {}
        self._strategy_fallback: list[_StrategyRow] = []
        self._preferred_outcomes: set[MemoryUid] = set()
        self._suppressed_outcomes: set[MemoryUid] = set()
        self._parents: dict[MemoryUid, set[MemoryUid]] = {}
        self._node_by_uid: dict[MemoryUid, NodeRecord] = {}
        self._refined_action_scores: dict[tuple[int, int], tuple[int, float]] = {}
        for arena in (*self._nodes, *self._edges):
            self._stable_records_with_version(arena)

    def close(self) -> None:
        for arena in (*self._nodes, *self._edges, *self._actions):
            arena.close()

    def score_actions(self, context_signature: int, action_ids: Iterable[int]) -> tuple[ActionScore, ...]:
        self._refresh_strategy_cache()
        rows: list[ActionScore] = []
        for raw_action in action_ids:
            action = int(raw_action)
            support = 0
            total_score = 0.0
            total_weight = 0.0
            evidence_shards = 0
            for arena in self._actions:
                record = arena.lookup(context_signature, action)
                if record is None:
                    continue
                evidence_shards += 1
                support += int(record.support_count)
                total_score += float(record.score_sum)
                total_weight += float(record.score_weight)
            refined_support, refined_score = self._refined_action_scores.get(
                (int(context_signature), action), (0, 0.0)
            )
            if refined_support > 0:
                support += refined_support
                total_score += refined_score * refined_support
                total_weight += refined_support
            score = 0.0 if total_weight <= 0 else total_score / total_weight
            rows.append(ActionScore(action, support, score, evidence_shards))
        return tuple(rows)

    def best_action(self, context_signature: int, action_ids: Iterable[int]) -> int | None:
        scored = self.score_actions(context_signature, action_ids)
        if not scored:
            return None
        seen = [row for row in scored if row.support_count > 0]
        if not seen:
            return None
        return min(seen, key=lambda row: (-row.score, -row.support_count, row.action_id)).action_id

    def outcome_distribution(
        self,
        context_signature: int,
        action_id: int,
        *,
        min_support: int = 3,
        stability_threshold: float = 0.60,
    ) -> dict[int, float]:
        counts: dict[int, int] = {}
        total = 0
        context = int(context_signature)
        action = int(action_id)
        for row in self.node_records(level=MemoryLevel.M1):
            if len(row.key_parts) < 3:
                continue
            if int(row.key_parts[0]) != context or signed_u64(int(row.key_parts[1])) != action:
                continue
            support = max(0, int(row.support_count))
            outcome = int(row.key_parts[2])
            counts[outcome] = counts.get(outcome, 0) + support
            total += support
        if total < max(1, int(min_support)) or not counts:
            return {}
        dominant = max(counts.values()) / total
        if dominant < float(stability_threshold):
            return {}
        return {outcome: count / total for outcome, count in counts.items()}

    def _stable_records_with_version(self, arena, *, timeout: float = 1.0):
        cache_key = id(arena)
        cached = self._record_cache.get(cache_key)
        deadline = time.monotonic() + max(0.01, float(timeout))
        while time.monotonic() < deadline:
            before = arena.sequence
            if before & 1:
                if cached is not None:
                    return cached
                time.sleep(0.0005)
                continue
            rows = tuple(arena.records())
            after = arena.sequence
            if before == after and not (after & 1):
                snapshot = (rows, int(after))
                self._record_cache[cache_key] = snapshot
                return snapshot
            if cached is not None:
                return cached
            time.sleep(0)
        if cached is not None:
            return cached
        raise RuntimeError(f"could not obtain coherent live {arena.kind} records")

    def _stable_records(self, arena, *, timeout: float = 1.0):
        rows, _version = self._stable_records_with_version(arena, timeout=timeout)
        return rows

    def node_records(self, *, level: MemoryLevel | int | None = None) -> tuple[NodeRecord, ...]:
        selected = []
        wanted = None if level is None else int(level)
        for arena in self._nodes:
            for record in self._stable_records(arena):
                if wanted is None or int(record.level) == wanted:
                    selected.append(record)
        return tuple(selected)

    def edge_records(self) -> tuple[EdgeRecord, ...]:
        return tuple(record for arena in self._edges for record in self._stable_records(arena))

    def source_games(self, uid: MemoryUid, *, max_depth: int = 8) -> frozenset[int]:
        edges = self.edge_records()
        direct: dict[MemoryUid, set[int]] = {}
        parents: dict[MemoryUid, set[MemoryUid]] = {}
        for edge in edges:
            relation = int(edge.relation_type)
            if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
                direct.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
            elif relation in _LINEAGE_RELATIONS:
                parents.setdefault(edge.source_uid, set()).add(edge.target_uid)

        games = set(direct.get(uid, ()))
        frontier = {uid}
        visited = {uid}
        for _depth in range(max(0, int(max_depth))):
            following: set[MemoryUid] = set()
            for current in frontier:
                for parent in parents.get(current, ()):
                    games.update(direct.get(parent, ()))
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        return frozenset(games)

    def level_counts(self) -> dict[int, int]:
        result = {int(level): 0 for level in MemoryLevel}
        for record in self.node_records():
            result[int(record.level)] = result.get(int(record.level), 0) + 1
        return result

    @property
    def memory_count(self) -> int:
        return sum(arena.count for arena in self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(arena.count for arena in self._edges)

    def has_uid(self, uid: MemoryUid) -> bool:
        return any(record.uid == uid for record in self.node_records())

    def _has_transferable_ancestor(self, uid: MemoryUid, *, max_depth: int = 8) -> bool:
        frontier = {uid}
        visited = set(frontier)
        for _depth in range(max(0, int(max_depth))):
            following: set[MemoryUid] = set()
            for current in frontier:
                for parent in self._parents.get(current, ()):
                    row = self._node_by_uid.get(parent)
                    if row is not None and int(row.level) >= int(MemoryLevel.M3):
                        if (
                            row.game_evidence_count >= 2
                            or int(row.validation_state) >= int(ValidationState.VALIDATED)
                        ):
                            return True
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        return False

    def _refresh_strategy_cache(self) -> None:
        current_version = tuple(arena.sequence for arena in (*self._nodes, *self._edges))
        if current_version == self._strategy_version and not any(value & 1 for value in current_version):
            return

        nodes_list: list[NodeRecord] = []
        node_versions: list[int] = []
        for arena in self._nodes:
            rows, version = self._stable_records_with_version(arena)
            nodes_list.extend(rows)
            node_versions.append(version)

        edges_list: list[EdgeRecord] = []
        edge_versions: list[int] = []
        for arena in self._edges:
            rows, version = self._stable_records_with_version(arena)
            edges_list.extend(rows)
            edge_versions.append(version)

        nodes = tuple(nodes_list)
        edges = tuple(edges_list)
        version = tuple((*node_versions, *edge_versions))

        active_states = {
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }
        probe_states = active_states | {
            int(CognitiveState.CANDIDATE),
            int(CognitiveState.PROBATION),
        }
        active_uids = {row.uid for row in nodes if int(row.cognitive_state) in active_states}
        node_by_uid = {row.uid: row for row in nodes}
        parents: dict[MemoryUid, set[MemoryUid]] = {}
        preferred: set[MemoryUid] = set()
        suppressed: set[MemoryUid] = set()
        refined: dict[tuple[int, int], list[float]] = {}
        for edge in edges:
            relation = int(edge.relation_type)
            if relation in _LINEAGE_RELATIONS:
                parents.setdefault(edge.source_uid, set()).add(edge.target_uid)
            if relation == int(RelationType.PREFERENCE) and edge.source_uid in active_uids:
                preferred.add(edge.source_uid)
            if relation == int(RelationType.SUPERSEDES) and edge.source_uid in active_uids:
                suppressed.add(edge.target_uid)
            if relation == int(RelationType.CONTEXT_REFINES):
                role = node_by_uid.get(edge.source_uid)
                source = node_by_uid.get(edge.target_uid)
                if (
                    role is not None
                    and source is not None
                    and int(role.memory_type) == int(MemoryType.CONTEXTUAL_ROLE)
                    and len(source.key_parts) >= 2
                    and int(role.cognitive_state) in probe_states
                    and role.support_count > 0
                ):
                    context = int(source.key_parts[0])
                    action = signed_u64(int(source.key_parts[1]))
                    value = max(0.0, float(role.learning_value), float(role.significance))
                    bucket = refined.setdefault((context, action), [0.0, 0.0])
                    bucket[0] += max(1, int(role.support_count))
                    bucket[1] += value * max(1, int(role.support_count))

        self._parents = parents
        self._node_by_uid = node_by_uid
        by_context: dict[int, list[_StrategyRow]] = {}
        fallback: list[_StrategyRow] = []
        for row in nodes:
            if (
                int(row.level) != int(MemoryLevel.M7)
                or int(row.memory_type) != int(MemoryType.STRATEGY)
                or len(row.key_parts) < 4
                or int(row.cognitive_state) not in probe_states
            ):
                continue
            action = signed_u64(int(row.key_parts[0]))
            outcome = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
            outcome_row = node_by_uid.get(outcome)
            if outcome_row is None or int(outcome_row.cognitive_state) not in probe_states:
                continue
            if outcome in suppressed:
                continue
            context_bucket = int(row.key_parts[3])
            if row.attempt_weight > 0:
                reliability = max(0.0, min(1.0, row.strategy_reliability))
                mean_cost = max(1e-9, row.strategy_mean_cost)
            else:
                reliability = min(0.55, max(0, int(row.support_count)) / 10.0)
                mean_cost = 1.0
            probationary = int(row.cognitive_state) not in active_states
            transferable = self._has_transferable_ancestor(row.uid)
            strategy = _StrategyRow(
                action,
                outcome,
                row.uid,
                int(row.support_count),
                reliability,
                mean_cost,
                context_bucket,
                probationary,
                transferable,
            )
            by_context.setdefault(context_bucket, []).append(strategy)
            if transferable:
                fallback.append(strategy)
        self._strategy_by_context = by_context
        self._strategy_fallback = fallback
        self._preferred_outcomes = preferred
        self._suppressed_outcomes = suppressed
        self._refined_action_scores = {
            key: (int(values[0]), values[1] / max(1.0, values[0]))
            for key, values in refined.items()
        }
        self._strategy_version = version

    def strategy_has_ancestor(
        self,
        strategy_uid: MemoryUid,
        ancestor_uid: MemoryUid,
        *,
        max_depth: int = 8,
    ) -> bool:
        self._refresh_strategy_cache()
        if strategy_uid == ancestor_uid:
            return True
        frontier = {strategy_uid}
        visited = set(frontier)
        for _depth in range(max(0, int(max_depth))):
            following: set[MemoryUid] = set()
            for uid in frontier:
                for parent in self._parents.get(uid, ()):
                    if parent == ancestor_uid:
                        return True
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                return False
            frontier = following
        return False

    def plan_candidates(
        self,
        context_signature: int,
        action_ids: Iterable[int],
        *,
        outcome_uid: MemoryUid | None = None,
        required_ancestor: MemoryUid | None = None,
        excluded_strategies: frozenset[MemoryUid] = frozenset(),
        ignore_preference: bool = False,
    ) -> tuple[PlannedAction, ...]:
        self._refresh_strategy_cache()
        context_bucket = stable_u64(int(context_signature), person=b"v8-context")
        available = {int(value) for value in action_ids}
        exact = list(self._strategy_by_context.get(context_bucket, ()))
        active_exact = [row for row in exact if not row.probationary]
        source = active_exact if active_exact else exact
        cross_context = False
        if not source:
            source = list(self._strategy_fallback)
            cross_context = True
        candidates: list[PlannedAction] = []
        for row in source:
            if row.action_id not in available or row.support <= 0:
                continue
            if row.strategy_uid in excluded_strategies:
                continue
            if outcome_uid is not None and row.outcome_uid != outcome_uid:
                continue
            if required_ancestor is not None and not self.strategy_has_ancestor(
                row.strategy_uid, required_ancestor
            ):
                continue
            preference_influenced = (
                not ignore_preference and row.outcome_uid in self._preferred_outcomes
            )
            preference_bonus = 0.25 if preference_influenced else 0.0
            support_prior = 0.05 * math.log1p(max(0, row.support))
            efficiency = 1.0 / max(1e-9, row.mean_cost)
            probation_penalty = 0.25 if row.probationary else 0.0
            transfer_penalty = 0.30 if cross_context else 0.0
            score = (
                row.reliability
                + 0.10 * efficiency
                + support_prior
                + preference_bonus
                - probation_penalty
                - transfer_penalty
            )
            candidates.append(
                PlannedAction(
                    row.action_id,
                    row.outcome_uid,
                    row.strategy_uid,
                    float(score),
                    preference_influenced,
                )
            )
        candidates.sort(key=lambda row: (-row.score, row.action_id, row.strategy_uid))
        return tuple(candidates)

    def planned_action(
        self,
        context_signature: int,
        action_ids: Iterable[int],
        *,
        outcome_uid: MemoryUid | None = None,
        required_ancestor: MemoryUid | None = None,
        excluded_strategies: frozenset[MemoryUid] = frozenset(),
        ignore_preference: bool = False,
    ) -> PlannedAction | None:
        candidates = self.plan_candidates(
            context_signature,
            action_ids,
            outcome_uid=outcome_uid,
            required_ancestor=required_ancestor,
            excluded_strategies=excluded_strategies,
            ignore_preference=ignore_preference,
        )
        return None if not candidates else candidates[0]

    def state_digest(self) -> str:
        digest = blake2b(digest_size=32, person=b"arc-v8-state")
        for descriptor, nodes, edges, actions in zip(
            self.descriptors, self._nodes, self._edges, self._actions, strict=True
        ):
            digest.update(nodes.snapshot_bytes())
            digest.update(edges.snapshot_bytes())
            digest.update(actions.snapshot_bytes())
        return digest.hexdigest()
