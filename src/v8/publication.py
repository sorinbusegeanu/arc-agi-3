from __future__ import annotations

import math
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


class LiveReadView:
    """Bounded-staleness live view over independently published RAM shards."""

    def __init__(self, descriptors: Iterable[ShardReadDescriptor]) -> None:
        self.descriptors = tuple(descriptors)
        self._nodes = tuple(SharedNodeArena.attach(d.nodes) for d in self.descriptors)
        self._edges = tuple(SharedEdgeArena.attach(d.edges) for d in self.descriptors)
        self._actions = tuple(SharedActionArena.attach(d.actions) for d in self.descriptors)
        self._strategy_version: tuple[int, ...] = ()
        self._strategy_by_context: dict[int, list[_StrategyRow]] = {}
        self._preferred_outcomes: set[MemoryUid] = set()
        self._suppressed_outcomes: set[MemoryUid] = set()
        self._parents: dict[MemoryUid, set[MemoryUid]] = {}

    def close(self) -> None:
        for arena in (*self._nodes, *self._edges, *self._actions):
            arena.close()

    def score_actions(self, context_signature: int, action_ids: Iterable[int]) -> tuple[ActionScore, ...]:
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
        """Return an expectation only after it is supported and stable.

        Actors call this before the transition. Returning an empty mapping suppresses
        prediction-error generation until an expectation actually exists, removing the
        bootstrap/circularity problem from early sparse contingencies.
        """
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

    def node_records(self, *, level: MemoryLevel | int | None = None) -> tuple[NodeRecord, ...]:
        selected = []
        wanted = None if level is None else int(level)
        for arena in self._nodes:
            for record in arena.records():
                if wanted is None or int(record.level) == wanted:
                    selected.append(record)
        return tuple(selected)

    def edge_records(self) -> tuple[EdgeRecord, ...]:
        return tuple(record for arena in self._edges for record in arena.records())

    def source_games(self, uid: MemoryUid, *, max_depth: int = 8) -> frozenset[int]:
        """Return exact formation provenance, inherited through graph ancestry."""
        edges = self.edge_records()
        direct: dict[MemoryUid, set[int]] = {}
        parents: dict[MemoryUid, set[MemoryUid]] = {}
        lineage_relations = {
            int(RelationType.PROVENANCE),
            int(RelationType.EXPLAINS),
            int(RelationType.CONTEXT_REFINES),
            int(RelationType.TRANSFER_CORRESPONDENCE),
            int(RelationType.SUPERSEDES),
            int(RelationType.LEADS_TO),
        }
        for edge in edges:
            relation = int(edge.relation_type)
            if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
                direct.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
            elif relation in lineage_relations:
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

    def _refresh_strategy_cache(self) -> None:
        version = tuple(arena.sequence for arena in (*self._nodes, *self._edges))
        if version == self._strategy_version:
            return
        admissible = {
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }
        nodes = self.node_records()
        active_uids = {row.uid for row in nodes if int(row.cognitive_state) in admissible}
        edges = self.edge_records()
        parents: dict[MemoryUid, set[MemoryUid]] = {}
        preferred: set[MemoryUid] = set()
        suppressed: set[MemoryUid] = set()
        for edge in edges:
            relation = int(edge.relation_type)
            if relation == int(RelationType.GAME_PROVENANCE):
                continue
            parents.setdefault(edge.source_uid, set()).add(edge.target_uid)
            if relation == int(RelationType.PREFERENCE) and edge.source_uid in active_uids:
                preferred.add(edge.source_uid)
            if relation == int(RelationType.SUPERSEDES) and edge.source_uid in active_uids:
                suppressed.add(edge.target_uid)

        by_context: dict[int, list[_StrategyRow]] = {}
        for row in nodes:
            if (
                int(row.level) != int(MemoryLevel.M7)
                or int(row.memory_type) != int(MemoryType.STRATEGY)
                or len(row.key_parts) < 4
                or int(row.cognitive_state) not in admissible
            ):
                continue
            action = signed_u64(int(row.key_parts[0]))
            outcome = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
            # Strategies are cognitively admissible only while their represented
            # outcome class is itself active. This is required for operational M6
            # split/demotion: an invalidated coarse outcome cannot keep directing
            # behavior through an otherwise-active M7 strategy.
            if outcome not in active_uids:
                continue
            if outcome in suppressed:
                continue
            context_bucket = int(row.key_parts[3])
            if row.attempt_weight > 0:
                reliability = max(0.0, min(1.0, row.strategy_reliability))
                mean_cost = max(1e-9, row.strategy_mean_cost)
            else:
                reliability = min(0.75, max(0, int(row.support_count)) / 8.0)
                mean_cost = 1.0
            by_context.setdefault(context_bucket, []).append(
                _StrategyRow(
                    action,
                    outcome,
                    row.uid,
                    int(row.support_count),
                    reliability,
                    mean_cost,
                )
            )
        self._strategy_by_context = by_context
        self._preferred_outcomes = preferred
        self._suppressed_outcomes = suppressed
        self._parents = parents
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
        candidates: list[PlannedAction] = []
        for row in self._strategy_by_context.get(context_bucket, ()):
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
            score = row.reliability + 0.10 * efficiency + support_prior + preference_bonus
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
