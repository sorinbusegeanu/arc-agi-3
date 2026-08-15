from __future__ import annotations

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
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, signed_u64, stable_u64


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


class LiveReadView:
    """Bounded-staleness live view over independently published RAM shards."""

    def __init__(self, descriptors: Iterable[ShardReadDescriptor]) -> None:
        self.descriptors = tuple(descriptors)
        self._nodes = tuple(SharedNodeArena.attach(d.nodes) for d in self.descriptors)
        self._edges = tuple(SharedEdgeArena.attach(d.edges) for d in self.descriptors)
        self._actions = tuple(SharedActionArena.attach(d.actions) for d in self.descriptors)
        self._strategy_version: tuple[int, ...] = ()
        self._strategy_by_context: dict[int, list[tuple[int, MemoryUid, MemoryUid, int]]] = {}
        self._preferred_outcomes: set[MemoryUid] = set()

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
        by_context: dict[int, list[tuple[int, MemoryUid, MemoryUid, int]]] = {}
        for row in self.node_records(level=MemoryLevel.M7):
            if int(row.memory_type) != int(MemoryType.STRATEGY) or len(row.key_parts) < 4:
                continue
            action = signed_u64(int(row.key_parts[0]))
            outcome = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
            context_bucket = int(row.key_parts[3])
            by_context.setdefault(context_bucket, []).append((action, outcome, row.uid, int(row.support_count)))
        preferred: set[MemoryUid] = set()
        for edge in self.edge_records():
            if int(edge.relation_type) == int(RelationType.PREFERENCE):
                preferred.add(edge.source_uid)
        self._strategy_by_context = by_context
        self._preferred_outcomes = preferred
        self._strategy_version = version

    def planned_action(self, context_signature: int, action_ids: Iterable[int]) -> PlannedAction | None:
        self._refresh_strategy_cache()
        context_bucket = stable_u64(int(context_signature), person=b"v8-context")
        available = {int(value) for value in action_ids}
        candidates = []
        for action, outcome, strategy, support in self._strategy_by_context.get(context_bucket, ()):
            if action not in available or support <= 0:
                continue
            preference_bonus = 2.0 if outcome in self._preferred_outcomes else 0.0
            score = float(support) + preference_bonus
            candidates.append((score, action, outcome, strategy))
        if not candidates:
            return None
        score, action, outcome, strategy = max(candidates, key=lambda row: (row[0], -row[1]))
        return PlannedAction(action, outcome, strategy, score)

    def state_digest(self) -> str:
        digest = blake2b(digest_size=32, person=b"arc-v8-state")
        for descriptor, nodes, edges, actions in zip(
            self.descriptors, self._nodes, self._edges, self._actions, strict=True
        ):
            digest.update(nodes.snapshot_bytes())
            digest.update(edges.snapshot_bytes())
            digest.update(actions.snapshot_bytes())
        return digest.hexdigest()
