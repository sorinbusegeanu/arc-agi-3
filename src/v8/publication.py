from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable

from v8.arena import (
    ArenaDescriptor,
    NodeRecord,
    SharedActionArena,
    SharedEdgeArena,
    SharedNodeArena,
)
from v8.model import MemoryLevel, MemoryUid


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


class LiveReadView:
    """Bounded-staleness live view over independently published RAM shards."""

    def __init__(self, descriptors: Iterable[ShardReadDescriptor]) -> None:
        self.descriptors = tuple(descriptors)
        self._nodes = tuple(SharedNodeArena.attach(d.nodes) for d in self.descriptors)
        self._edges = tuple(SharedEdgeArena.attach(d.edges) for d in self.descriptors)
        self._actions = tuple(SharedActionArena.attach(d.actions) for d in self.descriptors)

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

    def state_digest(self) -> str:
        digest = blake2b(digest_size=32, person=b"arc-v8-state")
        for descriptor, nodes, edges, actions in zip(
            self.descriptors, self._nodes, self._edges, self._actions, strict=True
        ):
            digest.update(nodes.snapshot_bytes())
            digest.update(edges.snapshot_bytes())
            digest.update(actions.snapshot_bytes())
        return digest.hexdigest()
