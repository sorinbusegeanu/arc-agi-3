from __future__ import annotations

from array import array
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable

from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import ActionAggregate, CognitionIndexes


@dataclass(frozen=True, slots=True)
class PackedPairIndex:
    key_a: array
    key_b: array
    offsets: array
    lengths: array
    values: array

    @classmethod
    def build(cls, rows: Iterable[tuple[int, int, Iterable[MemoryId]]]) -> "PackedPairIndex":
        ordered = sorted((int(a), int(b), tuple(sorted(set(values), key=int))) for a, b, values in rows)
        key_a, key_b, offsets, lengths, values = array("q"), array("q"), array("Q"), array("Q"), array("Q")
        for a, b, ids in ordered:
            key_a.append(a); key_b.append(b); offsets.append(len(values)); lengths.append(len(ids)); values.extend(int(v) for v in ids)
        return cls(key_a, key_b, offsets, lengths, values)

    def lookup(self, a: int, b: int, limit: int | None = None) -> tuple[MemoryId, ...]:
        target = (int(a), int(b))
        lo = bisect_left(list(zip(self.key_a, self.key_b)), target)
        if lo >= len(self.key_a) or (int(self.key_a[lo]), int(self.key_b[lo])) != target:
            return ()
        start = int(self.offsets[lo]); stop = start + int(self.lengths[lo])
        if limit is not None:
            stop = min(stop, start + max(0, int(limit)))
        return tuple(MemoryId(int(v)) for v in self.values[start:stop])


@dataclass(frozen=True, slots=True)
class PackedRoleExactIndex:
    contexts: array
    actions: array
    families: array
    offsets: array
    lengths: array
    values: array

    @classmethod
    def build(cls, indexes: CognitionIndexes) -> "PackedRoleExactIndex":
        rows = sorted(indexes.role_by_context_action_family.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2])))
        contexts, actions, families, offsets, lengths, values = array("q"), array("q"), array("Q"), array("Q"), array("Q"), array("Q")
        for (context, action, family), ids in rows:
            contexts.append(int(context)); actions.append(int(action)); families.append(int(family)); offsets.append(len(values)); lengths.append(len(ids)); values.extend(int(v) for v in ids)
        return cls(contexts, actions, families, offsets, lengths, values)

    def lookup(self, context: int, action: int, family: MemoryId, limit: int) -> tuple[MemoryId, ...]:
        target = (int(context), int(action), int(family))
        keys = list(zip(self.contexts, self.actions, self.families))
        lo = bisect_left(keys, target)
        if lo >= len(keys) or tuple(int(v) for v in keys[lo]) != target:
            return ()
        start = int(self.offsets[lo]); stop = min(start + int(self.lengths[lo]), start + max(0, int(limit)))
        return tuple(MemoryId(int(v)) for v in self.values[start:stop])


@dataclass(frozen=True, slots=True)
class PackedActionAggregates:
    action_ids: array
    values: array

    @classmethod
    def build(cls, indexes: CognitionIndexes) -> "PackedActionAggregates":
        action_ids, values = array("q"), array("d")
        for action_id, aggregate in sorted(indexes.action_aggregates.items()):
            action_ids.append(int(action_id))
            values.extend((aggregate.future_option_sum, aggregate.future_option_count, aggregate.positive_count, aggregate.negative_count, aggregate.failure_count, aggregate.contradiction_count))
        return cls(action_ids, values)

    def get(self, action_id: int) -> ActionAggregate:
        pos = bisect_left(self.action_ids, int(action_id))
        if pos >= len(self.action_ids) or int(self.action_ids[pos]) != int(action_id):
            return ActionAggregate()
        base = pos * 6
        row = self.values[base:base + 6]
        return ActionAggregate(float(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))


@dataclass(frozen=True, slots=True)
class PackedCognitionIndexes:
    contingencies: PackedPairIndex
    roles_fallback: PackedPairIndex
    roles_exact: PackedRoleExactIndex
    concepts_by_role: PackedPairIndex
    action_aggregates: PackedActionAggregates

    @classmethod
    def build(cls, indexes: CognitionIndexes) -> "PackedCognitionIndexes":
        return cls(
            contingencies=PackedPairIndex.build((context, action, ids) for (context, action), ids in indexes.contingency_by_context_action.items()),
            roles_fallback=PackedPairIndex.build((context, action, ids) for (context, action), ids in indexes.role_by_context_action.items()),
            roles_exact=PackedRoleExactIndex.build(indexes),
            concepts_by_role=PackedPairIndex.build((int(role), 0, ids) for role, ids in indexes.concepts_by_role.items()),
            action_aggregates=PackedActionAggregates.build(indexes),
        )

    def concepts(self, role_id: MemoryId, limit: int) -> tuple[MemoryId, ...]:
        return self.concepts_by_role.lookup(int(role_id), 0, limit)
