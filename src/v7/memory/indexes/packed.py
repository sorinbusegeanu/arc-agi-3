from __future__ import annotations

from array import array
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import ActionAggregate, ActionScoreInput, CognitionIndexes


def _pair_row(a_values: Sequence[int], b_values: Sequence[int], a: int, b: int) -> int:
    target = (int(a), int(b))
    lo, hi = 0, len(a_values)
    while lo < hi:
        mid = (lo + hi) // 2
        current = (int(a_values[mid]), int(b_values[mid]))
        if current < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(a_values) and (int(a_values[lo]), int(b_values[lo])) == target:
        return lo
    return -1


def _triple_row(a_values: Sequence[int], b_values: Sequence[int], c_values: Sequence[int], a: int, b: int, c: int) -> int:
    target = (int(a), int(b), int(c))
    lo, hi = 0, len(a_values)
    while lo < hi:
        mid = (lo + hi) // 2
        current = (int(a_values[mid]), int(b_values[mid]), int(c_values[mid]))
        if current < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(a_values) and (int(a_values[lo]), int(b_values[lo]), int(c_values[lo])) == target:
        return lo
    return -1


@dataclass(frozen=True, slots=True)
class PackedPairIndex:
    key_a: Sequence[int]
    key_b: Sequence[int]
    offsets: Sequence[int]
    lengths: Sequence[int]
    values: Sequence[int]

    @classmethod
    def build(cls, rows: Iterable[tuple[int, int, Iterable[MemoryId]]]) -> "PackedPairIndex":
        ordered = sorted((int(a), int(b), tuple(sorted(set(values), key=int))) for a, b, values in rows)
        key_a, key_b, offsets, lengths, values = array("q"), array("q"), array("Q"), array("Q"), array("Q")
        for a, b, ids in ordered:
            key_a.append(a)
            key_b.append(b)
            offsets.append(len(values))
            lengths.append(len(ids))
            values.extend(int(v) for v in ids)
        return cls(key_a, key_b, offsets, lengths, values)

    def lookup(self, a: int, b: int, limit: int | None = None) -> tuple[MemoryId, ...]:
        row = _pair_row(self.key_a, self.key_b, a, b)
        if row < 0:
            return ()
        start = int(self.offsets[row])
        stop = start + int(self.lengths[row])
        if limit is not None:
            stop = min(stop, start + max(0, int(limit)))
        return tuple(MemoryId(int(v)) for v in self.values[start:stop])


@dataclass(frozen=True, slots=True)
class PackedRoleExactIndex:
    contexts: Sequence[int]
    actions: Sequence[int]
    families: Sequence[int]
    offsets: Sequence[int]
    lengths: Sequence[int]
    values: Sequence[int]

    @classmethod
    def build(cls, indexes: CognitionIndexes) -> "PackedRoleExactIndex":
        rows = sorted(indexes.role_by_context_action_family.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2])))
        contexts, actions, families, offsets, lengths, values = array("q"), array("q"), array("Q"), array("Q"), array("Q"), array("Q")
        for (context, action, family), ids in rows:
            contexts.append(int(context))
            actions.append(int(action))
            families.append(int(family))
            offsets.append(len(values))
            lengths.append(len(ids))
            values.extend(int(v) for v in ids)
        return cls(contexts, actions, families, offsets, lengths, values)

    def lookup(self, context: int, action: int, family: MemoryId, limit: int) -> tuple[MemoryId, ...]:
        row = _triple_row(self.contexts, self.actions, self.families, context, action, int(family))
        if row < 0:
            return ()
        start = int(self.offsets[row])
        stop = min(start + int(self.lengths[row]), start + max(0, int(limit)))
        return tuple(MemoryId(int(v)) for v in self.values[start:stop])


@dataclass(frozen=True, slots=True)
class PackedActionAggregates:
    action_ids: Sequence[int]
    values: Sequence[float]

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
        return ActionAggregate(float(self.values[base]), int(self.values[base + 1]), int(self.values[base + 2]), int(self.values[base + 3]), int(self.values[base + 4]), int(self.values[base + 5]))


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

    def score_inputs(
        self,
        *,
        context_signature: int,
        action_ids: Iterable[int],
        family_ids_by_action: Mapping[int, MemoryId] | None = None,
        role_limit: int = 64,
        concept_limit: int = 128,
    ) -> tuple[ActionScoreInput, ...]:
        if role_limit < 0 or concept_limit < 0:
            raise ValueError("limits must be non-negative")
        family_ids_by_action = family_ids_by_action or {}
        rows: list[ActionScoreInput] = []
        for raw_action_id in action_ids:
            action_id = int(raw_action_id)
            contingencies = self.contingencies.lookup(context_signature, action_id)
            family = family_ids_by_action.get(action_id)
            roles = self.roles_exact.lookup(context_signature, action_id, family, role_limit) if family is not None else ()
            if not roles:
                roles = self.roles_fallback.lookup(context_signature, action_id, role_limit)
            concepts: list[MemoryId] = []
            seen: set[MemoryId] = set()
            for role_id in roles:
                remaining = concept_limit - len(concepts)
                if remaining <= 0:
                    break
                for concept_id in self.concepts(role_id, remaining):
                    if concept_id in seen:
                        continue
                    seen.add(concept_id)
                    concepts.append(concept_id)
                    if len(concepts) >= concept_limit:
                        break
            rows.append(ActionScoreInput(action_id, contingencies, self.action_aggregates.get(action_id), roles, tuple(concepts)))
        return tuple(rows)
