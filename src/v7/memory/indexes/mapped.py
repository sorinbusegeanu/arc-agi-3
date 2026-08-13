from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import ActionAggregate, ActionScoreInput


def _lower_bound_pair(a: memoryview, b: memoryview, target_a: int, target_b: int) -> int:
    lo, hi = 0, len(a)
    target = (int(target_a), int(target_b))
    while lo < hi:
        mid = (lo + hi) // 2
        current = (int(a[mid]), int(b[mid]))
        if current < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _lower_bound_triple(a: memoryview, b: memoryview, c: memoryview, ta: int, tb: int, tc: int) -> int:
    lo, hi = 0, len(a)
    target = (int(ta), int(tb), int(tc))
    while lo < hi:
        mid = (lo + hi) // 2
        current = (int(a[mid]), int(b[mid]), int(c[mid]))
        if current < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _lower_bound(values: memoryview, target: int) -> int:
    lo, hi = 0, len(values)
    target = int(target)
    while lo < hi:
        mid = (lo + hi) // 2
        if int(values[mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@dataclass(frozen=True, slots=True)
class MappedPairIndex:
    key_a: memoryview
    key_b: memoryview
    offsets: memoryview
    lengths: memoryview
    values: memoryview

    def lookup(self, a: int, b: int, limit: int | None = None) -> tuple[MemoryId, ...]:
        row = _lower_bound_pair(self.key_a, self.key_b, a, b)
        if row >= len(self.key_a) or (int(self.key_a[row]), int(self.key_b[row])) != (int(a), int(b)):
            return ()
        start = int(self.offsets[row])
        stop = start + int(self.lengths[row])
        if limit is not None:
            stop = min(stop, start + max(0, int(limit)))
        return tuple(MemoryId(int(v)) for v in self.values[start:stop])


@dataclass(frozen=True, slots=True)
class MappedRoleExactIndex:
    contexts: memoryview
    actions: memoryview
    families: memoryview
    offsets: memoryview
    lengths: memoryview
    values: memoryview

    def lookup(self, context: int, action: int, family: MemoryId, limit: int) -> tuple[MemoryId, ...]:
        row = _lower_bound_triple(self.contexts, self.actions, self.families, context, action, int(family))
        target = (int(context), int(action), int(family))
        if row >= len(self.contexts) or (int(self.contexts[row]), int(self.actions[row]), int(self.families[row])) != target:
            return ()
        start = int(self.offsets[row])
        stop = min(start + int(self.lengths[row]), start + max(0, int(limit)))
        return tuple(MemoryId(int(v)) for v in self.values[start:stop])


@dataclass(frozen=True, slots=True)
class MappedActionAggregates:
    action_ids: memoryview
    values: memoryview

    def get(self, action_id: int) -> ActionAggregate:
        row = _lower_bound(self.action_ids, action_id)
        if row >= len(self.action_ids) or int(self.action_ids[row]) != int(action_id):
            return ActionAggregate()
        base = row * 6
        values = self.values[base:base + 6]
        return ActionAggregate(
            future_option_sum=float(values[0]),
            future_option_count=int(values[1]),
            positive_count=int(values[2]),
            negative_count=int(values[3]),
            failure_count=int(values[4]),
            contradiction_count=int(values[5]),
        )


@dataclass(frozen=True, slots=True)
class MappedPackedCognitionIndexes:
    contingencies: MappedPairIndex
    roles_fallback: MappedPairIndex
    roles_exact: MappedRoleExactIndex
    concepts_by_role: MappedPairIndex
    action_aggregates: MappedActionAggregates
    owners: tuple[object, ...] = ()

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
                remaining = max(0, concept_limit - len(concepts))
                if remaining == 0:
                    break
                for concept_id in self.concepts(role_id, remaining):
                    if concept_id in seen:
                        continue
                    seen.add(concept_id)
                    concepts.append(concept_id)
                    if len(concepts) >= concept_limit:
                        break
            rows.append(ActionScoreInput(
                action_id=action_id,
                contingency_ids=contingencies,
                aggregate=self.action_aggregates.get(action_id),
                role_ids=roles,
                concept_ids=tuple(concepts),
            ))
        return tuple(rows)
