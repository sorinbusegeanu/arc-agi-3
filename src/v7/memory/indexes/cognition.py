from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from v7.memory.ids import MemoryId


@dataclass(frozen=True, slots=True)
class ContingencyIndexMutation:
    context_signature: int
    action_id: int
    memory_id: MemoryId


@dataclass(frozen=True, slots=True)
class RoleIndexMutation:
    context_signature: int
    action_id: int
    role_id: MemoryId
    family_id: MemoryId | None = None


@dataclass(frozen=True, slots=True)
class RoleConceptIndexMutation:
    role_id: MemoryId
    concept_id: MemoryId


@dataclass(frozen=True, slots=True)
class ActionAggregateDelta:
    action_id: int
    future_option_sum_delta: float = 0.0
    future_option_count_delta: int = 0
    positive_count_delta: int = 0
    negative_count_delta: int = 0
    failure_count_delta: int = 0
    contradiction_count_delta: int = 0


@dataclass(frozen=True, slots=True)
class ActionAggregate:
    future_option_sum: float = 0.0
    future_option_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    failure_count: int = 0
    contradiction_count: int = 0

    @property
    def future_option_mean(self) -> float:
        if self.future_option_count <= 0:
            return 0.0
        return self.future_option_sum / self.future_option_count


@dataclass(frozen=True, slots=True)
class ActionScoreInput:
    action_id: int
    contingency_ids: tuple[MemoryId, ...]
    aggregate: ActionAggregate
    role_ids: tuple[MemoryId, ...]
    concept_ids: tuple[MemoryId, ...]


@dataclass(frozen=True, slots=True)
class CognitionIndexes:
    contingency_by_context_action: Mapping[tuple[int, int], tuple[MemoryId, ...]]
    role_by_context_action_family: Mapping[tuple[int, int, MemoryId], tuple[MemoryId, ...]]
    role_by_context_action: Mapping[tuple[int, int], tuple[MemoryId, ...]]
    concepts_by_role: Mapping[MemoryId, tuple[MemoryId, ...]]
    action_aggregates: Mapping[int, ActionAggregate]

    @classmethod
    def empty(cls) -> "CognitionIndexes":
        empty: Mapping[object, object] = MappingProxyType({})
        return cls(
            contingency_by_context_action=empty,
            role_by_context_action_family=empty,
            role_by_context_action=empty,
            concepts_by_role=empty,
            action_aggregates=empty,
        )

    @classmethod
    def freeze(
        cls,
        *,
        contingency_by_context_action: Mapping[tuple[int, int], Iterable[MemoryId]],
        role_by_context_action_family: Mapping[tuple[int, int, MemoryId], Iterable[MemoryId]],
        role_by_context_action: Mapping[tuple[int, int], Iterable[MemoryId]],
        concepts_by_role: Mapping[MemoryId, Iterable[MemoryId]],
        action_aggregates: Mapping[int, ActionAggregate],
    ) -> "CognitionIndexes":
        def freeze_ids(mapping):
            return MappingProxyType(
                {
                    key: tuple(sorted(set(values), key=int))
                    for key, values in mapping.items()
                }
            )

        return cls(
            contingency_by_context_action=freeze_ids(contingency_by_context_action),
            role_by_context_action_family=freeze_ids(role_by_context_action_family),
            role_by_context_action=freeze_ids(role_by_context_action),
            concepts_by_role=freeze_ids(concepts_by_role),
            action_aggregates=MappingProxyType(dict(action_aggregates)),
        )

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
        context_signature = int(context_signature)

        for raw_action_id in action_ids:
            action_id = int(raw_action_id)
            contingency_ids = self.contingency_by_context_action.get(
                (context_signature, action_id), ()
            )
            family_id = family_ids_by_action.get(action_id)
            if family_id is not None:
                role_ids = self.role_by_context_action_family.get(
                    (context_signature, action_id, family_id), ()
                )
            else:
                role_ids = ()
            if not role_ids:
                role_ids = self.role_by_context_action.get(
                    (context_signature, action_id), ()
                )
            role_ids = role_ids[:role_limit]

            concepts: list[MemoryId] = []
            seen: set[MemoryId] = set()
            for role_id in role_ids:
                for concept_id in self.concepts_by_role.get(role_id, ()):
                    if concept_id in seen:
                        continue
                    seen.add(concept_id)
                    concepts.append(concept_id)
                    if len(concepts) >= concept_limit:
                        break
                if len(concepts) >= concept_limit:
                    break

            rows.append(
                ActionScoreInput(
                    action_id=action_id,
                    contingency_ids=contingency_ids,
                    aggregate=self.action_aggregates.get(action_id, ActionAggregate()),
                    role_ids=role_ids,
                    concept_ids=tuple(concepts),
                )
            )
        return tuple(rows)


class CognitionIndexBuilder:
    """Mutable writer-owned index state; publication produces an immutable snapshot."""

    def __init__(self) -> None:
        self._contingencies: dict[tuple[int, int], set[MemoryId]] = {}
        self._roles_exact: dict[tuple[int, int, MemoryId], set[MemoryId]] = {}
        self._roles_fallback: dict[tuple[int, int], set[MemoryId]] = {}
        self._concepts_by_role: dict[MemoryId, set[MemoryId]] = {}
        self._action_aggregates: dict[int, ActionAggregate] = {}

    def apply_contingency_batch(self, mutations: Iterable[ContingencyIndexMutation]) -> int:
        unique: set[tuple[int, int, MemoryId]] = set()
        for mutation in mutations:
            key = (int(mutation.context_signature), int(mutation.action_id), mutation.memory_id)
            unique.add(key)
        for context_signature, action_id, memory_id in unique:
            self._contingencies.setdefault((context_signature, action_id), set()).add(memory_id)
        return len(unique)

    def apply_role_batch(self, mutations: Iterable[RoleIndexMutation]) -> int:
        unique: set[tuple[int, int, MemoryId | None, MemoryId]] = set()
        for mutation in mutations:
            unique.add(
                (
                    int(mutation.context_signature),
                    int(mutation.action_id),
                    mutation.family_id,
                    mutation.role_id,
                )
            )
        for context_signature, action_id, family_id, role_id in unique:
            self._roles_fallback.setdefault((context_signature, action_id), set()).add(role_id)
            if family_id is not None:
                self._roles_exact.setdefault(
                    (context_signature, action_id, family_id), set()
                ).add(role_id)
        return len(unique)

    def apply_role_concept_batch(self, mutations: Iterable[RoleConceptIndexMutation]) -> int:
        unique = {(mutation.role_id, mutation.concept_id) for mutation in mutations}
        for role_id, concept_id in unique:
            self._concepts_by_role.setdefault(role_id, set()).add(concept_id)
        return len(unique)

    def apply_action_aggregate_batch(self, deltas: Iterable[ActionAggregateDelta]) -> int:
        coalesced: dict[int, ActionAggregateDelta] = {}
        for delta in deltas:
            action_id = int(delta.action_id)
            prior = coalesced.get(action_id)
            if prior is None:
                coalesced[action_id] = delta
                continue
            coalesced[action_id] = ActionAggregateDelta(
                action_id=action_id,
                future_option_sum_delta=prior.future_option_sum_delta + delta.future_option_sum_delta,
                future_option_count_delta=prior.future_option_count_delta + delta.future_option_count_delta,
                positive_count_delta=prior.positive_count_delta + delta.positive_count_delta,
                negative_count_delta=prior.negative_count_delta + delta.negative_count_delta,
                failure_count_delta=prior.failure_count_delta + delta.failure_count_delta,
                contradiction_count_delta=prior.contradiction_count_delta + delta.contradiction_count_delta,
            )

        next_values: dict[int, ActionAggregate] = {}
        for action_id, delta in coalesced.items():
            current = self._action_aggregates.get(action_id, ActionAggregate())
            candidate = ActionAggregate(
                future_option_sum=current.future_option_sum + float(delta.future_option_sum_delta),
                future_option_count=current.future_option_count + int(delta.future_option_count_delta),
                positive_count=current.positive_count + int(delta.positive_count_delta),
                negative_count=current.negative_count + int(delta.negative_count_delta),
                failure_count=current.failure_count + int(delta.failure_count_delta),
                contradiction_count=current.contradiction_count + int(delta.contradiction_count_delta),
            )
            if min(
                candidate.future_option_count,
                candidate.positive_count,
                candidate.negative_count,
                candidate.failure_count,
                candidate.contradiction_count,
            ) < 0:
                raise ValueError("action aggregate counts cannot be negative")
            next_values[action_id] = candidate

        self._action_aggregates.update(next_values)
        return len(coalesced)

    def freeze(self) -> CognitionIndexes:
        return CognitionIndexes.freeze(
            contingency_by_context_action=self._contingencies,
            role_by_context_action_family=self._roles_exact,
            role_by_context_action=self._roles_fallback,
            concepts_by_role=self._concepts_by_role,
            action_aggregates=self._action_aggregates,
        )
