from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import blake2b
from typing import Iterable

import numpy as np

from v8.model import stable_u64


MAX_NORMALIZED_FACTS_PER_EVENT = 8
_M1N_MARKER = 1 << 63
_M1N_HASH_MASK = (1 << 55) - 1
_HISTORY_WINDOW = 16
_MAX_RELATION_COMPONENTS = 24


class NormalizedPrimitive(IntEnum):
    NO_CHANGE = 1
    COMPONENT_CREATED = 2
    COMPONENT_REMOVED = 3
    COMPONENT_RELOCATED = 4
    COMPONENT_GEOMETRY_CHANGED = 5
    COMPONENT_ATTRIBUTE_CHANGED = 6
    RELATION_APPEARED = 7
    RELATION_DISAPPEARED = 8
    ACTION_BECAME_AVAILABLE = 9
    ACTION_BECAME_UNAVAILABLE = 10
    DELAYED_CHANGE = 11
    AUTONOMOUS_CHANGE = 12


@dataclass(frozen=True, slots=True)
class StructuralFact:
    kind: NormalizedPrimitive
    structure_signature: int
    relation_signature: int = 0
    temporal_bucket: int = 0
    magnitude_bucket: int = 0

    @property
    def token(self) -> int:
        payload = stable_u64(
            int(self.structure_signature),
            int(self.relation_signature),
            int(self.temporal_bucket),
            int(self.magnitude_bucket),
            person=b"v8.6-m1n-fact",
        )
        return int(
            _M1N_MARKER
            | ((payload & _M1N_HASH_MASK) << 8)
            | (int(self.kind) & 0xFF)
        )


def is_normalized_fact_token(value: int) -> bool:
    raw = int(value)
    if raw < 0 or not (raw & _M1N_MARKER):
        return False
    try:
        NormalizedPrimitive(raw & 0xFF)
    except ValueError:
        return False
    return True


def normalized_fact_kind(value: int) -> NormalizedPrimitive:
    if not is_normalized_fact_token(value):
        raise ValueError("not a normalized M1N fact token")
    return NormalizedPrimitive(int(value) & 0xFF)


def normalized_family_key(value: int) -> tuple[int, int]:
    """Return the M2 grouping key for an M1N token.

    M1N identity preserves a 55-bit structural descriptor while M2 intentionally
    groups by primitive observable change class. More specific structure remains
    available through each M1N parent and can be rediscovered as carriers/roles.
    """
    kind = normalized_fact_kind(value)
    return (int(kind), 0)


def _digest_array(*parts: bytes, person: bytes) -> int:
    digest = blake2b(digest_size=8, person=person[:16])
    for part in parts:
        digest.update(len(part).to_bytes(4, "little"))
        digest.update(part)
    return int.from_bytes(digest.digest(), "little")


def _magnitude_bucket(value: int) -> int:
    count = max(0, int(value))
    if count == 0:
        return 0
    if count == 1:
        return 1
    if count <= 4:
        return 2
    if count <= 16:
        return 3
    if count <= 64:
        return 4
    return 5


def _temporal_bucket(elapsed_since_change: int) -> int:
    elapsed = max(0, int(elapsed_since_change))
    if elapsed == 0:
        return 0
    if elapsed == 1:
        return 1
    if elapsed <= 4:
        return 2
    if elapsed <= 16:
        return 3
    return 4


def _normalized_changed_geometry(before: np.ndarray, after: np.ndarray) -> tuple[int, int]:
    left = np.asarray(before, dtype=np.int64)
    right = np.asarray(after, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 2:
        return (
            _digest_array(
                np.asarray(left.shape, dtype=np.int64).tobytes(),
                np.asarray(right.shape, dtype=np.int64).tobytes(),
                person=b"v8.6-shape",
            ),
            max(left.size, right.size),
        )
    changed = np.argwhere(left != right)
    if changed.size == 0:
        return (
            _digest_array(np.asarray(left.shape, dtype=np.int64).tobytes(), person=b"v8.6-nochange"),
            0,
        )
    mins = changed.min(axis=0)
    normalized = (changed - mins).astype(np.int16, copy=False)
    order = np.lexsort((normalized[:, 1], normalized[:, 0]))
    normalized = normalized[order]
    return (
        _digest_array(
            np.asarray(left.shape, dtype=np.int64).tobytes(),
            normalized.tobytes(),
            person=b"v8.6-change",
        ),
        int(changed.shape[0]),
    )


def _canonical_value_pattern(values: np.ndarray) -> bytes:
    labels: dict[int, int] = {}
    result = np.empty(values.size, dtype=np.int16)
    for index, value in enumerate(values.reshape(-1).tolist()):
        raw = int(value)
        if raw not in labels:
            labels[raw] = len(labels)
        result[index] = labels[raw]
    return result.tobytes()


def _main_change_fact(
    before: np.ndarray,
    after: np.ndarray,
    *,
    elapsed_since_change: int,
) -> StructuralFact:
    left = np.asarray(before, dtype=np.int64)
    right = np.asarray(after, dtype=np.int64)
    structure, changed = _normalized_changed_geometry(left, right)
    temporal = _temporal_bucket(elapsed_since_change)
    magnitude = _magnitude_bucket(changed)
    if changed == 0:
        return StructuralFact(NormalizedPrimitive.NO_CHANGE, structure, 0, temporal, 0)
    if left.shape != right.shape or left.ndim != 2:
        return StructuralFact(
            NormalizedPrimitive.COMPONENT_GEOMETRY_CHANGED,
            structure,
            stable_u64(left.size, right.size, person=b"v8.6-size-change"),
            temporal,
            magnitude,
        )

    before_occ = left != 0
    after_occ = right != 0
    added = np.logical_and(~before_occ, after_occ)
    removed = np.logical_and(before_occ, ~after_occ)
    added_count = int(np.count_nonzero(added))
    removed_count = int(np.count_nonzero(removed))

    if added_count and not removed_count:
        kind = NormalizedPrimitive.COMPONENT_CREATED
    elif removed_count and not added_count:
        kind = NormalizedPrimitive.COMPONENT_REMOVED
    elif added_count and removed_count and added_count == removed_count:
        kind = NormalizedPrimitive.COMPONENT_RELOCATED
    elif np.array_equal(before_occ, after_occ):
        kind = NormalizedPrimitive.COMPONENT_ATTRIBUTE_CHANGED
    else:
        kind = NormalizedPrimitive.COMPONENT_GEOMETRY_CHANGED

    changed_mask = left != right
    relation = _digest_array(
        _canonical_value_pattern(left[changed_mask]),
        _canonical_value_pattern(right[changed_mask]),
        person=b"v8.6-value-relation",
    )
    return StructuralFact(kind, structure, relation, temporal, magnitude)


@dataclass(frozen=True, slots=True)
class _Component:
    shape_signature: int
    cells: frozenset[tuple[int, int]]


def _components(grid: np.ndarray) -> tuple[_Component, ...]:
    array = np.asarray(grid, dtype=np.int64)
    if array.ndim != 2:
        return ()
    height, width = map(int, array.shape)
    seen: set[tuple[int, int]] = set()
    result: list[_Component] = []
    for y in range(height):
        for x in range(width):
            if (y, x) in seen or int(array[y, x]) == 0:
                continue
            color = int(array[y, x])
            stack = [(y, x)]
            cells: list[tuple[int, int]] = []
            seen.add((y, x))
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < height and 0 <= nx < width):
                        continue
                    if (ny, nx) in seen or int(array[ny, nx]) != color:
                        continue
                    seen.add((ny, nx))
                    stack.append((ny, nx))
            min_y = min(v[0] for v in cells)
            min_x = min(v[1] for v in cells)
            normalized = np.asarray(
                sorted((cy - min_y, cx - min_x) for cy, cx in cells), dtype=np.int16
            )
            signature = _digest_array(normalized.tobytes(), person=b"v8.6-component")
            result.append(_Component(signature, frozenset(cells)))
            if len(result) >= _MAX_RELATION_COMPONENTS:
                return tuple(result)
    return tuple(result)


def _relation_set(grid: np.ndarray) -> set[int]:
    components = _components(grid)
    if len(components) < 2:
        return set()
    cell_owner: dict[tuple[int, int], int] = {}
    for index, component in enumerate(components):
        for cell in component.cells:
            cell_owner[cell] = index
    relations: set[int] = set()
    for (y, x), source in cell_owner.items():
        for dy, dx in ((1, 0), (0, 1)):
            target = cell_owner.get((y + dy, x + dx))
            if target is None or target == source:
                continue
            left, right = sorted(
                (components[source].shape_signature, components[target].shape_signature)
            )
            relations.add(stable_u64(left, right, dy, dx, person=b"v8.6-relation"))
    return relations


def _base_action_ids(actions: Iterable[int]) -> frozenset[int]:
    result = set()
    for value in actions:
        raw = int(value)
        result.add(raw if 0 <= raw <= 0xFF else raw & 0xFF)
    return frozenset(result)


def native_action_set_signature(actions: Iterable[int]) -> int:
    values = tuple(sorted(_base_action_ids(actions)))
    return stable_u64(*values, person=b"v8.6-action-set") if values else 0


def observable_history_signature(entries: Iterable[tuple[int, int, int, int]]) -> int:
    rows = tuple(entries)[-_HISTORY_WINDOW:]
    if not rows:
        return 0
    value = stable_u64(len(rows), person=b"v8.6-history")
    for index, row in enumerate(rows):
        value = stable_u64(value, index, *map(int, row), person=b"v8.6-history")
    return int(value)


def grounded_context_signature(grid_signature: int, history_signature: int) -> int:
    return stable_u64(
        int(grid_signature), int(history_signature), person=b"v8.6-grounded-context"
    )


def normalized_facts_signature(tokens: Iterable[int]) -> int:
    values = tuple(int(v) for v in tokens)
    return stable_u64(*values, person=b"v8.6-facts") if values else 0


def extract_structural_facts(
    before: np.ndarray,
    after: np.ndarray,
    *,
    before_actions: Iterable[int] = (),
    after_actions: Iterable[int] = (),
    elapsed_since_change: int = 0,
) -> tuple[StructuralFact, ...]:
    """Extract bounded observable structural changes without game-semantic labels."""
    facts: list[StructuralFact] = [
        _main_change_fact(before, after, elapsed_since_change=elapsed_since_change)
    ]
    left_relations = _relation_set(before)
    right_relations = _relation_set(after)
    temporal = _temporal_bucket(elapsed_since_change)
    for relation in sorted(right_relations - left_relations):
        facts.append(
            StructuralFact(
                NormalizedPrimitive.RELATION_APPEARED,
                relation,
                relation,
                temporal,
                1,
            )
        )
        if len(facts) >= MAX_NORMALIZED_FACTS_PER_EVENT:
            return tuple(facts)
    for relation in sorted(left_relations - right_relations):
        facts.append(
            StructuralFact(
                NormalizedPrimitive.RELATION_DISAPPEARED,
                relation,
                relation,
                temporal,
                1,
            )
        )
        if len(facts) >= MAX_NORMALIZED_FACTS_PER_EVENT:
            return tuple(facts)

    before_ids = _base_action_ids(before_actions)
    after_ids = _base_action_ids(after_actions)
    if after_ids - before_ids:
        facts.append(
            StructuralFact(
                NormalizedPrimitive.ACTION_BECAME_AVAILABLE,
                stable_u64(len(before_ids), len(after_ids), person=b"v8.6-action-change"),
                0,
                temporal,
                _magnitude_bucket(len(after_ids - before_ids)),
            )
        )
    if len(facts) < MAX_NORMALIZED_FACTS_PER_EVENT and before_ids - after_ids:
        facts.append(
            StructuralFact(
                NormalizedPrimitive.ACTION_BECAME_UNAVAILABLE,
                stable_u64(len(before_ids), len(after_ids), person=b"v8.6-action-change"),
                0,
                temporal,
                _magnitude_bucket(len(before_ids - after_ids)),
            )
        )

    if (
        len(facts) < MAX_NORMALIZED_FACTS_PER_EVENT
        and facts[0].kind != NormalizedPrimitive.NO_CHANGE
        and int(elapsed_since_change) > 0
    ):
        facts.append(
            StructuralFact(
                NormalizedPrimitive.DELAYED_CHANGE,
                facts[0].structure_signature,
                facts[0].relation_signature,
                temporal,
                facts[0].magnitude_bucket,
            )
        )
    return tuple(facts[:MAX_NORMALIZED_FACTS_PER_EVENT])


def extract_normalized_fact_tokens(
    before: np.ndarray,
    after: np.ndarray,
    *,
    before_actions: Iterable[int] = (),
    after_actions: Iterable[int] = (),
    elapsed_since_change: int = 0,
) -> tuple[int, ...]:
    return tuple(
        fact.token
        for fact in extract_structural_facts(
            before,
            after,
            before_actions=before_actions,
            after_actions=after_actions,
            elapsed_since_change=elapsed_since_change,
        )
    )
