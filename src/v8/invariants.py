from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from v8.arena import EdgeRecord, NodeRecord
from v8.model import CognitiveState, MemoryUid


def validate_runtime_invariants(nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> tuple[str, ...]:
    """Return invariant violations without mutating live state."""
    nodes = tuple(nodes)
    edges = tuple(edges)
    errors: list[str] = []
    by_uid: dict[MemoryUid, NodeRecord] = {}
    key_by_uid: dict[MemoryUid, tuple[int, int, tuple[int, ...]]] = {}
    for row in nodes:
        key = (int(row.level), int(row.memory_type), tuple(row.key_parts))
        prior = key_by_uid.get(row.uid)
        if prior is not None and prior != key:
            errors.append(f"uid_collision:{row.uid.hex()}")
        key_by_uid[row.uid] = key
        by_uid[row.uid] = row
    for edge in edges:
        if edge.source_uid not in by_uid:
            errors.append(f"dangling_source:{edge.source_uid.hex()}")
        if edge.target_uid not in by_uid:
            errors.append(f"dangling_target:{edge.target_uid.hex()}")
    for row in nodes:
        if int(row.cognitive_state) == int(CognitiveState.RETIRED) and int(row.validation_state) < 0:
            errors.append(f"invalid_retired_state:{row.uid.hex()}")
    return tuple(sorted(set(errors)))
