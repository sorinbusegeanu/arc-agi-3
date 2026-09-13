from __future__ import annotations

from collections import deque

from v9.memory.identity import MemoryUid
from v9.memory.relations import EdgeAuthority, RelationEdge


def bounded_reachability(origin: MemoryUid, edges: tuple[RelationEdge, ...], *, maximum_depth: int, node_budget: int) -> tuple[MemoryUid, ...]:
    adjacency: dict[MemoryUid, list[MemoryUid]] = {}
    for edge in edges:
        if edge.authority is EdgeAuthority.ACTIVE:
            adjacency.setdefault(edge.source, []).append(edge.target)
    seen = {origin}
    queue = deque([(origin, 0)])
    while queue and len(seen) < node_budget:
        node, depth = queue.popleft()
        if depth >= maximum_depth:
            continue
        for target in sorted(adjacency.get(node, [])):
            if target not in seen:
                seen.add(target)
                queue.append((target, depth + 1))
                if len(seen) >= node_budget:
                    break
    return tuple(sorted(seen - {origin}))

