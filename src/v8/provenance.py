from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from v8.arena import EdgeRecord, NodeRecord
from v8.model import MemoryLevel, MemoryUid


def exact_game_provenance(nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> dict[MemoryUid, frozenset[int]]:
    """Reconstruct exact source-game sets through provenance/explanation ancestry.

    New M0 keys store source_game_hash as key part 2. Older restored M0 records without
    that field remain representable but contribute no exact game identity.
    """
    nodes = tuple(nodes)
    by_uid = {row.uid: row for row in nodes}
    parents: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    for edge in edges:
        if edge.source_uid in by_uid and edge.target_uid in by_uid:
            parents[edge.source_uid].add(edge.target_uid)

    memo: dict[MemoryUid, frozenset[int]] = {}

    def visit(uid: MemoryUid, stack: set[MemoryUid]) -> frozenset[int]:
        cached = memo.get(uid)
        if cached is not None:
            return cached
        if uid in stack:
            return frozenset()
        row = by_uid.get(uid)
        if row is None:
            return frozenset()
        if int(row.level) == int(MemoryLevel.M0):
            value = frozenset({int(row.key_parts[2])}) if len(row.key_parts) >= 3 and int(row.key_parts[2]) != 0 else frozenset()
            memo[uid] = value
            return value
        stack.add(uid)
        games: set[int] = set()
        for parent in parents.get(uid, ()):
            games.update(visit(parent, stack))
        stack.remove(uid)
        result = frozenset(games)
        memo[uid] = result
        return result

    for row in nodes:
        visit(row.uid, set())
    return memo
