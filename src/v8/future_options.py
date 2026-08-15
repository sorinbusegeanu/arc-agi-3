from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryUid


@dataclass(frozen=True, slots=True)
class FutureOptionEvidence:
    uid: MemoryUid
    before_reach: int
    after_reach: int
    delta: int


class FutureOptionEstimator:
    """Bounded learned reachability with local dirty-cache invalidation."""

    def __init__(self, *, horizon: int = 3) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self._cache: dict[int, int] = {}
        self._adjacency: dict[int, tuple[int, ...]] = {}

    @staticmethod
    def _graph(rows: tuple[NodeRecord, ...]) -> dict[int, set[int]]:
        graph: dict[int, set[int]] = defaultdict(set)
        for row in rows:
            if int(row.level) != int(MemoryLevel.M1) or len(row.key_parts) < 4:
                continue
            before = int(row.key_parts[0])
            after = int(row.key_parts[3])
            if after != 0:
                graph[before].add(after)
        return graph

    def _invalidate_changed_neighborhood(self, graph: dict[int, set[int]]) -> None:
        current = {node: tuple(sorted(targets)) for node, targets in graph.items()}
        changed = {
            node
            for node in set(current) | set(self._adjacency)
            if current.get(node, ()) != self._adjacency.get(node, ())
        }
        if not changed:
            return
        reverse: dict[int, set[int]] = defaultdict(set)
        for source, targets in graph.items():
            for target in targets:
                reverse[target].add(source)
        affected = set(changed)
        frontier = deque((node, 0) for node in changed)
        while frontier:
            node, depth = frontier.popleft()
            if depth >= self.horizon:
                continue
            for parent in reverse.get(node, ()):
                if parent in affected:
                    continue
                affected.add(parent)
                frontier.append((parent, depth + 1))
        for node in affected:
            self._cache.pop(node, None)
        self._adjacency = current

    def _reach(self, graph: dict[int, set[int]], start: int) -> int:
        cached = self._cache.get(int(start))
        if cached is not None:
            return cached
        visited = {int(start)}
        frontier = deque([(int(start), 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= self.horizon:
                continue
            for nxt in graph.get(node, ()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                frontier.append((nxt, depth + 1))
        value = max(0, len(visited) - 1)
        self._cache[int(start)] = value
        return value

    def evaluate(self, rows: tuple[NodeRecord, ...]) -> tuple[FutureOptionEvidence, ...]:
        m1 = tuple(row for row in rows if int(row.level) == int(MemoryLevel.M1))
        graph = self._graph(m1)
        self._invalidate_changed_neighborhood(graph)
        result: list[FutureOptionEvidence] = []
        for row in m1:
            if len(row.key_parts) < 4:
                continue
            before = int(row.key_parts[0])
            after = int(row.key_parts[3])
            if after == 0:
                continue
            before_reach = self._reach(graph, before)
            after_reach = self._reach(graph, after)
            result.append(
                FutureOptionEvidence(
                    row.uid,
                    before_reach,
                    after_reach,
                    after_reach - before_reach,
                )
            )
        return tuple(result)
