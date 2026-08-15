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
    """Bounded learned reachability over the discovered M1 context-transition graph."""

    def __init__(self, *, horizon: int = 3) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self._cache: dict[tuple[int, int], int] = {}
        self._graph_version = -1

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

    def _reach(self, graph: dict[int, set[int]], start: int, version: int) -> int:
        cache_key = (int(start), int(version))
        cached = self._cache.get(cache_key)
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
        self._cache[cache_key] = value
        return value

    def evaluate(self, rows: tuple[NodeRecord, ...]) -> tuple[FutureOptionEvidence, ...]:
        m1 = tuple(row for row in rows if int(row.level) == int(MemoryLevel.M1))
        version = max((int(row.updated_watermark) for row in m1), default=0)
        if version != self._graph_version:
            self._cache.clear()
            self._graph_version = version
        graph = self._graph(m1)
        result: list[FutureOptionEvidence] = []
        for row in m1:
            if len(row.key_parts) < 4:
                continue
            before = int(row.key_parts[0])
            after = int(row.key_parts[3])
            if after == 0:
                continue
            before_reach = self._reach(graph, before, version)
            after_reach = self._reach(graph, after, version)
            result.append(FutureOptionEvidence(row.uid, before_reach, after_reach, after_reach - before_reach))
        return tuple(result)
