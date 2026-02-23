from __future__ import annotations

from collections import deque

from .memory_episodic import EpisodicMemory


class Planner:
    """BFS planner over discovered transition graph."""

    def __init__(self, memory: EpisodicMemory) -> None:
        self.memory = memory

    def plan_to_any_goal(self, start_hash: str, goal_hashes: set[str]) -> list[str]:
        if start_hash in goal_hashes:
            return [start_hash]
        q: deque[str] = deque([start_hash])
        parent: dict[str, str | None] = {start_hash: None}

        while q:
            node = q.popleft()
            for nxt in self.memory.adj.get(node, set()):
                if nxt in parent:
                    continue
                parent[nxt] = node
                if nxt in goal_hashes:
                    return _reconstruct(parent, nxt)
                q.append(nxt)
        return []


def _reconstruct(parent: dict[str, str | None], end: str) -> list[str]:
    out: list[str] = []
    cur: str | None = end
    while cur is not None:
        out.append(cur)
        cur = parent[cur]
    out.reverse()
    return out
