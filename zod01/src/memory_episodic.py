from __future__ import annotations

from collections import defaultdict, deque

from .types import NormalizedAction, TransitionDelta, TransitionEdge


class EpisodicMemory:
    def __init__(self, max_states: int = 100_000) -> None:
        self.max_states = max_states
        self.visited_order: deque[str] = deque()
        self.visited: set[str] = set()
        self.transitions: dict[tuple[str, str], TransitionEdge] = {}
        self.adj: dict[str, set[str]] = defaultdict(set)

    def register_state(self, state_hash: str) -> None:
        if state_hash in self.visited:
            return
        self.visited.add(state_hash)
        self.visited_order.append(state_hash)
        while len(self.visited_order) > self.max_states:
            stale = self.visited_order.popleft()
            self.visited.discard(stale)

    def add_transition(
        self,
        src_hash: str,
        action: NormalizedAction,
        dst_hash: str,
        delta: TransitionDelta,
        won: bool,
    ) -> None:
        key = (src_hash, action.key())
        edge = self.transitions.get(key)
        if edge is None:
            edge = TransitionEdge(src_hash=src_hash, action=action, dst_hash=dst_hash, delta=delta)
            self.transitions[key] = edge
        edge.stats.visits += 1
        if won:
            edge.stats.success_visits += 1
        if delta.no_op:
            edge.stats.no_op_visits += 1
        edge.dst_hash = dst_hash
        edge.delta = delta
        self.adj[src_hash].add(dst_hash)

    def get_edge(self, src_hash: str, action: NormalizedAction) -> TransitionEdge | None:
        return self.transitions.get((src_hash, action.key()))

    def action_visits(self, src_hash: str, action: NormalizedAction) -> int:
        edge = self.get_edge(src_hash, action)
        return 0 if edge is None else edge.stats.visits

    def unique_states(self) -> int:
        return len(self.visited)
