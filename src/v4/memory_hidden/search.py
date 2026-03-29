from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

from .typedState import MemoryHiddenTypedStateV4


@dataclass(frozen=True)
class MemoryHiddenSearchOutcomeV4:
    status: str
    plan: tuple[int, ...] = ()
    explored_nodes: int = 0
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MemoryHiddenSearchV4:
    def path_within_safe_region(self, state: MemoryHiddenTypedStateV4, goal_cells: set[tuple[int, int]]) -> MemoryHiddenSearchOutcomeV4:
        deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        safe = set(state.common.traversable_safe_cells)
        queue = deque([(state.common.avatar_position, ())])
        seen = {state.common.avatar_position}
        explored = 0
        while queue:
            pos, plan = queue.popleft()
            explored += 1
            if pos in goal_cells:
                return MemoryHiddenSearchOutcomeV4(status="found", plan=plan, explored_nodes=explored)
            for action_id, delta in deltas.items():
                if action_id not in state.common.current_legal_actions:
                    continue
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if nxt in seen or nxt not in safe:
                    continue
                seen.add(nxt)
                queue.append((nxt, plan + (action_id,)))
        return MemoryHiddenSearchOutcomeV4(status="exhausted", explored_nodes=explored, failure_reason="no safe-region path found")
