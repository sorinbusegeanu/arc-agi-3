from __future__ import annotations

from collections import deque

from .types import NormalizedAction, TransitionDelta


class SafetyGuard:
    def __init__(self, max_noop_streak: int = 6, cycle_window: int = 12, cycle_repeat: int = 3) -> None:
        self.max_noop_streak = max_noop_streak
        self.cycle_window = cycle_window
        self.cycle_repeat = cycle_repeat
        self.noop_streak = 0
        self.recent: deque[str] = deque(maxlen=cycle_window)

    def observe(self, state_hash: str, delta: TransitionDelta | None) -> None:
        self.recent.append(state_hash)
        if delta is None:
            return
        if delta.no_op:
            self.noop_streak += 1
        else:
            self.noop_streak = 0

    def penalties(self, action: NormalizedAction) -> tuple[float, tuple[str, ...]]:
        score = 0.0
        tags: list[str] = []
        if self.noop_streak >= self.max_noop_streak and action.name in {"ACTION5", "ACTION6"}:
            score += 0.75
            tags.append("noop-streak")
        if self._cycle_risk():
            if action.name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
                score += 0.25
            tags.append("cycle-risk")
        return score, tuple(tags)

    def _cycle_risk(self) -> bool:
        if len(self.recent) < self.cycle_window:
            return False
        freq: dict[str, int] = {}
        for h in self.recent:
            freq[h] = freq.get(h, 0) + 1
        return max(freq.values(), default=0) >= self.cycle_repeat
