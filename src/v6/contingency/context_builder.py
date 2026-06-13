from __future__ import annotations

from collections import deque


class ContextBuilder:
    def __init__(self, context_length: int = 3) -> None:
        self.context_length = int(context_length)
        self.transformation_families: deque[int] = deque(maxlen=self.context_length)
        self.actions: deque[int] = deque(maxlen=self.context_length)

    def signature(self) -> tuple[int | None, ...]:
        families = _left_pad(tuple(self.transformation_families), self.context_length)
        actions = _left_pad(tuple(self.actions), self.context_length)
        return families + actions

    def multi_scale_signatures(self, current_action: int, max_level: int = 3) -> dict[int, tuple[int | None, ...]]:
        families = tuple(self.transformation_families)
        actions = tuple(self.actions)
        signatures: dict[int, tuple[int | None, ...]] = {}
        for level in range(max(0, int(max_level)) + 1):
            if level == 0:
                signatures[level] = (int(current_action),)
            else:
                signatures[level] = (
                    _left_pad(families, level)
                    + _left_pad(actions, level)
                    + (int(current_action),)
                )
        return signatures

    def update(self, transformation_family: int, action: int) -> None:
        self.transformation_families.append(int(transformation_family))
        self.actions.append(int(action))


def _left_pad(values: tuple[int, ...], length: int) -> tuple[int | None, ...]:
    if length <= 0:
        return ()
    padding = (None,) * max(0, length - len(values))
    return padding + values[-length:]
