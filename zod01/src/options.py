from __future__ import annotations

from collections.abc import Iterator

from .types import NormalizedAction


def navigate_to(src: tuple[int, int], dst: tuple[int, int]) -> Iterator[NormalizedAction]:
    """Macro option: greedy manhattan movement with cardinal actions."""
    x, y = src
    tx, ty = dst
    while x < tx:
        x += 1
        yield NormalizedAction("ACTION4")
    while x > tx:
        x -= 1
        yield NormalizedAction("ACTION3")
    while y < ty:
        y += 1
        yield NormalizedAction("ACTION2")
    while y > ty:
        y -= 1
        yield NormalizedAction("ACTION1")


def interact_near() -> list[NormalizedAction]:
    return [NormalizedAction("ACTION5")]


def select_tool_and_click(x: int, y: int) -> list[NormalizedAction]:
    return [NormalizedAction("ACTION5"), NormalizedAction("ACTION6", x=x, y=y)]
