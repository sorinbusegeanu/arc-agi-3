from __future__ import annotations

from typing import Any

from arcengine import GameAction

from .types import NormalizedAction


COORD_MIN = 0
COORD_MAX = 63


def normalize_game_action(action: GameAction) -> NormalizedAction:
    data = action.action_data.model_dump() if hasattr(action, "action_data") else {}
    x = data.get("x")
    y = data.get("y")
    return NormalizedAction(name=action.name, x=x, y=y, reasoning=data.get("reasoning"))


def to_game_action(action: NormalizedAction, game_id: str = "") -> GameAction:
    ga = GameAction.from_name(action.name)
    if ga.is_complex():
        x = COORD_MIN if action.x is None else max(COORD_MIN, min(COORD_MAX, action.x))
        y = COORD_MIN if action.y is None else max(COORD_MIN, min(COORD_MAX, action.y))
        ga.set_data({"game_id": game_id, "x": x, "y": y})
    if action.reasoning is not None:
        ga.reasoning = action.reasoning  # type: ignore[assignment]
    return ga


def build_action(
    name: str,
    x: int | None = None,
    y: int | None = None,
    reasoning: str | dict[str, Any] | None = None,
) -> NormalizedAction:
    if name == "ACTION6":
        if x is None:
            x = 32
        if y is None:
            y = 32
    return NormalizedAction(name=name, x=x, y=y, reasoning=reasoning)
