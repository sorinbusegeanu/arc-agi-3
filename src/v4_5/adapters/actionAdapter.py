from __future__ import annotations

import re
from dataclasses import dataclass

from v4.agentContract.types import V4Action


_CLICK_PATTERN = re.compile(r"^CLICK(?:[@:](?P<x>-?\d+),(?P<y>-?\d+))?$")
_ACTION_ID_TO_NAME = {
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "WAIT",
    6: "CLICK",
    7: "NOOP",
}
_MOVEMENT_TOKENS = {
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4,
}


@dataclass(frozen=True)
class ActionTranslationContext:
    available_action_ids: tuple[int, ...]
    coordinate_action_id: int | None = None
    coordinate_bounds: tuple[int, int, int, int] | None = None


class ActionAdapter:
    reused_modules = ("src/v4/agentContract/*",)

    def action_name_for_id(self, action_id: int) -> str:
        return _ACTION_ID_TO_NAME.get(int(action_id), f"ACTION{int(action_id)}")

    def available_primitive_actions(self, available_action_ids: tuple[int, ...]) -> tuple[str, ...]:
        names: list[str] = []
        for action_id in available_action_ids:
            name = self.action_name_for_id(action_id)
            if name not in names:
                names.append(name)
        return tuple(names)

    def translate_prefix(self, prefix: tuple[str, ...], context: ActionTranslationContext) -> tuple[V4Action, ...]:
        translated = []
        for token in prefix:
            translated.append(self.translate_token(token, context))
        return tuple(translated)

    def translate_token(self, token: str, context: ActionTranslationContext) -> V4Action:
        normalized = str(token or "").strip().upper()
        if not normalized:
            raise ValueError("unsupported empty action token")
        if normalized in _MOVEMENT_TOKENS:
            return self._primitive_action(_MOVEMENT_TOKENS[normalized], normalized, context)
        if normalized in {"WAIT", "NOOP"}:
            return self._translate_wait_like(normalized, context)
        if normalized.startswith("ACTION") and normalized[6:].isdigit():
            action_id = int(normalized[6:])
            return self._primitive_action(action_id, self.action_name_for_id(action_id), context)
        click_match = _CLICK_PATTERN.match(normalized)
        if click_match is not None:
            return self._translate_click(click_match, context)
        raise ValueError(f"unsupported action token: {token}")

    def _primitive_action(self, action_id: int, action_name: str, context: ActionTranslationContext) -> V4Action:
        if int(action_id) not in set(context.available_action_ids):
            raise ValueError(f"action token not available in current observation: {action_name}")
        return V4Action(action_id=int(action_id), action_name=f"ACTION{int(action_id)}")

    def _translate_wait_like(self, token: str, context: ActionTranslationContext) -> V4Action:
        preferred = 5 if token == "WAIT" else 7
        fallback = 7 if token == "WAIT" else 5
        if preferred in set(context.available_action_ids):
            return V4Action(action_id=preferred, action_name=f"ACTION{preferred}")
        if fallback in set(context.available_action_ids):
            return V4Action(action_id=fallback, action_name=f"ACTION{fallback}")
        raise ValueError(f"{token} is not available in current observation")

    def _translate_click(self, match: re.Match[str], context: ActionTranslationContext) -> V4Action:
        coordinate_action_id = context.coordinate_action_id
        if coordinate_action_id is None or coordinate_action_id not in set(context.available_action_ids):
            raise ValueError("click is not available in current observation")
        if match.group("x") is None or match.group("y") is None:
            if context.coordinate_bounds is not None:
                raise ValueError("click token requires explicit coordinates for this environment")
            return V4Action(action_id=coordinate_action_id, action_name=f"ACTION{coordinate_action_id}")
        payload = {"x": int(match.group("x")), "y": int(match.group("y"))}
        bounds = context.coordinate_bounds
        if bounds is not None:
            min_x, min_y, max_x, max_y = bounds
            if not (min_x <= payload["x"] <= max_x and min_y <= payload["y"] <= max_y):
                raise ValueError("click token is outside coordinate bounds")
        return V4Action(action_id=coordinate_action_id, action_name=f"ACTION{coordinate_action_id}", payload=payload)
