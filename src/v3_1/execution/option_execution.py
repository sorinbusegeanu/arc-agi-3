from __future__ import annotations

import random


MOVEMENT_ALIASES = {
    "action1": "up",
    "action2": "down",
    "action3": "left",
    "action4": "right",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "interact": "interact",
    "action5": "interact",
    "hold": "hold",
    "stop": "hold",
}


def _action_name(action_row: dict) -> str:
    return str(action_row.get("name", "")).lower()


def action_alias(action_row: dict) -> str:
    return MOVEMENT_ALIASES.get(_action_name(action_row), _action_name(action_row))


def _match_action(available_actions, *names: str):
    lowered = {name.lower() for name in names}
    for action in available_actions:
        if action_alias(action) in lowered or _action_name(action) in lowered:
            return action
    return None


def choose_probe_action(available_actions, history: list[object], *, recent_no_change_actions: list[str] | None = None, rng: random.Random | None = None) -> object:
    if not available_actions:
        raise RuntimeError("probe policy requires at least one available action")
    rng = rng or random.Random()
    recent_no_change_actions = recent_no_change_actions or []
    last_alias = action_alias(history[-1]) if history and isinstance(history[-1], dict) else None
    prev_alias = action_alias(history[-2]) if len(history) >= 2 and isinstance(history[-2], dict) else None
    weights = []
    for action in available_actions:
        alias = action_alias(action)
        weight = 1.0
        if alias == last_alias:
            weight *= 0.25
        if len(history) >= 3 and last_alias is not None and prev_alias is not None:
            older_alias = action_alias(history[-3]) if isinstance(history[-3], dict) else None
            if older_alias == last_alias and alias == prev_alias:
                weight *= 0.3
        if recent_no_change_actions.count(alias) >= 2:
            weight *= 0.2
        seen_in_episode = any(isinstance(prev, dict) and action_alias(prev) == alias for prev in history)
        if not seen_in_episode:
            weight *= 1.5
        weights.append(max(weight, 1e-6))
    return rng.choices(list(available_actions), weights=weights, k=1)[0]


def choose_directed_action(selected_action: dict | None, routed_action: dict | None, available_actions, history: list[object]) -> object:
    del history
    if not available_actions:
        raise RuntimeError("directed execution requires at least one available action")
    if routed_action is not None and routed_action.get("terminal"):
        desired = str(routed_action.get("desired_action_name", "")).lower()
        matched = _match_action(available_actions, desired)
        if matched is not None:
            return matched
        raise RuntimeError(f"directed terminal action unavailable: {desired}")
    if routed_action is not None and routed_action.get("movement"):
        desired = str(routed_action.get("desired_action_name", "")).lower()
        matched = _match_action(available_actions, desired)
        if matched is not None:
            return matched
        raise RuntimeError(f"directed movement action unavailable: {desired}")
    if selected_action is not None:
        action_type = str(selected_action.get("type", "")).lower()
        if action_type in {"interact", "inspect", "inspect_local"}:
            matched = _match_action(available_actions, "interact")
            if matched is not None:
                return matched
            raise RuntimeError("interact action unavailable for directed execution")
        if action_type in {"hold_position", "position_only"}:
            raise RuntimeError("position-only candidate reached target; no terminal action should be executed")
    raise RuntimeError("directed execution could not resolve an action")
