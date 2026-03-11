from __future__ import annotations


def _action_name(action_row: dict) -> str:
    return str(action_row.get("name", "")).lower()


def _match_action(available_actions, *names: str):
    lowered = {name.lower() for name in names}
    for action in available_actions:
        name = _action_name(action)
        if name in lowered:
            return action
    return None


def choose_probe_action(available_actions, history: list[object]) -> object:
    preferred_order = ["right", "down", "left", "up", "interact", "noop"]
    recent_names = [_action_name(action) for action in history[-2:] if isinstance(action, dict)]
    for name in preferred_order:
        candidate = _match_action(available_actions, name)
        if candidate is None:
            continue
        if recent_names.count(name) >= 2 and len(available_actions) > 1:
            continue
        return candidate
    return available_actions[0] if available_actions else {"id": 0, "name": "noop"}


def choose_directed_action(selected_action: dict | None, routed_action: dict | None, available_actions, history: list[object]) -> object:
    if routed_action is not None:
        desired = str(routed_action.get("desired_action_name", "")).lower()
        matched = _match_action(available_actions, desired)
        if matched is not None:
            return matched
        if "action_id" in routed_action:
            for action in available_actions:
                if int(action.get("id", -1)) == int(routed_action["action_id"]):
                    return action
    if selected_action is not None:
        explicit = selected_action.get("action")
        if explicit is not None:
            return explicit
        action_type = str(selected_action.get("type", "")).lower()
        if action_type in {"interact", "inspect", "inspect_local"}:
            matched = _match_action(available_actions, "interact")
            if matched is not None:
                return matched
        if action_type in {"hold_position"}:
            matched = _match_action(available_actions, "noop")
            if matched is not None:
                return matched
    return choose_probe_action(available_actions, history)
