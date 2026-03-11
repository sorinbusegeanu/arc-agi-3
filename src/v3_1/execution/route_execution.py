from __future__ import annotations


def route_instruction(decision_action: dict | None, *, current_observation, info: dict | None = None) -> dict | None:
    if decision_action is None:
        return None
    info = dict(info or {})
    target = decision_action.get("centroid")
    avatar = info.get("avatar")
    if not isinstance(target, (list, tuple)) or len(target) != 2 or not isinstance(avatar, (list, tuple)) and not isinstance(avatar, tuple):
        return {"desired_action_name": "interact" if decision_action.get("type") in {"interact", "inspect"} else "noop"}
    tx, ty = float(target[0]), float(target[1])
    ax, ay = float(avatar[0]), float(avatar[1])
    dx = tx - ax
    dy = ty - ay
    if abs(dx) <= 0.5 and abs(dy) <= 0.5:
        desired = "interact" if decision_action.get("type") != "hold_position" else "noop"
        return {"desired_action_name": desired, "distance": 0.0}
    if abs(dx) >= abs(dy):
        desired = "right" if dx > 0 else "left"
    else:
        desired = "down" if dy > 0 else "up"
    return {
        "desired_action_name": desired,
        "distance": abs(dx) + abs(dy),
        "target_centroid": [tx, ty],
        "avatar": [ax, ay],
    }
