from __future__ import annotations

from collections import deque

from v3_1.config.defaults import DEFAULT_CONFIG
from v3_1.execution.option_execution import action_alias


ACTION_DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

HIGH_CONFIDENCE_THRESHOLD = 0.6
MODERATE_CONFIDENCE_THRESHOLD = 0.35


def _avatar_from_tracker(info: dict) -> tuple[list[int] | None, float, str, bool]:
    avatar = info.get("avatar")
    confidence = float(info.get("avatar_confidence", 0.0) or 0.0)
    source = str(info.get("avatar_source") or "unknown")
    ambiguous = bool(info.get("avatar_ambiguous", False))
    if isinstance(avatar, (list, tuple)) and len(avatar) == 2:
        return [int(float(avatar[0])), int(float(avatar[1]))], confidence, source, ambiguous
    return None, confidence, source, ambiguous


def _terminal_distance(required_action_family: str) -> float:
    execution_cfg = DEFAULT_CONFIG.execution
    if required_action_family == "interact":
        return float(getattr(execution_cfg, "interact_terminal_distance_cells", 0))
    if required_action_family == "click_at":
        return float(getattr(execution_cfg, "click_terminal_distance_cells", 0))
    return float(getattr(execution_cfg, "move_terminal_distance_cells", 0))


def _passable_cells(observation):
    if not isinstance(observation, list):
        return set()
    passable = set()
    for y, row in enumerate(observation):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            if int(value) in {0, 1, 2}:
                passable.add((x, y))
    return passable


def _bounded_bfs_next_step(*, avatar: tuple[int, int], target: tuple[int, int], observation, max_depth: int = 12):
    passable = _passable_cells(observation)
    if not passable:
        return None
    queue = deque([(avatar, [])])
    seen = {avatar}
    while queue:
        cell, path = queue.popleft()
        if len(path) > max_depth:
            continue
        if cell == target and path:
            return path[0]
        for action_name, (dx, dy) in ACTION_DELTAS.items():
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in seen or nxt not in passable:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [action_name]))
    return None


def route_instruction(decision_action: dict | None, *, current_observation, info: dict | None = None) -> dict | None:
    if decision_action is None:
        return None
    info = dict(info or {})
    action_type = str(decision_action.get("type") or "").lower()
    if action_type in {"hold_position", "position_only"}:
        return {"terminal": True, "stop": True, "distance": 0.0, "target_reached": True}
    avatar_mode_status = str(info.get("avatar_mode_status") or "unknown")
    avatar_status = str(info.get("avatar_status") or "unknown")
    target = decision_action.get("centroid")
    avatar, avatar_confidence, avatar_source, avatar_ambiguous = _avatar_from_tracker(info)
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        return {"failed": True, "failure_reason": "missing_target"}
    if avatar_mode_status != "movement_avatar" or avatar_status != "present":
        return {
            "failed": True,
            "failure_reason": "avatar_mode_unsupported",
            "avatar_mode_status": avatar_mode_status,
            "avatar_status": avatar_status,
        }
    if not isinstance(avatar, (list, tuple)) or len(avatar) != 2:
        return {"failed": True, "failure_reason": "missing_avatar"}
    if avatar_confidence < MODERATE_CONFIDENCE_THRESHOLD:
        return {
            "failed": True,
            "failure_reason": "avatar_localization_low_confidence",
            "avatar": list(avatar),
            "avatar_confidence": avatar_confidence,
            "avatar_source": avatar_source,
            "avatar_ambiguous": avatar_ambiguous,
        }
    required_action_family = str(decision_action.get("required_action_family") or "unknown").lower()
    click_target_coordinates = decision_action.get("click_target_coordinates") or decision_action.get("coordinates") or target
    tx, ty = int(float(target[0])), int(float(target[1]))
    ax, ay = int(float(avatar[0])), int(float(avatar[1]))
    current_distance = abs(tx - ax) + abs(ty - ay)
    if required_action_family == "click_at":
        return {
            "terminal": True,
            "desired_action_name": "click_at",
            "click_target_coordinates": [float(click_target_coordinates[0]), float(click_target_coordinates[1])] if isinstance(click_target_coordinates, (list, tuple)) and len(click_target_coordinates) == 2 else None,
            "distance": current_distance,
            "target_reached": current_distance <= _terminal_distance(required_action_family),
            "avatar_confidence": avatar_confidence,
            "avatar_source": avatar_source,
            "avatar_ambiguous": avatar_ambiguous,
        }
    if current_distance <= _terminal_distance(required_action_family):
        action_type = str(decision_action.get("type", "")).lower()
        if required_action_family == "move" or action_type in {"hold_position", "position_only"}:
            return {"terminal": True, "stop": True, "distance": 0.0, "target_reached": True}
        desired = "interact" if required_action_family == "interact" else None
        if desired is None:
            return {"terminal": True, "stop": True, "distance": 0.0, "target_reached": True}
        return {"terminal": True, "desired_action_name": desired, "distance": 0.0, "target_reached": True}

    available_actions = list(info.get("available_actions", []))
    reducing = []
    for action in available_actions:
        alias = action_alias(action)
        if alias not in ACTION_DELTAS:
            continue
        dx, dy = ACTION_DELTAS[alias]
        next_distance = abs(tx - (ax + dx)) + abs(ty - (ay + dy))
        if next_distance < current_distance:
            reducing.append((next_distance, alias))
    if reducing:
        reducing.sort(key=lambda row: (row[0], row[1]))
        return {
            "desired_action_name": reducing[0][1],
            "distance": current_distance,
            "target_centroid": [float(tx), float(ty)],
            "avatar": [float(ax), float(ay)],
            "movement": True,
            "required_action_family": required_action_family,
            "routing_mode": "greedy_reducing",
            "avatar_confidence": avatar_confidence,
            "avatar_source": avatar_source,
            "avatar_ambiguous": avatar_ambiguous,
        }

    next_step = _bounded_bfs_next_step(avatar=(ax, ay), target=(tx, ty), observation=current_observation)
    if next_step is not None:
        return {
            "desired_action_name": next_step,
            "distance": current_distance,
            "target_centroid": [float(tx), float(ty)],
            "avatar": [float(ax), float(ay)],
            "movement": True,
            "required_action_family": required_action_family,
            "routing_mode": "bounded_bfs",
            "temporary_distance_increase_allowed": True,
            "avatar_confidence": avatar_confidence,
            "avatar_source": avatar_source,
            "avatar_ambiguous": avatar_ambiguous,
        }
    return {
        "failed": True,
        "failure_reason": "blocked" if available_actions else "unreachable",
        "distance": current_distance,
        "target_centroid": [float(tx), float(ty)],
        "avatar": [float(ax), float(ay)],
        "avatar_confidence": avatar_confidence,
        "avatar_source": avatar_source,
        "avatar_ambiguous": avatar_ambiguous,
    }
