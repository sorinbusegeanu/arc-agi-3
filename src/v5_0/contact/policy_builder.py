from __future__ import annotations

from v5_0.contracts.avatar_types import ContactPolicy, POICandidate, ProbeTransitionRecord
from v5_0.route.trajectory_enumerator import (
    compute_action_space_delta,
    enumerate_routes_between_points,
    validate_route_actions_for_action_delta,
)


def build_contact_policies_for_poi(
    selected_avatar,
    poi_candidate: POICandidate,
    transitions: tuple[ProbeTransitionRecord, ...],
    episode_index: int,
) -> tuple[ContactPolicy, ...]:
    if _is_border_locked_poi(poi_candidate, transitions=transitions):
        return tuple()

    avatar_center = selected_avatar.selected_center or (0.0, 0.0)
    poi_center = poi_candidate.center
    avatar_bbox = getattr(selected_avatar, "selected_bbox", None)
    poi_bbox = getattr(poi_candidate, "bbox", None)
    dx, dy, step_scale = compute_action_space_delta(
        start_center=avatar_center,
        target_center=poi_center,
        start_bbox=avatar_bbox,
        target_bbox=poi_bbox,
    )
    start_action_center = (float(avatar_center[0]) / float(step_scale), float(avatar_center[1]) / float(step_scale))
    target_action_center = (float(poi_center[0]) / float(step_scale), float(poi_center[1]) / float(step_scale))
    shortest = abs(dx) + abs(dy)
    if shortest > 8:
        exploratory = _bounded_exploration_actions(dx=dx, dy=dy, budget=8)
        if not exploratory:
            return tuple()
        return (
            ContactPolicy(
                policy_id=f"contact:{episode_index}:{poi_candidate.poi_id}:explore_000",
                poi_id=poi_candidate.poi_id,
                episode_index=int(episode_index),
                planned_actions=exploratory,
                max_steps=int(len(exploratory)),
                stop_on_contact=True,
                stop_on_screen_change=False,
                stop_on_terminal=True,
            ),
        )
    policies: list[ContactPolicy] = []
    for mode, route_dx, route_dy, route_target in _interaction_route_targets(
        start_action_center=start_action_center,
        target_action_center=target_action_center,
        center_dx=dx,
        center_dy=dy,
    ):
        routes = enumerate_routes_between_points(
            start_center=start_action_center,
            target_center=route_target,
            max_extra_steps=2,
            max_routes=12,
        )
        for route in routes:
            planned = tuple(route.actions)
            if str(mode) == "touch" and not planned:
                continue
            if not _route_is_valid(planned, dx=route_dx, dy=route_dy):
                continue
            policies.append(
                ContactPolicy(
                    policy_id=f"contact:{episode_index}:{poi_candidate.poi_id}:{mode}:{route.route_id}",
                    poi_id=poi_candidate.poi_id,
                    episode_index=int(episode_index),
                    planned_actions=planned,
                    max_steps=int(len(planned)),
                    stop_on_contact=True,
                    stop_on_screen_change=False,
                    stop_on_terminal=True,
                    interaction_mode=str(mode),
                )
            )
    return tuple(_dedupe_policies_by_actions(policies))


def _interaction_route_targets(
    *,
    start_action_center: tuple[float, float],
    target_action_center: tuple[float, float],
    center_dx: int,
    center_dy: int,
) -> tuple[tuple[str, int, int, tuple[float, float]], ...]:
    # Try adjacent/touch positions first. If touching is not sufficient, the
    # runner will continue to the overlap route that steps onto the POI center.
    offsets = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
    candidates: list[tuple[str, int, int, tuple[float, float]]] = []
    for ox, oy in offsets:
        target = (float(target_action_center[0]) + ox, float(target_action_center[1]) + oy)
        dx = int(round(target[0] - float(start_action_center[0])))
        dy = int(round(target[1] - float(start_action_center[1])))
        if (dx, dy) == (int(center_dx), int(center_dy)):
            continue
        candidates.append(("touch", dx, dy, target))
    candidates.sort(key=lambda item: (abs(item[1]) + abs(item[2]), abs(item[1] - int(center_dx)) + abs(item[2] - int(center_dy)), item[1], item[2]))
    candidates.append(("overlap", int(center_dx), int(center_dy), target_action_center))
    seen: set[tuple[str, int, int]] = set()
    ordered: list[tuple[str, int, int, tuple[float, float]]] = []
    for item in candidates:
        key = (item[0], item[1], item[2])
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return tuple(ordered)


def _dedupe_policies_by_actions(policies: list[ContactPolicy]) -> tuple[ContactPolicy, ...]:
    survivors: dict[tuple[str, ...], ContactPolicy] = {}
    for policy in policies:
        key = tuple(str(item) for item in tuple(getattr(policy, "planned_actions", ())))
        existing = survivors.get(key)
        if existing is None:
            survivors[key] = policy
            continue
        existing_mode = str(getattr(existing, "interaction_mode", "overlap"))
        current_mode = str(getattr(policy, "interaction_mode", "overlap"))
        if existing_mode != "touch" and current_mode == "touch":
            survivors[key] = policy
    return tuple(sorted(survivors.values(), key=lambda item: (0 if str(getattr(item, "interaction_mode", "")) == "touch" else 1, len(tuple(item.planned_actions)), str(item.policy_id))))


def _route_is_valid(actions: tuple[str, ...], *, dx: int, dy: int) -> bool:
    ok, _ = validate_route_actions_for_action_delta(
        tuple(actions),
        dx=dx,
        dy=dy,
        budget=12,
        max_length=12,
        hint_source=None,
        allow_exploratory=False,
    )
    return bool(ok)


def _bounded_exploration_actions(*, dx: int, dy: int, budget: int) -> tuple[str, ...]:
    if int(budget) <= 0:
        return tuple()
    horizontal = "RIGHT" if int(dx) >= 0 else "LEFT"
    vertical = "DOWN" if int(dy) >= 0 else "UP"
    actions: list[str] = []
    if abs(int(dx)) >= abs(int(dy)):
        actions.extend([horizontal] * min(abs(int(dx)), int(budget)))
        remaining = int(budget) - len(actions)
        if remaining > 0:
            actions.extend([vertical] * min(abs(int(dy)), remaining))
    else:
        actions.extend([vertical] * min(abs(int(dy)), int(budget)))
        remaining = int(budget) - len(actions)
        if remaining > 0:
            actions.extend([horizontal] * min(abs(int(dx)), remaining))
    return tuple(str(item) for item in actions[: int(budget)])


def build_contact_policy_for_poi(
    selected_avatar,
    poi_candidate: POICandidate,
    transitions: tuple[ProbeTransitionRecord, ...],
    episode_index: int,
) -> ContactPolicy | None:
    policies = build_contact_policies_for_poi(
        selected_avatar=selected_avatar,
        poi_candidate=poi_candidate,
        transitions=transitions,
        episode_index=episode_index,
    )
    return policies[0] if policies else None


def _is_border_locked_poi(
    poi_candidate: POICandidate,
    *,
    transitions: tuple[ProbeTransitionRecord, ...] = (),
) -> bool:
    if "border_locked" in set(getattr(poi_candidate, "ambiguity_flags", ())):
        return True
    x0, y0, x1, y1 = poi_candidate.bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    strip_like = bw <= 2 or bh <= 2 or bw >= 3 * bh or bh >= 3 * bw
    tiny = int(poi_candidate.area) <= 8
    if not (tiny or strip_like):
        return False
    dims = _latest_frame_dimensions(transitions)
    if dims is None:
        return bool(x0 == 0 or y0 == 0)
    width, height = dims
    if x1 >= int(width) or y1 >= int(height):
        return bool(x0 == 0 or y0 == 0)
    touches_left = x0 <= 0
    touches_top = y0 <= 0
    touches_right = x1 >= (int(width) - 1)
    touches_bottom = y1 >= (int(height) - 1)
    return bool(touches_left or touches_top or touches_right or touches_bottom)


def _latest_frame_dimensions(transitions: tuple[ProbeTransitionRecord, ...]) -> tuple[int, int] | None:
    for item in reversed(tuple(transitions or ())):
        for frame in (getattr(item, "post_frame", None), getattr(item, "pre_frame", None)):
            if frame is None:
                continue
            h = len(frame)
            w = len(frame[0]) if h else 0
            if w > 0 and h > 0:
                return int(w), int(h)
    return None
