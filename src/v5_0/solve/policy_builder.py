from __future__ import annotations

from dataclasses import replace

from v5_0.route.trajectory_enumerator import (
    RouteCandidate,
    compute_action_space_delta,
    enumerate_routes_between_points,
    validate_route_actions_for_action_delta,
)


def build_solve_policy_for_target(
    selected_avatar_result,
    target_poi,
    current_frame,
    step_budget_remaining,
) -> tuple[str, ...]:
    candidates = _build_candidate_solve_routes(
        selected_avatar_result=selected_avatar_result,
        target_poi=target_poi,
        current_frame=current_frame,
        step_budget_remaining=step_budget_remaining,
        route_hints=None,
    )
    return tuple(candidates[0].actions) if candidates else tuple()


def build_adaptive_policy_for_target(
    selected_avatar_result,
    target_poi,
    current_frame,
    step_budget_remaining,
    route_hints=None,
) -> tuple[RouteCandidate, ...]:
    return _build_candidate_solve_routes(
        selected_avatar_result=selected_avatar_result,
        target_poi=target_poi,
        current_frame=current_frame,
        step_budget_remaining=step_budget_remaining,
        route_hints=route_hints,
    )


def _build_candidate_solve_routes(
    *,
    selected_avatar_result,
    target_poi,
    current_frame,
    step_budget_remaining,
    route_hints=None,
) -> tuple[RouteCandidate, ...]:
    budget = max(0, int(step_budget_remaining))
    if budget <= 0 or target_poi is None:
        return tuple()

    # The live solver already refreshes avatar geometry before calling the policy
    # builder. Re-tracking here without a histogram can drift back onto the wrong
    # component and collapse the route set on refreshed frontier states.
    avatar_bbox = getattr(selected_avatar_result, "selected_bbox", None)
    avatar_center_value = getattr(selected_avatar_result, "selected_center", None)
    if avatar_center_value is not None:
        avatar_center = tuple(avatar_center_value)
    elif avatar_bbox is not None:
        avatar_center = ((avatar_bbox[0] + avatar_bbox[2]) / 2.0, (avatar_bbox[1] + avatar_bbox[3]) / 2.0)
    else:
        avatar_center = (0.0, 0.0)

    target_center_value = getattr(target_poi, "center", None)
    if target_center_value is not None:
        target_center = tuple(target_center_value)
    elif getattr(target_poi, "bbox", None) is not None:
        bbox = getattr(target_poi, "bbox")
        target_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    else:
        target_center = avatar_center
    avatar_bbox_for_scale = avatar_bbox or getattr(selected_avatar_result, "selected_bbox", None)
    target_bbox_for_scale = getattr(target_poi, "bbox", None)
    dx, dy, step_scale = compute_action_space_delta(
        start_center=avatar_center,
        target_center=target_center,
        start_bbox=avatar_bbox_for_scale,
        target_bbox=target_bbox_for_scale,
    )
    avatar_action_center = (float(avatar_center[0]) / float(step_scale), float(avatar_center[1]) / float(step_scale))
    target_action_center = (float(target_center[0]) / float(step_scale), float(target_center[1]) / float(step_scale))
    shortest = abs(dx) + abs(dy)
    enumerated = _enumerate_interaction_routes(
        start_action_center=avatar_action_center,
        target_action_center=target_action_center,
        center_dx=dx,
        center_dy=dy,
    )
    route_candidates = [
        route
        for route in enumerated
        if route.length > 0 and route.length <= max(int(budget), int(shortest))
        and _route_is_sane(route, dx=int(route.net_dx), dy=int(route.net_dy), budget=max(int(budget), int(shortest)))
    ]
    if int(budget) <= 1 and shortest > int(budget):
        route_candidates = list(_exploratory_routes(budget, dx=dx, dy=dy))
    elif target_center_value is None or (shortest == 0 and getattr(target_poi, "bbox", None) is None):
        route_candidates = list(_exploratory_routes(budget, dx=dx, dy=dy))
    elif shortest > max(int(budget) + 1, 12):
        route_candidates = list(_exploratory_routes(budget, dx=dx, dy=dy))
    if not route_candidates:
        route_candidates = []
    hint_candidates = list(_hint_prefix_routes(route_hints=route_hints, budget=budget, dx=dx, dy=dy))
    combined = tuple(hint_candidates) + tuple(route_candidates)
    if not combined:
        return tuple()
    seen: set[tuple[str, ...]] = set()
    ordered_routes: list[RouteCandidate] = []
    for route in combined:
        key = tuple(route.actions)
        if key in seen:
            continue
        seen.add(key)
        ordered_routes.append(route)
    ordered_routes.sort(
        key=lambda route: (
            _hint_priority(route, route_hints=route_hints),
            0 if str(getattr(route, "route_id", "")).startswith("overlap:") else 1,
            int(route.length),
            int(route.turn_count),
            tuple(route.actions),
        )
    )
    return tuple(ordered_routes)


def _hint_priority(route: RouteCandidate, *, route_hints=None) -> int:
    route_actions = tuple(str(item) for item in tuple(getattr(route, "actions", ())))
    route_id = str(getattr(route, "route_id", ""))
    for hint in tuple(route_hints or ()):
        if not isinstance(hint, dict):
            continue
        hint_actions = tuple(str(item) for item in tuple(hint.get("actions", ())))
        hint_route_id = str(hint.get("route_id", ""))
        if hint_actions and route_actions == hint_actions:
            return 0
        if hint_route_id and (route_id == hint_route_id or route_id.endswith(f":{hint_route_id}")):
            return 1
    return 2


def _hint_prefix_routes(*, route_hints=None, budget: int, dx: int, dy: int) -> tuple[RouteCandidate, ...]:
    out: list[RouteCandidate] = []
    for hint in tuple(route_hints or ()):
        if not isinstance(hint, dict):
            continue
        hint_route_id = str(hint.get("route_id", ""))
        actions = tuple(str(item) for item in tuple(hint.get("actions", ())))
        if not actions:
            continue
        prefix_length = int(hint.get("suggested_prefix_length", len(actions)))
        prefix_actions = actions[: max(0, min(len(actions), prefix_length, int(budget)))]
        if not prefix_actions:
            continue
        candidate = _route_from_actions(
            route_id=(f"hint:{hint_route_id}" if hint_route_id else "hint:route"),
            actions=tuple(prefix_actions),
        )
        if _route_is_sane(candidate, dx=int(dx), dy=int(dy), budget=int(budget)):
            out.append(candidate)
        if int(dx) == 0 and int(dy) == 0 and actions:
            continuation = _route_from_actions(
                route_id=(f"hint:{hint_route_id}:continue" if hint_route_id else "hint:route:continue"),
                actions=(str(actions[-1]),),
            )
            if _route_is_sane(continuation, dx=int(dx), dy=int(dy), budget=int(budget)):
                out.append(continuation)
    return tuple(out)


def _route_from_actions(*, route_id: str, actions: tuple[str, ...]) -> RouteCandidate:
    dx = 0
    dy = 0
    turns = 0
    prev_axis = None
    points = [(0, 0)]
    for action in actions:
        axis = "H" if action in {"LEFT", "RIGHT"} else "V"
        if prev_axis is not None and axis != prev_axis:
            turns += 1
        prev_axis = axis
        if action == "LEFT":
            dx -= 1
        elif action == "RIGHT":
            dx += 1
        elif action == "UP":
            dy -= 1
        elif action == "DOWN":
            dy += 1
        points.append((dx, dy))
    return RouteCandidate(
        route_id=str(route_id),
        actions=tuple(actions),
        length=int(len(actions)),
        net_dx=int(dx),
        net_dy=int(dy),
        first_action=(actions[0] if actions else None),
        turn_count=int(turns),
        axis_order="NONE" if not actions else ("H_ONLY" if all(a in {"LEFT", "RIGHT"} for a in actions) else ("V_ONLY" if all(a in {"UP", "DOWN"} for a in actions) else "MIXED")),
        waypoints=tuple(points),
        score_components={"length": float(len(actions)), "turn_count": float(turns)},
    )


def _enumerate_interaction_routes(
    *,
    start_action_center: tuple[float, float],
    target_action_center: tuple[float, float],
    center_dx: int,
    center_dy: int,
) -> tuple[RouteCandidate, ...]:
    routes: list[RouteCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for mode, target in _interaction_targets(
        start_action_center=start_action_center,
        target_action_center=target_action_center,
        center_dx=center_dx,
        center_dy=center_dy,
    ):
        for route in enumerate_routes_between_points(
            start_center=start_action_center,
            target_center=target,
            max_extra_steps=2,
            max_routes=16,
        ):
            key = tuple(route.actions)
            if key in seen:
                continue
            seen.add(key)
            score = dict(route.score_components)
            score[f"interaction_{mode}"] = 1.0
            routes.append(replace(route, route_id=f"{mode}:{route.route_id}", score_components=score))
    routes.sort(key=lambda route: (0 if str(route.route_id).startswith("overlap:") else 1, int(route.length), int(route.turn_count), tuple(route.actions)))
    return tuple(routes)


def _interaction_targets(
    *,
    start_action_center: tuple[float, float],
    target_action_center: tuple[float, float],
    center_dx: int,
    center_dy: int,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    del start_action_center, center_dx, center_dy
    return (("overlap", target_action_center),)


def _route_is_sane(route: RouteCandidate, *, dx: int, dy: int, budget: int) -> bool:
    hint_source = str(getattr(route, "route_id", "")) if str(getattr(route, "route_id", "")).startswith("hint:") else None
    allow_exploratory = bool(getattr(route, "score_components", {}).get("reason_probe"))
    ok, _ = validate_route_actions_for_action_delta(
        tuple(route.actions),
        dx=dx,
        dy=dy,
        budget=int(budget),
        max_length=14,
        hint_source=hint_source,
        allow_exploratory=allow_exploratory,
    )
    return bool(ok)


def _exploratory_routes(budget: int, *, dx: int = 0, dy: int = 0) -> tuple[RouteCandidate, ...]:
    if int(budget) < 1:
        return tuple()
    horizontal = "RIGHT" if int(dx) >= 0 else "LEFT"
    vertical = "DOWN" if int(dy) >= 0 else "UP"
    probes = [
        (horizontal,),
        (vertical,),
        (horizontal, horizontal),
        (vertical, vertical),
        (horizontal, vertical),
        (vertical, horizontal),
    ]
    out: list[RouteCandidate] = []
    for idx, seq in enumerate(probes):
        actions = tuple(seq[: max(0, min(len(seq), int(budget)))])
        if not actions:
            continue
        if len(actions) > int(budget):
            continue
        route = _route_from_actions(route_id=f"explore_{idx:03d}", actions=actions)
        route.score_components["reason_probe"] = 1.0
        out.append(route)
    out.sort(key=lambda route: (int(route.length), int(route.turn_count), tuple(route.actions)))
    return tuple(out[:1])
