from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

MAX_SHORTEST_PATH = 12
MAX_FINAL_ROUTE = 14


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    actions: tuple[str, ...]
    length: int
    net_dx: int
    net_dy: int
    first_action: str | None
    turn_count: int
    axis_order: str
    waypoints: tuple[tuple[int, int], ...]
    score_components: dict[str, float]


def estimate_action_step_scale(primary_bbox, secondary_bbox) -> int:
    sizes = []
    for bbox in (primary_bbox, secondary_bbox):
        if bbox is None:
            continue
        w = max(1, int(bbox[2]) - int(bbox[0]) + 1)
        h = max(1, int(bbox[3]) - int(bbox[1]) + 1)
        sizes.append((w + h) / 2.0)
    if not sizes:
        return 8
    return max(1, int(round(sum(sizes) / len(sizes))))


def compute_action_space_delta(
    *,
    start_center,
    target_center,
    start_bbox=None,
    target_bbox=None,
) -> tuple[int, int, int]:
    step_scale = estimate_action_step_scale(start_bbox, target_bbox)
    sx = float(start_center[0]) if start_center is not None else 0.0
    sy = float(start_center[1]) if start_center is not None else 0.0
    tx = float(target_center[0]) if target_center is not None else sx
    ty = float(target_center[1]) if target_center is not None else sy
    dx = int(round((tx - sx) / float(step_scale)))
    dy = int(round((ty - sy) / float(step_scale)))
    return dx, dy, step_scale


def validate_route_actions_for_action_delta(
    actions: tuple[str, ...],
    *,
    dx: int | None,
    dy: int | None,
    budget: int | None = None,
    max_length: int = MAX_FINAL_ROUTE,
    hint_source: str | None = None,
    allow_exploratory: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    acts = tuple(str(action) for action in tuple(actions))
    if budget is not None and len(acts) > int(budget):
        reasons.append("excessive_length")
    if len(acts) > int(max_length):
        reasons.append("excessive_length")
    if len(acts) >= 2 and (acts[0], acts[1]) in {("LEFT", "RIGHT"), ("RIGHT", "LEFT"), ("UP", "DOWN"), ("DOWN", "UP")}:
        reasons.append("early_cancel_pattern")
    net_dx, net_dy = _net_displacement(acts)
    allow_prefix = bool(hint_source)
    if dx is not None:
        if allow_prefix:
            if int(dx) > 0 and net_dx < 0:
                reasons.append("impossible_displacement")
            if int(dx) < 0 and net_dx > 0:
                reasons.append("impossible_displacement")
            if abs(net_dx) > abs(int(dx)) + 1:
                reasons.append("impossible_displacement")
        elif not allow_exploratory and net_dx != int(dx):
            reasons.append("impossible_displacement")
    if dy is not None:
        if allow_prefix:
            if int(dy) > 0 and net_dy < 0:
                reasons.append("impossible_displacement")
            if int(dy) < 0 and net_dy > 0:
                reasons.append("impossible_displacement")
            if abs(net_dy) > abs(int(dy)) + 1:
                reasons.append("impossible_displacement")
        elif not allow_exploratory and net_dy != int(dy):
            reasons.append("impossible_displacement")
    first = acts[0] if acts else None
    if not allow_prefix and not allow_exploratory:
        if first == "LEFT" and dx is not None and int(dx) > 0:
            reasons.append("anti_target_first_move")
        if first == "RIGHT" and dx is not None and int(dx) < 0:
            reasons.append("anti_target_first_move")
        if first == "UP" and dy is not None and int(dy) > 0:
            reasons.append("anti_target_first_move")
        if first == "DOWN" and dy is not None and int(dy) < 0:
            reasons.append("anti_target_first_move")
    return (len(reasons) == 0, tuple(dict.fromkeys(reasons)))


def enumerate_routes_between_points(
    start_center,
    target_center,
    max_extra_steps=6,
    max_routes=32,
) -> tuple[RouteCandidate, ...]:
    sx = float(start_center[0]) if start_center is not None else 0.0
    sy = float(start_center[1]) if start_center is not None else 0.0
    tx = float(target_center[0]) if target_center is not None else sx
    ty = float(target_center[1]) if target_center is not None else sy

    dx = int(round(tx - sx))
    dy = int(round(ty - sy))
    h_token = "RIGHT" if dx >= 0 else "LEFT"
    v_token = "DOWN" if dy >= 0 else "UP"
    h_steps = abs(dx)
    v_steps = abs(dy)
    shortest = h_steps + v_steps
    if shortest == 0:
        return (
            RouteCandidate(
                route_id="route_000",
                actions=tuple(),
                length=0,
                net_dx=0,
                net_dy=0,
                first_action=None,
                turn_count=0,
                axis_order="NONE",
                waypoints=((0, 0),),
                score_components={"length": 0.0, "turn_count": 0.0, "detour_steps": 0.0},
            ),
        )

    action_sequences: set[tuple[str, ...]] = set()
    base = _enumerate_shortest_interleavings(h_steps=h_steps, v_steps=v_steps, h_token=h_token, v_token=v_token)
    action_sequences.update(base)

    # Geometry-aware bounded generation: shortest first, then tiny near-shortest/one-turn variants.
    if shortest > MAX_SHORTEST_PATH:
        base = tuple(sorted(base, key=lambda seq: (_turn_count(seq), seq))[:4])
    action_sequences.update(base)

    # Keep the enumerator bounded: shortest paths only. The remaining
    # plausibility filters keep the output deterministic and small.
    _ = max(0, min(int(max_extra_steps), 6))

    sorted_actions = sorted(
        (seq for seq in action_sequences if len(seq) <= MAX_FINAL_ROUTE and _is_plausible_route(seq, dx=dx, dy=dy)),
        key=lambda seq: (len(seq), _turn_count(seq), _monotonic_progress_penalty(seq, dx, dy), seq),
    )

    routes: list[RouteCandidate] = []
    route_cap = max(1, min(int(max_routes), 32))
    for index, actions in enumerate(sorted_actions[: route_cap]):
        net_dx, net_dy = _net_displacement(actions)
        route = RouteCandidate(
            route_id=f"route_{index:03d}",
            actions=actions,
            length=len(actions),
            net_dx=net_dx,
            net_dy=net_dy,
            first_action=(actions[0] if actions else None),
            turn_count=_turn_count(actions),
            axis_order=_axis_order(actions),
            waypoints=_waypoints(actions),
            score_components={
                "length": float(len(actions)),
                "turn_count": float(_turn_count(actions)),
                "detour_steps": float(max(0, len(actions) - shortest)),
                "monotonic_penalty": float(_monotonic_progress_penalty(actions, dx, dy)),
            },
        )
        routes.append(route)
    return tuple(routes)


def _enumerate_shortest_interleavings(*, h_steps: int, v_steps: int, h_token: str, v_token: str) -> tuple[tuple[str, ...], ...]:
    if h_steps == 0 and v_steps == 0:
        return (tuple(),)
    if h_steps == 0:
        return (tuple(v_token for _ in range(v_steps)),)
    if v_steps == 0:
        return (tuple(h_token for _ in range(h_steps)),)
    total = h_steps + v_steps
    out: list[tuple[str, ...]] = []
    for h_positions in combinations(range(total), h_steps):
        h_pos = set(h_positions)
        seq = tuple(h_token if i in h_pos else v_token for i in range(total))
        out.append(seq)
    out.sort()
    return tuple(out)


def _turn_count(actions: tuple[str, ...]) -> int:
    turns = 0
    prev_axis = None
    for action in actions:
        axis = "H" if action in {"LEFT", "RIGHT"} else "V"
        if prev_axis is not None and axis != prev_axis:
            turns += 1
        prev_axis = axis
    return turns


def _axis_order(actions: tuple[str, ...]) -> str:
    if not actions:
        return "NONE"
    axes = ["H" if action in {"LEFT", "RIGHT"} else "V" for action in actions]
    unique = tuple(dict.fromkeys(axes))
    if unique == ("H",):
        return "H_ONLY"
    if unique == ("V",):
        return "V_ONLY"
    if unique[:2] == ("H", "V") and all(a == "H" for a in axes[: axes.index("V")]):
        return "H_FIRST"
    if unique[:2] == ("V", "H") and all(a == "V" for a in axes[: axes.index("H")]):
        return "V_FIRST"
    return "MIXED"


def _waypoints(actions: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    x = 0
    y = 0
    points: list[tuple[int, int]] = [(x, y)]
    for action in actions:
        if action == "LEFT":
            x -= 1
        elif action == "RIGHT":
            x += 1
        elif action == "UP":
            y -= 1
        elif action == "DOWN":
            y += 1
        points.append((x, y))
    return tuple(points)


def _net_displacement(actions: tuple[str, ...]) -> tuple[int, int]:
    x = 0
    y = 0
    for action in actions:
        if action == "LEFT":
            x -= 1
        elif action == "RIGHT":
            x += 1
        elif action == "UP":
            y -= 1
        elif action == "DOWN":
            y += 1
    return x, y


def _is_plausible_route(actions: tuple[str, ...], *, dx: int, dy: int) -> bool:
    if not actions:
        return dx == 0 and dy == 0
    if len(actions) > MAX_FINAL_ROUTE:
        return False
    net = _net_displacement(actions)
    if net != (dx, dy):
        return False
    # avoid anti-target first move unless no movement in that axis
    first = actions[0]
    if dx > 0 and first == "LEFT":
        return False
    if dx < 0 and first == "RIGHT":
        return False
    if dy > 0 and first == "UP":
        return False
    if dy < 0 and first == "DOWN":
        return False
    # avoid early oscillation
    if len(actions) >= 2:
        a0, a1 = actions[0], actions[1]
        if (a0, a1) in {("LEFT", "RIGHT"), ("RIGHT", "LEFT"), ("UP", "DOWN"), ("DOWN", "UP")}:
            return False
    return True


def _monotonic_progress_penalty(actions: tuple[str, ...], dx: int, dy: int) -> int:
    x = 0
    y = 0
    goal = abs(dx) + abs(dy)
    penalty = 0
    best = goal
    for action in actions:
        if action == "LEFT":
            x -= 1
        elif action == "RIGHT":
            x += 1
        elif action == "UP":
            y -= 1
        elif action == "DOWN":
            y += 1
        dist = abs(dx - x) + abs(dy - y)
        if dist > best:
            penalty += 1
        best = min(best, dist)
    return penalty
