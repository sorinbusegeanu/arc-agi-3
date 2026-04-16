from __future__ import annotations

from v5_0.contracts.avatar_types import ProbeTransitionRecord


def score_repeated_edge_band_changes(
    transitions: tuple[ProbeTransitionRecord, ...],
    edge_band_components: tuple[dict[str, object], ...],
) -> dict[str, dict[str, float]]:
    world_change_steps = _world_change_steps(transitions)
    frame_h, frame_w = _frame_shape(transitions)
    scores: dict[str, dict[str, float]] = {}
    for item in edge_band_components:
        region_id = str(item["hud_region_id"])
        seen_steps = tuple(int(v) for v in item.get("seen_step_indices", ()))
        changed_steps = tuple(int(v) for v in item.get("change_step_indices", ()))
        distinct_seen = len(set(seen_steps))
        distinct_changed = len(set(changed_steps))
        seen_ratio = min(1.0, distinct_seen / max(1.0, float(len(transitions))))
        changed_ratio = min(1.0, distinct_changed / max(1.0, float(len(transitions))))
        edge_lock = max(0.0, min(1.0, float(item.get("edge_locked_fraction", 0.0))))
        positional_stability = _edge_positional_stability(item.get("bbox"), str(item.get("edge_side", "")), frame_w, frame_h)
        same_side_stability = _same_side_stability(item.get("bbox"), str(item.get("edge_side", "")), frame_w, frame_h)

        persistence_score = max(
            0.0,
            min(
                1.0,
                0.45 * seen_ratio + 0.25 * same_side_stability + 0.20 * positional_stability + 0.10 * edge_lock,
            ),
        )

        recurrence = 0.0
        if distinct_seen > 0:
            recurrence = min(1.0, distinct_changed / float(distinct_seen))
        coupled = sum(1 for step in set(changed_steps) if step in world_change_steps)
        coupled_ratio = coupled / max(1, distinct_changed)
        change_repeat_score = max(
            0.0,
            min(
                1.0,
                0.55 * changed_ratio + 0.35 * recurrence + 0.10 * edge_lock - 0.15 * coupled_ratio,
            ),
        )
        stability_score = max(
            0.0,
            min(
                1.0,
                0.72 * persistence_score + 0.18 * edge_lock + 0.07 * positional_stability + 0.03 * change_repeat_score,
            ),
        )
        scores[region_id] = {
            "stability_score": float(stability_score),
            "persistence_score": float(persistence_score),
            "change_repeat_score": float(change_repeat_score),
            "positional_stability": float(positional_stability),
            "same_side_stability": float(same_side_stability),
            "edge_lock_score": float(edge_lock),
            "world_coupling_ratio": float(coupled_ratio),
        }
    return scores


def _frame_shape(transitions: tuple[ProbeTransitionRecord, ...]) -> tuple[int, int]:
    for record in transitions:
        frame = record.pre_frame if record.pre_frame is not None else record.post_frame
        if frame is not None:
            return len(frame), (len(frame[0]) if frame else 0)
    return (0, 0)


def _edge_positional_stability(bbox, edge_side: str, width: int, height: int) -> float:
    if bbox is None or width <= 0 or height <= 0:
        return 0.0
    x0, y0, x1, y1 = bbox
    if edge_side == "top":
        dist = float(y0)
    elif edge_side == "bottom":
        dist = float((height - 1) - y1)
    elif edge_side == "left":
        dist = float(x0)
    elif edge_side == "right":
        dist = float((width - 1) - x1)
    else:
        return 0.0
    return max(0.0, min(1.0, 1.0 - dist / 6.0))


def _same_side_stability(bbox, edge_side: str, width: int, height: int) -> float:
    if bbox is None or width <= 0 or height <= 0:
        return 0.0
    x0, y0, x1, y1 = bbox
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    if edge_side in {"top", "bottom"}:
        thickness = h
        span = w / max(1.0, float(width))
    else:
        thickness = w
        span = h / max(1.0, float(height))
    thickness_score = max(0.0, min(1.0, 1.0 - (thickness - 1.0) / 8.0))
    span_score = max(0.0, min(1.0, 1.0 - max(0.0, span - 0.45)))
    return max(0.0, min(1.0, 0.65 * thickness_score + 0.35 * span_score))


def _world_change_steps(transitions: tuple[ProbeTransitionRecord, ...]) -> set[int]:
    steps: set[int] = set()
    for record in transitions:
        if record.pre_frame is None or record.post_frame is None:
            continue
        if _any_world_change(record.pre_frame, record.post_frame):
            steps.add(int(record.step_index))
    return steps


def _any_world_change(pre: tuple[tuple[int, ...], ...], post: tuple[tuple[int, ...], ...], edge_band: int = 1) -> bool:
    h = min(len(pre), len(post))
    w = min(len(pre[0]), len(post[0])) if h else 0
    if h == 0 or w == 0:
        return False
    for y in range(h):
        for x in range(w):
            if int(pre[y][x]) == int(post[y][x]):
                continue
            if edge_band <= x < (w - edge_band) and edge_band <= y < (h - edge_band):
                return True
    return False
