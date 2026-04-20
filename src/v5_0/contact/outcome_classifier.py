from __future__ import annotations

from v5_0.contact.frame_tracker import detect_new_object_appeared, detect_object_removed
from v5_0.contracts.avatar_types import ContactOutcome


def classify_contact_outcome(
    tested_poi_result,
    initial_poi_candidate,
    selected_avatar,
) -> ContactOutcome:
    steps = tuple(tested_poi_result.steps)
    if not steps:
        return ContactOutcome(
            outcome_type="no_effect",
            confidence=0.2,
            contact_step_index=None,
            screen_change_step_indices=(),
            reward_change_step_indices=(),
            object_removed=False,
            new_object_appeared=False,
            level_transition=False,
            terminal=False,
            hud_change_only=False,
            notes=("no_steps",),
        )

    screen_changes = tuple(step.step_index for step in steps if step.screen_changed)
    reward_changes = tuple(
        step.step_index
        for step in steps
        if step.reward_before is not None and step.reward_after is not None and step.reward_before != step.reward_after
    )
    level_transition = any(step.levels_completed_after > step.levels_completed_before for step in steps)
    terminal = any(step.terminal for step in steps)
    hud_only = bool(screen_changes) and all(step.hud_changed_only for step in steps if step.screen_changed)
    contact_step = next((step.step_index for step in steps if step.contact_detected), None)

    object_removed = False
    new_object_appeared = False
    for step in steps:
        object_removed = object_removed or detect_object_removed(
            step.poi_bbox_before,
            step.poi_bbox_after,
            bool(step.contact_detected),
        )
        new_object_appeared = new_object_appeared or detect_new_object_appeared(
            step.pre_frame,
            step.post_frame,
            step.avatar_bbox_after,
        )

    if level_transition:
        outcome = "level_transition"
        confidence = 1.0
    elif terminal:
        outcome = "terminal"
        confidence = 0.95
    elif reward_changes:
        outcome = "reward_change"
        confidence = 0.9
    elif object_removed:
        outcome = "object_removed"
        confidence = 0.8
    elif new_object_appeared:
        outcome = "new_object_appeared"
        confidence = 0.75
    elif _door_opens(steps):
        outcome = "door_opens"
        confidence = 0.7
    elif hud_only:
        outcome = "hud_change_only"
        confidence = 0.6
    else:
        outcome = "no_effect"
        confidence = 0.4

    return ContactOutcome(
        outcome_type=outcome,
        confidence=float(confidence),
        contact_step_index=contact_step,
        screen_change_step_indices=screen_changes,
        reward_change_step_indices=reward_changes,
        object_removed=bool(object_removed),
        new_object_appeared=bool(new_object_appeared),
        level_transition=bool(level_transition),
        terminal=bool(terminal),
        hud_change_only=bool(hud_only),
        notes=(),
    )


def _door_opens(steps) -> bool:
    for step in steps:
        if step.contact_detected and step.screen_changed and not step.hud_changed_only:
            if _has_opening_like_topology_change(step):
                return True
    return False


def _has_opening_like_topology_change(step) -> bool:
    pre = step.pre_frame
    post = step.post_frame
    if pre is None or post is None:
        return False
    h = min(len(pre), len(post))
    w = min(len(pre[0]), len(post[0])) if h else 0
    if h == 0 or w == 0:
        return False
    pre_background = _dominant_value(pre)
    post_background = _dominant_value(post)
    changed_to_background = 0
    changed_cells = 0
    for y in range(h):
        for x in range(w):
            pre_v = int(pre[y][x])
            post_v = int(post[y][x])
            if pre_v == post_v:
                continue
            changed_cells += 1
            if pre_v != pre_background and post_v == post_background:
                changed_to_background += 1
    if changed_cells <= 0:
        return False
    if changed_to_background < 2:
        return False
    return (changed_to_background / changed_cells) >= 0.35


def _dominant_value(frame: tuple[tuple[int, ...], ...]) -> int:
    counts: dict[int, int] = {}
    for row in frame:
        for value in row:
            key = int(value)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 0
    return max(sorted(counts), key=lambda key: counts[key])


def is_useful_world_change(outcome: ContactOutcome) -> bool:
    outcome_type = str(getattr(outcome, "outcome_type", ""))
    return outcome_type in {
        "reward_change",
        "object_removed",
        "door_opens",
        "level_transition",
        "terminal_success",
    }


def build_route_hint_from_contact_outcome(partial_result, outcome) -> dict[str, object]:
    policy = getattr(partial_result, "policy", None)
    actions = tuple(str(item) for item in tuple(getattr(policy, "planned_actions", ())))
    route_id = str(getattr(policy, "policy_id", ""))
    contact_step_index = getattr(outcome, "contact_step_index", None)
    outcome_type = str(getattr(outcome, "outcome_type", "no_effect"))
    world_change_reached = bool(
        outcome_type in {"reward_change", "object_removed", "door_opens", "level_transition", "terminal", "terminal_success"}
        or bool(getattr(outcome, "new_object_appeared", False))
    )
    screen_steps = tuple(int(v) for v in tuple(getattr(outcome, "screen_change_step_indices", ())))
    reward_steps = tuple(int(v) for v in tuple(getattr(outcome, "reward_change_step_indices", ())))
    useful_world_steps = tuple(sorted(set(screen_steps + reward_steps)))
    step_count = len(tuple(getattr(partial_result, "steps", ())))
    if contact_step_index is not None:
        stop_step_index = int(contact_step_index)
        suggested_prefix_length = int(contact_step_index) + 1
        best_post_frame_index = int(contact_step_index)
    elif world_change_reached and useful_world_steps:
        stop_step_index = int(useful_world_steps[0])
        suggested_prefix_length = int(useful_world_steps[0]) + 1
        best_post_frame_index = int(useful_world_steps[0])
    else:
        stop_step_index = (len(actions) - 1) if actions else None
        suggested_prefix_length = len(actions)
        best_post_frame_index = (step_count - 1) if step_count > 0 else None
    steps = tuple(getattr(partial_result, "steps", ()))
    last_step = steps[-1] if steps else None
    last_avatar_bbox = getattr(partial_result, "final_avatar_bbox", None)
    last_poi_bbox = getattr(partial_result, "final_poi_bbox", None)
    if last_avatar_bbox is None and last_step is not None:
        last_avatar_bbox = getattr(last_step, "avatar_bbox_after", None)
    if last_poi_bbox is None and last_step is not None:
        last_poi_bbox = getattr(last_step, "poi_bbox_after", None)
    return {
        "route_id": route_id,
        "actions": actions,
        "length": int(len(actions)),
        "contact_reached": bool(contact_step_index is not None),
        "world_change_reached": bool(world_change_reached),
        "outcome_type": outcome_type,
        "stop_step_index": stop_step_index,
        "start_avatar_bbox": getattr(partial_result, "initial_avatar_bbox", None),
        "start_poi_bbox": getattr(partial_result, "initial_poi_bbox", None),
        "last_avatar_bbox": last_avatar_bbox,
        "last_poi_bbox": last_poi_bbox,
        "suggested_prefix_length": int(max(0, suggested_prefix_length)),
        "best_post_frame_index": best_post_frame_index,
    }
