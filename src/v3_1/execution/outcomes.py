from __future__ import annotations


def _avatar_from_observation(observation):
    if not isinstance(observation, list):
        return None
    for y, row in enumerate(observation):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            if value == 1:
                return [x, y]
    return None


def summarize_outcome(*, steps, request, routed_history: list[dict], rewards: list[float]) -> dict:
    avatar_positions = [_avatar_from_observation(step.observation) for step in steps]
    avatar_positions = [position for position in avatar_positions if position is not None]
    initial_distance = None
    final_distance = None
    target = request.metadata.get("target_centroid") if isinstance(request.metadata, dict) else None
    if isinstance(target, (list, tuple)) and len(target) == 2 and avatar_positions:
        initial_distance = abs(float(avatar_positions[0][0]) - float(target[0])) + abs(float(avatar_positions[0][1]) - float(target[1]))
        final_distance = abs(float(avatar_positions[-1][0]) - float(target[0])) + abs(float(avatar_positions[-1][1]) - float(target[1]))
    progress = 0.0
    if initial_distance is not None and final_distance is not None:
        progress = initial_distance - final_distance
    unique_positions = {tuple(position) for position in avatar_positions}
    noop_steps = max(0, sum(1 for previous, current in zip(avatar_positions, avatar_positions[1:]) if previous == current))
    stalled = len(unique_positions) <= 2 and len(avatar_positions) >= 3
    blocked = stalled and progress <= 0.0
    route_success = final_distance is not None and final_distance <= 0.5
    route_failure = bool(request.mode == "directed" and blocked and not route_success)
    termination_reason = "done" if steps and steps[-1].done else "step_budget_exhausted"
    if route_failure:
        termination_reason = "route_failed"
    elif blocked:
        termination_reason = "blocked"
    elif stalled:
        termination_reason = "stalled"
    return {
        "success": bool(steps and steps[-1].done),
        "reward_delta": float(sum(rewards)),
        "termination_reason": termination_reason,
        "progress": progress,
        "blocked": blocked,
        "stalled": stalled,
        "noop_steps": noop_steps,
        "route_success": route_success,
        "route_failure": route_failure,
        "avatar_positions": avatar_positions,
        "routed_actions": routed_history,
        "consequence_summary": {
            "reward_total": float(sum(rewards)),
            "positive_reward_steps": sum(1 for reward in rewards if reward > 0),
            "terminal": bool(steps and steps[-1].done),
        },
    }
