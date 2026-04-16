from __future__ import annotations

from collections import Counter

from v5_0.contracts.avatar_types import MechanicDecision


def select_target_from_mechanic_memory(
    mechanic_memory,
    hud_targeting_report,
    ranked_poi_candidates,
) -> MechanicDecision:
    if mechanic_memory is None:
        return MechanicDecision(None, ("no_mechanic_memory",), 0.0, False)

    retired = set(str(v) for v in tuple(getattr(mechanic_memory, "retired_poi_ids", ())))
    states_by_id = {str(item.poi_id): item for item in tuple(getattr(mechanic_memory, "poi_states", ()))}
    candidate_ids = [str(item.poi_id) for item in tuple(ranked_poi_candidates or ()) if str(item.poi_id) not in retired]
    if not candidate_ids:
        return MechanicDecision(None, ("no_active_candidate",), 0.0, True)

    hud_selected = None
    hud_ambiguous = True
    if hud_targeting_report is not None:
        selected = getattr(hud_targeting_report, "selected", None)
        hud_selected = getattr(selected, "selected_poi_id", None) if selected is not None else None
        hud_ambiguous = bool(getattr(selected, "ambiguous", True)) if selected is not None else True

    scored = []
    for idx, poi_id in enumerate(candidate_ids):
        state = states_by_id.get(poi_id)
        label = str(getattr(state, "mechanic_label", "unknown")) if state is not None else "unknown"
        if label == "decoy":
            continue
        if label == "hazard":
            # Allow only when nothing safer remains.
            safer_exists = any(
                str(getattr(states_by_id.get(other), "mechanic_label", "unknown")) not in {"hazard", "decoy"}
                for other in candidate_ids
            )
            if safer_exists:
                continue
        base = _label_priority(label)
        conf = float(getattr(state, "confidence", 0.0)) if state is not None else 0.0
        memory_priority = float(getattr(state, "priority_score", 0.0)) if state is not None else 0.0
        hud_bonus = 0.0
        if hud_selected is not None and str(hud_selected) == poi_id:
            hud_bonus = 0.15 if not hud_ambiguous else 0.07
            if label == "decoy":
                hud_bonus = -0.3
        score = 0.5 * base + 0.3 * memory_priority + 0.15 * conf + hud_bonus
        scored.append((score, idx, poi_id, label))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not scored:
        return MechanicDecision(None, ("all_candidates_filtered",), 0.0, True)

    best = scored[0]
    reason = [f"label:{best[3]}"]
    if hud_selected is not None and str(hud_selected) == best[2]:
        reason.append("hud_supported")
    return MechanicDecision(
        selected_poi_id=str(best[2]),
        reason_codes=tuple(reason),
        confidence=float(max(0.0, min(1.0, best[0]))),
        retarget_required=False,
    )


def update_target_after_step(
    current_target_id,
    solve_episode_history,
    blocked_streak,
) -> MechanicDecision:
    if current_target_id is None:
        return MechanicDecision(None, ("no_current_target",), 0.0, True)
    history = tuple(solve_episode_history or ())
    target_steps = [s for s in history if str(getattr(s, "target_poi_id", "")) == str(current_target_id)]
    no_effect_streak = _tail_count(target_steps, lambda s: str(getattr(s, "outcome_type", "")) in {"no_effect", "hud_change_only"})
    useful_streak = _tail_count(target_steps, lambda s: str(getattr(s, "outcome_type", "")) in {"reward_change", "object_removed", "new_object_appeared", "door_opens", "level_transition"})
    if useful_streak >= 1:
        return MechanicDecision(str(current_target_id), ("useful_progress",), 0.8, False)
    if no_effect_streak >= 3:
        return MechanicDecision(str(current_target_id), ("repeated_no_effect",), 0.9, True)
    if int(blocked_streak) >= 2:
        return MechanicDecision(str(current_target_id), ("repeated_block",), 0.8, True)
    return MechanicDecision(str(current_target_id), ("continue",), 0.5, False)


def _label_priority(label: str) -> float:
    return {
        "exit": 1.0,
        "target": 0.9,
        "collectible": 0.75,
        "door_or_switch": 0.65,
        "unknown": 0.5,
        "hazard": 0.1,
        "decoy": 0.0,
    }.get(label, 0.4)


def _tail_count(items, predicate) -> int:
    count = 0
    for item in reversed(tuple(items)):
        if not predicate(item):
            break
        count += 1
    return count
