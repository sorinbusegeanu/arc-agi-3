from __future__ import annotations

from v5_0.contracts.avatar_types import POIMechanicState


def classify_poi_mechanics(evidence_items) -> tuple[POIMechanicState, ...]:
    out: list[POIMechanicState] = []
    for item in sorted(tuple(evidence_items or ()), key=lambda v: str(v.poi_id)):
        label = "unknown"
        confidence = 0.2

        if int(item.level_transition_count) >= 2 or int(item.terminal_count) >= 2:
            label = "exit"
            confidence = min(0.95, 0.6 + 0.15 * max(int(item.level_transition_count), int(item.terminal_count)))
        elif int(item.door_opens_count) >= 2 and int(item.level_transition_count) == 0:
            label = "door_or_switch"
            confidence = min(0.9, 0.55 + 0.1 * int(item.door_opens_count))
        elif int(item.reward_change_count) + int(item.object_removed_count) >= 2:
            label = "collectible"
            confidence = min(0.9, 0.5 + 0.12 * (int(item.reward_change_count) + int(item.object_removed_count)))
        elif int(item.no_effect_count) >= 3 and int(item.useful_change_count) == 0:
            label = "decoy"
            confidence = min(0.95, 0.55 + 0.1 * int(item.no_effect_count))
        elif int(item.terminal_count) >= 2 and int(item.useful_change_count) == 0:
            label = "hazard"
            confidence = min(0.9, 0.5 + 0.12 * int(item.terminal_count))

        priority = _priority_score(label, confidence, float(item.hud_match_confidence))
        out.append(
            POIMechanicState(
                poi_id=str(item.poi_id),
                mechanic_label=label,
                confidence=float(max(0.0, min(1.0, confidence))),
                priority_score=float(priority),
                attempt_count=int(item.contact_count),
                success_count=int(item.useful_change_count),
                failure_count=int(item.no_effect_count),
                last_outcome_type=_last_outcome_hint(item),
                active=True,
            )
        )
    return tuple(out)


def _priority_score(label: str, confidence: float, hud_bonus: float) -> float:
    base = {
        "exit": 1.0,
        "target": 0.9,
        "collectible": 0.75,
        "door_or_switch": 0.65,
        "unknown": 0.45,
        "decoy": 0.05,
        "hazard": 0.02,
    }.get(label, 0.3)
    return max(0.0, min(1.0, base * 0.75 + confidence * 0.2 + float(hud_bonus) * 0.05))


def _last_outcome_hint(item) -> str | None:
    if int(item.level_transition_count) > 0:
        return "level_transition"
    if int(item.terminal_count) > 0:
        return "terminal"
    if int(item.door_opens_count) > 0:
        return "door_opens"
    if int(item.object_removed_count) > 0:
        return "object_removed"
    if int(item.reward_change_count) > 0:
        return "reward_change"
    if int(item.no_effect_count) > 0:
        return "no_effect"
    return None
