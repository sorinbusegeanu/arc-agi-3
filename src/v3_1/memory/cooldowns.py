from __future__ import annotations


def advance_cooldowns(cooldowns: dict[str, dict | int]) -> dict[str, dict]:
    next_state: dict[str, dict] = {}
    for key, value in cooldowns.items():
        if isinstance(value, int):
            remaining = value - 1
            payload = {"remaining_rounds": remaining, "scope": "candidate", "reason": "legacy"}
        else:
            payload = dict(value)
            remaining = int(payload.get("remaining_rounds", 0)) - 1
            payload["remaining_rounds"] = remaining
        if remaining > 0:
            next_state[key] = payload
    return next_state


def apply_failure_cooldowns(
    cooldowns: dict[str, dict | int],
    *,
    candidate_id: str | None,
    target_entity_id: str | None,
    target_area_id: str | None,
    cooldown_rounds: int,
    reason: str,
) -> dict[str, dict]:
    next_state = advance_cooldowns(cooldowns)
    rows = [
        (candidate_id, "candidate", cooldown_rounds),
        (target_entity_id, "target", cooldown_rounds + 1 if target_entity_id else cooldown_rounds),
        (target_area_id, "area", cooldown_rounds if target_area_id else cooldown_rounds),
    ]
    for key, scope, ttl in rows:
        if not key:
            continue
        next_state[str(key)] = {
            "remaining_rounds": max(1, ttl),
            "scope": scope,
            "reason": reason,
        }
    return next_state
