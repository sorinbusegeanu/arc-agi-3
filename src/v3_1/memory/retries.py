from __future__ import annotations


def aggregate_retry_patterns(retries: dict[str, dict]) -> dict[str, dict]:
    aggregates: dict[str, dict] = {}
    for key, row in retries.items():
        scope = str(row.get("scope", "candidate"))
        pattern_key = f"{scope}:{key}"
        aggregates[pattern_key] = {
            "pattern_key": pattern_key,
            "scope": scope,
            "attempts": int(row.get("attempts", 0)),
            "failures": int(row.get("failures", 0)),
            "recent_failures": int(row.get("recent_failures", 0)),
            "reasons": dict(row.get("reasons", {})),
        }
    return aggregates


def update_retry_ledgers(retries: dict[str, dict], *, candidate_id: str | None, target_entity_id: str | None, target_area_id: str | None, success: bool, termination_reason: str | None) -> dict[str, dict]:
    next_state = {key: dict(value) for key, value in retries.items()}
    if success:
        for key in (candidate_id, target_entity_id, target_area_id):
            if key and key in next_state:
                next_state[key]["recent_failures"] = 0
        return next_state
    for key, scope in ((candidate_id, "candidate"), (target_entity_id, "target"), (target_area_id, "area")):
        if not key:
            continue
        row = dict(next_state.get(str(key), {"scope": scope, "attempts": 0, "failures": 0, "recent_failures": 0, "reasons": {}}))
        row["attempts"] = int(row.get("attempts", 0)) + 1
        row["failures"] = int(row.get("failures", 0)) + 1
        row["recent_failures"] = int(row.get("recent_failures", 0)) + 1
        reasons = dict(row.get("reasons", {}))
        if termination_reason:
            reasons[str(termination_reason)] = int(reasons.get(str(termination_reason), 0)) + 1
        row["reasons"] = reasons
        next_state[str(key)] = row
    return next_state
