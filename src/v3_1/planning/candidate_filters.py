from __future__ import annotations


def filter_candidates(candidates: list[dict], belief: dict) -> tuple[list[dict], list[dict]]:
    cooldowns = dict(belief.get("cooldowns", {}))
    exhausted = set(belief.get("exhausted", set()))
    failed_candidates = dict(belief.get("failed_candidates", {}))
    blocked_targets = {str(row["entity_id"]) for row in belief.get("blocked_targets", [])}
    survivors: list[dict] = []
    blocked_rows: list[dict] = []

    def _cooldown_active(key: str | None) -> bool:
        if not key:
            return False
        row = cooldowns.get(str(key), 0)
        if isinstance(row, dict):
            return int(row.get("remaining_rounds", 0)) > 0
        return int(row) > 0

    for candidate in candidates:
        row = dict(candidate)
        reasons = list(row.get("blocked_reasons", []))
        candidate_id = str(row["candidate_id"])
        target_entity_id = row.get("target_entity_id")

        if candidate_id in exhausted:
            reasons.append("exhausted")
        if _cooldown_active(candidate_id) or _cooldown_active(target_entity_id):
            reasons.append("cooldown")
        if target_entity_id in blocked_targets and not bool(row.get("reachable_later")):
            reasons.append("unreachable")
        if row.get("target_entity_id") is not None and row.get("target_entity_id") not in belief.get("indexes", {}).get("reachable_targets", []) and not row.get("reachable_later"):
            reasons.append("invalid_target")
        if int(failed_candidates.get(candidate_id, 0)) >= 2:
            reasons.append("repeated_failure")
        if int(failed_candidates.get(str(target_entity_id), 0)) >= 2:
            reasons.append("target_repeated_failure")

        row["blocked_reasons"] = sorted(set(reasons))
        row["blocked"] = bool(row["blocked_reasons"])
        if row["blocked"]:
            blocked_rows.append(row)
            continue
        survivors.append(row)
    return survivors, blocked_rows
