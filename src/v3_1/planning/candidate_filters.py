from __future__ import annotations

FILTER_REASON_TAXONOMY = {
    "hard_cooldown_active": {"severity": "hard", "provenance": "memory"},
    "hard_exhausted_scope": {"severity": "hard", "provenance": "memory"},
    "hard_unreachable_target": {"severity": "hard", "provenance": "blackboard"},
    "hard_invalid_target": {"severity": "hard", "provenance": "blackboard"},
    "hard_contradiction_current_evidence": {"severity": "hard", "provenance": "blackboard"},
    "soft_repeated_failure": {"severity": "soft", "provenance": "memory"},
    "soft_target_repeated_failure": {"severity": "soft", "provenance": "memory"},
    "soft_local_class_repeat": {"severity": "soft", "provenance": "local_context"},
    "soft_local_target_repeat": {"severity": "soft", "provenance": "local_context"},
    "soft_route_repeat": {"severity": "soft", "provenance": "local_context"},
    "soft_trigger_repeat": {"severity": "soft", "provenance": "local_context"},
    "soft_area_repeated_failure": {"severity": "soft", "provenance": "local_context"},
    "soft_stale_support_decay": {"severity": "soft", "provenance": "blackboard"},
    "soft_uncertain_contradiction": {"severity": "soft", "provenance": "blackboard"},
}


def filter_candidates(candidates: list[dict], belief: dict) -> tuple[list[dict], list[dict]]:
    cooldowns = dict(belief.get("cooldowns", {}))
    exhaustion_map = dict(belief.get("exhaustion_map", {}))
    exhausted = set(belief.get("exhausted_keys", belief.get("exhausted", set())))
    failed_candidates = dict(belief.get("failed_candidates", {}))
    blocked_targets = {str(row["entity_id"]) for row in belief.get("blocked_targets", []) if row.get("entity_id") is not None}
    localized_context = dict(belief.get("localized_context", {}))
    local_context = dict(belief.get("local_context", {}))
    evidence_index = dict(belief.get("indexes", {}).get("evidence_index", {}))
    prior_failure_success_context = dict(belief.get("prior_failure_success_context", {}))
    candidate_counts_before: dict[str, int] = {}
    candidate_counts_after: dict[str, int] = {}
    block_counts_by_reason: dict[str, int] = {}
    downgrade_counts_by_reason: dict[str, int] = {}
    survivors: list[dict] = []
    blocked_rows: list[dict] = []

    def _cooldown_row(key: str | None):
        if not key:
            return None
        return cooldowns.get(str(key), 0)

    def _cooldown_active(key: str | None, *, scopes: set[str] | None = None) -> bool:
        row = _cooldown_row(key)
        if isinstance(row, dict):
            if scopes is not None and str(row.get("scope", "candidate")) not in scopes:
                return False
            return int(row.get("remaining_rounds", 0)) > 0
        return scopes is None and int(row or 0) > 0

    def _exhausted_in_scope(key: str | None, scope: str) -> bool:
        if not key:
            return False
        return str(key) in {str(value) for value in list(exhaustion_map.get(scope, []) or [])}

    for candidate in candidates:
        row = dict(candidate)
        reason_details = list(row.get("blocked_reason_details", []))
        candidate_id = str(row["candidate_id"])
        candidate_class = str(row.get("candidate_class") or "unknown")
        candidate_counts_before[candidate_class] = candidate_counts_before.get(candidate_class, 0) + 1
        target_entity_id = row.get("target_entity_id")
        target_area_id = row.get("target_area_id")
        route_signature = row.get("route_signature")
        trigger_zone_id = row.get("trigger_zone_id")
        local_zone_key = f"{target_area_id or local_context.get('current_area_id') or 'global'}:{target_entity_id or candidate_id}"
        local_zone = dict(localized_context.get("by_zone", {}).get(local_zone_key, {}))
        local_area = dict(localized_context.get("by_area", {}).get(str(target_area_id or local_context.get("current_area_id") or "global"), {}))
        target_history = list(prior_failure_success_context.get("recent_target_entity_ids", []))
        contradiction_flags = dict(row.get("contradiction_flags", {}))
        stale_support_flags = dict(row.get("stale_support_flags", {}))
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []

        if candidate_id in exhausted or (route_signature is not None and str(route_signature) in exhausted):
            hard_reasons.append("hard_exhausted_scope")
        if _cooldown_active(candidate_id, scopes={"candidate"}) or _cooldown_active(route_signature, scopes={"candidate"}):
            hard_reasons.append("hard_cooldown_active")
        if _exhausted_in_scope(target_entity_id, "target") or _exhausted_in_scope(target_area_id, "area"):
            soft_reasons.append("soft_target_repeated_failure")
        if _cooldown_active(target_entity_id, scopes={"target"}) or _cooldown_active(target_area_id, scopes={"area"}) or _cooldown_active(trigger_zone_id, scopes={"target", "area"}):
            soft_reasons.append("soft_area_repeated_failure")
        if target_entity_id in blocked_targets and not bool(row.get("reachable_later")):
            hard_reasons.append("hard_unreachable_target")
        if row.get("target_entity_id") is not None and row.get("target_entity_id") not in belief.get("indexes", {}).get("reachable_targets", []) and not row.get("reachable_later"):
            hard_reasons.append("hard_invalid_target")
        if int(failed_candidates.get(candidate_id, 0)) >= 2:
            soft_reasons.append("soft_repeated_failure")
        if int(failed_candidates.get(str(target_entity_id), 0)) >= 2:
            soft_reasons.append("soft_target_repeated_failure")
        if int(local_zone.get("failures", 0)) >= 2 and int(local_zone.get("successes", 0)) == 0:
            soft_reasons.append("soft_local_target_repeat")
        if int(local_area.get("failures", 0)) >= 3 and int(local_area.get("successes", 0)) == 0:
            soft_reasons.append("soft_area_repeated_failure")
        if route_signature and int(failed_candidates.get(str(route_signature), 0)) >= 2:
            soft_reasons.append("soft_route_repeat")
        if trigger_zone_id and int(failed_candidates.get(str(trigger_zone_id), 0)) >= 2:
            soft_reasons.append("soft_trigger_repeat")
        if target_entity_id and target_history.count(str(target_entity_id)) >= 2:
            soft_reasons.append("soft_local_target_repeat")
        if int(local_area.get("candidate_ids", []).count(candidate_id)) >= 2:
            soft_reasons.append("soft_local_class_repeat")
        supporting_refs = list(row.get("supporting_evidence_refs", []))
        if supporting_refs and not any(str(ref) in evidence_index for ref in supporting_refs):
            hard_reasons.append("hard_contradiction_current_evidence")
        if contradiction_flags.get("stale_target") or contradiction_flags.get("topology_invalidation"):
            soft_reasons.append("soft_uncertain_contradiction")
        if stale_support_flags.get("support_refs_missing") or contradiction_flags.get("evidence_decay"):
            soft_reasons.append("soft_stale_support_decay")

        class_soft_limits = {
            "fallback_action": 99,
            "recovery_move": 3,
            "frontier_move": 2,
            "route_probe": 2,
            "trigger_probe": 2,
            "local_probe": 2,
            "target": 2,
            "click_target": 2,
        }
        dedup_soft = sorted(set(soft_reasons))
        if len(dedup_soft) >= class_soft_limits.get(candidate_class, 2):
            hard_reasons.append("soft_stale_support_decay" if "soft_stale_support_decay" in dedup_soft else dedup_soft[0])
            dedup_soft = [reason for reason in dedup_soft if reason not in hard_reasons]

        for reason in sorted(set(hard_reasons)):
            detail = dict(FILTER_REASON_TAXONOMY.get(reason, {}))
            detail["reason_code"] = reason
            reason_details.append(detail)
            block_counts_by_reason[reason] = block_counts_by_reason.get(reason, 0) + 1
        for reason in dedup_soft:
            detail = dict(FILTER_REASON_TAXONOMY.get(reason, {}))
            detail["reason_code"] = reason
            reason_details.append(detail)
            downgrade_counts_by_reason[reason] = downgrade_counts_by_reason.get(reason, 0) + 1

        row["blocked_reasons"] = sorted(set(hard_reasons))
        row["soft_filter_reasons"] = dedup_soft
        row["blocked_reason_details"] = reason_details
        row["filter_provenance"] = sorted(set(detail.get("provenance", "unknown") for detail in reason_details))
        row["blocked"] = bool(row["blocked_reasons"])
        if row["blocked"]:
            blocked_rows.append(row)
            continue
        if row.get("soft_filter_reasons"):
            row["score_penalty_soft_filters"] = 0.08 * len(row["soft_filter_reasons"])
        candidate_counts_after[candidate_class] = candidate_counts_after.get(candidate_class, 0) + 1
        survivors.append(row)

    audit = {
        "candidate_counts_by_class_before": candidate_counts_before,
        "candidate_counts_by_class_after": candidate_counts_after,
        "block_counts_by_reason": block_counts_by_reason,
        "downgrade_counts_by_reason": downgrade_counts_by_reason,
    }
    for row in survivors:
        row["filter_audit"] = audit
    for row in blocked_rows:
        row["filter_audit"] = audit
    return survivors, blocked_rows
