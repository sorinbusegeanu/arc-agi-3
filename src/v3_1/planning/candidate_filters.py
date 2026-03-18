from __future__ import annotations

FILTER_REASON_TAXONOMY = {
    "hard.cooldown.active": {"severity": "hard", "provenance": "tactical_memory"},
    "hard.exhaustion.scope": {"severity": "hard", "provenance": "tactical_memory"},
    "hard.target.unreachable": {"severity": "hard", "provenance": "observed_world"},
    "hard.target.invalid": {"severity": "hard", "provenance": "observed_world"},
    "hard.evidence.contradicted": {"severity": "hard", "provenance": "observed_world"},
    "soft.repeat.candidate_area": {"severity": "soft", "provenance": "tactical_context"},
    "soft.repeat.target_area": {"severity": "soft", "provenance": "tactical_context"},
    "soft.repeat.route": {"severity": "soft", "provenance": "tactical_context"},
    "soft.repeat.trigger": {"severity": "soft", "provenance": "tactical_context"},
    "soft.support.stale": {"severity": "soft", "provenance": "uncertainty_context"},
    "soft.contradiction.uncertain": {"severity": "soft", "provenance": "hypothesized_world"},
}

WEAKNESS_BY_OBJECTIVE = {
    "interact": 2,
    "gather_local_info": 2,
    "explore_frontier": 2,
    "probe_route": 2,
    "test_trigger": 2,
    "recover": 3,
    "fallback": 99,
}


def _target_ids(rows: list[dict]) -> set[str]:
    return {str(row.get("entity_id") or row.get("target_entity_id") or "") for row in list(rows or []) if str(row.get("entity_id") or row.get("target_entity_id") or "")}


def _row_lookup(rows: list[dict], *, key_name: str) -> dict[str, dict]:
    lookup = {}
    for row in list(rows or []):
        key = str(row.get(key_name) or "")
        if key:
            lookup[key] = dict(row)
    return lookup


def _cooldown_active(cooldowns: dict, keys: list[str]) -> bool:
    for key in keys:
        if not key:
            continue
        row = cooldowns.get(str(key))
        if isinstance(row, dict):
            if int(row.get("remaining_rounds", 0) or 0) > 0:
                return True
        elif int(row or 0) > 0:
            return True
    return False


def _mark_filter_usage(row: dict, *, observed: bool, hypothesis: bool, compatibility: bool) -> None:
    row["used_observed_support"] = bool(row.get("used_observed_support")) or observed
    row["used_hypothesis_support"] = bool(row.get("used_hypothesis_support")) or hypothesis
    row["used_compatibility_fallback"] = bool(row.get("used_compatibility_fallback")) or compatibility
    row["filter_input_mode"] = "split_world_native" if not row["used_compatibility_fallback"] else "compatibility_fallback"


def _append_reason(row: dict, *, code: str, detail: dict | None = None, hard: bool) -> None:
    key = "blocked_reasons" if hard else "soft_filter_reasons"
    row[key].append(code)
    row["blocked_reason_details"].append({"code": code, **dict(detail or {})})
    row["filter_provenance"].append(dict(FILTER_REASON_TAXONOMY.get(code, {})))
    audit_key = "block_counts_by_reason" if hard else "downgrade_counts_by_reason"
    audit = row.setdefault("filter_audit", {})
    counts = audit.setdefault(audit_key, {})
    counts[code] = int(counts.get(code, 0) or 0) + 1


def filter_candidates(
    candidates: list[dict],
    *,
    observed_world: dict | None = None,
    hypothesized_world: dict | None = None,
    uncertainty_context: dict | None = None,
    belief_fallback: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    belief_fallback = dict(belief_fallback or {})
    observed_world = dict(observed_world or belief_fallback.get("observed_world", {}) or {})
    hypothesized_world = dict(hypothesized_world or belief_fallback.get("hypothesized_world", {}) or {})
    uncertainty_context = dict(uncertainty_context or belief_fallback.get("uncertainty_context", {}) or {})
    tactical_memory = dict(uncertainty_context.get("tactical_memory", {}) or belief_fallback.get("tactical_memory_view", {}) or {})
    tactical_context = dict(tactical_memory.get("tactical_context", {}) or {})
    cooldowns = dict(tactical_memory.get("cooldowns", {}) or {})
    exhausted_keys = {str(key) for key in list(tactical_memory.get("exhausted_keys", []) or []) if key}
    observed_blocked = _target_ids(list(observed_world.get("blocked_targets", [])))
    observed_reachable = _target_ids(list(observed_world.get("reachable_targets", [])))
    hypothesis_reachable = _target_ids(list(hypothesized_world.get("reachable_targets", [])))
    observed_entities = _row_lookup(list(dict(observed_world.get("entities", {})).values()) + list(observed_world.get("reachable_targets", [])) + list(observed_world.get("promising_pois", [])) + list(observed_world.get("trigger_candidates", [])), key_name="entity_id")
    hypothesized_entities = _row_lookup(list(dict(hypothesized_world.get("entities", {})).values()) + list(hypothesized_world.get("reachable_targets", [])) + list(hypothesized_world.get("promising_pois", [])) + list(hypothesized_world.get("trigger_candidates", [])), key_name="entity_id")
    survivors: list[dict] = []
    blocked: list[dict] = []
    counts_before: dict[str, int] = {}
    counts_after: dict[str, int] = {}

    for candidate in candidates:
        row = dict(candidate)
        row.setdefault("blocked_reasons", [])
        row.setdefault("soft_filter_reasons", [])
        row.setdefault("blocked_reason_details", [])
        row.setdefault("filter_provenance", [])
        row["used_observed_support"] = False
        row["used_hypothesis_support"] = False
        row["used_compatibility_fallback"] = str(row.get("seed_contract") or "") == "compatibility_fallback"
        row["filter_input_mode"] = "compatibility_fallback" if row["used_compatibility_fallback"] else "split_world_native"
        row["blocked"] = False
        candidate_class = str(row.get("candidate_class") or "unknown")
        counts_before[candidate_class] = int(counts_before.get(candidate_class, 0) or 0) + 1
        counts_after.setdefault(candidate_class, counts_before[candidate_class])
        row["filter_audit"] = {
            "candidate_counts_by_class_before": dict(counts_before),
            "candidate_counts_by_class_after": dict(counts_after),
            "block_counts_by_reason": {},
            "downgrade_counts_by_reason": {},
        }

        target_entity_id = str(row.get("target_entity_id") or "")
        target_area_id = str(row.get("target_area_id") or "")
        route_signature = str(row.get("route_signature") or "")
        trigger_zone_id = str(row.get("trigger_zone_id") or "")
        local_area = str(row.get("candidate_context", {}).get("local_area") or target_area_id or uncertainty_context.get("current_area_id") or "global")
        local_key = f"{candidate_class}:{local_area}"
        target_local_key = f"{target_entity_id or 'none'}:{local_area}"
        recent_local_outcomes = dict(tactical_context.get("recent_local_outcomes", {}))
        repeat_patterns = dict(tactical_context.get("repeat_pattern_state", {}))

        _mark_filter_usage(row, observed=target_entity_id in observed_reachable or target_entity_id in observed_blocked, hypothesis=target_entity_id in hypothesis_reachable, compatibility=row["used_compatibility_fallback"])

        if _cooldown_active(cooldowns, [str(row.get("candidate_id") or ""), target_entity_id, target_area_id, route_signature, trigger_zone_id]):
            _append_reason(row, code="hard.cooldown.active", detail={"target_entity_id": target_entity_id}, hard=True)
        if any(key in exhausted_keys for key in [str(row.get("candidate_id") or ""), target_entity_id, target_area_id, route_signature, trigger_zone_id] if key):
            _append_reason(row, code="hard.exhaustion.scope", detail={"target_entity_id": target_entity_id}, hard=True)
        if target_entity_id and target_entity_id in observed_blocked and not row.get("reachable_now") and not row.get("reachable_later"):
            _append_reason(row, code="hard.target.unreachable", detail={"target_entity_id": target_entity_id}, hard=True)
        graph_backed_target = bool(
            target_entity_id
            and (
                str(target_entity_id).startswith("poi:")
                or str(target_entity_id).startswith("trigger:")
                or str(target_entity_id).startswith("mg:")
            )
            and (
                list(row.get("candidate_step_plan", []) or [])
                or list(row.get("supporting_graph_node_ids", []) or [])
                or str(row.get("candidate_class") or "") in {"unlock_then_exit", "unlock_trigger", "verify_panel_state", "mechanic_chain_deterministic", "mechanic_chain_llm"}
            )
            and isinstance(dict(row.get("action", {}) or {}).get("centroid"), list)
        )
        if target_entity_id and not (target_entity_id in observed_entities or target_entity_id in hypothesized_entities or graph_backed_target or row.get("objective_type") in {"explore_frontier", "recover", "fallback"}):
            _append_reason(row, code="hard.target.invalid", detail={"target_entity_id": target_entity_id}, hard=True)
        if any(bool(value) for value in dict(row.get("contradiction_flags", {})).values()):
            hard_contradiction = bool(dict(row.get("contradiction_flags", {})).get("hard_contradiction")) or bool(dict(row.get("contradiction_flags", {})).get("stale_target"))
            _append_reason(row, code="hard.evidence.contradicted" if hard_contradiction else "soft.contradiction.uncertain", detail={"target_entity_id": target_entity_id}, hard=hard_contradiction)
        if bool(dict(row.get("stale_support_flags", {})).get("support_refs_missing")) or bool(dict(row.get("stale_support_flags", {})).get("target_stale")):
            _append_reason(row, code="soft.support.stale", detail={"target_entity_id": target_entity_id}, hard=False)

        local_failures = int(dict(recent_local_outcomes.get(target_local_key, {})).get("failures", 0) or 0)
        class_local_state = dict(repeat_patterns.get("by_candidate_area", {})).get(local_key, {})
        class_local_failures = int(dict(class_local_state).get("failures", 0) or 0)
        if class_local_failures >= WEAKNESS_BY_OBJECTIVE.get(str(row.get("objective_type") or "fallback"), 99):
            _append_reason(row, code="soft.repeat.candidate_area", detail={"count": class_local_failures}, hard=False)
        if local_failures >= 2:
            _append_reason(row, code="soft.repeat.target_area", detail={"count": local_failures}, hard=False)
        route_repeat_state = dict(repeat_patterns.get("by_route_signature", {})).get(route_signature, {})
        if route_signature and int(dict(route_repeat_state).get("failures", 0) or 0) >= 2:
            _append_reason(row, code="soft.repeat.route", detail={"route_signature": route_signature}, hard=False)
        trigger_repeat_state = dict(repeat_patterns.get("by_trigger_zone", {})).get(trigger_zone_id, {})
        if trigger_zone_id and int(dict(trigger_repeat_state).get("failures", 0) or 0) >= 2:
            _append_reason(row, code="soft.repeat.trigger", detail={"trigger_zone_id": trigger_zone_id}, hard=False)

        if row["blocked_reasons"]:
            row["blocked"] = True
            blocked.append(row)
            counts_after[candidate_class] = max(0, int(counts_after.get(candidate_class, 1) or 1) - 1)
            row["filter_audit"]["candidate_counts_by_class_after"] = dict(counts_after)
            continue

        row["score_penalty_soft_filters"] = 0.06 * len(list(row.get("soft_filter_reasons", [])))
        row["filter_audit"]["candidate_counts_by_class_after"] = dict(counts_after)
        survivors.append(row)

    return survivors, blocked
