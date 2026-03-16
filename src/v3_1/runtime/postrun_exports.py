from __future__ import annotations

from collections import Counter
from v3_1.storage.serialization import dumps

import ray

from v3_1.visualization.heatmaps import (
    build_poi_heatmap,
    build_visit_heatmap,
    render_heatmap_debug_png,
    render_observation_png,
    render_overlay_png,
)
from v3_1.visualization.summaries import build_run_summary


def _persist(storage_agent, **kwargs) -> str:
    persist = getattr(storage_agent, "persist", None)
    if persist is not None and hasattr(persist, "remote"):
        return ray.get(persist.remote(**kwargs))
    return storage_agent.persist(**kwargs)


def _persist_visualization(storage_agent, **kwargs) -> str:
    persist = getattr(storage_agent, "persist_visualization_bytes", None)
    if persist is not None and hasattr(persist, "remote"):
        return ray.get(persist.remote(**kwargs))
    return storage_agent.persist_visualization_bytes(**kwargs)


def _persist_session(storage_agent, **kwargs) -> str:
    persist = getattr(storage_agent, "persist_session_json", None)
    if persist is not None and hasattr(persist, "remote"):
        return ray.get(persist.remote(**kwargs))
    return storage_agent.persist_session_json(**kwargs)


def _persist_session_bytes(storage_agent, **kwargs) -> str:
    persist = getattr(storage_agent, "persist_session_bytes", None)
    if persist is not None and hasattr(persist, "remote"):
        return ray.get(persist.remote(**kwargs))
    return storage_agent.persist_session_bytes(**kwargs)


def _first_validated_round(proposals: dict, validation_state: dict) -> int | None:
    validated_rounds = [
        int(row.get("round_id", 0) or 0)
        for row in dict(proposals or {}).values()
        if str(dict(validation_state or {}).get(str(row.get("proposal_id")), "")) == "validated"
    ]
    return min(validated_rounds) if validated_rounds else None


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _selected_candidate(decision: dict) -> dict:
    metadata = dict(decision.get("metadata", {})) if isinstance(decision, dict) else {}
    selected = metadata.get("selected_candidate", {})
    return dict(selected) if isinstance(selected, dict) else {}


def _decision_candidates(decision: dict) -> list[dict]:
    if not isinstance(decision, dict):
        return []
    metadata = dict(decision.get("metadata", {})) if isinstance(decision.get("metadata"), dict) else {}
    rows = list(decision.get("ranked_candidates", ()) or [])
    rows.extend(list(metadata.get("fallback_candidates", []) or []))
    rows.extend(list(metadata.get("blocked_candidates", []) or []))
    selected = metadata.get("selected_candidate", {})
    if isinstance(selected, dict) and selected:
        rows.append(selected)
    return [dict(row) for row in rows if isinstance(row, dict)]


def _score_breakdown(candidate: dict) -> dict:
    breakdown = candidate.get("score_breakdown", {})
    return dict(breakdown) if isinstance(breakdown, dict) else {}


def _blocked_reasons(candidate: dict) -> set[str]:
    return {str(reason) for reason in list(candidate.get("blocked_reasons", []) or []) if reason}


def _candidate_targets(candidate: dict) -> tuple[str | None, str | None, str | None]:
    return (
        str(candidate.get("candidate_id")) if candidate.get("candidate_id") is not None else None,
        str(candidate.get("target_entity_id")) if candidate.get("target_entity_id") is not None else None,
        str(candidate.get("target_area_id")) if candidate.get("target_area_id") is not None else None,
    )


def _target_key(candidate: dict) -> str | None:
    _, target_entity_id, target_area_id = _candidate_targets(candidate)
    return target_entity_id or target_area_id


def _memory_signature(state: dict) -> dict:
    working = dict((state or {}).get("working_memory", {}))
    plan_memory = dict(working.get("plan_memory", {}))
    return {
        "cooldowns": dict(working.get("cooldowns", {})),
        "retries": dict(working.get("retries", {})),
        "exhausted": list(working.get("exhausted", [])),
        "exhaustion_map": dict(working.get("exhaustion_map", {})),
        "recovery_memory": dict(plan_memory.get("recovery_memory", {})),
        "route_patterns": dict(plan_memory.get("route_patterns", {})),
        "repeated_failures": dict(plan_memory.get("repeated_failures", {})),
    }


def _active_cooldown_keys(state: dict) -> set[str]:
    cooldowns = dict(((state or {}).get("working_memory", {}) or {}).get("cooldowns", {}))
    active = set()
    for key, value in cooldowns.items():
        if isinstance(value, dict):
            if int(value.get("remaining_rounds", 0) or 0) > 0:
                active.add(str(key))
        elif int(value or 0) > 0:
            active.add(str(key))
    return active


def _exhausted_keys(state: dict) -> set[str]:
    working = dict((state or {}).get("working_memory", {}))
    exhausted = {str(key) for key in list(working.get("exhausted", []) or []) if key}
    exhaustion_map = dict(working.get("exhaustion_map", {}))
    for rows in exhaustion_map.values():
        for key in list(rows or []):
            if key:
                exhausted.add(str(key))
    return exhausted


def _nonzero_memory_features(candidate: dict) -> dict[str, float]:
    breakdown = _score_breakdown(candidate)
    return {
        "prior_success_rate": float(breakdown.get("prior_success_rate", 0.0) or 0.0),
        "prior_failure_rate": float(breakdown.get("prior_failure_rate", 0.0) or 0.0),
        "prior_route_failure_risk": float(breakdown.get("prior_route_failure_risk", 0.0) or 0.0),
        "prior_recovery_usefulness": float(breakdown.get("prior_recovery_usefulness", 0.0) or 0.0),
        "retry_penalty": float(breakdown.get("retry_penalty", 0.0) or 0.0),
        "cooldown_penalty": float(breakdown.get("cooldown_penalty", 0.0) or 0.0),
        "exhaustion_penalty": float(breakdown.get("exhaustion_penalty", 0.0) or 0.0),
    }


def _candidate_has_memory_signal(candidate: dict) -> bool:
    return any(abs(value) > 0.0 for value in _nonzero_memory_features(candidate).values())


def _skill_rate(skill: dict, field: str) -> float:
    stats = dict(skill.get("execution_stats", {}))
    attempts = int(stats.get("attempts", 0) or 0)
    return _safe_rate(int(stats.get(field, 0) or 0), attempts)


def _top_skill_rows(skill_library: dict[str, dict], *, sort_key) -> list[dict]:
    rows = []
    for skill_id, skill in skill_library.items():
        stats = dict(skill.get("execution_stats", {}))
        rows.append(
            {
                "skill_id": str(skill_id),
                "skill_type": skill.get("skill_type"),
                "attempts": int(stats.get("attempts", 0) or 0),
                "successes": int(stats.get("successes", 0) or 0),
                "failures": int(stats.get("failures", 0) or 0),
                "success_rate": _skill_rate(skill, "successes"),
                "failure_rate": _skill_rate(skill, "failures"),
            }
        )
    rows.sort(key=sort_key, reverse=True)
    return rows[:10]


def _memory_telemetry(state: dict) -> dict:
    working = dict((state or {}).get("working_memory", {}))
    telemetry = working.get("memory_telemetry", {})
    return dict(telemetry) if isinstance(telemetry, dict) else {}


def _collect_direct_memory_telemetry(round_records: list[dict]) -> tuple[list[dict], list[dict]]:
    events: list[dict] = []
    outcomes: list[dict] = []
    previous_event_count = 0
    previous_outcome_count = 0
    for record in round_records:
        telemetry = _memory_telemetry(dict(record.get("post_memory_state", {})))
        post_events = list(telemetry.get("events", []) or [])
        post_outcomes = list(telemetry.get("outcomes", []) or [])
        if len(post_events) >= previous_event_count:
            events.extend(dict(row) for row in post_events[previous_event_count:])
            previous_event_count = len(post_events)
        else:
            events.extend(dict(row) for row in post_events)
            previous_event_count = len(post_events)
        if len(post_outcomes) >= previous_outcome_count:
            outcomes.extend(dict(row) for row in post_outcomes[previous_outcome_count:])
            previous_outcome_count = len(post_outcomes)
        else:
            outcomes.extend(dict(row) for row in post_outcomes)
            previous_outcome_count = len(post_outcomes)
    return events, outcomes


def _build_memory_summary(*, round_records: list[dict], latest_memory_version: str) -> dict:
    direct_events, direct_outcomes = _collect_direct_memory_telemetry(round_records)
    direct_event_counts = Counter(str(row.get("event_type") or "") for row in direct_events)
    direct_events_available = bool(direct_events or direct_outcomes)
    direct_retry_by_scope = Counter(str(row.get("scope") or "unknown") for row in direct_events if str(row.get("event_type") or "") == "retry_increment")
    post_versions = [str(row.get("post_memory_version") or "") for row in round_records if row.get("post_memory_version")]
    directed_attempt_count = len(round_records)
    attempt_count = directed_attempt_count
    success_count = 0
    failure_count = 0
    outcome_counts = Counter()
    retry_penalty_nonzero_count = 0
    retry_penalty_applied_before_reselect_count = 0
    cooldown_set_count = 0
    cooldown_active_count = 0
    cooldown_block_count = 0
    cooldown_penalty_nonzero_count = 0
    cooldown_violation_count = 0
    exhaustion_block_count = 0
    exhaustion_penalty_nonzero_count = 0
    repeated_selection_after_exhaustion_count = 0
    recovery_mode_entry_count = 0
    recovery_after_failure_count = 0
    recovery_success_count = 0
    recovery_failure_count = 0
    prior_recovery_usefulness_nonzero_count = 0
    route_failure_count = 0
    route_stall_count = 0
    no_progress_count = 0
    unreachable_count = 0
    blocked_count = 0
    contact_no_effect_count = 0
    contact_no_reach_count = 0
    contact_boundary_without_progress_count = 0
    prior_route_failure_risk_nonzero_count = 0
    route_failure_risk_changed_after_repeat_count = 0
    attempts_with_memory_features_present = 0
    candidates_with_nonzero_prior_success_rate = 0
    candidates_with_nonzero_prior_failure_rate = 0
    candidates_with_nonzero_prior_route_failure_risk = 0
    candidates_with_nonzero_prior_recovery_usefulness = 0
    candidates_with_nonzero_retry_penalty = 0
    candidates_with_nonzero_cooldown_penalty = 0
    candidates_with_nonzero_exhaustion_penalty = 0
    candidates_blocked_by_cooldown = 0
    candidates_blocked_by_exhaustion = 0
    candidates_blocked_by_invalid_target = 0
    candidates_blocked_by_unreachable = 0
    failed_target_later_penalized_count = 0
    failed_target_later_blocked_count = 0
    successful_target_later_boosted_count = 0
    repeated_failed_route_later_given_higher_route_risk_count = 0
    recovery_failure_later_increased_recovery_bias_count = 0
    memory_influenced_decision_count = 0
    all_zero_memory_feature_rounds = 0
    memory_version_advanced_but_features_unchanged_rounds = 0
    attempt_logged_without_later_memory_effect_count = 0
    decision_used_default_zero_priors_count = 0
    selected_candidate_with_known_failed_target_and_no_penalty_count = 0
    selected_candidate_with_active_cooldown_count = 0
    selected_candidate_from_exhausted_cluster_count = 0
    rounds_with_memory_version_advance = 0
    seen_failed_targets: set[str] = set()
    seen_successful_targets: set[str] = set()
    seen_route_failed_targets: set[str] = set()
    seen_recovery_failed_targets: set[str] = set()
    pending_failed_targets_without_effect: set[str] = set()
    last_route_risk_by_target: dict[str, float] = {}

    for record in round_records:
        decision = dict(record.get("decision", {}))
        outcome = dict(record.get("outcome", {}))
        pre_memory_state = dict(record.get("pre_memory_state", {}))
        post_memory_state = dict(record.get("post_memory_state", {}))
        pre_version = str(record.get("pre_memory_version") or "")
        post_version = str(record.get("post_memory_version") or "")
        if pre_version and post_version and pre_version != post_version:
            rounds_with_memory_version_advance += 1
            if _memory_signature(pre_memory_state) == _memory_signature(post_memory_state):
                memory_version_advanced_but_features_unchanged_rounds += 1

        active_cooldowns = _active_cooldown_keys(pre_memory_state)
        post_active_cooldowns = _active_cooldown_keys(post_memory_state)
        cooldown_active_count += len(active_cooldowns)
        if not direct_events_available:
            cooldown_set_count += len(post_active_cooldowns - active_cooldowns)
        exhausted_keys = _exhausted_keys(pre_memory_state)

        selected = _selected_candidate(decision)
        selected_features = _nonzero_memory_features(selected)
        if _candidate_has_memory_signal(selected):
            attempts_with_memory_features_present += 1
            memory_influenced_decision_count += 1
        else:
            decision_used_default_zero_priors_count += 1

        selected_candidate_id, selected_target_entity_id, selected_target_area_id = _candidate_targets(selected)
        selected_target_key = _target_key(selected)
        if selected_candidate_id in active_cooldowns or (selected_target_entity_id and selected_target_entity_id in active_cooldowns) or (selected_target_area_id and selected_target_area_id in active_cooldowns):
            cooldown_violation_count += 1
            selected_candidate_with_active_cooldown_count += 1
        if selected_candidate_id in exhausted_keys or (selected_target_entity_id and selected_target_entity_id in exhausted_keys) or (selected_target_area_id and selected_target_area_id in exhausted_keys):
            repeated_selection_after_exhaustion_count += 1
            selected_candidate_from_exhausted_cluster_count += 1
        if selected_target_key in seen_failed_targets and selected_features["retry_penalty"] <= 0.0 and selected_features["prior_failure_rate"] <= 0.0 and selected_features["cooldown_penalty"] <= 0.0:
            selected_candidate_with_known_failed_target_and_no_penalty_count += 1

        candidates = _decision_candidates(decision)
        if candidates and not any(_candidate_has_memory_signal(candidate) for candidate in candidates):
            all_zero_memory_feature_rounds += 1
        for candidate in candidates:
            features = _nonzero_memory_features(candidate)
            reasons = _blocked_reasons(candidate)
            target_key = _target_key(candidate)
            if features["prior_success_rate"] > 0.0:
                candidates_with_nonzero_prior_success_rate += 1
            if features["prior_failure_rate"] > 0.0:
                candidates_with_nonzero_prior_failure_rate += 1
            if features["prior_route_failure_risk"] > 0.0:
                candidates_with_nonzero_prior_route_failure_risk += 1
                prior_route_failure_risk_nonzero_count += 1
            if abs(features["prior_recovery_usefulness"]) > 0.0:
                candidates_with_nonzero_prior_recovery_usefulness += 1
                prior_recovery_usefulness_nonzero_count += 1
            if features["retry_penalty"] > 0.0:
                candidates_with_nonzero_retry_penalty += 1
                retry_penalty_nonzero_count += 1
            if features["cooldown_penalty"] > 0.0:
                candidates_with_nonzero_cooldown_penalty += 1
                cooldown_penalty_nonzero_count += 1
            if features["exhaustion_penalty"] > 0.0:
                candidates_with_nonzero_exhaustion_penalty += 1
                exhaustion_penalty_nonzero_count += 1
            if "cooldown" in reasons:
                candidates_blocked_by_cooldown += 1
                cooldown_block_count += 1
            if "exhausted" in reasons:
                candidates_blocked_by_exhaustion += 1
                exhaustion_block_count += 1
            if "invalid_target" in reasons:
                candidates_blocked_by_invalid_target += 1
            if "unreachable" in reasons:
                candidates_blocked_by_unreachable += 1
                unreachable_count += 1
            if target_key in seen_failed_targets and (features["retry_penalty"] > 0.0 or features["prior_failure_rate"] > 0.0):
                failed_target_later_penalized_count += 1
                pending_failed_targets_without_effect.discard(str(target_key))
            if target_key in seen_failed_targets and reasons:
                failed_target_later_blocked_count += 1
                pending_failed_targets_without_effect.discard(str(target_key))
            if target_key in seen_successful_targets and features["prior_success_rate"] > 0.0:
                successful_target_later_boosted_count += 1
            if target_key in seen_route_failed_targets:
                previous_risk = last_route_risk_by_target.get(str(target_key), 0.0)
                if features["prior_route_failure_risk"] > previous_risk:
                    route_failure_risk_changed_after_repeat_count += 1
                if features["prior_route_failure_risk"] > 0.0:
                    repeated_failed_route_later_given_higher_route_risk_count += 1
            if target_key in seen_recovery_failed_targets and candidate.get("candidate_class") == "recovery_move" and abs(features["prior_recovery_usefulness"]) > 0.0:
                recovery_failure_later_increased_recovery_bias_count += 1

        success = bool(outcome.get("success") or dict(outcome.get("outcome", {})).get("success"))
        termination_reason = str(outcome.get("termination_reason") or dict(outcome.get("outcome", {})).get("termination_reason") or ("success" if success else "unknown"))
        outcome_summary = dict(outcome.get("outcome", {}))
        progress = float(outcome_summary.get("progress", 0.0) or 0.0)
        route_failed = bool(outcome_summary.get("route_failure"))
        if success:
            success_count += 1
            outcome_counts["success"] += 1
        else:
            failure_count += 1
            outcome_counts[termination_reason] += 1
        if route_failed:
            route_failure_count += 1
        if termination_reason == "stalled":
            route_stall_count += 1
        if progress <= 0.0 and not success:
            no_progress_count += 1
        if termination_reason == "blocked":
            blocked_count += 1
        required_family = str((decision.get("selected_action") or {}).get("required_action_family") or (decision.get("selected_action") or {}).get("type") or "").lower()
        if required_family in {"interact", "click_at"} and not success and float(outcome.get("reward_delta", 0.0) or 0.0) <= 0.0:
            contact_no_effect_count += 1
        if required_family in {"interact", "click_at"} and termination_reason in {"blocked", "stalled", "noop"}:
            contact_no_reach_count += 1
        if required_family in {"interact", "click_at"} and progress <= 0.0 and not success:
            contact_boundary_without_progress_count += 1

        if selected.get("candidate_class") == "recovery_move":
            recovery_mode_entry_count += 1
            if seen_failed_targets:
                recovery_after_failure_count += 1
            if success:
                recovery_success_count += 1
            else:
                recovery_failure_count += 1

        if not success and selected_features["retry_penalty"] > 0.0 and selected_target_key is not None:
            retry_penalty_applied_before_reselect_count += 1

        if selected_target_key:
            if success:
                seen_successful_targets.add(str(selected_target_key))
            else:
                seen_failed_targets.add(str(selected_target_key))
                pending_failed_targets_without_effect.add(str(selected_target_key))
            if route_failed:
                seen_route_failed_targets.add(str(selected_target_key))
            if selected.get("candidate_class") == "recovery_move" and not success:
                seen_recovery_failed_targets.add(str(selected_target_key))
        if selected_target_key and selected_features["prior_route_failure_risk"] > 0.0:
            last_route_risk_by_target[str(selected_target_key)] = selected_features["prior_route_failure_risk"]

    attempt_logged_without_later_memory_effect_count = len(pending_failed_targets_without_effect)

    final_memory_state = dict(round_records[-1].get("post_memory_state", {})) if round_records else {}
    working_memory = dict(final_memory_state.get("working_memory", {}))
    retries = dict(working_memory.get("retries", {}))
    target_retry_rows = [dict(row) for row in retries.values() if isinstance(row, dict) and str(row.get("scope", "")) == "target"]
    target_retry_attempts = [int(row.get("attempts", 0) or 0) for row in target_retry_rows if int(row.get("attempts", 0) or 0) > 0]
    if direct_events_available:
        target_retry_latest: dict[str, int] = {}
        for row in direct_events:
            if str(row.get("event_type") or "") != "retry_increment" or str(row.get("scope") or "") != "target":
                continue
            key = str(row.get("key") or "")
            if not key:
                continue
            target_retry_latest[key] = max(int(row.get("attempts_after", 0) or 0), target_retry_latest.get(key, 0))
        if target_retry_latest:
            target_retry_attempts = [count for count in target_retry_latest.values() if count > 0]
            target_retry_rows = [{"scope": "target", "attempts": count} for count in target_retry_attempts]
    exhausted_keys_seen = set()
    for record in round_records:
        exhausted_keys_seen |= _exhausted_keys(dict(record.get("pre_memory_state", {})))
        exhausted_keys_seen |= _exhausted_keys(dict(record.get("post_memory_state", {})))
    skill_library = dict(final_memory_state.get("skill_library", {}))
    attempted_skills = [skill for skill in skill_library.values() if int(dict(skill.get("execution_stats", {})).get("attempts", 0) or 0) > 0]
    skills_with_success = [skill for skill in attempted_skills if int(dict(skill.get("execution_stats", {})).get("successes", 0) or 0) > 0]
    skills_with_failure = [skill for skill in attempted_skills if int(dict(skill.get("execution_stats", {})).get("failures", 0) or 0) > 0]

    working_signals = [
        rounds_with_memory_version_advance > 0,
        attempts_with_memory_features_present > 0,
        candidates_with_nonzero_retry_penalty > 0 or candidates_with_nonzero_prior_failure_rate > 0 or candidates_with_nonzero_prior_success_rate > 0,
        memory_influenced_decision_count > 0,
        failed_target_later_penalized_count > 0 or failed_target_later_blocked_count > 0 or successful_target_later_boosted_count > 0,
        retry_penalty_nonzero_count > 0 or cooldown_block_count > 0 or exhaustion_block_count > 0,
    ]
    anti_signals = [
        all_zero_memory_feature_rounds == directed_attempt_count and directed_attempt_count > 0,
        decision_used_default_zero_priors_count == directed_attempt_count and directed_attempt_count > 0,
        memory_version_advanced_but_features_unchanged_rounds > 0,
        selected_candidate_with_known_failed_target_and_no_penalty_count > 0,
        selected_candidate_with_active_cooldown_count > 0,
        selected_candidate_from_exhausted_cluster_count > 0,
    ]
    memory_working_score = max(
        0.0,
        min(
            1.0,
            _safe_rate(sum(1 for value in working_signals if value), len(working_signals))
            - (0.12 * sum(1 for value in anti_signals if value)),
        ),
    )
    if memory_working_score >= 0.67:
        memory_working_verdict = "working"
    elif memory_working_score >= 0.34:
        memory_working_verdict = "partially_working"
    else:
        memory_working_verdict = "not_working"

    material_skill_update_events = [
        row
        for row in direct_events
        if str(row.get("event_type") or "") == "skill_stat_update"
        and (
            int(row.get("attempts_after", 0) or 0) > int(row.get("attempts_before", 0) or 0)
            or int(row.get("successes_after", 0) or 0) > int(row.get("successes_before", 0) or 0)
            or int(row.get("failures_after", 0) or 0) > int(row.get("failures_before", 0) or 0)
        )
    ]
    cooldown_set_count_value = direct_event_counts.get("cooldown_set", 0) if direct_events_available else cooldown_set_count
    cooldown_clear_count_value = direct_event_counts.get("cooldown_clear", 0)
    retry_increment_event_count = direct_event_counts.get("retry_increment", 0)
    exhaustion_set_count = direct_event_counts.get("exhaustion_set", 0)
    exhaustion_clear_count = direct_event_counts.get("exhaustion_clear", 0)
    recovery_history_write_count = direct_event_counts.get("recovery_history_write", 0)
    route_failure_write_count = direct_event_counts.get("route_failure_write", 0)
    skill_stat_update_count = len(material_skill_update_events)
    memory_write_event_count = direct_event_counts.get("memory_write", 0)
    exhausted_cluster_count_value = len({str(row.get("key")) for row in direct_events if str(row.get("event_type")) == "exhaustion_set" and row.get("key")}) if direct_events_available else len(exhausted_keys_seen)
    skills_with_updated_stats_count_value = (
        len({str(row.get("skill_id")) for row in material_skill_update_events if row.get("skill_id")})
        if direct_events_available
        else len(attempted_skills)
    )

    return {
        "memory_version_count": len(set(post_versions)),
        "rounds_with_memory_version_advance": rounds_with_memory_version_advance,
        "directed_attempt_count": directed_attempt_count,
        "attempts_with_memory_features_present": attempts_with_memory_features_present,
        "attempt_count": attempt_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": _safe_rate(success_count, attempt_count),
        "failure_rate": _safe_rate(failure_count, attempt_count),
        "outcome_counts_by_class": dict(sorted(outcome_counts.items())),
        "targets_retried_count": sum(1 for row in target_retry_rows if int(row.get("attempts", 0) or 0) > 1),
        "mean_retries_per_target": sum(target_retry_attempts) / float(len(target_retry_attempts)) if target_retry_attempts else 0.0,
        "max_retries_per_target": max(target_retry_attempts) if target_retry_attempts else 0,
        "retry_penalty_nonzero_count": retry_penalty_nonzero_count,
        "retry_penalty_applied_before_reselect_count": retry_penalty_applied_before_reselect_count,
        "cooldown_set_count": cooldown_set_count_value,
        "cooldown_active_count": cooldown_active_count,
        "cooldown_block_count": cooldown_block_count,
        "cooldown_penalty_nonzero_count": cooldown_penalty_nonzero_count,
        "cooldown_violation_count": cooldown_violation_count,
        "exhausted_cluster_count": exhausted_cluster_count_value,
        "exhaustion_block_count": exhaustion_block_count,
        "exhaustion_penalty_nonzero_count": exhaustion_penalty_nonzero_count,
        "repeated_selection_after_exhaustion_count": repeated_selection_after_exhaustion_count,
        "recovery_mode_entry_count": recovery_mode_entry_count,
        "recovery_after_failure_count": recovery_after_failure_count,
        "recovery_success_count": recovery_success_count,
        "recovery_failure_count": recovery_failure_count,
        "recovery_success_rate": _safe_rate(recovery_success_count, recovery_mode_entry_count),
        "prior_recovery_usefulness_nonzero_count": prior_recovery_usefulness_nonzero_count,
        "route_failure_count": route_failure_count,
        "route_stall_count": route_stall_count,
        "no_progress_count": no_progress_count,
        "unreachable_count": unreachable_count,
        "blocked_count": blocked_count,
        "contact_no_effect_count": contact_no_effect_count,
        "contact_no_reach_count": contact_no_reach_count,
        "contact_boundary_without_progress_count": contact_boundary_without_progress_count,
        "prior_route_failure_risk_nonzero_count": prior_route_failure_risk_nonzero_count,
        "route_failure_risk_changed_after_repeat_count": route_failure_risk_changed_after_repeat_count,
        "skill_count": len(skill_library),
        "skills_attempted_count": len(attempted_skills),
        "skills_with_success_count": len(skills_with_success),
        "skills_with_failure_count": len(skills_with_failure),
        "skills_with_updated_stats_count": skills_with_updated_stats_count_value,
        "top_skills_by_attempts": _top_skill_rows(skill_library, sort_key=lambda row: (row["attempts"], row["successes"])),
        "top_skills_by_success_rate": _top_skill_rows(skill_library, sort_key=lambda row: (row["success_rate"], row["attempts"])),
        "top_skills_by_failure_rate": _top_skill_rows(skill_library, sort_key=lambda row: (row["failure_rate"], row["attempts"])),
        "candidates_with_nonzero_prior_success_rate": candidates_with_nonzero_prior_success_rate,
        "candidates_with_nonzero_prior_failure_rate": candidates_with_nonzero_prior_failure_rate,
        "candidates_with_nonzero_prior_route_failure_risk": candidates_with_nonzero_prior_route_failure_risk,
        "candidates_with_nonzero_prior_recovery_usefulness": candidates_with_nonzero_prior_recovery_usefulness,
        "candidates_with_nonzero_retry_penalty": candidates_with_nonzero_retry_penalty,
        "candidates_with_nonzero_cooldown_penalty": candidates_with_nonzero_cooldown_penalty,
        "candidates_with_nonzero_exhaustion_penalty": candidates_with_nonzero_exhaustion_penalty,
        "candidates_blocked_by_cooldown": candidates_blocked_by_cooldown,
        "candidates_blocked_by_exhaustion": candidates_blocked_by_exhaustion,
        "candidates_blocked_by_invalid_target": candidates_blocked_by_invalid_target,
        "candidates_blocked_by_unreachable": candidates_blocked_by_unreachable,
        "failed_target_later_penalized_count": failed_target_later_penalized_count,
        "failed_target_later_blocked_count": failed_target_later_blocked_count,
        "successful_target_later_boosted_count": successful_target_later_boosted_count,
        "repeated_failed_route_later_given_higher_route_risk_count": repeated_failed_route_later_given_higher_route_risk_count,
        "recovery_failure_later_increased_recovery_bias_count": recovery_failure_later_increased_recovery_bias_count,
        "memory_influenced_decision_count": memory_influenced_decision_count,
        "all_zero_memory_feature_rounds": all_zero_memory_feature_rounds,
        "memory_version_advanced_but_features_unchanged_rounds": memory_version_advanced_but_features_unchanged_rounds,
        "attempt_logged_without_later_memory_effect_count": attempt_logged_without_later_memory_effect_count,
        "decision_used_default_zero_priors_count": decision_used_default_zero_priors_count,
        "selected_candidate_with_known_failed_target_and_no_penalty_count": selected_candidate_with_known_failed_target_and_no_penalty_count,
        "selected_candidate_with_active_cooldown_count": selected_candidate_with_active_cooldown_count,
        "selected_candidate_from_exhausted_cluster_count": selected_candidate_from_exhausted_cluster_count,
        "memory_write_event_count": memory_write_event_count,
        "retry_increment_event_count": retry_increment_event_count,
        "candidate_retry_increment_event_count": int(direct_retry_by_scope.get("candidate", 0)),
        "target_retry_increment_event_count": int(direct_retry_by_scope.get("target", 0)),
        "area_retry_increment_event_count": int(direct_retry_by_scope.get("area", 0)),
        "cooldown_clear_count": cooldown_clear_count_value,
        "exhaustion_set_count": exhaustion_set_count,
        "exhaustion_clear_count": exhaustion_clear_count,
        "recovery_history_write_count": recovery_history_write_count,
        "route_failure_write_count": route_failure_write_count,
        "skill_stat_update_count": skill_stat_update_count,
        "memory_event_count": len(direct_events),
        "memory_update_record_count": len(direct_outcomes),
        "memory_summary_uses_direct_events": direct_events_available,
        "memory_working_score": memory_working_score,
        "memory_working_verdict": memory_working_verdict,
        "latest_memory_version": latest_memory_version,
    }


def _visit_debug_payload(episodes: list[dict], sequence: list[list[int]]) -> dict:
    counts_by_episode = {}
    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        episode_sequence = []
        for step in episode.get("steps", []):
            cell = step.get("avatar_cell")
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                continue
            episode_sequence.append([int(cell[0]), int(cell[1])])
        counts_by_episode[episode_id] = {
            "visit_count": len(episode_sequence),
            "unique_visit_count": len({(cell[0], cell[1]) for cell in episode_sequence}),
            "first_10": episode_sequence[:10],
        }
    counter = Counter((int(cell[0]), int(cell[1])) for cell in sequence if isinstance(cell, list) and len(cell) == 2)
    xs = [cell[0] for cell in sequence]
    ys = [cell[1] for cell in sequence]
    return {
        "total_visit_coordinates": len(sequence),
        "unique_visit_cells": len(counter),
        "visit_counts_per_episode": counts_by_episode,
        "first_20_visit_cells": sequence[:20],
        "min_x": min(xs) if xs else None,
        "max_x": max(xs) if xs else None,
        "min_y": min(ys) if ys else None,
        "max_y": max(ys) if ys else None,
        "raw_visit_cells": sequence,
        "top_cells": [
            {"cell": [x, y], "count": count}
            for (x, y), count in counter.most_common(20)
        ],
    }


def _target_effect_fallbacks(round_records: list[dict]) -> dict[str, dict]:
    by_target: dict[str, dict] = {}
    for record in list(round_records or []):
        decision = dict(record.get("decision", {}))
        metadata = dict(decision.get("metadata", {})) if isinstance(decision.get("metadata"), dict) else {}
        selected = dict(metadata.get("selected_candidate", {})) if isinstance(metadata.get("selected_candidate"), dict) else {}
        target_id = selected.get("target_entity_id") or selected.get("target")
        if not target_id:
            continue
        analysis_summary = dict(record.get("analysis_summary", {}))
        move_steps = int(analysis_summary.get("move_steps_count", 0) or 0)
        movement_steps_with_change = int(analysis_summary.get("movement_steps_with_change", 0) or 0)
        interact_steps = int(analysis_summary.get("interact_steps_count", 0) or 0)
        interact_steps_with_change = int(analysis_summary.get("interact_steps_with_change", 0) or 0)
        click_steps = int(analysis_summary.get("click_steps_count", 0) or 0)
        click_steps_with_change = int(analysis_summary.get("click_steps_with_change", 0) or 0)
        movement_score = (float(movement_steps_with_change) / float(move_steps)) if move_steps > 0 else 0.0
        interact_score = (float(interact_steps_with_change) / float(interact_steps)) if interact_steps > 0 else 0.0
        click_score = (float(click_steps_with_change) / float(click_steps)) if click_steps > 0 else 0.0
        current = by_target.setdefault(
            str(target_id),
            {
                "movement_effect_score": 0.0,
                "interact_effect_score": 0.0,
                "click_effect_score": 0.0,
                "candidate_effect_score": 0.0,
            },
        )
        current["movement_effect_score"] = max(float(current.get("movement_effect_score", 0.0)), movement_score)
        current["interact_effect_score"] = max(float(current.get("interact_effect_score", 0.0)), interact_score)
        current["click_effect_score"] = max(float(current.get("click_effect_score", 0.0)), click_score)
        current["candidate_effect_score"] = max(
            float(current.get("candidate_effect_score", 0.0)),
            movement_score,
            interact_score,
            click_score,
        )
    return by_target


def build_session_summary(*, round_id: int, won: bool, blackboard_version: str, memory_version: str, blackboard_state: dict, selected_target_entity_ids: list[str] | None, round_records: list[dict] | None = None) -> dict:
    target_ids = {str(target_id) for target_id in (selected_target_entity_ids or []) if target_id}
    total_entities = len(dict(blackboard_state.get("entities", {})))
    entities = dict(blackboard_state.get("entities", {}))
    effect_fallbacks = _target_effect_fallbacks(list(round_records or []))
    targeted_entities = []
    for target_id in sorted(target_ids):
        if target_id not in entities:
            continue
        row = dict(entities[target_id])
        fallback = dict(effect_fallbacks.get(target_id, {}))
        if fallback:
            row["movement_effect_score"] = max(float(row.get("movement_effect_score", 0.0) or 0.0), float(fallback.get("movement_effect_score", 0.0) or 0.0))
            row["interact_effect_score"] = max(float(row.get("interact_effect_score", 0.0) or 0.0), float(fallback.get("interact_effect_score", 0.0) or 0.0))
            row["click_effect_score"] = max(float(row.get("click_effect_score", 0.0) or 0.0), float(fallback.get("click_effect_score", 0.0) or 0.0))
            row["candidate_effect_score"] = max(float(row.get("candidate_effect_score", 0.0) or 0.0), float(fallback.get("candidate_effect_score", 0.0) or 0.0))
        targeted_entities.append(row)
    effectful_targets = [row for row in targeted_entities if float(row.get("candidate_effect_score", 0.0)) > 0.0]
    movement_effectful_targets = [row for row in targeted_entities if float(row.get("movement_effect_score", 0.0)) > 0.0]
    interact_effectful_targets = [row for row in targeted_entities if float(row.get("interact_effect_score", 0.0)) > 0.0]
    click_effectful_targets = [row for row in targeted_entities if float(row.get("click_effect_score", 0.0)) > 0.0]
    summary = build_run_summary(
        rounds_completed=round_id,
        won=won,
        latest_blackboard_version=blackboard_version,
        latest_memory_version=memory_version,
        unique_target_entity_ids=len(target_ids),
        total_number_of_entities=total_entities,
    )
    summary["percentage_targets_with_effect"] = float(len(effectful_targets)) / float(len(targeted_entities)) if targeted_entities else 0.0
    summary["average_effect_strength"] = (
        sum(float(row.get("candidate_effect_score", 0.0)) for row in targeted_entities) / float(len(targeted_entities))
        if targeted_entities
        else 0.0
    )
    summary["percentage_targets_with_movement_effect"] = float(len(movement_effectful_targets)) / float(len(targeted_entities)) if targeted_entities else 0.0
    summary["percentage_targets_with_interact_effect"] = float(len(interact_effectful_targets)) / float(len(targeted_entities)) if targeted_entities else 0.0
    summary["percentage_targets_with_click_effect"] = float(len(click_effectful_targets)) / float(len(targeted_entities)) if targeted_entities else 0.0
    summary["average_movement_effect_strength"] = (
        sum(float(row.get("movement_effect_score", 0.0)) for row in targeted_entities) / float(len(targeted_entities))
        if targeted_entities
        else 0.0
    )
    summary["average_interact_effect_strength"] = (
        sum(float(row.get("interact_effect_score", 0.0)) for row in targeted_entities) / float(len(targeted_entities))
        if targeted_entities
        else 0.0
    )
    summary["average_click_effect_strength"] = (
        sum(float(row.get("click_effect_score", 0.0)) for row in targeted_entities) / float(len(targeted_entities))
        if targeted_entities
        else 0.0
    )
    return summary


def build_heatmap_payloads(*, episodes: list[dict], blackboard_state: dict, width: int, height: int) -> dict:
    visit_bundle = build_visit_heatmap(episodes, width=width, height=height)
    poi_bundle = build_poi_heatmap(blackboard_state, width=width, height=height)
    return {
        "visit_bundle": visit_bundle,
        "poi_bundle": poi_bundle,
        "visit_heatmap": visit_bundle["counts"],
        "poi_heatmap": poi_bundle["accepted_counts"],
        "visit_debug_payload": _visit_debug_payload(episodes, list(visit_bundle.get("sequence", []))),
    }


def _ledger_export_views(session_ledger) -> dict:
    records = list(getattr(session_ledger, "records", []) or [])
    stage_order: dict[int, list[str]] = {}
    decision_outcome_links: list[dict] = []
    version_transitions: list[dict] = []
    durable_flush_chronology: list[dict] = []
    stop_reason_chronology: list[dict] = []
    planner_contract_tracking: list[dict] = []
    evidence_provenance_summaries: list[dict] = []
    durable_certification_summaries: list[dict] = []
    for record in records:
        round_id = int(getattr(record, "round_id", 0) or 0)
        event_type = str(getattr(record, "event_type", "") or "")
        payload = dict(getattr(record, "payload", {}) or {})
        stage_order.setdefault(round_id, []).append(event_type)
        version_transitions.append(
            {
                "round_id": round_id,
                "pass_id": int(getattr(record, "pass_id", 0) or 0),
                "event_type": event_type,
                "blackboard_version": getattr(record, "blackboard_version", None),
                "memory_version": getattr(record, "memory_version", None),
                "plan_context_id": getattr(record, "plan_context_id", None),
            }
        )
        if getattr(record, "decision_id", None) or getattr(record, "outcome_id", None):
            decision_outcome_links.append(
                {
                    "round_id": round_id,
                    "event_type": event_type,
                    "decision_id": getattr(record, "decision_id", None),
                    "outcome_id": getattr(record, "outcome_id", None),
                    "episode_id": getattr(record, "episode_id", None),
                }
            )
        if event_type in {"durable flush requested", "durable flush completed"}:
            durable_flush_chronology.append(
                {
                    "round_id": round_id,
                    "pass_id": int(getattr(record, "pass_id", 0) or 0),
                    "event_type": event_type,
                    "timestamp": getattr(record, "timestamp", None),
                    "payload": payload,
                }
            )
        if event_type == "stop decision made":
            stop_reason_chronology.append(
                {
                    "round_id": round_id,
                    "timestamp": getattr(record, "timestamp", None),
                    "stop_reason": payload.get("stop_reason"),
                    "payload": payload,
                }
            )
        if event_type in {"probe plan selected", "directed plan selected"}:
            planner_contract_tracking.append(
                {
                    "round_id": round_id,
                    "event_type": event_type,
                    "planner_contract_mode": payload.get("planner_contract_mode"),
                }
            )
        if event_type in {"probe episode executed", "directed episode executed"}:
            evidence_provenance_summaries.append(
                {
                    "round_id": round_id,
                    "event_type": event_type,
                    "outcome_id": getattr(record, "outcome_id", None),
                    "outcome_evidence_provenance_summary": dict(payload.get("outcome_evidence_provenance_summary", {})),
                }
            )
        if event_type in {"probe memory reconcile completed", "directed memory reconcile completed"}:
            durable_certification_summaries.append(
                {
                    "round_id": round_id,
                    "event_type": event_type,
                    "durable_eligibility_summary": dict(payload.get("durable_eligibility_summary", {})),
                }
            )
    return {
        "per_round_stage_order": {str(key): value for key, value in sorted(stage_order.items())},
        "decision_outcome_links": decision_outcome_links,
        "version_transitions": version_transitions,
        "durable_flush_chronology": durable_flush_chronology,
        "stop_reason_chronology": stop_reason_chronology,
        "planner_contract_tracking": planner_contract_tracking,
        "evidence_provenance_summaries": evidence_provenance_summaries,
        "durable_certification_summaries": durable_certification_summaries,
    }


def persist_postrun_outputs(
    storage_agent,
    *,
    session_id: str,
    round_id: int,
    game_id: str,
    summary: dict,
    memory_events: list[dict],
    memory_summary: dict,
    heatmap_payloads: dict,
    first_observation: list[list[int]] | None,
    export_png: bool,
    width: int,
    height: int,
    session_ledger_payload: dict | None = None,
    mechanic_graph_payload: dict | None = None,
    mechanic_paths_payload: dict | None = None,
    mechanic_relations_summary: dict | None = None,
    deterministic_hypotheses_payload: dict | None = None,
    llm_hypotheses_payload: dict | None = None,
    hypothesis_agreement_payload: dict | None = None,
    hypothesis_validation_summary: dict | None = None,
    path_to_victory_candidates: dict | None = None,
    llm_usage_summary: dict | None = None,
    hypothesis_lifecycle_summary: dict | None = None,
    experiment_results_summary: dict | None = None,
) -> dict:
    summary_path = _persist_session(storage_agent, session_id=session_id, kind="report", name="summary.json", payload=summary)
    memory_events_jsonl_path = _persist_session_bytes(
        storage_agent,
        session_id=session_id,
        kind="report",
        name="memory_events.jsonl",
        payload=("".join(f"{dumps(row)}\n" for row in memory_events)).encode("utf-8"),
    )
    memory_summary_path = _persist_session(
        storage_agent,
        session_id=session_id,
        kind="report",
        name="memory_summary.json",
        payload=memory_summary,
    )
    heatmap_path = _persist(storage_agent, session_id=session_id, round_id=round_id, kind="heatmap", name="visit_heatmap.json", payload=heatmap_payloads["visit_heatmap"])
    poi_heatmap_path = _persist(storage_agent, session_id=session_id, round_id=round_id, kind="heatmap", name="poi_heatmap.json", payload=heatmap_payloads["poi_bundle"])
    visit_debug_path = _persist(storage_agent, session_id=session_id, round_id=round_id, kind="report", name="visit_heatmap_debug.json", payload=heatmap_payloads["visit_debug_payload"])
    exports = {
        "summary_path": summary_path,
        "memory_events_jsonl_path": memory_events_jsonl_path,
        "memory_summary_path": memory_summary_path,
        "heatmap_path": heatmap_path,
        "poi_heatmap_path": poi_heatmap_path,
        "visit_heatmap_debug_path": visit_debug_path,
    }
    if mechanic_graph_payload is not None:
        exports["mechanic_graph_path"] = _persist_session(
            storage_agent,
            session_id=session_id,
            kind="report",
            name="mechanic_graph.json",
            payload=mechanic_graph_payload,
        )
    if mechanic_paths_payload is not None:
        exports["mechanic_paths_to_exit_path"] = _persist_session(
            storage_agent,
            session_id=session_id,
            kind="report",
            name="mechanic_paths_to_exit.json",
            payload=mechanic_paths_payload,
        )
    if mechanic_relations_summary is not None:
        exports["mechanic_relations_summary_path"] = _persist_session(
            storage_agent,
            session_id=session_id,
            kind="report",
            name="mechanic_relations_summary.json",
            payload=mechanic_relations_summary,
        )
    if deterministic_hypotheses_payload is not None:
        exports["deterministic_hypotheses_path"] = _persist_session(storage_agent, session_id=session_id, kind="report", name="deterministic_hypotheses.json", payload=deterministic_hypotheses_payload)
    if llm_hypotheses_payload is not None:
        exports["llm_hypotheses_path"] = _persist_session(storage_agent, session_id=session_id, kind="report", name="llm_hypotheses.json", payload=llm_hypotheses_payload)
    if hypothesis_agreement_payload is not None:
        exports["hypothesis_agreement_path"] = _persist_session(storage_agent, session_id=session_id, kind="report", name="hypothesis_agreement.json", payload=hypothesis_agreement_payload)
    if hypothesis_validation_summary is not None:
        exports["hypothesis_validation_summary_path"] = _persist_session(storage_agent, session_id=session_id, kind="report", name="hypothesis_validation_summary.json", payload=hypothesis_validation_summary)
    if path_to_victory_candidates is not None:
        exports["path_to_victory_candidates_path"] = _persist_session(storage_agent, session_id=session_id, kind="report", name="path_to_victory_candidates.json", payload=path_to_victory_candidates)
    if llm_usage_summary is not None:
        exports["llm_usage_summary_path"] = _persist_session(storage_agent, session_id=session_id, kind="report", name="llm_usage_summary.json", payload=llm_usage_summary)
    if hypothesis_lifecycle_summary is not None:
        exports["hypothesis_lifecycle_summary_path"] = _persist_session(storage_agent, session_id=session_id, kind="report", name="hypothesis_lifecycle_summary.json", payload=hypothesis_lifecycle_summary)
    if experiment_results_summary is not None:
        exports["experiment_results_summary_path"] = _persist_session(storage_agent, session_id=session_id, kind="report", name="experiment_results_summary.json", payload=experiment_results_summary)
    if session_ledger_payload is not None:
        exports["session_ledger_path"] = _persist_session(
            storage_agent,
            session_id=session_id,
            kind="report",
            name="session_ledger.json",
            payload=session_ledger_payload,
        )
    if first_observation is not None:
        game_map_path = _persist_visualization(
            storage_agent,
            session_id=session_id,
            kind="visualization",
            name=f"{game_id}.png",
            payload=render_observation_png(first_observation, width=width, height=height, scale=15),
        )
        exports["game_map_png_path"] = game_map_path
    if export_png:
        png_path = _persist_visualization(
            storage_agent,
            session_id=session_id,
            kind="visualization",
            name="visit_heatmap.png",
            payload=render_overlay_png(first_observation, heatmap_payloads["visit_heatmap"], overlay_kind="visit", width=width, height=height, scale=15, start=heatmap_payloads["visit_bundle"].get("start"), end=heatmap_payloads["visit_bundle"].get("end")),
        )
        poi_png_path = _persist_visualization(
            storage_agent,
            session_id=session_id,
            kind="visualization",
            name="poi_heatmap.png",
            payload=render_overlay_png(first_observation, heatmap_payloads["poi_heatmap"], overlay_kind="poi", width=width, height=height, scale=15),
        )
        poi_debug_png_path = _persist_visualization(
            storage_agent,
            session_id=session_id,
            kind="visualization",
            name="poi_heatmap_debug.png",
            payload=render_heatmap_debug_png(heatmap_payloads["poi_heatmap"], overlay_kind="poi", width=width, height=height, scale=15),
        )
        visit_debug_png_path = _persist_visualization(
            storage_agent,
            session_id=session_id,
            kind="visualization",
            name="visit_heatmap_debug.png",
            payload=render_heatmap_debug_png(heatmap_payloads["visit_heatmap"], overlay_kind="visit", width=width, height=height, scale=15, start=heatmap_payloads["visit_bundle"].get("start"), end=heatmap_payloads["visit_bundle"].get("end")),
        )
        exports["heatmap_png_path"] = png_path
        exports["poi_heatmap_png_path"] = poi_png_path
        exports["poi_heatmap_debug_png_path"] = poi_debug_png_path
        exports["visit_heatmap_debug_png_path"] = visit_debug_png_path
    return exports


def export_postrun(storage_agent, *, session_id: str, round_id: int, game_id: str, episodes: list[dict], blackboard_state: dict, won: bool, blackboard_version: str, memory_version: str, width: int, height: int, export_png: bool = False, first_observation: list[list[int]] | None = None, selected_target_entity_ids: list[str] | None = None, round_records: list[dict] | None = None, session_ledger=None, mechanic_graph_state: dict | None = None, mechanic_graph_version: str | None = None) -> dict:
    direct_events, _ = _collect_direct_memory_telemetry(list(round_records or []))
    ledger_views = _ledger_export_views(session_ledger) if session_ledger is not None else None
    summary = build_session_summary(
        round_id=round_id,
        won=won,
        blackboard_version=blackboard_version,
        memory_version=memory_version,
        blackboard_state=blackboard_state,
        selected_target_entity_ids=selected_target_entity_ids,
        round_records=list(round_records or []),
    )
    if ledger_views is not None:
        summary["session_ledger_embedded"] = False
        summary["source_of_truth"] = "session_ledger"
        summary["fallback_fields_used"] = ["episodes_for_heatmaps", "round_records_for_memory_summary"]
    heatmap_payloads = build_heatmap_payloads(
        episodes=episodes,
        blackboard_state=blackboard_state,
        width=width,
        height=height,
    )
    memory_summary = _build_memory_summary(round_records=list(round_records or []), latest_memory_version=memory_version)
    if ledger_views is not None:
        memory_summary["session_ledger_embedded"] = False
        memory_summary["source_of_truth"] = "session_ledger"
        memory_summary["fallback_fields_used"] = ["round_records_for_memory_summary"]
    mechanic_graph_state = dict(mechanic_graph_state or {})
    mechanic_edges = list(dict(mechanic_graph_state.get("edges_by_id", {})).values())
    strongest_paths = sorted(mechanic_edges, key=lambda row: (-int(row.get("support_count", 0) or 0), -float(row.get("confidence", 0.0) or 0.0), str(row.get("edge_id", ""))))[:20]
    relations_summary = {
        "mechanic_graph_version": mechanic_graph_version,
        "strongest_trigger_exit_chains": [row for row in strongest_paths if str(row.get("edge_kind") or "") in {"requires", "enables_exit", "opens"}],
        "strongest_panel_gate_match_relations": [row for row in strongest_paths if str(row.get("edge_kind") or "") in {"matches", "controls_access", "displays"}],
        "contradicted_mechanic_hypotheses": [row for row in mechanic_edges if int(row.get("contradiction_count", 0) or 0) > 0],
        "graph_coverage_by_round": dict(Counter(int(round_id) for row in mechanic_edges for round_id in list(row.get("source_round_ids", []) or []))),
        "percentage_targets_with_movement_effect": summary.get("percentage_targets_with_effect", 0.0),
        "percentage_targets_with_interact_effect": _safe_rate(sum(1 for row in episodes if any(float(poi.get("interact_effect_score", 0.0) or 0.0) > 0.0 for poi in list(row.get("pois", []) or []))), max(1, len(list(episodes or [])))),
        "percentage_targets_with_click_effect": _safe_rate(sum(1 for row in episodes if any(float(poi.get("click_effect_score", 0.0) or 0.0) > 0.0 for poi in list(row.get("pois", []) or []))), max(1, len(list(episodes or [])))),
        "average_movement_effect_strength": _safe_rate(sum(float(poi.get("movement_effect_score", 0.0) or 0.0) for row in episodes for poi in list(row.get("pois", []) or [])), max(1, sum(len(list(row.get("pois", []) or [])) for row in episodes))),
        "average_interact_effect_strength": _safe_rate(sum(float(poi.get("interact_effect_score", 0.0) or 0.0) for row in episodes for poi in list(row.get("pois", []) or [])), max(1, sum(len(list(row.get("pois", []) or [])) for row in episodes))),
        "average_click_effect_strength": _safe_rate(sum(float(poi.get("click_effect_score", 0.0) or 0.0) for row in episodes for poi in list(row.get("pois", []) or [])), max(1, sum(len(list(row.get("pois", []) or [])) for row in episodes))),
    }
    latest_registry = dict((list(round_records or [])[-1].get("hypothesis_registry_snapshot", {}) if round_records else {}) or {})
    deterministic_payload = {"proposals": list(dict(latest_registry.get("deterministic_proposals", {})).values())}
    llm_payload = {"proposals": list(dict(latest_registry.get("llm_proposals", {})).values())}
    agreement_payload = {
        "deterministic_count": len(dict(latest_registry.get("deterministic_proposals", {}))),
        "llm_count": len(dict(latest_registry.get("llm_proposals", {}))),
        "validation_state": dict(latest_registry.get("validation_state", {})),
    }
    validation_summary = {
        "proposals_by_source": {
            "deterministic_hypothesis": len(dict(latest_registry.get("deterministic_proposals", {}))),
            "llm_hypothesis": len(dict(latest_registry.get("llm_proposals", {}))),
        },
        "validated_proposals_by_source": {
            "deterministic_hypothesis": sum(1 for proposal_id in dict(latest_registry.get("deterministic_proposals", {})) if str(dict(latest_registry.get("validation_state", {})).get(proposal_id, "")) == "validated"),
            "llm_hypothesis": sum(1 for proposal_id in dict(latest_registry.get("llm_proposals", {})) if str(dict(latest_registry.get("validation_state", {})).get(proposal_id, "")) == "validated"),
        },
        "contradiction_rate_by_source": {
            "deterministic_hypothesis": _safe_rate(sum(1 for row in dict(latest_registry.get("deterministic_proposals", {})).values() if len(list(row.get("contradiction_refs", []) or [])) > 0), max(1, len(dict(latest_registry.get("deterministic_proposals", {}))))),
            "llm_hypothesis": _safe_rate(sum(1 for row in dict(latest_registry.get("llm_proposals", {})).values() if len(list(row.get("contradiction_refs", []) or [])) > 0), max(1, len(dict(latest_registry.get("llm_proposals", {}))))),
        },
        "first_correct_path_round_by_source": {
            "deterministic_hypothesis": _first_validated_round(
                dict(latest_registry.get("deterministic_proposals", {})),
                dict(latest_registry.get("validation_state", {})),
            ),
            "llm_hypothesis": _first_validated_round(
                dict(latest_registry.get("llm_proposals", {})),
                dict(latest_registry.get("validation_state", {})),
            ),
        },
        "source_agreement_on_final_winning_path": len(dict(latest_registry.get("deterministic_proposals", {}))) > 0 and len(dict(latest_registry.get("llm_proposals", {}))) > 0,
    }
    path_to_victory_payload = {
        "candidates": [row for row in strongest_paths if str(row.get("edge_kind") or "") in {"requires", "enables_exit", "opens"}],
    }
    llm_proposals = dict(latest_registry.get("llm_proposals", {}))
    deterministic_proposals = dict(latest_registry.get("deterministic_proposals", {}))
    ledger_records = list(getattr(session_ledger, "records", []) or []) if session_ledger is not None else []
    llm_event_payloads = [
        dict(getattr(record, "payload", {}) or {})
        for record in ledger_records
        if str(getattr(record, "event_type", "")) in {"llm call skipped", "llm call attempted", "llm call failed", "llm call succeeded"}
    ]
    llm_task_roles = Counter(str(payload.get("gating_reason") or payload.get("provider_name") or "unknown") for payload in llm_event_payloads)
    llm_prompt_modes = Counter(str(payload.get("prompt_mode") or "unknown") for payload in llm_event_payloads if payload)
    llm_query_target_kinds = Counter()
    prompt_char_counts = [int(payload.get("prompt_char_count", 0) or 0) for payload in llm_event_payloads if int(payload.get("prompt_char_count", 0) or 0) > 0]
    if not prompt_char_counts:
        prompt_char_counts = [
            int(dict(row.get("metadata", {}) or {}).get("prompt_char_count", 0) or 0)
            for row in llm_proposals.values()
            if int(dict(row.get("metadata", {}) or {}).get("prompt_char_count", 0) or 0) > 0
        ]
    if not llm_prompt_modes:
        llm_prompt_modes = Counter(
            str(dict(row.get("metadata", {}) or {}).get("prompt_mode") or "unknown")
            for row in llm_proposals.values()
            if row
        )
    prompt_trim_count = sum(1 for payload in llm_event_payloads if bool(payload.get("prompt_trim_applied", False)))
    prompt_budget_skip_count = sum(1 for payload in llm_event_payloads if str(payload.get("skip_reason") or payload.get("gating_reason") or "") in {"prompt_too_large_after_trimming", "prompt_budget_exceeded"})
    for payload in llm_event_payloads:
        query_target_id = str(payload.get("query_target_id") or "")
        if not query_target_id:
            llm_query_target_kinds["unknown"] += 1
            continue
        target_kind = query_target_id.split(":", 1)[0] if ":" in query_target_id else "unknown"
        llm_query_target_kinds[target_kind] += 1
    if not llm_query_target_kinds:
        for row in llm_proposals.values():
            query_target_id = str(dict(row.get("metadata", {}) or {}).get("query_target_id") or "")
            if not query_target_id:
                llm_query_target_kinds["unknown"] += 1
                continue
            target_kind = query_target_id.split(":", 1)[0] if ":" in query_target_id else "unknown"
            llm_query_target_kinds[target_kind] += 1
    accepted_llm_by_role = Counter(str(dict(row.get("metadata", {}) or {}).get("task_role") or "unknown") for row in llm_proposals.values())
    llm_rejection_counts = Counter()
    for row in llm_proposals.values():
        llm_rejection_counts.update(dict(dict(row.get("metadata", {}) or {}).get("rejection_reason_counts", {}) or {}))
    llm_usage_summary = {
        "proposals_generated_by_source": {
            "deterministic_hypothesis": len(deterministic_proposals),
            "llm_hypothesis": len(llm_proposals),
        },
        "proposals_validated_by_source": {
            "deterministic_hypothesis": sum(1 for proposal_id in deterministic_proposals if str(dict(latest_registry.get("validation_state", {})).get(proposal_id, "")) == "validated"),
            "llm_hypothesis": sum(1 for proposal_id in llm_proposals if str(dict(latest_registry.get("validation_state", {})).get(proposal_id, "")) == "validated"),
        },
        "proposals_contradicted_by_source": {
            "deterministic_hypothesis": sum(1 for proposal_id in deterministic_proposals if str(dict(latest_registry.get("validation_state", {})).get(proposal_id, "")) == "contradicted"),
            "llm_hypothesis": sum(1 for proposal_id in llm_proposals if str(dict(latest_registry.get("validation_state", {})).get(proposal_id, "")) == "contradicted"),
        },
        "llm_call_attempt_count": sum(1 for record in list(getattr(session_ledger, "records", []) or []) if str(getattr(record, "event_type", "")) == "llm call attempted") if session_ledger is not None else 0,
        "llm_call_success_count": sum(1 for record in list(getattr(session_ledger, "records", []) or []) if str(getattr(record, "event_type", "")) == "llm call succeeded") if session_ledger is not None else 0,
        "parameter_preset_used": sorted({str(dict(row.get("metadata", {}) or {}).get("parameter_preset_used") or "") for row in llm_proposals.values() if dict(row.get("metadata", {}) or {}).get("parameter_preset_used")}),
        "calls_by_task_role": dict(llm_task_roles),
        "schema_valid_response_rate": _safe_rate(sum(1 for payload in llm_event_payloads if payload.get("error_code") in {None, ""}), max(1, len([payload for payload in llm_event_payloads if payload]))),
        "think_rejection_count": int(llm_rejection_counts.get("think_content", 0)),
        "prose_plus_json_rejection_count": int(llm_rejection_counts.get("non_json_wrapper_text", 0)),
        "accepted_proposal_count_by_task_role": dict(accepted_llm_by_role),
        "average_prompt_size": (sum(prompt_char_counts) / float(len(prompt_char_counts))) if prompt_char_counts else 0.0,
        "max_prompt_size": max(prompt_char_counts) if prompt_char_counts else 0,
        "prompt_trim_count": int(prompt_trim_count),
        "prompt_budget_skip_count": int(prompt_budget_skip_count),
        "calls_per_prompt_mode": dict(llm_prompt_modes),
        "calls_per_query_target_kind": dict(llm_query_target_kinds),
    }
    hypothesis_lifecycle_summary = {
        "proposal_lifecycle_state": dict(latest_registry.get("proposal_lifecycle_state", {})),
        "first_support_round": dict(latest_registry.get("first_support_round", {})),
        "first_contradiction_round": dict(latest_registry.get("first_contradiction_round", {})),
        "first_validation_round": dict(latest_registry.get("first_validation_round", {})),
        "source_agreement_groups": dict(latest_registry.get("source_agreement_groups", {})),
        "first_supported_path_round": min([int(value) for value in dict(latest_registry.get("first_support_round", {})).values()]) if dict(latest_registry.get("first_support_round", {})) else None,
        "first_validated_winning_path_round": min([int(value) for value in dict(latest_registry.get("first_validation_round", {})).values()]) if dict(latest_registry.get("first_validation_round", {})) else None,
    }
    experiment_results_summary = {
        "experiments_run_by_source": {
            "deterministic_hypothesis": sum(1 for row in deterministic_proposals.values() if str(row.get("proposal_kind", "")) == "test"),
            "llm_hypothesis": sum(1 for row in llm_proposals.values() if str(row.get("proposal_kind", "")) == "test"),
        },
        "experiment_supports_by_source": {
            "deterministic_hypothesis": sum(1 for row in deterministic_proposals.values() if list(dict(row.get("metadata", {}) or {}).get("experiment_supports_hypothesis_ids", []) or [])),
            "llm_hypothesis": sum(1 for row in llm_proposals.values() if list(dict(row.get("metadata", {}) or {}).get("experiment_supports_hypothesis_ids", []) or [])),
        },
        "experiment_contradicts_by_source": {
            "deterministic_hypothesis": sum(1 for row in deterministic_proposals.values() if list(dict(row.get("metadata", {}) or {}).get("experiment_contradicts_hypothesis_ids", []) or [])),
            "llm_hypothesis": sum(1 for row in llm_proposals.values() if list(dict(row.get("metadata", {}) or {}).get("experiment_contradicts_hypothesis_ids", []) or [])),
        },
    }
    return persist_postrun_outputs(
        storage_agent,
        session_id=session_id,
        round_id=round_id,
        game_id=game_id,
        summary=summary,
        memory_events=direct_events,
        memory_summary=memory_summary,
        heatmap_payloads=heatmap_payloads,
        first_observation=first_observation,
        export_png=export_png,
        width=width,
        height=height,
        session_ledger_payload={"records": session_ledger.to_dicts(), "views": ledger_views} if session_ledger is not None else None,
        mechanic_graph_payload={"mechanic_graph_version": mechanic_graph_version, **mechanic_graph_state},
        mechanic_paths_payload={"mechanic_graph_version": mechanic_graph_version, "paths": strongest_paths},
        mechanic_relations_summary=relations_summary,
        deterministic_hypotheses_payload=deterministic_payload,
        llm_hypotheses_payload=llm_payload,
        hypothesis_agreement_payload=agreement_payload,
        hypothesis_validation_summary=validation_summary,
        path_to_victory_candidates=path_to_victory_payload,
        llm_usage_summary=llm_usage_summary,
        hypothesis_lifecycle_summary=hypothesis_lifecycle_summary,
        experiment_results_summary=experiment_results_summary,
    )
