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


def export_postrun(storage_agent, *, session_id: str, round_id: int, game_id: str, episodes: list[dict], blackboard_state: dict, won: bool, blackboard_version: str, memory_version: str, width: int, height: int, export_png: bool = False, first_observation: list[list[int]] | None = None, selected_target_entity_ids: list[str] | None = None, round_records: list[dict] | None = None) -> dict:
    direct_events, _ = _collect_direct_memory_telemetry(list(round_records or []))
    target_ids = {str(target_id) for target_id in (selected_target_entity_ids or []) if target_id}
    total_entities = len(dict(blackboard_state.get("entities", {})))
    entities = dict(blackboard_state.get("entities", {}))
    targeted_entities = [entities[target_id] for target_id in sorted(target_ids) if target_id in entities]
    effectful_targets = [row for row in targeted_entities if float(row.get("candidate_effect_score", 0.0)) > 0.0]
    movement_effectful_targets = [row for row in targeted_entities if float(row.get("movement_effect_score", 0.0)) > 0.0]
    interact_effectful_targets = [row for row in targeted_entities if float(row.get("interact_effect_score", 0.0)) > 0.0]
    click_effectful_targets = [row for row in targeted_entities if float(row.get("click_effect_score", 0.0)) > 0.0]
    percentage_targets_with_effect = float(len(effectful_targets)) / float(len(targeted_entities)) if targeted_entities else 0.0
    percentage_targets_with_movement_effect = float(len(movement_effectful_targets)) / float(len(targeted_entities)) if targeted_entities else 0.0
    percentage_targets_with_interact_effect = float(len(interact_effectful_targets)) / float(len(targeted_entities)) if targeted_entities else 0.0
    percentage_targets_with_click_effect = float(len(click_effectful_targets)) / float(len(targeted_entities)) if targeted_entities else 0.0
    average_effect_strength = (
        sum(float(row.get("candidate_effect_score", 0.0)) for row in targeted_entities) / float(len(targeted_entities))
        if targeted_entities
        else 0.0
    )
    average_movement_effect_strength = (
        sum(float(row.get("movement_effect_score", 0.0)) for row in targeted_entities) / float(len(targeted_entities))
        if targeted_entities
        else 0.0
    )
    average_interact_effect_strength = (
        sum(float(row.get("interact_effect_score", 0.0)) for row in targeted_entities) / float(len(targeted_entities))
        if targeted_entities
        else 0.0
    )
    average_click_effect_strength = (
        sum(float(row.get("click_effect_score", 0.0)) for row in targeted_entities) / float(len(targeted_entities))
        if targeted_entities
        else 0.0
    )
    summary = build_run_summary(
        rounds_completed=round_id,
        won=won,
        latest_blackboard_version=blackboard_version,
        latest_memory_version=memory_version,
        unique_target_entity_ids=len(target_ids),
        total_number_of_entities=total_entities,
    )
    summary["percentage_targets_with_effect"] = percentage_targets_with_effect
    summary["average_effect_strength"] = average_effect_strength
    summary["percentage_targets_with_movement_effect"] = percentage_targets_with_movement_effect
    summary["percentage_targets_with_interact_effect"] = percentage_targets_with_interact_effect
    summary["percentage_targets_with_click_effect"] = percentage_targets_with_click_effect
    summary["average_movement_effect_strength"] = average_movement_effect_strength
    summary["average_interact_effect_strength"] = average_interact_effect_strength
    summary["average_click_effect_strength"] = average_click_effect_strength
    visit_bundle = build_visit_heatmap(episodes, width=width, height=height)
    poi_bundle = build_poi_heatmap(blackboard_state, width=width, height=height)
    heatmap = visit_bundle["counts"]
    poi_heatmap = poi_bundle["accepted_counts"]
    summary_path = _persist_session(storage_agent, session_id=session_id, kind="report", name="summary.json", payload=summary)
    memory_events_jsonl_path = _persist_session_bytes(
        storage_agent,
        session_id=session_id,
        kind="report",
        name="memory_events.jsonl",
        payload=("".join(f"{dumps(row)}\n" for row in direct_events)).encode("utf-8"),
    )
    memory_summary_path = _persist_session(
        storage_agent,
        session_id=session_id,
        kind="report",
        name="memory_summary.json",
        payload=_build_memory_summary(round_records=list(round_records or []), latest_memory_version=memory_version),
    )
    heatmap_path = _persist(storage_agent, session_id=session_id, round_id=round_id, kind="heatmap", name="visit_heatmap.json", payload=heatmap)
    poi_heatmap_path = _persist(storage_agent, session_id=session_id, round_id=round_id, kind="heatmap", name="poi_heatmap.json", payload=poi_bundle)
    visit_debug_path = _persist(storage_agent, session_id=session_id, round_id=round_id, kind="report", name="visit_heatmap_debug.json", payload=_visit_debug_payload(episodes, list(visit_bundle.get("sequence", []))))
    exports = {"summary_path": summary_path, "memory_events_jsonl_path": memory_events_jsonl_path, "memory_summary_path": memory_summary_path, "heatmap_path": heatmap_path, "poi_heatmap_path": poi_heatmap_path, "visit_heatmap_debug_path": visit_debug_path}
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
            payload=render_overlay_png(first_observation, heatmap, overlay_kind="visit", width=width, height=height, scale=15, start=visit_bundle.get("start"), end=visit_bundle.get("end")),
        )
        poi_png_path = _persist_visualization(
            storage_agent,
            session_id=session_id,
            kind="visualization",
            name="poi_heatmap.png",
            payload=render_overlay_png(first_observation, poi_heatmap, overlay_kind="poi", width=width, height=height, scale=15),
        )
        poi_debug_png_path = _persist_visualization(
            storage_agent,
            session_id=session_id,
            kind="visualization",
            name="poi_heatmap_debug.png",
            payload=render_heatmap_debug_png(poi_heatmap, overlay_kind="poi", width=width, height=height, scale=15),
        )
        visit_debug_png_path = _persist_visualization(
            storage_agent,
            session_id=session_id,
            kind="visualization",
            name="visit_heatmap_debug.png",
            payload=render_heatmap_debug_png(heatmap, overlay_kind="visit", width=width, height=height, scale=15, start=visit_bundle.get("start"), end=visit_bundle.get("end")),
        )
        exports["heatmap_png_path"] = png_path
        exports["poi_heatmap_png_path"] = poi_png_path
        exports["poi_heatmap_debug_png_path"] = poi_debug_png_path
        exports["visit_heatmap_debug_png_path"] = visit_debug_png_path
    return exports
