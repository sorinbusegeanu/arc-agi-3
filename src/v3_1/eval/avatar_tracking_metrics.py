from __future__ import annotations


def build_avatar_tracking_metrics(events: list[dict]) -> dict:
    total = len(list(events or []))
    confident = [row for row in list(events or []) if bool(row.get("avatar_localization_confident", False))]
    fallback = [row for row in list(events or []) if str(row.get("avatar_source_after") or row.get("avatar_source") or "") == "static_fallback"]
    ambiguous = [row for row in list(events or []) if bool(row.get("avatar_localization_ambiguous", False))]
    route_confident = [row for row in confident if str(row.get("termination_reason") or "") not in {"missing_avatar", "avatar_localization_low_confidence"}]
    trigger_confident = [row for row in confident if bool(row.get("trigger_contact_based_on_confident_avatar", False))]
    exit_confident = [row for row in confident if bool(row.get("target_approach_based_on_confident_avatar", False))]
    stale_static_failures = [row for row in fallback if str(row.get("termination_reason") or "") in {"blocked", "stalled", "avatar_localization_low_confidence"}]
    return {
        "confident_localization_rate": (len(confident) / float(total)) if total else 0.0,
        "fallback_localization_rate": (len(fallback) / float(total)) if total else 0.0,
        "ambiguous_localization_rate": (len(ambiguous) / float(total)) if total else 0.0,
        "route_step_success_with_confident_avatar": (len(route_confident) / float(len(confident))) if confident else 0.0,
        "trigger_contact_support_with_confident_avatar": (len(trigger_confident) / float(len(confident))) if confident else 0.0,
        "exit_attempt_support_with_confident_avatar": (len(exit_confident) / float(len(confident))) if confident else 0.0,
        "stale_static_cell_failure_count": len(stale_static_failures),
    }
