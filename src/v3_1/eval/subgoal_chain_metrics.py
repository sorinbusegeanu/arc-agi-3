from __future__ import annotations

from collections import Counter


def compute_subgoal_chain_metrics(chain_events: list[dict], *, total_rounds: int | None = None) -> dict:
    rows = [dict(row) for row in list(chain_events or [])]
    payloads = [dict(row.get("payload", {}) or {}) for row in rows]
    started = [row for row in rows if str(row.get("event_type") or "") == "subgoal chain started"]
    completed = [row for row in rows if str(row.get("event_type") or "") == "subgoal chain completed"]
    aborted = [row for row in rows if str(row.get("event_type") or "") == "subgoal chain aborted"]
    step_completed = [row for row in rows if str(row.get("event_type") or "") == "subgoal chain step completed"]
    step_failed = [row for row in rows if str(row.get("event_type") or "") == "subgoal chain step failed"]
    verification_step_events = [row for row in rows if str(dict(row.get("payload", {}) or {}).get("step_kind") or "") in {"verify_panel", "verify_gate"}]
    trigger_step_events = [row for row in rows if str(dict(row.get("payload", {}) or {}).get("step_kind") or "") in {"go_to_trigger", "retry_trigger"}]
    verification_step_successes = [row for row in step_completed if str(dict(row.get("payload", {}) or {}).get("step_kind") or "") in {"verify_panel", "verify_gate"}]
    trigger_step_successes = [row for row in step_completed if str(dict(row.get("payload", {}) or {}).get("step_kind") or "") in {"go_to_trigger", "retry_trigger"}]
    panel_verification_yield = len([row for row in verification_step_successes if str(dict(row.get("payload", {}) or {}).get("step_kind") or "") == "verify_panel"])
    gate_verification_yield = len([row for row in verification_step_successes if str(dict(row.get("payload", {}) or {}).get("step_kind") or "") == "verify_gate"])
    abort_reasons = Counter(str(dict(row.get("payload", {}) or {}).get("failure_reason") or "unknown") for row in aborted)
    started_chain_ids = {str(dict(row.get("payload", {}) or {}).get("chain_id") or "") for row in started if dict(row.get("payload", {}) or {}).get("chain_id")}
    completed_chain_ids = {str(dict(row.get("payload", {}) or {}).get("chain_id") or "") for row in completed if dict(row.get("payload", {}) or {}).get("chain_id")}
    false_chain_pursuit_count = len([row for row in aborted if str(dict(row.get("payload", {}) or {}).get("failure_reason") or "") not in {"", "all_steps_completed"}])
    replan_related = [row for row in aborted if str(dict(row.get("payload", {}) or {}).get("advancement_reason") or "") in {"chain_aborted", "replan"}]
    return {
        "chain_selection_rate": float(len(started)) / float(max(1, int(total_rounds or len({row.get('round_id') for row in rows}) or 1))),
        "chain_completion_rate": float(len(completed_chain_ids)) / float(max(1, len(started_chain_ids))),
        "chain_abort_rate": float(len(aborted)) / float(max(1, len(started))),
        "verification_step_success_rate": float(len(verification_step_successes)) / float(max(1, len(verification_step_events))),
        "trigger_step_success_rate": float(len(trigger_step_successes)) / float(max(1, len(trigger_step_events))),
        "panel_verification_yield": panel_verification_yield,
        "gate_verification_yield": gate_verification_yield,
        "exit_success_after_completed_chain": len([row for row in completed if str(dict(row.get("payload", {}) or {}).get("step_kind") or "") == "attempt_exit"]),
        "false_chain_pursuit_count": false_chain_pursuit_count,
        "average_replan_count_per_chain": float(len(replan_related)) / float(max(1, len(started_chain_ids))),
        "abort_reason_counts": dict(abort_reasons),
        "step_completion_count": len(step_completed),
        "step_failure_count": len(step_failed),
        "unique_started_chain_count": len(started_chain_ids),
    }
