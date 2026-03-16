from __future__ import annotations


def fallback_candidates(candidates: list[dict], blocked_candidates: list[dict], belief: dict) -> list[dict]:
    fallback_rows = [dict(row) for row in candidates if str(row.get("objective_type")) == "fallback"]
    if fallback_rows:
        return fallback_rows
    blocked = list(blocked_candidates or [])
    current_area_id = dict(belief or {}).get("current_area_id")
    reason = "all_main_candidates_blocked" if blocked else "no_generated_candidates"
    return [
        {
            "candidate_id": f"fallback:{current_area_id or 'global'}:{reason}",
            "candidate_class": "fallback_action",
            "objective_type": "fallback",
            "execution_mode": "move",
            "navigation_mode": "hold",
            "target_area_id": current_area_id,
            "required_action_family": "move",
            "reachable_now": True,
            "reachable_later": True,
            "action": {"type": "hold_position", "area_id": current_area_id},
            "fallback_reason": reason,
            "blocked_reasons": [],
            "rationale": reason,
            "fallback_baseline_penalty": 2.0,
        }
    ]
