from __future__ import annotations


def fallback_candidates(candidates: list[dict], blocked_candidates: list[dict], belief: dict) -> list[dict]:
    fallbacks = []
    seen_ids: set[str] = set()
    del blocked_candidates
    for row in candidates:
        if row["candidate_id"] in seen_ids:
            continue
        if row.get("blocked") or list(row.get("blocked_reasons", []) or []):
            continue
        if row.get("candidate_class") in {"local_probe", "frontier_move", "recovery_move", "route_probe", "fallback_action"}:
            fallback = dict(row)
            fallback["fallback_reason"] = "structured_fallback"
            fallbacks.append(fallback)
            seen_ids.add(row["candidate_id"])
    if not fallbacks:
        current_area_id = belief.get("current_area_id")
        fallbacks.append(
            {
                "candidate_id": "fallback:hold_position",
                "candidate_class": "fallback_action",
                "target_entity_id": None,
                "target_area_id": current_area_id,
                "target_key": f"target:fallback:{current_area_id or 'none'}:none",
                "skill_id": None,
                "skill_type": None,
                "required_action_family": "move",
                "effect_action_family": "move",
                "expected_progress_type": "fallback",
                "route_required": False,
                "route_signature": f"route:fallback:{current_area_id or 'none'}:none",
                "candidate_context": {
                    "avatar_area": belief.get("local_context", {}).get("current_area_id"),
                    "local_area": current_area_id,
                    "route_signature": f"route:fallback:{current_area_id or 'none'}:none",
                    "trigger_zone_id": None,
                    "target_entity_class": "fallback",
                },
                "expected_outcomes": {"expected_state_change": 0.0, "expected_evidence_gain": 0.05, "expected_route_progress": 0.0},
                "support_strength": {"direct_support": 0.0, "indirect_support": 0.0, "prior_support": 0.0},
                "contradiction_flags": {},
                "stale_support_flags": {},
                "supporting_evidence_refs": [],
                "action": {"type": "hold_position", "area_id": current_area_id, "skill_id": None},
                "confidence": 0.0,
                "utility": 0.0,
                "novelty": 0.0,
                "reachable_now": True,
                "reachable_later": False,
                "score": -1.0,
                "final_score": -1.0,
                "blocked_reasons": [],
                "blocked_reason_details": [],
                "fallback_reason": "no_valid_candidates",
            }
        )
    return fallbacks
