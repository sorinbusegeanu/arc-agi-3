from __future__ import annotations


def fallback_candidates(candidates: list[dict], blocked_candidates: list[dict], belief: dict) -> list[dict]:
    fallbacks = []
    seen_ids: set[str] = set()
    for row in candidates + blocked_candidates:
        if row["candidate_id"] in seen_ids:
            continue
        if row.get("candidate_class") in {"local_probe", "frontier_move", "recovery_move", "route_probe"}:
            fallback = dict(row)
            fallback["fallback_reason"] = "structured_fallback"
            fallbacks.append(fallback)
            seen_ids.add(row["candidate_id"])
    if not fallbacks:
        current_area_id = belief.get("current_area_id")
        fallbacks.append(
            {
                "candidate_id": "fallback:hold_position",
                "candidate_class": "fallback_hold",
                "target_entity_id": None,
                "target_area_id": current_area_id,
                "action": {"type": "hold_position", "area_id": current_area_id},
                "confidence": 0.0,
                "utility": 0.0,
                "novelty": 0.0,
                "reachable_now": True,
                "reachable_later": False,
                "score": -1.0,
                "final_score": -1.0,
                "blocked_reasons": [],
                "fallback_reason": "no_valid_candidates",
            }
        )
    return fallbacks
