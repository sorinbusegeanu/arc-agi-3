from __future__ import annotations

CLASS_PRIORITY = {
    "target": 0,
    "click_target": 1,
    "trigger_probe": 2,
    "frontier_move": 3,
    "route_probe": 4,
    "local_probe": 5,
    "recovery_move": 6,
    "fallback_action": 7,
    "fallback_hold": 8,
}


def rerank_candidates(candidates: list[dict], helper_proposals: list[dict], belief: dict) -> list[dict]:
    helper_boosts: dict[str, float] = {}
    helper_penalties: dict[str, float] = {}

    for helper_result in helper_proposals:
        for proposal in helper_result.get("proposals", []):
            candidate_id = proposal.get("candidate_id")
            if not candidate_id:
                continue
            helper_boosts[candidate_id] = helper_boosts.get(candidate_id, 0.0) + float(proposal.get("score_delta", 0.0))
            helper_penalties[candidate_id] = helper_penalties.get(candidate_id, 0.0) + float(proposal.get("risk_delta", 0.0))

    pre_score_order = [str(row.get("candidate_id")) for row in candidates]
    reranked = []
    for row in candidates:
        candidate = dict(row)
        helper_boost = helper_boosts.get(candidate["candidate_id"], 0.0)
        helper_penalty = helper_penalties.get(candidate["candidate_id"], 0.0)
        retry_row = belief.get("retries", {}).get(candidate["candidate_id"], 0)
        retry_count = int(retry_row.get("recent_failures", 0)) if isinstance(retry_row, dict) else int(retry_row or 0)
        tie_break = {
            "reachable_now_bonus": 0.15 if candidate.get("reachable_now") else 0.0,
            "priority_bonus": max(0.0, 0.08 - (0.01 * CLASS_PRIORITY.get(str(candidate.get("candidate_class")), 99))),
            "utility_bonus": 0.04 * float(candidate.get("utility", 0.0)),
            "retry_penalty": -0.05 * float(retry_count),
        }
        final_score = float(candidate.get("score", 0.0)) + helper_boost - helper_penalty + sum(tie_break.values())
        candidate["helper_boost"] = helper_boost
        candidate["helper_penalty"] = helper_penalty
        candidate["final_score"] = final_score
        candidate["rerank_diagnostics"] = {
            "tie_break": tie_break,
            "pre_score_rank_hint": pre_score_order.index(candidate["candidate_id"]) if candidate["candidate_id"] in pre_score_order else None,
        }
        reranked.append(candidate)

    reranked.sort(
        key=lambda item: (
            -float(item.get("final_score", 0.0)),
            CLASS_PRIORITY.get(str(item.get("candidate_class")), 99),
            not bool(item.get("reachable_now")),
            not bool(item.get("reachable_later")),
            -float(item.get("utility", 0.0)),
            item["candidate_id"],
        )
    )
    post_score_order = [str(row.get("candidate_id")) for row in reranked]
    decisive_terms = {}
    if reranked:
        winner = reranked[0]
        breakdown = dict(winner.get("score_breakdown", {}))
        decisive_terms = dict(sorted(
            ((str(key), float(value)) for key, value in breakdown.items() if isinstance(value, (int, float))),
            key=lambda item: -abs(item[1]),
        )[:5])
    for row in reranked:
        diagnostics = dict(row.get("rerank_diagnostics", {}))
        diagnostics["pre_score_order"] = pre_score_order
        diagnostics["post_score_order"] = post_score_order
        diagnostics["decisive_terms_for_winner"] = decisive_terms
        row["rerank_diagnostics"] = diagnostics
    return reranked
