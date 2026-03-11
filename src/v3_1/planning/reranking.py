from __future__ import annotations


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

    reranked = []
    for row in candidates:
        candidate = dict(row)
        helper_boost = helper_boosts.get(candidate["candidate_id"], 0.0)
        helper_penalty = helper_penalties.get(candidate["candidate_id"], 0.0)
        tie_break = (
            0.15 if candidate.get("reachable_now") else 0.0,
            0.08 if candidate.get("candidate_class") == "trigger_probe" else 0.0,
            0.04 * float(candidate.get("utility", 0.0)),
            -0.05 * float(belief.get("retries", {}).get(candidate["candidate_id"], 0)),
        )
        final_score = float(candidate.get("score", 0.0)) + helper_boost - helper_penalty + sum(tie_break)
        candidate["helper_boost"] = helper_boost
        candidate["helper_penalty"] = helper_penalty
        candidate["final_score"] = final_score
        reranked.append(candidate)

    reranked.sort(
        key=lambda item: (
            -float(item.get("final_score", 0.0)),
            not bool(item.get("reachable_now")),
            not bool(item.get("reachable_later")),
            -float(item.get("utility", 0.0)),
            item["candidate_id"],
        )
    )
    return reranked
