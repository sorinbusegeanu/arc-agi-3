from __future__ import annotations

from collections import Counter


def score_deterministic_proposal(proposal) -> dict:
    episode_support_count = len({ref.ref_id.split(":")[0] for ref in proposal.support_refs})
    round_support_count = 1 if proposal.round_id else 0
    observed_support_count = sum(1 for ref in proposal.support_refs if str(ref.evidence_tier) == "observed")
    directed_outcome_support_count = sum(1 for ref in proposal.support_refs if str(ref.provenance) in {"env_native", "analysis"})
    contradiction_count = len(tuple(proposal.contradiction_refs))
    contradiction_recency_score = 1.0 if contradiction_count > 0 else 0.0
    lag_consistency_score = 1.0 if len(tuple(proposal.support_refs)) <= 1 else 0.8
    support_consistency_score = min(1.0, 0.25 * len(tuple(proposal.support_refs)))
    confidence = min(
        1.0,
        0.15
        + (0.12 * observed_support_count)
        + (0.08 * directed_outcome_support_count)
        + (0.08 * episode_support_count)
        + (0.08 * support_consistency_score)
        - (0.15 * contradiction_count),
    )
    return {
        "episode_support_count": episode_support_count,
        "round_support_count": round_support_count,
        "observed_support_count": observed_support_count,
        "directed_outcome_support_count": directed_outcome_support_count,
        "contradiction_count": contradiction_count,
        "contradiction_recency_score": contradiction_recency_score,
        "lag_consistency_score": lag_consistency_score,
        "support_consistency_score": support_consistency_score,
        "confidence": confidence,
    }


def summarize_support(proposals: list) -> dict:
    counts = Counter(str(proposal.proposal_kind) for proposal in list(proposals or []))
    return {"proposal_kind_counts": dict(counts), "proposal_count": sum(counts.values())}
