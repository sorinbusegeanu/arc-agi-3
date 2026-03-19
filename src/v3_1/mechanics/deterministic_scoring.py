from __future__ import annotations

from collections import Counter


def score_deterministic_proposal(proposal) -> dict:
    episode_support_count = len({ref.ref_id.split(":")[0] for ref in proposal.support_refs})
    round_support_count = len({getattr(proposal, "round_id", 0)})
    observed_support_count = sum(1 for ref in proposal.support_refs if str(ref.evidence_tier) == "observed")
    directed_outcome_support_count = sum(1 for ref in proposal.support_refs if str(ref.provenance) in {"env_native", "analysis"})
    contradiction_count = len(tuple(proposal.contradiction_refs))
    contradiction_recency_score = 1.0 if contradiction_count > 0 else 0.0
    lag_consistency_score = 1.0 if len(tuple(proposal.support_refs)) <= 2 else 0.65
    support_consistency_score = min(1.0, 0.2 * len(tuple(proposal.support_refs)))
    probe_only_support = int(directed_outcome_support_count <= 0 and observed_support_count > 0)
    one_episode_only_penalty = 0.15 if episode_support_count <= 1 else 0.0
    unsupported_long_chain_penalty = 0.12 if len(tuple(getattr(proposal, "edge_kinds", ()))) >= 3 and observed_support_count <= 1 else 0.0
    exit_attempt_evidence_bonus = 0.12 if any("exit" in str(ref.ref_kind or "") for ref in proposal.support_refs) else 0.0
    counterfactual_bonus = 0.14 if any("counterfactual" in str(ref.ref_kind or "") for ref in proposal.support_refs) else 0.0
    repeated_trigger_before_exit_bonus = 0.1 if any("trigger_before_exit" in str(ref.ref_kind or "") for ref in proposal.support_refs) and observed_support_count >= 2 else 0.0
    repeated_detector_poi_support_bonus = 0.12 if any("poi_visit" in str(ref.ref_kind or "") for ref in proposal.support_refs) and observed_support_count >= 2 else 0.0
    post_visit_remote_change_bonus = 0.1 if any("poi_visit_then_remote_change" in str(ref.ref_kind or "") for ref in proposal.support_refs) else 0.0
    post_visit_panel_gate_bonus = 0.1 if any(str(ref.ref_kind or "") in {"poi_visit_then_panel_change", "poi_visit_then_gate_change"} for ref in proposal.support_refs) else 0.0
    post_visit_exit_shift_bonus = 0.1 if any("poi_visit_then_exit_attempt_change" in str(ref.ref_kind or "") for ref in proposal.support_refs) else 0.0
    repeated_probe_no_effect_penalty = 0.16 if any("repeated_probe_without_new_effect" in str(ref.ref_kind or "") for ref in proposal.support_refs) else 0.0
    no_downstream_link_penalty = 0.1 if any("poi_visit" in str(ref.ref_kind or "") for ref in proposal.support_refs) and not any("poi_visit_then_" in str(ref.ref_kind or "") for ref in proposal.support_refs) else 0.0
    confidence = min(
        1.0,
        0.15
        + (0.12 * observed_support_count)
        + (0.08 * directed_outcome_support_count)
        + (0.08 * episode_support_count)
        + (0.08 * support_consistency_score)
        + exit_attempt_evidence_bonus
        + counterfactual_bonus
        + repeated_trigger_before_exit_bonus
        + repeated_detector_poi_support_bonus
        + post_visit_remote_change_bonus
        + post_visit_panel_gate_bonus
        + post_visit_exit_shift_bonus
        - one_episode_only_penalty
        - (0.12 * probe_only_support)
        - unsupported_long_chain_penalty
        - repeated_probe_no_effect_penalty
        - no_downstream_link_penalty
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
        "probe_only_support": probe_only_support,
        "one_episode_only_penalty": one_episode_only_penalty,
        "unsupported_long_chain_penalty": unsupported_long_chain_penalty,
        "exit_attempt_evidence_bonus": exit_attempt_evidence_bonus,
        "counterfactual_bonus": counterfactual_bonus,
        "repeated_trigger_before_exit_bonus": repeated_trigger_before_exit_bonus,
        "repeated_detector_poi_support_bonus": repeated_detector_poi_support_bonus,
        "post_visit_remote_change_bonus": post_visit_remote_change_bonus,
        "post_visit_panel_gate_bonus": post_visit_panel_gate_bonus,
        "post_visit_exit_shift_bonus": post_visit_exit_shift_bonus,
        "repeated_probe_no_effect_penalty": repeated_probe_no_effect_penalty,
        "no_downstream_link_penalty": no_downstream_link_penalty,
        "confidence": confidence,
    }


def summarize_support(proposals: list) -> dict:
    counts = Counter(str(proposal.proposal_kind) for proposal in list(proposals or []))
    return {"proposal_kind_counts": dict(counts), "proposal_count": sum(counts.values())}
