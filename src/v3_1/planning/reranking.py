from __future__ import annotations


OBJECTIVE_PRIORITY = {
    "interact": 0,
    "test_trigger": 1,
    "explore_frontier": 2,
    "probe_route": 3,
    "gather_local_info": 4,
    "recover": 5,
    "fallback": 6,
}


def rerank_candidates(
    candidates: list[dict],
    helper_proposals: list[dict],
    *,
    observed_world: dict | None = None,
    hypothesized_world: dict | None = None,
    uncertainty_context: dict | None = None,
    durable_prior_context: dict | None = None,
    belief_fallback: dict | None = None,
) -> list[dict]:
    belief_fallback = dict(belief_fallback or {})
    observed_world = dict(observed_world or belief_fallback.get("observed_world", {}))
    hypothesized_world = dict(hypothesized_world or belief_fallback.get("hypothesized_world", {}))
    uncertainty_context = dict(uncertainty_context or belief_fallback.get("uncertainty_context", {}))
    durable_prior_context = dict(durable_prior_context or belief_fallback.get("durable_prior_context", belief_fallback.get("durable_prior_view", {})))
    helper_boosts: dict[str, float] = {}
    helper_penalties: dict[str, float] = {}
    helper_warning_codes: dict[str, set[str]] = {}
    helper_contradictions: dict[str, dict] = {}
    helper_support_adjustments: dict[str, dict] = {}
    helper_confidence_samples: dict[str, list[float]] = {}
    for helper_result in helper_proposals:
        for proposal in helper_result.get("proposals", []):
            candidate_id = proposal.get("candidate_id")
            if not candidate_id:
                continue
            helper_boosts[candidate_id] = helper_boosts.get(candidate_id, 0.0) + float(proposal.get("score_delta", 0.0))
            helper_penalties[candidate_id] = helper_penalties.get(candidate_id, 0.0) + float(proposal.get("risk_delta", 0.0))
            helper_warning_codes.setdefault(candidate_id, set()).update(str(code) for code in list(proposal.get("hard_warning_reason_codes", [])) if code)
            helper_confidence_samples.setdefault(candidate_id, []).append(float(proposal.get("confidence", 0.0)))
            existing_contradictions = helper_contradictions.setdefault(candidate_id, {})
            for key, value in dict(proposal.get("contradiction_flags", {})).items():
                existing_contradictions[str(key)] = bool(existing_contradictions.get(str(key), False) or bool(value))
            existing_support = helper_support_adjustments.setdefault(candidate_id, {})
            for key, value in dict(proposal.get("support_strength_adjustments", {})).items():
                existing_support[str(key)] = float(existing_support.get(str(key), 0.0)) + float(value)

    pre_score_order = [str(row.get("candidate_id")) for row in candidates]
    reranked = []
    for row in candidates:
        candidate = dict(row)
        seed_observed = bool(list(candidate.get("seed_observed_row_ids", [])))
        seed_hypothesis = bool(list(candidate.get("seed_hypothesis_row_ids", [])))
        compatibility_fallback = bool(str(candidate.get("seed_contract") or "") == "compatibility_fallback")
        dependency_chain_bonus = 0.0
        chosen_dependency_path = list(candidate.get("prerequisite_chain", []) or [])
        if str(candidate.get("candidate_class") or "") in {"unlock_then_exit", "trigger_then_target"} and len(chosen_dependency_path) >= 3:
            dependency_chain_bonus = 0.2 + (0.03 * len(chosen_dependency_path))
        contradiction_status = bool(candidate.get("depends_on_hypothesized_only_edges")) or bool(dict(candidate.get("helper_contradiction_flags", {})))
        selected_hypothesis_source = str(dict(candidate.get("action", {})).get("hypothesis_source") or candidate.get("hypothesis_source") or "")
        validation_state = str(dict(candidate.get("score_breakdown", {})).get("validation_state") or "none")
        graph_support_strength = float(candidate.get("candidate_effect_score", 0.0) or 0.0)
        agreement_score = float(dict(candidate.get("score_breakdown", {})).get("agreement_score", 0.0) or 0.0)
        chain_verification_count = len(list(candidate.get("candidate_verification_points", []) or []))
        chain_counterfactual_strength = float(dict(candidate.get("score_breakdown", {})).get("chain_counterfactual_strength", 0.0) or 0.0)
        chain_executability_score = float(dict(candidate.get("score_breakdown", {})).get("chain_step_executability_score", 0.0) or 0.0)
        chain_evidence_diversity = float(min(1.0, len(set(list(candidate.get("supporting_graph_node_ids", []) or []) + list(candidate.get("supporting_graph_edge_ids", []) or []))) / 6.0))
        chain_identity_stability = float(dict(candidate.get("score_breakdown", {})).get("chain_identity_stability", candidate.get("identity_confidence", 0.0)) or 0.0)
        planner_usable_bonus = float(dict(candidate.get("score_breakdown", {})).get("planner_usable_hypothesis_bonus", 0.0) or 0.0)
        helper_boost = helper_boosts.get(candidate["candidate_id"], 0.0)
        helper_penalty = helper_penalties.get(candidate["candidate_id"], 0.0)
        prior_target = dict(durable_prior_context.get("per_target", {}).get(str(candidate.get("target_key") or ""), {}))
        prior_bonus = 0.02 * float(prior_target.get("success_rate", 0.0) or 0.0)
        uncertainty_penalty = 0.05 * float(candidate.get("score_uncertainty", 0.0) or 0.0)
        final_score = float(candidate.get("score", 0.0)) + helper_boost - helper_penalty + prior_bonus - uncertainty_penalty + dependency_chain_bonus + (0.05 * chain_executability_score) + (0.03 * chain_evidence_diversity) + (0.03 * min(1.0, chain_counterfactual_strength)) + (0.03 * min(1.0, chain_identity_stability)) + planner_usable_bonus
        candidate["helper_boost"] = helper_boost
        candidate["helper_penalty"] = helper_penalty
        candidate["helper_warning_reason_codes"] = sorted(helper_warning_codes.get(candidate["candidate_id"], set()))
        candidate["helper_contradiction_flags"] = dict(helper_contradictions.get(candidate["candidate_id"], {}))
        candidate["helper_support_strength_adjustments"] = dict(helper_support_adjustments.get(candidate["candidate_id"], {}))
        candidate["helper_confidence"] = (
            sum(helper_confidence_samples.get(candidate["candidate_id"], [])) / float(max(1, len(helper_confidence_samples.get(candidate["candidate_id"], []))))
            if helper_confidence_samples.get(candidate["candidate_id"])
            else 0.0
        )
        candidate["final_score"] = final_score
        candidate["rerank_diagnostics"] = {
            "pre_score_rank_hint": pre_score_order.index(candidate["candidate_id"]) if candidate["candidate_id"] in pre_score_order else None,
            "rerank_contract_mode": "split_world_native",
            "rerank_used_observed_support": seed_observed,
            "rerank_used_hypothesis_support": seed_hypothesis,
            "rerank_used_compatibility_fallback": compatibility_fallback,
            "uncertainty_penalty": uncertainty_penalty,
            "durable_prior_bonus": prior_bonus,
            "dependency_chain_bonus": dependency_chain_bonus,
            "observed_support_count": len(list(candidate.get("seed_observed_row_ids", []))),
            "hypothesis_support_count": len(list(candidate.get("seed_hypothesis_row_ids", []))),
            "chosen_dependency_path": chosen_dependency_path,
            "path_support_strength": float(candidate.get("candidate_effect_score", 0.0) or 0.0),
            "prerequisite_completeness": max(0.0, min(1.0, float(len(chosen_dependency_path)) / 4.0)),
            "contradiction_status": contradiction_status,
            "selected_hypothesis_source": selected_hypothesis_source,
            "agreement_score": agreement_score,
            "validation_state": validation_state,
            "graph_support_strength": graph_support_strength,
            "chain_executability_score": chain_executability_score,
            "chain_verification_count": chain_verification_count,
            "chain_counterfactual_strength": chain_counterfactual_strength,
            "chain_evidence_diversity": chain_evidence_diversity,
            "chain_identity_stability": chain_identity_stability,
            "planner_usable_bonus": planner_usable_bonus,
            "uncertainty_versions": dict(uncertainty_context.get("versions", {})),
            "observed_world_counts": {
                "reachable": len(list(observed_world.get("reachable_targets", []))),
                "trigger_candidates": len(list(observed_world.get("trigger_candidates", []))),
            },
            "hypothesized_world_counts": {
                "reachable": len(list(hypothesized_world.get("reachable_targets", []))),
                "trigger_candidates": len(list(hypothesized_world.get("trigger_candidates", []))),
            },
        }
        reranked.append(candidate)

    reranked.sort(
        key=lambda item: (
            -float(item.get("final_score", 0.0)),
            OBJECTIVE_PRIORITY.get(str(item.get("objective_type")), 99),
            item["candidate_id"],
        )
    )
    post_score_order = [str(row.get("candidate_id")) for row in reranked]
    decisive_terms = {}
    if reranked:
        winner = reranked[0]
        breakdown = dict(winner.get("score_breakdown", {}))
        decisive_terms = dict(sorted(((str(key), float(value)) for key, value in breakdown.items() if isinstance(value, (int, float))), key=lambda item: -abs(item[1]))[:5])
    for row in reranked:
        row["rerank_diagnostics"]["pre_score_order"] = pre_score_order
        row["rerank_diagnostics"]["post_score_order"] = post_score_order
        row["rerank_diagnostics"]["decisive_terms_for_winner"] = decisive_terms
    return reranked
