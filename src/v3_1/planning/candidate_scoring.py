from __future__ import annotations

DEFAULT_CLASS_WEIGHTS = {
    "interact": {"utility": 1.1, "progress": 1.0, "novelty": 0.5},
    "gather_local_info": {"utility": 0.8, "progress": 0.7, "novelty": 0.95},
    "explore_frontier": {"utility": 0.7, "progress": 1.05, "novelty": 1.0},
    "probe_route": {"utility": 0.65, "progress": 1.0, "novelty": 0.85},
    "test_trigger": {"utility": 1.0, "progress": 0.85, "novelty": 0.6},
    "recover": {"utility": 0.6, "progress": 0.8, "novelty": 0.4},
    "fallback": {"utility": 0.1, "progress": 0.1, "novelty": 0.0},
}


def score_candidates(
    candidates: list[dict],
    route_features: dict[str, dict],
    planning_cfg,
    *,
    observed_world: dict | None = None,
    hypothesized_world: dict | None = None,
    uncertainty_context: dict | None = None,
    durable_prior_context: dict | None = None,
    belief_fallback: dict | None = None,
) -> list[dict]:
    belief_fallback = dict(belief_fallback or {})
    observed_world = dict(observed_world or belief_fallback.get("observed_world", {}) or {})
    hypothesized_world = dict(hypothesized_world or belief_fallback.get("hypothesized_world", {}) or {})
    uncertainty_context = dict(uncertainty_context or belief_fallback.get("uncertainty_context", {}) or {})
    tactical_memory_view = dict(uncertainty_context.get("tactical_memory", {}) or {})
    tactical_context = dict(tactical_memory_view.get("tactical_context", {}))
    retries = dict(tactical_memory_view.get("retries", {}))
    cooldowns = dict(tactical_memory_view.get("cooldowns", {}))
    exhausted_keys = {str(key) for key in set(tactical_memory_view.get("exhausted_keys", set()))}
    observed_consequences = list(observed_world.get("consequences", {}).values()) if isinstance(observed_world.get("consequences"), dict) else list(observed_world.get("consequences", []))
    hypothesized_consequences = list(hypothesized_world.get("consequences", {}).values()) if isinstance(hypothesized_world.get("consequences"), dict) else list(hypothesized_world.get("consequences", []))
    observed_triggers = dict(observed_world.get("trigger_zones", {}))
    hypothesized_triggers = dict(hypothesized_world.get("trigger_zones", {}))
    evidence_index = dict(uncertainty_context.get("evidence_index", {}) or {})
    durable_prior_view = dict(durable_prior_context or belief_fallback.get("durable_prior_context", {}) or {})
    candidate_outcome_priors = dict(durable_prior_view.get("candidate_outcomes", {}))
    poi_priors = dict(durable_prior_view.get("poi_patterns", {}))
    trigger_priors = dict(durable_prior_view.get("trigger_patterns", {}))
    consequence_priors = dict(durable_prior_view.get("consequence_patterns", {}))
    recovery_priors = dict(durable_prior_view.get("recovery_patterns", {}))
    versions = dict(uncertainty_context.get("versions", {}) or belief_fallback.get("versions", {}) or {})

    scored = []
    for row in candidates:
        candidate = dict(row)
        objective_type = str(candidate.get("objective_type") or "fallback")
        weights = dict(DEFAULT_CLASS_WEIGHTS.get(objective_type, {}))
        target_entity_id = candidate.get("target_entity_id")
        route_row = route_features.get(candidate["candidate_id"], {})
        novelty = float(candidate.get("novelty", 0.0)) * float(weights.get("novelty", 1.0))
        reachability = 1.0 if candidate.get("reachable_now") else (0.45 if candidate.get("reachable_later") else -0.5)
        progress = float(route_row.get("progress_potential", 0.0)) * float(weights.get("progress", 1.0))
        utility = float(candidate.get("utility", 0.0)) * float(weights.get("utility", 1.0))
        candidate_retry = retries.get(candidate["candidate_id"], 0)
        target_retry = retries.get(str(target_entity_id), 0)
        candidate_retry_count = int(candidate_retry.get("recent_failures", 0)) if isinstance(candidate_retry, dict) else int(candidate_retry or 0)
        target_retry_count = int(target_retry.get("recent_failures", 0)) if isinstance(target_retry, dict) else int(target_retry or 0)
        retry_penalty = float(getattr(planning_cfg, "retry_penalty_weight", 0.18)) * float(candidate_retry_count + target_retry_count)

        active_cooldown = False
        for key in (candidate.get("candidate_id"), target_entity_id, candidate.get("target_area_id"), candidate.get("route_signature"), candidate.get("trigger_zone_id")):
            if key is None:
                continue
            row_value = cooldowns.get(str(key), 0)
            if isinstance(row_value, dict) and int(row_value.get("remaining_rounds", 0) or 0) > 0:
                active_cooldown = True
                break
            if not isinstance(row_value, dict) and int(row_value or 0) > 0:
                active_cooldown = True
                break
        cooldown_penalty = 0.25 if active_cooldown else 0.0
        exhaustion_penalty = 0.35 if any(str(key) in exhausted_keys for key in (candidate.get("candidate_id"), target_entity_id, candidate.get("target_area_id")) if key is not None) else 0.0
        neighborhood_exhaustion_penalty = 0.15 if any(str(key) in exhausted_keys for key in (candidate.get("route_signature"), candidate.get("trigger_zone_id"), candidate.get("target_area_id")) if key is not None) else 0.0

        route_risk = float(route_row.get("risk", 0.0))
        route_cost = float(route_row.get("cost", 0.0))
        route_uncertainty = float(route_row.get("uncertainty", route_risk * 0.5))
        trigger_bonus = 0.08 * (
            int(str(candidate.get("trigger_zone_id") or "") in observed_triggers)
            + sum(1 for row in observed_triggers.values() if str(row.get("entity_id") or "") == str(target_entity_id or ""))
        )
        trigger_uncertainty = 0.0 if trigger_bonus else (0.3 if objective_type == "test_trigger" else 0.0)
        action_hint = str(dict(candidate.get("action", {})).get("action_hint") or dict(candidate.get("action", {})).get("action_name") or "")
        observed_consequence_count = sum(1 for row in observed_consequences if str(row.get("action_name") or row.get("action_key") or "") == action_hint)
        hypothesized_consequence_count = sum(1 for row in hypothesized_consequences if str(row.get("action_name") or row.get("action_key") or "") == action_hint)
        consequence_bonus = 0.05 * observed_consequence_count

        full_support_refs = list(candidate.get("full_supporting_evidence_refs", candidate.get("supporting_evidence_refs", [])))
        support_freshness = 1.0 if full_support_refs and all(str(ref) in evidence_index for ref in full_support_refs) else (0.5 if full_support_refs else 0.0)
        contradiction_penalty = 0.0
        contradiction_count = sum(1 for value in dict(candidate.get("contradiction_flags", {})).values() if value)
        if full_support_refs and support_freshness < 1.0:
            contradiction_penalty += 0.35
        contradiction_penalty += 0.08 * contradiction_count
        graph_hop_count = int(candidate.get("hop_count", 0) or 0)
        graph_observed_bonus = 0.08 if graph_hop_count > 0 and not bool(candidate.get("depends_on_hypothesized_only_edges")) else 0.0
        graph_path_bonus = 0.12 if str(candidate.get("candidate_class") or "") in {"unlock_then_exit", "trigger_then_target", "unlock_trigger"} else 0.0
        graph_pattern_bonus = 0.08 if str(candidate.get("candidate_class") or "") in {"verify_panel_state", "verify_gate_match", "state_sync_probe"} else 0.0
        graph_hypothesis_penalty = 0.18 if bool(candidate.get("depends_on_hypothesized_only_edges")) else 0.0
        graph_long_chain_penalty = 0.05 * max(0, graph_hop_count - 2)
        graph_stale_penalty = 0.05 if contradiction_count > 0 and graph_hop_count > 0 else 0.0
        hypothesis_source = str(dict(candidate.get("action", {})).get("hypothesis_source") or candidate.get("hypothesis_source") or "")
        registry_snapshot = dict(belief_fallback.get("hypothesis_registry_snapshot", {}) or {})
        validation_state = "validated" if any(str(registry_snapshot.get("validation_state", {}).get(hypothesis_id, "")) == "validated" for hypothesis_id in list(dict(candidate.get("action", {})).get("supporting_hypothesis_ids", []) or [])) else "new"
        agreement_groups = dict(registry_snapshot.get("agreement_groups", {}) or {})
        agreement_score = 0.2 if any(hypothesis_id in agreement_groups for hypothesis_id in list(dict(candidate.get("action", {})).get("supporting_hypothesis_ids", []) or [])) else 0.0
        llm_only_penalty = 0.22 if hypothesis_source == "llm" and validation_state != "validated" else 0.0
        deterministic_priority_bonus = 0.16 if hypothesis_source == "deterministic" and validation_state == "validated" else 0.08 if hypothesis_source == "deterministic" else 0.0
        validated_llm_bonus = 0.06 if hypothesis_source == "llm" and validation_state == "validated" else 0.0
        contradiction_hypothesis_penalty = 0.15 if contradiction_count > 0 and hypothesis_source in {"deterministic", "llm"} else 0.0

        target_area_id = candidate.get("target_area_id")
        target_area_key = f"{target_area_id or uncertainty_context.get('current_area_id') or belief_fallback.get('current_area_id') or 'global'}:{target_entity_id or 'none'}"
        local_outcome = dict(tactical_context.get("recent_local_outcomes", {}).get(target_area_key, {}))
        local_failures = int(local_outcome.get("failures", 0))
        local_successes = int(local_outcome.get("successes", 0))
        local_failure_risk = float(local_failures) / float(max(1, local_failures + local_successes))

        progress_type_score = 0.0
        expected_progress_type = str(candidate.get("expected_progress_type", "fallback"))
        if expected_progress_type == "state_change":
            progress_type_score = 0.25 + (0.35 * float(candidate.get("candidate_effect_score", 0.0)))
        elif expected_progress_type == "evidence_gain":
            progress_type_score = 0.15 + (0.3 * float(candidate.get("expected_outcomes", {}).get("expected_evidence_gain", 0.0)))
        elif expected_progress_type == "route_progress":
            progress_type_score = 0.15 + (0.3 * float(candidate.get("expected_outcomes", {}).get("expected_route_progress", 0.0)))
        else:
            progress_type_score = 0.02

        target_key = str(candidate.get("target_key") or "")
        target_prior = dict(durable_prior_view.get("per_target", {}).get(target_key, {}))
        poi_signature = str(candidate.get("target_entity_id") or "")
        prior_candidate = dict(candidate_outcome_priors.get(objective_type, {}))
        prior_attempts = max(1, int(prior_candidate.get("attempts", 0) or 0))
        prior_success_rate = float(prior_candidate.get("successes", 0)) / float(prior_attempts)
        prior_failure_rate = float(prior_candidate.get("failures", 0)) / float(prior_attempts)
        prior_route_failure_risk = float(prior_candidate.get("route_failures", 0)) / float(prior_attempts)
        prior_poi = dict(target_prior.get("poi_pattern", {})) or dict(poi_priors.get(poi_signature, {}))
        prior_poi_utility = float(prior_poi.get("utility_total", 0.0)) / float(max(1, int(prior_poi.get("observations", 0) or 0)))
        prior_trigger = dict(trigger_priors.get(str(candidate.get("trigger_zone_id") or target_entity_id or ""), {}))
        prior_trigger_bonus = float(prior_trigger.get("confidence_total", 0.0)) / float(max(1, int(prior_trigger.get("observations", 0) or 0)))
        prior_consequence = dict(consequence_priors.get(str(candidate.get("candidate_id")), {}))
        prior_consequence_bonus = float(prior_consequence.get("reward_total", 0.0)) / float(max(1, int(prior_consequence.get("observations", 0) or 0)))
        prior_recovery = dict(recovery_priors.get(str(candidate.get("target_area_id") or target_entity_id or "global"), {}))
        recovery_usefulness = float(prior_recovery.get("successes", 0)) - float(prior_recovery.get("failures", 0))
        durable_prior_strength = max(0.0, prior_success_rate + prior_poi_utility + prior_trigger_bonus)
        seed_requires_hypothesis_penalty = 0.18 if bool(candidate.get("seed_requires_hypothesis")) else 0.0
        zero_observed_support_penalty = 0.14 if len(list(candidate.get("seed_observed_row_ids", []))) == 0 else 0.0
        contradiction_seed_penalty = 0.12 if contradiction_count > 0 else 0.0
        compatibility_alias_penalty = 0.1 if str(candidate.get("seed_contract") or "") == "compatibility_fallback" else 0.0
        direct_observed_support_bonus = 0.12 if len(list(candidate.get("seed_observed_row_ids", []))) > 0 else 0.0
        repeated_observed_support_bonus = 0.08 if len(list(candidate.get("seed_observed_row_ids", []))) > 1 else 0.0
        directed_outcome_backed_bonus = 0.1 if float(candidate.get("support_strength", {}).get("direct_support", 0.0)) > 0.5 and objective_type in {"interact", "test_trigger", "recover"} else 0.0
        score_used_compatibility_fallback = bool(str(candidate.get("seed_contract") or "") == "compatibility_fallback")

        score = (
            novelty * float(getattr(planning_cfg, "novelty_weight", 0.6))
            + utility * float(getattr(planning_cfg, "utility_weight", 1.0))
            + float(getattr(planning_cfg, "reachability_weight", 0.55)) * reachability
            + float(getattr(planning_cfg, "progress_weight", 1.0)) * progress
            + float(getattr(planning_cfg, "progress_type_weight", 0.12)) * progress_type_score
            + float(getattr(planning_cfg, "support_freshness_weight", 0.08)) * support_freshness
            + float(getattr(planning_cfg, "trigger_bonus_weight", 0.08)) * (trigger_bonus / 0.08 if trigger_bonus else 0.0)
            + float(getattr(planning_cfg, "consequence_bonus_weight", 0.05)) * (consequence_bonus / 0.05 if consequence_bonus else 0.0)
            + float(getattr(planning_cfg, "durable_prior_strength_weight", 0.06)) * durable_prior_strength
            + 0.06 * prior_poi_utility
            + 0.05 * prior_trigger_bonus
            + 0.04 * prior_consequence_bonus
            + (0.05 * recovery_usefulness if objective_type == "recover" else 0.0)
            + direct_observed_support_bonus
            + repeated_observed_support_bonus
            + directed_outcome_backed_bonus
            + graph_observed_bonus
            + graph_path_bonus
            + graph_pattern_bonus
            + deterministic_priority_bonus
            + validated_llm_bonus
            + agreement_score
            - retry_penalty
            - cooldown_penalty
            - exhaustion_penalty
            - neighborhood_exhaustion_penalty
            - float(getattr(planning_cfg, "route_risk_weight", 0.35)) * route_risk
            - float(getattr(planning_cfg, "route_cost_weight", 0.12)) * route_cost
            - float(getattr(planning_cfg, "route_uncertainty_weight", 0.09)) * route_uncertainty
            - float(getattr(planning_cfg, "trigger_uncertainty_weight", 0.07)) * trigger_uncertainty
            - float(getattr(planning_cfg, "local_failure_risk_weight", 0.14)) * local_failure_risk
            - float(getattr(planning_cfg, "contradiction_penalty_weight", 1.0)) * contradiction_penalty
            - graph_hypothesis_penalty
            - graph_long_chain_penalty
            - graph_stale_penalty
            - llm_only_penalty
            - contradiction_hypothesis_penalty
            - seed_requires_hypothesis_penalty
            - zero_observed_support_penalty
            - contradiction_seed_penalty
            - compatibility_alias_penalty
            - 0.08 * prior_failure_rate
            - 0.07 * prior_route_failure_risk
            - (0.05 * hypothesized_consequence_count)
            - float(candidate.get("score_penalty_soft_filters", 0.0))
            - float(candidate.get("fallback_baseline_penalty", 0.0))
        )

        uncertainty = min(1.0, max(route_uncertainty, trigger_uncertainty, (1.0 - support_freshness if full_support_refs else 0.5)))
        confidence = max(0.0, min(1.0, 0.5 + (0.25 * support_freshness) + (0.2 * prior_success_rate) - (0.2 * uncertainty)))
        exploration_bias = max(0.0, uncertainty - confidence) if objective_type in {"gather_local_info", "explore_frontier", "probe_route"} else 0.0
        recovery_bias = max(0.0, uncertainty - 0.4) if objective_type == "recover" else 0.0
        helper_escalation = bool(uncertainty >= 0.65 and objective_type in {"test_trigger", "probe_route", "recover"})
        score += float(getattr(planning_cfg, "exploration_bias_weight", 0.12)) * exploration_bias
        score += float(getattr(planning_cfg, "recovery_bias_weight", 0.1)) * recovery_bias

        candidate["score"] = score
        candidate["score_confidence"] = confidence
        candidate["score_uncertainty"] = uncertainty
        candidate["helper_escalation"] = helper_escalation
        candidate["score_contract_mode"] = "split_world_native" if not score_used_compatibility_fallback else "compatibility_fallback"
        candidate["score_used_compatibility_fallback"] = score_used_compatibility_fallback
        candidate["score_breakdown"] = {
            "schema_version": "v3_1_planner_score_v3",
            "objective_type": objective_type,
            "novelty": novelty,
            "reachability": reachability,
            "progress": progress,
            "utility": utility,
            "progress_type_score": progress_type_score,
            "support_freshness": support_freshness,
            "contradiction_penalty": contradiction_penalty,
            "local_failure_risk": local_failure_risk,
            "neighborhood_exhaustion_penalty": neighborhood_exhaustion_penalty,
            "route_risk": route_risk,
            "route_cost": route_cost,
            "route_uncertainty": route_uncertainty,
            "trigger_bonus": trigger_bonus,
            "trigger_uncertainty": trigger_uncertainty,
            "consequence_bonus": consequence_bonus,
            "retry_penalty": retry_penalty,
            "cooldown_penalty": cooldown_penalty,
            "exhaustion_penalty": exhaustion_penalty,
            "durable_prior_strength": durable_prior_strength,
            "seed_requires_hypothesis_penalty": seed_requires_hypothesis_penalty,
            "zero_observed_support_penalty": zero_observed_support_penalty,
            "contradiction_seed_penalty": contradiction_seed_penalty,
            "compatibility_alias_penalty": compatibility_alias_penalty,
            "direct_observed_support_bonus": direct_observed_support_bonus,
            "repeated_observed_support_bonus": repeated_observed_support_bonus,
            "directed_outcome_backed_bonus": directed_outcome_backed_bonus,
            "graph_observed_bonus": graph_observed_bonus,
            "graph_path_bonus": graph_path_bonus,
            "graph_pattern_bonus": graph_pattern_bonus,
            "graph_hypothesis_penalty": graph_hypothesis_penalty,
            "graph_long_chain_penalty": graph_long_chain_penalty,
            "graph_stale_penalty": graph_stale_penalty,
            "hypothesis_source": hypothesis_source,
            "validation_state": validation_state,
            "agreement_score": agreement_score,
            "llm_only_penalty": llm_only_penalty,
            "deterministic_priority_bonus": deterministic_priority_bonus,
            "validated_llm_bonus": validated_llm_bonus,
            "contradiction_hypothesis_penalty": contradiction_hypothesis_penalty,
            "score_contract_mode": candidate["score_contract_mode"],
            "score_used_compatibility_fallback": score_used_compatibility_fallback,
            "observed_consequence_support_count": observed_consequence_count,
            "hypothesized_consequence_support_count": hypothesized_consequence_count,
            "observed_trigger_support_count": int(trigger_bonus / 0.08) if trigger_bonus else 0,
            "hypothesized_trigger_support_count": int(str(candidate.get("trigger_zone_id") or "") in hypothesized_triggers),
            "prior_success_rate": prior_success_rate,
            "prior_failure_rate": prior_failure_rate,
            "prior_route_failure_risk": prior_route_failure_risk,
            "prior_poi_utility": prior_poi_utility,
            "prior_trigger_bonus": prior_trigger_bonus,
            "prior_consequence_bonus": prior_consequence_bonus,
            "prior_recovery_usefulness": recovery_usefulness,
            "score_confidence": confidence,
            "score_uncertainty": uncertainty,
            "exploration_bias": exploration_bias,
            "recovery_bias": recovery_bias,
            "helper_escalation": helper_escalation,
            "config_hash_inputs": {
                "novelty_weight": float(getattr(planning_cfg, "novelty_weight", 0.6)),
                "utility_weight": float(getattr(planning_cfg, "utility_weight", 1.0)),
                "progress_weight": float(getattr(planning_cfg, "progress_weight", 1.0)),
            },
            "freshness_versions": versions,
        }
        scored.append(candidate)
    return scored
