from __future__ import annotations

CLASS_WEIGHT_OVERRIDES = {
    "target": {"utility": 1.1, "progress": 1.0, "novelty": 0.5},
    "click_target": {"utility": 1.05, "progress": 0.9, "novelty": 0.45},
    "local_probe": {"utility": 0.8, "progress": 0.7, "novelty": 0.95},
    "frontier_move": {"utility": 0.7, "progress": 1.05, "novelty": 1.0},
    "route_probe": {"utility": 0.65, "progress": 1.0, "novelty": 0.85},
    "trigger_probe": {"utility": 1.0, "progress": 0.85, "novelty": 0.6},
    "recovery_move": {"utility": 0.6, "progress": 0.8, "novelty": 0.4},
    "fallback_action": {"utility": 0.1, "progress": 0.1, "novelty": 0.0},
}


def score_candidates(candidates: list[dict], belief: dict, route_features: dict[str, dict], planning_cfg) -> list[dict]:
    scored = []
    retries = dict(belief.get("retries", {}))
    cooldowns = dict(belief.get("cooldowns", {}))
    exhausted_keys = {str(key) for key in set(belief.get("exhausted_keys", set()))}
    trigger_support = dict(belief.get("trigger_support", {}))
    candidate_outcome_priors = dict(belief.get("persistent_candidate_outcomes", {}))
    poi_priors = dict(belief.get("persistent_poi_patterns", {}))
    trigger_priors = dict(belief.get("persistent_trigger_patterns", {}))
    consequence_priors = dict(belief.get("persistent_consequence_patterns", {}))
    recovery_priors = dict(belief.get("persistent_recovery_patterns", {}))
    localized_context = dict(belief.get("localized_context", {}))
    evidence_index = dict(belief.get("indexes", {}).get("evidence_index", {}))
    versions = dict(belief.get("versions", {}))

    for row in candidates:
        candidate = dict(row)
        candidate_class = str(candidate.get("candidate_class") or "unknown")
        class_weights = dict(CLASS_WEIGHT_OVERRIDES.get(candidate_class, {}))
        target_entity_id = candidate.get("target_entity_id")
        route_row = route_features.get(candidate["candidate_id"], {})
        novelty = float(candidate.get("novelty", 0.0)) * float(class_weights.get("novelty", 1.0))
        reachability = 1.0 if candidate.get("reachable_now") else (0.45 if candidate.get("reachable_later") else -0.5)
        progress = float(route_row.get("progress_potential", 0.0)) * float(class_weights.get("progress", 1.0))
        candidate_retry = retries.get(candidate["candidate_id"], 0)
        target_retry = retries.get(str(target_entity_id), 0)
        candidate_retry_count = int(candidate_retry.get("recent_failures", 0)) if isinstance(candidate_retry, dict) else int(candidate_retry or 0)
        target_retry_count = int(target_retry.get("recent_failures", 0)) if isinstance(target_retry, dict) else int(target_retry or 0)
        retry_penalty = 0.18 * float(candidate_retry_count + target_retry_count)
        target_area_id = candidate.get("target_area_id")
        active_cooldown = False
        for key in (candidate.get("candidate_id"), target_entity_id, target_area_id, candidate.get("route_signature"), candidate.get("trigger_zone_id")):
            if key is None:
                continue
            cooldown_row = cooldowns.get(str(key), 0)
            if isinstance(cooldown_row, dict) and int(cooldown_row.get("remaining_rounds", 0) or 0) > 0:
                active_cooldown = True
                break
            if not isinstance(cooldown_row, dict) and int(cooldown_row or 0) > 0:
                active_cooldown = True
                break
        cooldown_penalty = 0.25 if active_cooldown else 0.0
        exhaustion_penalty = 0.35 if any(str(key) in exhausted_keys for key in (candidate.get("candidate_id"), target_entity_id, target_area_id) if key is not None) else 0.0
        utility = float(candidate.get("utility", 0.0)) * float(class_weights.get("utility", 1.0))
        movement_effect_score = float(candidate.get("movement_effect_score", 0.0))
        interact_effect_score = float(candidate.get("interact_effect_score", 0.0))
        click_effect_score = float(candidate.get("click_effect_score", 0.0))
        candidate_effect_score = float(candidate.get("candidate_effect_score", 0.0))
        distance_score = float(candidate.get("distance_score", 0.0))
        motion_score = float(candidate.get("motion_score", 0.0))
        route_risk = float(route_row.get("risk", 0.0))
        route_cost = float(route_row.get("cost", 0.0))
        route_uncertainty = float(route_row.get("uncertainty", route_risk * 0.5))
        expected_progress_type = str(candidate.get("expected_progress_type", "movement"))
        if expected_progress_type == "interaction":
            progress_type_score = 0.2 + (0.3 * interact_effect_score)
        elif expected_progress_type == "route_probe":
            progress_type_score = 0.15 + (0.2 * progress)
        elif expected_progress_type == "fallback":
            progress_type_score = 0.02
        else:
            progress_type_score = 0.15 + (0.2 * movement_effect_score)
        trigger_bonus = 0.08 * len(trigger_support.get(str(target_entity_id), []))
        trigger_uncertainty = 0.0 if trigger_bonus else (0.3 if candidate_class == "trigger_probe" else 0.0)
        consequence_bonus = 0.05 * len(belief.get("consequence_support", {}).get(str(candidate.get("action")), []))
        support_refs = list(candidate.get("supporting_evidence_refs", []))
        support_freshness = 1.0 if support_refs and all(str(ref) in evidence_index for ref in support_refs) else (0.5 if support_refs else 0.0)
        contradiction_penalty = 0.35 if support_refs and support_freshness < 1.0 else 0.0
        local_zone_key = f"{candidate.get('target_area_id') or belief.get('current_area_id') or 'global'}:{target_entity_id or candidate.get('candidate_id')}"
        local_zone = dict(localized_context.get("by_zone", {}).get(local_zone_key, {}))
        local_failures = int(local_zone.get("failures", 0))
        local_successes = int(local_zone.get("successes", 0))
        local_failure_risk = float(local_failures) / float(max(1, local_failures + local_successes))
        neighborhood_exhaustion_penalty = 0.15 if any(str(key) in exhausted_keys for key in (candidate.get("route_signature"), candidate.get("trigger_zone_id"), target_area_id) if key is not None) else 0.0
        prior_candidate = dict(candidate_outcome_priors.get(str(candidate.get("candidate_class")), {}))
        prior_attempts = max(1, int(prior_candidate.get("attempts", 0) or 0))
        prior_success_rate = float(prior_candidate.get("successes", 0)) / float(prior_attempts)
        prior_failure_rate = float(prior_candidate.get("failures", 0)) / float(prior_attempts)
        prior_route_failure_risk = float(prior_candidate.get("route_failures", 0)) / float(prior_attempts)
        prior_poi = dict(poi_priors.get(str(candidate.get("signature") or target_entity_id or ""), {}))
        prior_poi_utility = float(prior_poi.get("utility_total", 0.0)) / float(max(1, int(prior_poi.get("observations", 0) or 0)))
        prior_trigger = dict(trigger_priors.get(str(target_entity_id or ""), {}))
        prior_trigger_bonus = float(prior_trigger.get("confidence_total", 0.0)) / float(max(1, int(prior_trigger.get("observations", 0) or 0)))
        prior_consequence = dict(consequence_priors.get(str(candidate.get("candidate_id")), {}))
        prior_consequence_bonus = float(prior_consequence.get("reward_total", 0.0)) / float(max(1, int(prior_consequence.get("observations", 0) or 0)))
        prior_recovery = dict(recovery_priors.get(str(candidate.get("target_area_id") or target_entity_id or "global"), {}))
        recovery_usefulness = float(prior_recovery.get("successes", 0)) - float(prior_recovery.get("failures", 0))
        durable_prior_strength = max(0.0, prior_success_rate + prior_poi_utility + prior_trigger_bonus)
        contradiction_count = sum(1 for value in dict(candidate.get("contradiction_flags", {})).values() if value)
        contradiction_penalty += 0.08 * contradiction_count
        score = (
            novelty * float(getattr(planning_cfg, "novelty_weight", 0.6))
            + (utility * float(getattr(planning_cfg, "utility_weight", 1.0)))
            + (0.08 * candidate_effect_score)
            + (0.12 * progress_type_score)
            + (0.08 * support_freshness)
            + (float(getattr(planning_cfg, "reachability_weight", 0.55)) * reachability)
            + (float(getattr(planning_cfg, "progress_weight", 1.0)) * progress)
            + (float(getattr(planning_cfg, "trigger_bonus_weight", 0.08)) * (trigger_bonus / 0.08 if trigger_bonus else 0.0))
            + (float(getattr(planning_cfg, "consequence_bonus_weight", 0.05)) * (consequence_bonus / 0.05 if consequence_bonus else 0.0))
            + (0.12 * prior_success_rate)
            + (0.06 * prior_poi_utility)
            + (0.05 * prior_trigger_bonus)
            + (0.04 * prior_consequence_bonus)
            + (0.05 * recovery_usefulness if candidate_class == "recovery_move" else 0.0)
            + (0.06 * durable_prior_strength)
            - (float(getattr(planning_cfg, "retry_penalty_weight", 0.18)) * (retry_penalty / 0.18 if retry_penalty else 0.0))
            - cooldown_penalty
            - exhaustion_penalty
            - neighborhood_exhaustion_penalty
            - (float(getattr(planning_cfg, "route_risk_weight", 0.35)) * route_risk)
            - (float(getattr(planning_cfg, "route_cost_weight", 0.12)) * route_cost)
            - (0.09 * route_uncertainty)
            - (0.07 * trigger_uncertainty)
            - (0.14 * local_failure_risk)
            - contradiction_penalty
            - (0.08 * prior_failure_rate)
            - (0.07 * prior_route_failure_risk)
            - float(candidate.get("score_penalty_soft_filters", 0.0))
        )
        uncertainty = min(1.0, max(route_uncertainty, trigger_uncertainty, (1.0 - support_freshness if support_refs else 0.5)))
        confidence = max(0.0, min(1.0, 0.5 + (0.25 * support_freshness) + (0.2 * prior_success_rate) - (0.2 * uncertainty)))
        candidate["score"] = score
        candidate["score_confidence"] = confidence
        candidate["score_uncertainty"] = uncertainty
        candidate["score_breakdown"] = {
            "schema_version": "v3_1_planner_score_v2",
            "class_weights": class_weights,
            "novelty": novelty,
            "reachability": reachability,
            "progress": progress,
            "retry_penalty": retry_penalty,
            "cooldown_penalty": cooldown_penalty,
            "exhaustion_penalty": exhaustion_penalty,
            "utility": utility,
            "movement_effect_score": movement_effect_score,
            "interact_effect_score": interact_effect_score,
            "click_effect_score": click_effect_score,
            "candidate_effect_score": candidate_effect_score,
            "distance_score": distance_score,
            "motion_score": motion_score,
            "local_failure_risk": local_failure_risk,
            "neighborhood_exhaustion_penalty": neighborhood_exhaustion_penalty,
            "contradiction_penalty": contradiction_penalty,
            "support_freshness": support_freshness,
            "expected_progress_type": expected_progress_type,
            "progress_type_score": progress_type_score,
            "route_risk": route_risk,
            "route_cost": route_cost,
            "route_uncertainty": route_uncertainty,
            "trigger_bonus": trigger_bonus,
            "trigger_uncertainty": trigger_uncertainty,
            "consequence_bonus": consequence_bonus,
            "prior_success_rate": prior_success_rate,
            "prior_failure_rate": prior_failure_rate,
            "prior_route_failure_risk": prior_route_failure_risk,
            "prior_poi_utility": prior_poi_utility,
            "prior_trigger_bonus": prior_trigger_bonus,
            "prior_consequence_bonus": prior_consequence_bonus,
            "prior_recovery_usefulness": recovery_usefulness,
            "durable_prior_strength": durable_prior_strength,
            "score_confidence": confidence,
            "score_uncertainty": uncertainty,
            "freshness_versions": versions,
        }
        scored.append(candidate)
    return scored
