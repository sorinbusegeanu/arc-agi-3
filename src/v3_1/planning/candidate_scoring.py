from __future__ import annotations


def score_candidates(candidates: list[dict], belief: dict, route_features: dict[str, dict], planning_cfg) -> list[dict]:
    scored = []
    retries = dict(belief.get("retries", {}))
    trigger_support = dict(belief.get("trigger_support", {}))

    for row in candidates:
        candidate = dict(row)
        target_entity_id = candidate.get("target_entity_id")
        route_row = route_features.get(candidate["candidate_id"], {})
        novelty = float(candidate.get("novelty", 0.0))
        reachability = 1.0 if candidate.get("reachable_now") else (0.45 if candidate.get("reachable_later") else -0.5)
        progress = float(route_row.get("progress_potential", 0.0))
        retry_penalty = 0.18 * float(retries.get(candidate["candidate_id"], 0) + retries.get(str(target_entity_id), 0))
        cooldown_penalty = 0.25 if candidate.get("candidate_class") == "recovery_move" and candidate.get("reachable_now") else 0.0
        exhaustion_penalty = 0.0
        utility = float(candidate.get("utility", 0.0))
        route_risk = float(route_row.get("risk", 0.0))
        route_cost = float(route_row.get("cost", 0.0))
        trigger_bonus = 0.08 * len(trigger_support.get(str(target_entity_id), []))
        consequence_bonus = 0.05 * len(belief.get("consequence_support", {}).get(str(candidate.get("action")), []))
        score = (
            novelty * float(getattr(planning_cfg, "novelty_weight", 0.6))
            + (utility * float(getattr(planning_cfg, "utility_weight", 1.0)))
            + (float(getattr(planning_cfg, "reachability_weight", 0.55)) * reachability)
            + (float(getattr(planning_cfg, "progress_weight", 1.0)) * progress)
            + (float(getattr(planning_cfg, "trigger_bonus_weight", 0.08)) * (trigger_bonus / 0.08 if trigger_bonus else 0.0))
            + (float(getattr(planning_cfg, "consequence_bonus_weight", 0.05)) * (consequence_bonus / 0.05 if consequence_bonus else 0.0))
            - (float(getattr(planning_cfg, "retry_penalty_weight", 0.18)) * (retry_penalty / 0.18 if retry_penalty else 0.0))
            - cooldown_penalty
            - exhaustion_penalty
            - (float(getattr(planning_cfg, "route_risk_weight", 0.35)) * route_risk)
            - (float(getattr(planning_cfg, "route_cost_weight", 0.12)) * route_cost)
        )
        candidate["score"] = score
        candidate["score_breakdown"] = {
            "novelty": novelty,
            "reachability": reachability,
            "progress": progress,
            "retry_penalty": retry_penalty,
            "cooldown_penalty": cooldown_penalty,
            "exhaustion_penalty": exhaustion_penalty,
            "utility": utility,
            "route_risk": route_risk,
            "route_cost": route_cost,
            "trigger_bonus": trigger_bonus,
            "consequence_bonus": consequence_bonus,
        }
        scored.append(candidate)
    return scored
