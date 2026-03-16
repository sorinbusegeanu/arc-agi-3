from __future__ import annotations

from v3_1.contracts.messages import HelperTaskResult


def _result(request, proposals: tuple[dict, ...], *, metadata: dict | None = None) -> HelperTaskResult:
    return HelperTaskResult(
        session_id=request.session_id,
        run_id=request.run_id,
        game_id=request.game_id,
        round_id=request.round_id,
        pass_id=request.pass_id,
        helper_mode=request.helper_mode,
        plan_context_id=request.plan_context_id,
        blackboard_version=request.blackboard_version,
        memory_version=request.memory_version,
        policy_version=request.policy_version,
        ranker_version=request.ranker_version,
        proposal_id=f"helper:{request.helper_mode}:{request.plan_context_id}",
        proposals=proposals,
        metadata=metadata or {},
    )


def _support_slice(request) -> tuple[list[dict], dict, dict, dict, dict, dict]:
    payload = dict(request.payload)
    candidate_rows = list(payload.get("candidate_rows", []))
    belief_slices = dict(payload.get("belief_slices", {}))
    local_context = dict(payload.get("local_context", {}))
    trigger_support = dict(payload.get("trigger_support", {}))
    consequence_support = dict(payload.get("consequence_support", {}))
    route_facts = dict(payload.get("route_facts", {}))
    return candidate_rows, belief_slices, local_context, trigger_support, consequence_support, route_facts


def _base_proposal(*, candidate: dict, source: str, confidence: float, score_delta: float = 0.0, risk_delta: float = 0.0, hard_warning_reason_codes: list[str] | None = None, contradiction_flags: dict | None = None, support_strength_adjustments: dict | None = None, evidence: dict | None = None) -> dict:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "score_delta": float(score_delta),
        "risk_delta": float(risk_delta),
        "confidence": float(confidence),
        "source": source,
        "hard_warning_reason_codes": list(hard_warning_reason_codes or []),
        "contradiction_flags": dict(contradiction_flags or {}),
        "support_strength_adjustments": dict(support_strength_adjustments or {}),
        "evidence": dict(evidence or {}),
    }


def candidate_expansion(request) -> HelperTaskResult:
    candidate_rows, belief_slices, local_context, _, _, _ = _support_slice(request)
    if not candidate_rows:
        return _result(request, (), metadata={"suppressed": True, "reason": "missing_candidate_rows"})
    priors = dict(request.payload.get("durable_priors", {}))
    candidate_outcomes = dict(priors.get("candidate_outcomes", {}))
    proposals = []
    for candidate in candidate_rows:
        prior = dict(candidate_outcomes.get(str(candidate.get("objective_type") or candidate.get("candidate_class") or ""), {}))
        prior_strength = 0.03 if int(prior.get("successes", 0)) > int(prior.get("failures", 0)) else 0.0
        proposals.append(
            _base_proposal(
                candidate=candidate,
                source="candidate_expansion",
                confidence=0.55 + prior_strength,
                score_delta=0.05 + prior_strength,
                support_strength_adjustments={"prior_support_delta": prior_strength, "direct_support_delta": 0.02 if belief_slices else 0.0},
                evidence={"local_context_current_area": local_context.get("current_area_id"), "prior_attempts": int(prior.get("attempts", 0) or 0)},
            )
        )
    return _result(request, tuple(proposals), metadata={"used_inputs": ["candidate_rows", "durable_priors", "local_context"]})


def route_analysis(request) -> HelperTaskResult:
    candidate_rows, _, _, _, _, route_facts = _support_slice(request)
    if not candidate_rows or not route_facts:
        return _result(request, (), metadata={"suppressed": True, "reason": "missing_route_facts"})
    failure_priors = dict(request.payload.get("durable_priors", {}).get("recovery_patterns", {}))
    proposals = []
    for candidate in candidate_rows:
        feature_row = dict(route_facts.get(str(candidate.get("candidate_id")), {}))
        prior = dict(failure_priors.get(str(candidate.get("route_signature") or candidate.get("candidate_id")), {}))
        if not feature_row:
            continue
        contradiction_flags = {"route_uncertain": float(feature_row.get("uncertainty", 0.0)) > 0.7}
        proposals.append(
            _base_proposal(
                candidate=candidate,
                source="route_analysis",
                confidence=max(0.2, 1.0 - float(feature_row.get("uncertainty", 0.0))),
                score_delta=0.05 * float(feature_row.get("progress_potential", 0.0)),
                risk_delta=0.1 * float(feature_row.get("risk", 0.0)) + 0.03 * float(prior.get("failures", 0)),
                hard_warning_reason_codes=["helper.route.high_uncertainty"] if contradiction_flags["route_uncertain"] else [],
                contradiction_flags=contradiction_flags,
                support_strength_adjustments={"direct_support_delta": 0.03 if float(feature_row.get("reachable_now", 0.0)) else 0.0},
                evidence={"route_facts": feature_row, "prior_failures": float(prior.get("failures", 0))},
            )
        )
    return _result(request, tuple(proposals), metadata={"used_inputs": ["candidate_rows", "route_facts", "durable_priors"]})


def score_feature_computation(request) -> HelperTaskResult:
    candidate_rows, _, _, trigger_support, consequence_support, _ = _support_slice(request)
    if not candidate_rows:
        return _result(request, (), metadata={"suppressed": True, "reason": "missing_candidate_rows"})
    proposals = []
    for candidate in candidate_rows:
        target_entity_id = str(candidate.get("target_entity_id") or "")
        trigger_count = len(trigger_support.get(target_entity_id, []))
        consequence_count = len(consequence_support.get(str(candidate.get("action", {}).get("type")), []))
        proposals.append(
            _base_proposal(
                candidate=candidate,
                source="score_feature_computation",
                confidence=0.45,
                score_delta=(0.02 * trigger_count) + (0.01 * consequence_count),
                support_strength_adjustments={"indirect_support_delta": 0.02 * trigger_count, "direct_support_delta": 0.01 * consequence_count},
                evidence={"trigger_count": trigger_count, "consequence_count": consequence_count},
            )
        )
    return _result(request, tuple(proposals), metadata={"used_inputs": ["candidate_rows", "trigger_support", "consequence_support"]})


def hypothesis_proposal(request) -> HelperTaskResult:
    candidate_rows, belief_slices, _, trigger_support, consequence_support, _ = _support_slice(request)
    if not candidate_rows or not (trigger_support or consequence_support):
        return _result(request, (), metadata={"suppressed": True, "reason": "missing_hypothesis_support"})
    trigger_priors = dict(request.payload.get("durable_priors", {}).get("trigger_patterns", {}))
    proposals = []
    for candidate in candidate_rows:
        target_entity_id = str(candidate.get("target_entity_id") or "")
        prior = dict(trigger_priors.get(target_entity_id, {}))
        has_support = bool(trigger_support.get(target_entity_id)) or bool(consequence_support)
        proposals.append(
            _base_proposal(
                candidate=candidate,
                source="hypothesis_proposal",
                confidence=0.5 + (0.1 if has_support else 0.0),
                score_delta=0.02 + (0.03 if int(prior.get("observations", 0) or 0) > 0 else 0.0),
                contradiction_flags={"stale_support": not has_support},
                support_strength_adjustments={"prior_support_delta": 0.03 if prior else 0.0},
                evidence={"belief_versions": belief_slices.get("versions", {}), "trigger_prior_observations": int(prior.get("observations", 0) or 0)},
            )
        )
    return _result(request, tuple(proposals), metadata={"used_inputs": ["candidate_rows", "belief_slices", "trigger_support", "consequence_support"]})


def pruning_suggestion(request) -> HelperTaskResult:
    candidate_rows, _, _, _, _, _ = _support_slice(request)
    if not candidate_rows:
        return _result(request, (), metadata={"suppressed": True, "reason": "missing_candidate_rows"})
    high_retry_ids = set(request.payload.get("high_retry_ids", []))
    failure_patterns = dict(request.payload.get("recent_local_failure_patterns", {}))
    failure_priors = dict(request.payload.get("durable_priors", {}).get("failure_patterns", {}))
    proposals = []
    for candidate in candidate_rows:
        candidate_id = str(candidate.get("candidate_id"))
        route_signature = str(candidate.get("route_signature") or "")
        trigger_zone_id = str(candidate.get("trigger_zone_id") or "")
        hard_warning_reason_codes = []
        risk_delta = 0.0
        if candidate_id in high_retry_ids:
            risk_delta += 0.15
            hard_warning_reason_codes.append("helper.prune.high_retry")
        if route_signature and route_signature in failure_patterns:
            risk_delta += 0.08
        if trigger_zone_id and trigger_zone_id in failure_patterns:
            risk_delta += 0.08
        if candidate_id in failure_priors:
            risk_delta += 0.05
        proposals.append(
            _base_proposal(
                candidate=candidate,
                source="pruning_suggestion",
                confidence=0.6 if risk_delta > 0 else 0.3,
                risk_delta=risk_delta,
                hard_warning_reason_codes=hard_warning_reason_codes,
                contradiction_flags={"repeated_local_failure": risk_delta >= 0.15},
                evidence={"recent_local_failure_patterns": list(failure_patterns.keys())[:8]},
            )
        )
    return _result(request, tuple(proposals), metadata={"used_inputs": ["candidate_rows", "high_retry_ids", "recent_local_failure_patterns", "durable_priors"]})


def run_helper_mode(request) -> HelperTaskResult:
    mode = str(request.helper_mode)
    if mode == "candidate_expansion":
        return candidate_expansion(request)
    if mode == "route_analysis":
        return route_analysis(request)
    if mode == "score_feature_computation":
        return score_feature_computation(request)
    if mode == "hypothesis_proposal":
        return hypothesis_proposal(request)
    if mode == "pruning_suggestion":
        return pruning_suggestion(request)
    return _result(request, (), metadata={"suppressed": True, "reason": "unknown_mode"})
