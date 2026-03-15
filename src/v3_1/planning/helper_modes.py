from __future__ import annotations

from v3_1.contracts.messages import HelperTaskResult


def _result(request, proposals: tuple[dict, ...]) -> HelperTaskResult:
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
    )


def candidate_expansion(request) -> HelperTaskResult:
    priors = dict(request.payload.get("durable_priors", {}))
    candidate_outcomes = dict(priors.get("candidate_outcomes", {}))
    proposals = tuple({"candidate_id": candidate_id, "score_delta": 0.08, "source": "candidate_expansion"} for candidate_id in request.candidate_ids)
    boosted = []
    for proposal in proposals:
        row = dict(proposal)
        prior = dict(candidate_outcomes.get(str(candidate_id := row["candidate_id"]), {}))
        row["score_delta"] = float(row["score_delta"]) + (0.03 if int(prior.get("successes", 0)) > int(prior.get("failures", 0)) else 0.0)
        boosted.append(row)
    proposals = tuple(boosted)
    return _result(request, proposals)


def route_analysis(request) -> HelperTaskResult:
    proposals = []
    route_features = dict(request.payload.get("route_features", {}))
    failure_priors = dict(request.payload.get("durable_priors", {}).get("recovery_patterns", {}))
    for candidate_id in request.candidate_ids:
        feature_row = route_features.get(candidate_id, {})
        prior = dict(failure_priors.get(candidate_id, {}))
        proposals.append(
            {
                "candidate_id": candidate_id,
                "score_delta": 0.05 * float(feature_row.get("progress_potential", 0.0)),
                "risk_delta": 0.1 * float(feature_row.get("risk", 0.0)) + 0.03 * float(prior.get("failures", 0)),
                "source": "route_analysis",
            }
        )
    return _result(request, tuple(proposals))


def score_feature_computation(request) -> HelperTaskResult:
    proposals = tuple(
        {
            "candidate_id": candidate_id,
            "score_delta": 0.03,
            "feature_name": "helper_feature_bonus",
            "source": "score_feature_computation",
        }
        for candidate_id in request.candidate_ids
    )
    return _result(request, proposals)


def hypothesis_proposal(request) -> HelperTaskResult:
    priors = dict(request.payload.get("durable_priors", {}))
    trigger_priors = dict(priors.get("trigger_patterns", {}))
    proposals = tuple(
        {
            "candidate_id": candidate_id,
            "score_delta": 0.04 + (0.03 if trigger_priors else 0.0),
            "hypothesis": "possible_hidden_trigger" if not trigger_priors else "prior_trigger_pattern",
            "source": "hypothesis_proposal",
        }
        for candidate_id in request.candidate_ids
    )
    return _result(request, proposals)


def pruning_suggestion(request) -> HelperTaskResult:
    high_retry_ids = set(request.payload.get("high_retry_ids", []))
    failure_priors = dict(request.payload.get("durable_priors", {}).get("failure_patterns", {}))
    proposals = tuple(
        {
            "candidate_id": candidate_id,
            "risk_delta": (0.2 if candidate_id in high_retry_ids else 0.0) + (0.05 if candidate_id in failure_priors else 0.0),
            "source": "pruning_suggestion",
        }
        for candidate_id in request.candidate_ids
    )
    return _result(request, proposals)


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
    return _result(request, ())
