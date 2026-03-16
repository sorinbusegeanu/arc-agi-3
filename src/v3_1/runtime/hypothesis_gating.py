from __future__ import annotations


def llm_skip_reason(
    *,
    config,
    mechanic_graph_snapshot: dict | None,
    deterministic_bundle,
    repeated_failures: int,
    contradiction_level: int,
    deterministic_tied: bool,
    graph_ambiguity: float,
    current_call_count: int,
    last_successful_llm_call_accepted: bool = False,
    prompt_too_large_after_trimming: bool = False,
) -> str | None:
    if prompt_too_large_after_trimming:
        return "prompt_too_large_after_trimming"
    if not bool(getattr(config.hypothesis_generation, "enable_llm", False)):
        return "llm_disabled"
    if int(current_call_count) >= int(getattr(config.hypothesis_generation, "llm_call_budget_per_round", 0)):
        return "llm_call_budget_exhausted"
    if bool(last_successful_llm_call_accepted):
        return "recent_successful_llm_call_accepted"
    graph_state = dict((mechanic_graph_snapshot or {}).get("state", mechanic_graph_snapshot or {}))
    observed_edges = [
        row
        for row in list(dict(graph_state.get("edges_by_id", {})).values())
        if str(row.get("evidence_tier") or "") == "observed" and float(row.get("confidence", 0.0) or 0.0) >= 0.75
    ]
    validated_deterministic = [
        row
        for row in list(getattr(deterministic_bundle, "path_proposals", ()) or [])
        if float(getattr(row, "confidence", 0.0) or 0.0) >= 0.75 and not bool(getattr(row, "requires_validation", True))
    ]
    if observed_edges or validated_deterministic:
        return "strong_non_llm_explanation_available"
    should_trigger = bool(
        len(list(getattr(deterministic_bundle, "path_proposals", ()) or [])) == 0
        or int(repeated_failures) >= 2
        or int(contradiction_level) >= int(getattr(config.hypothesis_generation, "deterministic_contradiction_threshold", 2))
        or bool(deterministic_tied)
        or float(graph_ambiguity) >= 0.5
    )
    return None if should_trigger else "llm_trigger_conditions_not_met"


def should_call_llm(
    *,
    config,
    mechanic_graph_snapshot: dict | None,
    deterministic_bundle,
    repeated_failures: int,
    contradiction_level: int,
    deterministic_tied: bool,
    graph_ambiguity: float,
    current_call_count: int,
    last_successful_llm_call_accepted: bool = False,
    prompt_too_large_after_trimming: bool = False,
) -> bool:
    return llm_skip_reason(
        config=config,
        mechanic_graph_snapshot=mechanic_graph_snapshot,
        deterministic_bundle=deterministic_bundle,
        repeated_failures=repeated_failures,
        contradiction_level=contradiction_level,
        deterministic_tied=deterministic_tied,
        graph_ambiguity=graph_ambiguity,
        current_call_count=current_call_count,
        last_successful_llm_call_accepted=last_successful_llm_call_accepted,
        prompt_too_large_after_trimming=prompt_too_large_after_trimming,
    ) is None
