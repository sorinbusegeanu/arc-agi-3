from __future__ import annotations

from v3_1.contracts.messages import ExecutorRequest, PlannerDecision


def build_executor_request(decision: PlannerDecision, *, max_steps: int, mode: str, seed: int | None = None) -> ExecutorRequest:
    selected = dict(decision.metadata.get("selected_candidate", {})) if isinstance(decision.metadata, dict) else {}
    target_centroid = None
    if isinstance(decision.selected_action, dict):
        target_centroid = decision.selected_action.get("centroid")
    return ExecutorRequest(
        session_id=decision.session_id,
        run_id=decision.run_id,
        game_id=decision.game_id,
        round_id=decision.round_id,
        pass_id=decision.pass_id,
        plan_context_id=decision.plan_context_id,
        candidate_id=decision.selected_candidate_id,
        action=decision.selected_action,
        max_steps=max_steps,
        mode=mode,
        metadata={
            "candidate_class": selected.get("candidate_class"),
            "target_entity_id": selected.get("target_entity_id"),
            "target_area_id": selected.get("target_area_id"),
            "target_centroid": target_centroid,
            "rationale": selected.get("rationale") or decision.rationale,
            "fallback_candidates": list(decision.metadata.get("fallback_candidates", [])) if isinstance(decision.metadata, dict) else [],
            "seed": seed,
        },
    )
