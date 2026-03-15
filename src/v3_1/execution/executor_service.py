from __future__ import annotations

from v3_1.contracts.messages import ExecutorRequest, PlannerDecision


def build_executor_request(decision: PlannerDecision, *, max_steps: int, mode: str, seed: int | None = None) -> ExecutorRequest:
    selected = dict(decision.metadata.get("selected_candidate", {})) if isinstance(decision.metadata, dict) else {}
    target_centroid = None
    click_target_coordinates = None
    if isinstance(decision.selected_action, dict):
        target_centroid = decision.selected_action.get("centroid")
        click_target_coordinates = decision.selected_action.get("click_target_coordinates") or decision.selected_action.get("coordinates")
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
        action_id=selected.get("action_id"),
        action_name=selected.get("action_name"),
        action_family=str(selected.get("required_action_family") or "unknown"),
        required_action_family=str(selected.get("required_action_family") or "unknown"),
        target_entity_id=selected.get("target_entity_id"),
        target_centroid=target_centroid,
        click_target_coordinates=click_target_coordinates,
        metadata={
            "candidate_class": selected.get("candidate_class"),
            "target_entity_id": selected.get("target_entity_id"),
            "target_area_id": selected.get("target_area_id"),
            "skill_id": selected.get("skill_id"),
            "target_centroid": target_centroid,
            "click_target_coordinates": click_target_coordinates,
            "required_action_family": str(selected.get("required_action_family") or "unknown"),
            "rationale": selected.get("rationale") or decision.rationale,
            "fallback_candidates": list(decision.metadata.get("fallback_candidates", [])) if isinstance(decision.metadata, dict) else [],
            "seed": seed,
        },
    )
