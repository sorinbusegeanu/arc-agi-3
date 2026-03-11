from __future__ import annotations

from typing import Optional

from codex_baseline_v2.executor.online_executor import run_offline_local_execution
from codex_baseline_v2.shared.plan_records import PlanResultV1, SkillExecutionRecordV1, SkillSpecV1
from codex_baseline_v2.shared.schemas import ControllerInstructionV2, ExecutorOutcomeV2, TrajectoryEpisodeV2, SCHEMA_VERSION


def _extract_target_ref(values: list[str]) -> Optional[str]:
    for value in values:
        if value.startswith("poi:") or value.startswith("trigger_zone:"):
            return value
        if "poi:" in value:
            return value[value.index("poi:") :]
        if "trigger_zone:" in value:
            return value[value.index("trigger_zone:") :]
    return None


def _skill_success(skill: SkillSpecV1, outcome: ExecutorOutcomeV2) -> bool:
    if skill.skill_type == "go_to_region":
        return bool(outcome.reached)
    if skill.skill_type == "probe_hidden_trigger":
        return any(record.event_ids or record.cause_effect_link_ids for record in outcome.consequence_records)
    if skill.skill_type == "verify_mechanic":
        return any(record.event_ids or record.cause_effect_link_ids for record in outcome.consequence_records)
    return bool(outcome.reached or outcome.contact)


def _termination_reason(outcome: ExecutorOutcomeV2, success: bool) -> str:
    if success:
        return "success"
    if outcome.outcome_summary in {"invalid_target", "blocked", "route_stall", "no_progress", "unreachable_now", "contact_no_effect", "contact_no_reach", "contact_reached_boundary_without_route_progress"}:
        return outcome.outcome_summary
    if outcome.blocked:
        return "blocked"
    return "no_progress"


def instruction_from_skill(skill: SkillSpecV1, plan: PlanResultV1, game_id: str, round_id: int) -> ControllerInstructionV2:
    target_ref = _extract_target_ref(skill.precondition_ids)
    mode = {
        "go_to_region": "poi_approach",
        "contact_poi": "poi_interaction",
        "probe_hidden_trigger": "step_on_region",
        "cross_transition": "cross_boundary_edge",
        "dwell_on_region": "dwell_on_region",
        "perform_action_at_region": "action_in_region",
        "verify_mechanic": "repeat_route_fragment" if target_ref is None else "poi_approach",
        "return_to_anchor": "counterfactual_avoid_contact",
    }.get(skill.skill_type, "poi_approach")
    rationale_parts = [f"plan_id={plan.plan_id}", f"skill_id={skill.skill_id}", f"subgoal_id={plan.selected_subgoal_id}"]
    if target_ref and target_ref.startswith("trigger_zone:"):
        rationale_parts.append(f"trigger_zone_id={target_ref}")
    if target_ref and target_ref.startswith("poi:"):
        rationale_parts.append(f"target_poi_id={target_ref}")
    return ControllerInstructionV2(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        round_id=round_id,
        instruction_id=f"round{round_id:03d}:option:{skill.skill_id}",
        mode=mode,
        target_poi_id=target_ref if target_ref and target_ref.startswith("poi:") else None,
        target_region=None,
        target_type="trigger_zone" if target_ref and target_ref.startswith("trigger_zone:") else "option_skill",
        target_geometry=None,
        target_source_round=round_id,
        rationale=" ".join(rationale_parts),
        progress_metric="option_progress",
        stop_condition="option_budget",
        ranked_alternatives=list(plan.alternative_plan_node_ids),
    )


def execute_option(session, skill: SkillSpecV1, plan: PlanResultV1, blackboard, cfg) -> tuple[ExecutorOutcomeV2, TrajectoryEpisodeV2, SkillExecutionRecordV1]:
    instruction = instruction_from_skill(skill, plan, blackboard.game_id, blackboard.round_id + 1)
    outcome, episode = run_offline_local_execution(session, instruction, blackboard, cfg)
    record = skill_record_from_execution(skill, instruction, outcome, episode)
    return outcome, episode, record


def skill_record_from_execution(
    skill: SkillSpecV1,
    instruction: ControllerInstructionV2,
    outcome: ExecutorOutcomeV2,
    episode: TrajectoryEpisodeV2,
    execution_suffix: str = "",
) -> SkillExecutionRecordV1:
    execution_id = f"skill_execution:{instruction.instruction_id}"
    if execution_suffix:
        execution_id = f"{execution_id}:{execution_suffix}"
    success = _skill_success(skill, outcome)
    record = SkillExecutionRecordV1(
        schema_version="v2.3.2",
        execution_id=execution_id,
        skill_id=skill.skill_id,
        parameter_values=[value for value in skill.precondition_ids[:1]],
        start_step=episode.steps[0].step_idx if episode.steps else 0,
        end_step=episode.steps[-1].step_idx if episode.steps else 0,
        success=success,
        termination_reason=_termination_reason(outcome, success),
        observed_event_ids=[event_id for record in outcome.consequence_records for event_id in record.event_ids],
        observed_topology_delta_ids=[record.topology_delta_id for record in outcome.consequence_records if record.topology_delta_id],
        updated_confidence_delta=0.1 if success else (-0.02 if outcome.outcome_summary in {"contact_no_effect", "contact_no_reach", "contact_reached_boundary_without_route_progress"} else -0.1),
    )
    return record
