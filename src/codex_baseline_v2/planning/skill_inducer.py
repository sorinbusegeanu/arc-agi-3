from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from codex_baseline_v2.planning.skill_library import merge_skill_specs
from codex_baseline_v2.shared.plan_records import SkillSpecV1
from codex_baseline_v2.shared.schemas import BlackboardStateV2


def _has_verification_support(blackboard: BlackboardStateV2, target_ref: str, subgoal_id: str | None = None) -> bool:
    if target_ref.startswith("poi:"):
        poi = next((row for row in blackboard.poi_table if row.poi_id == target_ref), None)
        if poi is not None and poi.linked_mechanic_hypothesis_ids:
            return True
        if any(link.cause_poi_id == target_ref for link in blackboard.cause_effect_table):
            return True
        if any(chain.trigger_poi_id == target_ref for chain in blackboard.causal_chain_hypotheses):
            return True
        if any(trace.target_poi_id == target_ref for trace in blackboard.counterfactual_traces):
            return True
    if target_ref.startswith("trigger_zone:"):
        if any(hidden.trigger_zone_id == target_ref for hidden in blackboard.hidden_trigger_hypotheses):
            return True
        if any(chain.trigger_zone_id == target_ref for chain in blackboard.causal_chain_hypotheses):
            return True
    if subgoal_id and subgoal_id.startswith("subgoal:") and not subgoal_id.startswith("subgoal:reach:"):
        return True
    return False


def _skill_type_for_mode(mode: str) -> str:
    return {
        "poi_approach": "go_to_region",
        "poi_interaction": "contact_poi",
        "step_on_region": "probe_hidden_trigger",
        "dwell_on_region": "dwell_on_region",
        "action_in_region": "perform_action_at_region",
        "cross_boundary_edge": "cross_transition",
        "repeat_route_fragment": "verify_mechanic",
        "counterfactual_avoid_contact": "return_to_anchor",
    }.get(mode, "go_to_region")


def _target_ref_from_record(record) -> str | None:
    return record.target_trigger_zone_id or record.target_poi_id


def _target_recently_failed(existing_skills: List[SkillSpecV1], target_ref: str) -> bool:
    for skill in existing_skills:
        if target_ref not in skill.precondition_ids:
            continue
        if skill.total_attempt_count > 0 and skill.success_count == 0:
            return True
    return False


def _is_target_bound_skill(skill: SkillSpecV1) -> bool:
    return any(ref.startswith("poi:") or ref.startswith("trigger_zone:") for ref in skill.precondition_ids)


def _merge_with_existing(existing: SkillSpecV1 | None, induced: SkillSpecV1) -> SkillSpecV1:
    if existing is None:
        return induced
    return merge_skill_specs(existing, induced)


def _has_concrete_target_binding(blackboard: BlackboardStateV2, target_ref: str | None) -> bool:
    if not target_ref:
        return False
    if target_ref.startswith("poi:"):
        poi = next((row for row in blackboard.poi_table if row.poi_id == target_ref), None)
        return bool(poi is not None and poi.bbox is not None)
    if target_ref.startswith("trigger_zone:"):
        zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
        return bool(zone is not None and (zone.bbox is not None or zone.cells))
    return False


def induce_skills(blackboard: BlackboardStateV2, existing: List[SkillSpecV1] | None = None) -> list[SkillSpecV1]:
    existing_skills = list(existing or [])
    skills: Dict[str, SkillSpecV1] = {
        skill.skill_id: skill
        for skill in existing_skills
        if _is_target_bound_skill(skill)
        if not (skill.skill_type == "probe_hidden_trigger" and (not skill.precondition_ids or not any(ref.startswith("trigger_zone:") for ref in skill.precondition_ids)))
    }
    by_target = defaultdict(list)
    for record in blackboard.intervention_table:
        target_ref = _target_ref_from_record(record)
        if not target_ref:
            continue
        mode = record.probe_mode or record.intent_class or "poi_approach"
        skill_type = _skill_type_for_mode(mode)
        by_target[(skill_type, target_ref)].append(record)
    for (skill_type, target_ref), records in by_target.items():
        if len(records) < 1:
            continue
        precondition_ids = [target_ref]
        expected_effect_node_ids = sorted({event_id for record in records for event_id in record.effect_event_ids})
        if skill_type == "probe_hidden_trigger":
            if not target_ref or not precondition_ids or not any(ref.startswith("trigger_zone:") for ref in precondition_ids):
                continue
            if not expected_effect_node_ids and not any(record.target_trigger_zone_id for record in records):
                continue
        if skill_type == "verify_mechanic" and (target_ref is None or not _has_verification_support(blackboard, target_ref)):
            continue
        skill_id = f"skill:{skill_type}:{target_ref}"
        success_count = sum(1 for record in records if record.reached or record.contact or record.effect_event_ids)
        avg_duration = sum(max(0, record.end_step_idx - record.start_step_idx + 1) for record in records) / float(max(1, len(records)))
        induced = SkillSpecV1(
            schema_version="v2.3.2",
            skill_id=skill_id,
            skill_type=skill_type,
            parameter_names=["target_ref"],
            precondition_ids=precondition_ids,
            expected_effect_node_ids=expected_effect_node_ids,
            average_duration_steps=avg_duration,
            success_rate=0.02,
            failure_mode_labels=sorted({"null_effect" if record.null_effect else "blocked" if record.blocked else "no_effect" for record in records if not (record.reached or record.contact or record.effect_event_ids)}),
            source_trace_ids=[record.instruction_id for record in records],
            total_attempt_count=0,
            success_count=0,
            failure_count=0,
            active=True,
        )
        if skill_id in skills:
            skills[skill_id] = _merge_with_existing(skills[skill_id], induced)
        else:
            skills[skill_id] = induced
    if not skills:
        for record in blackboard.reachability_table:
            if record.status not in {"reachable", "reachable_now", "uncertain"}:
                continue
            if _target_recently_failed(existing_skills, record.poi_id):
                continue
            skill_id = f"skill:go_to_region:{record.poi_id}"
            induced = SkillSpecV1(
                schema_version="v2.3.2",
                skill_id=skill_id,
                skill_type="go_to_region",
                parameter_names=["target_ref"],
                precondition_ids=[record.poi_id],
                expected_effect_node_ids=[record.poi_id],
                average_duration_steps=max(1.0, float(record.distance_estimate or 1.0)),
                success_rate=0.02,
                failure_mode_labels=["blocked"] if record.status in {"blocked", "unreachable"} else [],
                source_trace_ids=[],
                total_attempt_count=0,
                success_count=0,
                failure_count=0,
                active=True,
            )
            if skill_id in skills:
                skills[skill_id] = _merge_with_existing(skills[skill_id], induced)
            else:
                skills[skill_id] = induced
        for zone in blackboard.trigger_zone_table:
            if _target_recently_failed(existing_skills, zone.trigger_zone_id):
                continue
            skill_id = f"skill:probe_hidden_trigger:{zone.trigger_zone_id}"
            induced = SkillSpecV1(
                schema_version="v2.3.2",
                skill_id=skill_id,
                skill_type="probe_hidden_trigger",
                parameter_names=["target_ref"],
                precondition_ids=[zone.trigger_zone_id],
                expected_effect_node_ids=[zone.trigger_zone_id],
                average_duration_steps=3.0,
                success_rate=0.03,
                failure_mode_labels=[],
                source_trace_ids=[],
                total_attempt_count=0,
                success_count=0,
                failure_count=0,
                active=True,
            )
            if skill_id in skills:
                skills[skill_id] = _merge_with_existing(skills[skill_id], induced)
            else:
                skills[skill_id] = induced
        if blackboard.dependency_graph is not None:
            for subgoal in blackboard.dependency_graph.subgoals:
                target_ref = subgoal.subgoal_id[len("subgoal:reach:") :] if subgoal.subgoal_id.startswith("subgoal:reach:poi:") else None
                if subgoal.subgoal_id.startswith("subgoal:reach:") and (target_ref is None or not _has_verification_support(blackboard, target_ref, subgoal.subgoal_id)):
                    continue
                if not subgoal.subgoal_id.startswith("subgoal:reach:"):
                    if target_ref is None or not _has_verification_support(blackboard, target_ref, subgoal.subgoal_id):
                        continue
                if target_ref is None or not _has_concrete_target_binding(blackboard, target_ref):
                    continue
                skill_id = f"skill:verify_mechanic:{target_ref}"
                induced = SkillSpecV1(
                    schema_version="v2.3.2",
                    skill_id=skill_id,
                    skill_type="verify_mechanic",
                    parameter_names=["target_ref"],
                    precondition_ids=[target_ref, subgoal.subgoal_id],
                    expected_effect_node_ids=list(subgoal.unlocks_ids),
                    average_duration_steps=4.0,
                    success_rate=0.02,
                    failure_mode_labels=[],
                    source_trace_ids=[],
                    total_attempt_count=0,
                    success_count=0,
                    failure_count=0,
                    active=True,
                )
                if skill_id in skills:
                    skills[skill_id] = _merge_with_existing(skills[skill_id], induced)
                else:
                    skills[skill_id] = induced
    return list(skills.values())
