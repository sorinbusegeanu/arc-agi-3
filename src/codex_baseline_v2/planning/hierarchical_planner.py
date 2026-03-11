from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.planning.plan_memory import (
    PlanMemoryStateV1,
    cluster_distance,
    cluster_key_for_skill,
    cluster_ledger_map,
    cluster_components,
    movement_cluster_ledger_map,
    )
from codex_baseline_v2.shared.plan_records import PlanNodeV1, PlanResultV1, PlannerBeliefStateV1, SkillSpecV1
from codex_baseline_v2.shared.schemas import BlackboardStateV2

PLANNER_MODULE_PATH = __file__
PLANNER_BUILD_ID = "planner_export_v20260309_01"


def _canonical_skill(skill: SkillSpecV1, skills_by_id: Dict[str, SkillSpecV1]) -> SkillSpecV1:
    if skill.skill_id and skill.skill_id in skills_by_id:
        return skills_by_id[skill.skill_id]
    return skill


def _observed_failure_penalty(skill: SkillSpecV1) -> float:
    if skill.total_attempt_count <= 0:
        return 0.0
    penalty = 0.0
    if "blocked" in skill.failure_mode_labels:
        penalty += 0.45
    if "route_stall" in skill.failure_mode_labels:
        penalty += 0.45
    if "no_progress" in skill.failure_mode_labels:
        penalty += 0.2
    if "contact_no_effect" in skill.failure_mode_labels:
        penalty += 0.03
    if "contact_no_reach" in skill.failure_mode_labels:
        penalty += 0.03
    if skill.success_count == 0 and skill.failure_count > 0:
        if skill.skill_type == "probe_hidden_trigger" and any(label in {"contact_no_effect", "contact_no_reach"} for label in skill.failure_mode_labels):
            penalty += min(0.25, 0.05 * float(skill.failure_count))
            return penalty
        penalty += min(1.25, 0.2 * float(skill.failure_count))
    return penalty


def _same_round_penalty(skill: SkillSpecV1) -> float:
    penalty = 0.0
    if skill.executions_this_round >= 1:
        penalty += 0.22 * float(skill.executions_this_round)
    if skill.latest_termination_reason_this_round in {"blocked", "route_stall", "invalid_target"}:
        penalty += 0.35
    elif skill.latest_termination_reason_this_round == "no_progress":
        penalty += 0.18
    elif skill.latest_termination_reason_this_round == "contact_no_effect":
        penalty += 0.12
    return penalty


def _go_to_region_route_risk_penalty(skill: SkillSpecV1) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    penalty = 0.0
    if "route_stall" in skill.failure_mode_labels:
        penalty += 0.45
    if "contact_no_reach" in skill.failure_mode_labels:
        penalty += 0.45
    if "blocked" in skill.failure_mode_labels:
        penalty += 0.35
    if "invalid_target" in skill.failure_mode_labels:
        penalty += 0.35
    flat_progress_failures = 0
    if skill.latest_termination_reason_this_round in {"route_stall", "contact_no_reach", "no_progress"}:
        flat_progress_failures += 1
    if skill.success_count == 0 and skill.failure_count > 0 and any(
        label in {"route_stall", "contact_no_reach", "no_progress"} for label in skill.failure_mode_labels
    ):
        flat_progress_failures += 1
    penalty += min(0.5, 0.2 * float(flat_progress_failures))
    return penalty


def _untouched_movement_bonus(skill: SkillSpecV1) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    return 0.28 if skill.total_attempt_count == 0 and skill.failure_count == 0 else 0.0


def _prior_contact_no_reach_penalty(skill: SkillSpecV1) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    count = sum(1 for label in skill.failure_mode_labels if label == "contact_no_reach")
    if count == 0 and skill.latest_termination_reason_this_round == "contact_no_reach":
        count = 1
    return min(0.45, 0.22 * float(count)) if count > 0 else 0.0


def _prior_route_stall_penalty(skill: SkillSpecV1) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    count = sum(1 for label in skill.failure_mode_labels if label == "route_stall")
    if count == 0 and skill.latest_termination_reason_this_round == "route_stall":
        count = 1
    return min(0.5, 0.25 * float(count)) if count > 0 else 0.0


def _same_target_repeat_penalty(skill: SkillSpecV1) -> float:
    if skill.skill_type != "go_to_region" or skill.total_attempt_count <= 0:
        return 0.0
    return min(0.35, 0.08 * float(skill.total_attempt_count))


def _same_target_no_progress_penalty(skill: SkillSpecV1) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    count = sum(1 for label in skill.failure_mode_labels if label in {"contact_no_reach", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"})
    if skill.latest_termination_reason_this_round in {"contact_no_reach", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"}:
        count += 1
    return min(0.7, 0.18 * float(count)) if count > 0 else 0.0


def _probe_target_retry_penalty(skill: SkillSpecV1) -> float:
    if skill.skill_type != "probe_hidden_trigger":
        return 0.0
    count = 0
    if skill.total_attempt_count > 0:
        count += 1
    count += sum(1 for label in skill.failure_mode_labels if label in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"})
    if skill.latest_termination_reason_this_round in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"}:
        count += 1
    return min(0.55, 0.12 * float(count)) if count > 0 else 0.0


def _probe_cluster_memory_penalty(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "probe_hidden_trigger" or plan_memory is None:
        return 0.0
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None:
        return 0.0
    row = cluster_ledger_map(plan_memory).get(cluster_key)
    penalty = 0.0
    if cluster_key in set(plan_memory.failed_cluster_keys_this_round):
        penalty += 0.18
    if cluster_key in set(plan_memory.recent_failed_cluster_keys):
        penalty += 0.12
    if cluster_key in set(plan_memory.excluded_cluster_cooldowns):
        penalty += 0.16
    if row is not None:
        penalty += min(0.28, 0.05 * float(row.contact_no_effect_count))
        penalty += min(0.2, 0.04 * float(row.repeated_no_effect_streak))
        if row.locally_exhausted:
            penalty += 0.24
    return min(0.8, penalty)


def _probe_row_band_retry_penalty(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "probe_hidden_trigger" or plan_memory is None:
        return 0.0
    cluster_key = cluster_key_for_skill(skill)
    area_id, _, qy = cluster_components(cluster_key)
    if area_id is None or qy is None:
        return 0.0
    penalty = 0.0
    for row in cluster_ledger_map(plan_memory).values():
        if row.area_id != area_id or row.quantized_y is None:
            continue
        if abs(int(row.quantized_y) - int(qy)) <= 1 and row.failure_count > 0:
            penalty += min(0.36, 0.07 * float(row.failure_count))
            penalty += min(0.2, 0.04 * float(row.contact_no_effect_count))
            penalty += min(0.15, 0.03 * float(row.repeated_no_effect_streak))
            if row.locally_exhausted:
                penalty += 0.18
            if row.last_failed_round is not None and row.repeated_no_effect_streak >= 2:
                penalty += 0.12
    return min(0.55, penalty)


def _probe_route_risk_penalty(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "probe_hidden_trigger":
        return 0.0
    penalty = 0.0
    count = sum(1 for label in skill.failure_mode_labels if label in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"})
    if skill.latest_prior_round_termination_reason in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"}:
        count += 1
    if skill.latest_termination_reason_this_round in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"}:
        count += 1
    penalty += min(0.6, 0.11 * float(count))
    penalty += 0.9 * _probe_row_band_retry_penalty(skill, plan_memory)
    if plan_memory is not None:
        row = cluster_ledger_map(plan_memory).get(cluster_key_for_skill(skill))
        if row is not None:
            penalty += min(0.24, 0.05 * float(row.contact_no_effect_count))
            penalty += min(0.18, 0.04 * float(row.repeated_no_effect_streak))
            if row.locally_exhausted:
                penalty += 0.22
    return min(0.9, penalty)


def _probe_history_penalty(skill: SkillSpecV1) -> float:
    if skill.skill_type != "probe_hidden_trigger":
        return 0.0
    count = float(skill.historical_contact_no_effect_count)
    if skill.latest_prior_round_termination_reason in {"contact_no_effect", "contact_reached_boundary_without_route_progress"}:
        count += 1.0
    if skill.latest_termination_reason_this_round in {"contact_no_effect", "contact_reached_boundary_without_route_progress"}:
        count += 1.0
    return min(0.7, 0.14 * count)


def _movement_route_family_unproven_bonus(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    if plan_memory is None:
        return 0.0
    movement_rows = list(movement_cluster_ledger_map(plan_memory).values())
    if any(row.success_count > 0 for row in movement_rows):
        return 0.0
    if skill.total_attempt_count > 0 or skill.failure_count > 0:
        return 0.0
    return 0.18


def _movement_row_band_retry_penalty(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "go_to_region" or plan_memory is None:
        return 0.0
    cluster_key = cluster_key_for_skill(skill)
    area_id, _, qy = cluster_components(cluster_key)
    if area_id is None or qy is None:
        return 0.0
    penalty = 0.0
    for row in movement_cluster_ledger_map(plan_memory).values():
        if row.area_id != area_id or row.quantized_y is None:
            continue
        if abs(int(row.quantized_y) - int(qy)) <= 1 and row.failure_count > 0:
            penalty += min(0.3, 0.06 * float(row.failure_count))
            if row.locally_exhausted:
                penalty += 0.15
    return min(0.45, penalty)


def _generalized_route_family_penalty(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "go_to_region" or plan_memory is None:
        return 0.0
    return min(
        0.8,
        _movement_cluster_retry_penalty(skill, plan_memory)
        + _failed_movement_area_retry_penalty(skill, plan_memory)
        + _movement_row_band_retry_penalty(skill, plan_memory),
    )


def _same_area_family_retry_penalty(skill: SkillSpecV1, blackboard: Optional[BlackboardStateV2], plan_memory: Optional[PlanMemoryStateV1] = None) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    cluster_key = cluster_key_for_skill(skill)
    area_id, _, _ = cluster_components(cluster_key)
    if plan_memory is not None and area_id is not None and area_id in set(plan_memory.failed_movement_area_ids_this_round):
        return 0.18 if skill.failure_count > 0 or skill.total_attempt_count > 0 else 0.08
    if blackboard is None or not blackboard.decision_history:
        return 0.0
    recent = list(blackboard.decision_history)[-2:]
    if not any(record.mode == "poi_approach" and record.outcome_summary in {"contact_no_reach", "route_stall"} for record in recent):
        return 0.0
    return 0.12 if skill.failure_count > 0 else 0.0


def _failed_movement_area_retry_penalty(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "go_to_region" or plan_memory is None:
        return 0.0
    cluster_key = cluster_key_for_skill(skill)
    area_id, _, _ = cluster_components(cluster_key)
    if area_id is None or area_id not in set(plan_memory.failed_movement_area_ids_this_round):
        return 0.0
    if skill.total_attempt_count == 0 and skill.failure_count == 0:
        return 0.18
    return 0.38


def _movement_cluster_retry_penalty(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "go_to_region" or plan_memory is None:
        return 0.0
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None:
        return 0.0
    penalty = 0.0
    if cluster_key in set(plan_memory.recent_failed_movement_cluster_keys):
        penalty += 0.4
    row = movement_cluster_ledger_map(plan_memory).get(cluster_key)
    if row is not None:
        penalty += min(0.35, 0.08 * float(row.failure_count))
        penalty += min(0.2, 0.05 * float(row.contact_no_effect_count))
        if row.locally_exhausted:
            penalty += 0.3
    return penalty


def _movement_cluster_novelty_bonus(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    if skill.total_attempt_count != 0 or skill.failure_count != 0:
        return 0.0
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None or plan_memory is None:
        return 0.18
    return 0.22 if cluster_key not in set(plan_memory.recent_failed_movement_cluster_keys) else 0.0


def _frontier_novelty_bonus(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    cluster_key = cluster_key_for_skill(skill)
    area_id, _, _ = cluster_components(cluster_key)
    if skill.total_attempt_count != 0 or skill.failure_count != 0:
        return 0.0
    if plan_memory is None:
        return 0.28
    if cluster_key in set(plan_memory.recent_failed_movement_cluster_keys):
        return 0.0
    if area_id is not None and area_id in set(plan_memory.failed_movement_area_ids_this_round):
        return 0.0
    return 0.34


def _cross_area_bonus(skill: SkillSpecV1, blackboard: Optional[BlackboardStateV2], plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    cluster_key = cluster_key_for_skill(skill)
    area_id, _, _ = cluster_components(cluster_key)
    if area_id is None:
        return 0.0
    current_area = blackboard.area_table[0].area_id if blackboard is not None and blackboard.area_table else None
    if current_area is None:
        return 0.0
    if area_id != current_area and skill.total_attempt_count == 0 and skill.failure_count == 0:
        return 0.3
    return 0.0


def _movement_default_risk_penalty(skill: SkillSpecV1) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    if skill.total_attempt_count > 0 or skill.failure_count > 0:
        return 0.0
    return 0.08


def _recovery_nonlocal_movement_bias(
    skill: SkillSpecV1,
    candidate_source: Optional[str],
    plan_memory: Optional[PlanMemoryStateV1],
    entries: List[Dict[str, object]],
) -> float:
    if skill.skill_type != "go_to_region":
        return 0.0
    if candidate_source != "non-local reachable frontier":
        return 0.0
    failed_clusters = len(plan_memory.recent_failed_movement_cluster_keys) if plan_memory is not None else 0
    failed_areas = len(plan_memory.failed_movement_area_ids_this_round) if plan_memory is not None else 0
    movement_memory_weak = failed_clusters <= 1 and failed_areas <= 1
    if not movement_memory_weak:
        return 0.0
    probe_memory_risk_active = bool(
        plan_memory is not None and (
            plan_memory.failed_cluster_keys_this_round
            or plan_memory.recent_failed_cluster_keys
            or plan_memory.excluded_cluster_cooldowns
        )
    )
    probe_penalty_active = any(
        entry.get("skill").skill_type == "probe_hidden_trigger"
        and (
            float(entry["node"].final_score_breakdown.get("probe_route_risk_penalty", 0.0)) > 0.0
            or float(entry["node"].final_score_breakdown.get("probe_cluster_retry_penalty", 0.0)) > 0.0
            or float(entry["node"].final_score_breakdown.get("probe_row_band_retry_penalty", 0.0)) > 0.0
        )
        for entry in entries
    )
    return 0.22 if (probe_penalty_active or probe_memory_risk_active) else 0.0


def _movement_cluster_locally_exhausted(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> bool:
    if skill.skill_type != "go_to_region" or plan_memory is None:
        return False
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None:
        return False
    row = movement_cluster_ledger_map(plan_memory).get(cluster_key)
    return bool(row is not None and row.locally_exhausted)


def _score(skill: SkillSpecV1, revisit_penalty: float, contradiction_penalty: float) -> float:
    zero_attempt_penalty = 0.35 if not skill.source_trace_ids else 0.0
    invalid_target_penalty = 0.85 if "invalid_target" in skill.failure_mode_labels else 0.0
    all_failed_penalty = 0.0
    if skill.total_attempt_count > 0 and skill.success_count == 0:
        if skill.skill_type == "probe_hidden_trigger" and any(label in {"contact_no_effect", "contact_no_reach"} for label in skill.failure_mode_labels):
            all_failed_penalty = 0.15
        else:
            all_failed_penalty = 0.6
    missing_provenance_penalty = 0.25 if not skill.source_trace_ids else 0.0
    movement_failure_penalty = 1.5 if skill.skill_type == "go_to_region" and skill.total_attempt_count > 0 and skill.success_count == 0 and any(label in {"route_stall", "blocked"} for label in skill.failure_mode_labels) else 0.0
    route_risk_penalty = _go_to_region_route_risk_penalty(skill)
    return (
        1.5 * skill.success_rate
        + 0.8 * float(bool(skill.expected_effect_node_ids))
        - 0.05 * skill.average_duration_steps
        - revisit_penalty
        - contradiction_penalty * float(bool(skill.failure_mode_labels))
        - zero_attempt_penalty
        - invalid_target_penalty
        - all_failed_penalty
        - missing_provenance_penalty
        - _observed_failure_penalty(skill)
        - movement_failure_penalty
        - route_risk_penalty
        - _same_round_penalty(skill)
    )


def _estimated_success(skill: SkillSpecV1, target_ref: Optional[str] = None, blackboard: Optional[BlackboardStateV2] = None) -> float:
    if skill.total_attempt_count > 0:
        if skill.skill_type == "probe_hidden_trigger" and any(label in {"contact_no_effect", "contact_no_reach"} for label in skill.failure_mode_labels) and not any(label in {"blocked", "route_stall", "invalid_target"} for label in skill.failure_mode_labels):
            base = max(0.04, 0.16 - 0.01 * float(skill.failure_count))
            base -= 0.03 * float(skill.repeated_contact_no_effect_count_this_round)
            return max(0.02, base)
        base = skill.success_rate
        if skill.failure_count > 0:
            base -= min(0.7, 0.12 * float(skill.failure_count))
        if "route_stall" in skill.failure_mode_labels:
            base -= 0.15
        if "blocked" in skill.failure_mode_labels:
            base -= 0.15
        if "no_progress" in skill.failure_mode_labels:
            base -= 0.05
        if skill.success_count == 0 and skill.failure_count > 0:
            base = min(base, max(0.01, 0.08 / float(skill.failure_count)))
        base -= 0.05 * float(skill.executions_this_round)
        return max(0.01, min(1.0, base))
    if skill.skill_type == "probe_hidden_trigger":
        base = 0.045
        if blackboard is not None and target_ref and target_ref.startswith("trigger_zone:"):
            zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
            if zone is not None:
                base += min(0.035, 0.05 * float(zone.hidden_trigger_confidence))
                base += 0.85 * _probe_distance_signal(target_ref, blackboard)
                base += _probe_access_signal(target_ref, blackboard)
                if zone.activation_count > 0:
                    base -= 0.03
                if zone.null_count > 0:
                    base -= min(0.04, 0.012 * float(zone.null_count))
                if zone.contradiction_count > 0:
                    base -= min(0.05, 0.015 * float(zone.contradiction_count))
        return max(0.02, min(0.18, base))
    if skill.success_rate > 0.0:
        return min(skill.success_rate, 0.1)
    if skill.skill_type == "go_to_region":
        return max(0.01, 0.04 - 0.02 * _go_to_region_route_risk_penalty(skill))
    return 0.04


def _estimated_information_gain(skill: SkillSpecV1, target_ref: Optional[str], blackboard: Optional[BlackboardStateV2]) -> float:
    if skill.skill_type == "go_to_region":
        base = 0.08
    elif skill.skill_type == "probe_hidden_trigger":
        base = 0.68
    else:
        base = 0.35 if skill.expected_effect_node_ids else 0.2
    if blackboard is not None and target_ref:
        if target_ref.startswith("poi:"):
            poi = next((row for row in blackboard.poi_table if row.poi_id == target_ref), None)
            if poi is not None:
                observed = max(1, int(poi.observation_count or poi.evidence_count or 1))
                base *= max(0.2, 1.0 / float(observed))
        elif target_ref.startswith("trigger_zone:"):
            zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
            if zone is not None:
                base *= max(0.35, 1.0 - min(0.55, 0.08 * float(zone.activation_count + zone.null_count + zone.contradiction_count)))
                if zone.hidden_trigger_confidence <= 0.0:
                    base *= 0.5
                cell_count = len(zone.cells or [])
                if cell_count > 0:
                    base *= max(0.7, min(1.15, 0.85 + 0.03 * float(min(cell_count, 10))))
                if zone.contradiction_count > 0:
                    base *= 0.85
                if zone.null_count > 0:
                    base *= max(0.6, 1.0 - 0.08 * float(zone.null_count))
                if zone.activation_count == 0 and zone.hidden_trigger_confidence > 0.0:
                    base *= 1.05
    if any(label in {"blocked", "route_stall", "no_progress"} for label in skill.failure_mode_labels):
        base *= 0.35
    if skill.skill_type == "probe_hidden_trigger" and any(label in {"contact_no_effect", "contact_no_reach"} for label in skill.failure_mode_labels):
        base *= 0.88
    if skill.skill_type == "probe_hidden_trigger":
        if skill.executions_this_round >= 1:
            base *= max(0.65, 1.0 - 0.12 * float(skill.executions_this_round))
        if skill.repeated_contact_no_effect_count_this_round >= 1:
            base *= max(0.7, 1.0 - 0.1 * float(skill.repeated_contact_no_effect_count_this_round))
    return max(0.05, min(0.95, base))


def _zone_centroid(blackboard: BlackboardStateV2, target_ref: str) -> Optional[Tuple[float, float]]:
    zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    if zone is None:
        return None
    if zone.bbox is not None:
        return ((zone.bbox.x1 + zone.bbox.x2) / 2.0, (zone.bbox.y1 + zone.bbox.y2) / 2.0)
    if zone.cells:
        xs = [cell[0] for cell in zone.cells]
        ys = [cell[1] for cell in zone.cells]
        return (sum(xs) / float(len(xs)), sum(ys) / float(len(ys)))
    return None


def _probe_locality_relation(
    blackboard: Optional[BlackboardStateV2],
    target_ref: Optional[str],
    other_ref: Optional[str],
) -> str:
    if blackboard is None or not target_ref or not other_ref:
        return "distant"
    if not target_ref.startswith("trigger_zone:") or not other_ref.startswith("trigger_zone:"):
        return "distant"
    if target_ref == other_ref:
        return "exact"
    center = _zone_centroid(blackboard, target_ref)
    other_center = _zone_centroid(blackboard, other_ref)
    if center is None or other_center is None:
        return "distant"
    distance = abs(center[0] - other_center[0]) + abs(center[1] - other_center[1])
    same_row = round(center[1]) == round(other_center[1])
    same_col = round(center[0]) == round(other_center[0])
    zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    other_zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == other_ref), None)
    same_area = bool(
        zone is not None
        and other_zone is not None
        and zone.area_id is not None
        and zone.area_id == other_zone.area_id
    )
    if distance <= 2.0:
        return "immediate"
    if same_row or same_col:
        if distance <= 8.0:
            return "immediate"
        return "cluster"
    if distance <= 5.0:
        return "immediate"
    if distance <= 10.0:
        return "cluster"
    if same_area and distance <= 12.0:
        return "cluster"
    return "distant"


def _cluster_failure_count(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> int:
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None or plan_memory is None:
        return 0
    row = movement_cluster_ledger_map(plan_memory).get(cluster_key) if skill.skill_type == "go_to_region" else cluster_ledger_map(plan_memory).get(cluster_key)
    return 0 if row is None else int(row.contact_no_effect_count)


def _cluster_exhausted(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> bool:
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None or plan_memory is None:
        return False
    if skill.skill_type == "go_to_region":
        row = movement_cluster_ledger_map(plan_memory).get(cluster_key)
        if row is not None and row.locally_exhausted:
            return True
        return cluster_key in set(plan_memory.recent_failed_movement_cluster_keys) and row is not None and row.failure_count >= 2
    row = cluster_ledger_map(plan_memory).get(cluster_key)
    if row is not None and row.locally_exhausted:
        return True
    if skill.skill_type == "probe_hidden_trigger" and cluster_key in set(plan_memory.recent_failed_cluster_keys):
        if skill.latest_termination_reason_this_round == "contact_no_effect" or skill.latest_prior_round_termination_reason == "contact_no_effect":
            return True
    return False


def _cooldown_applies(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> bool:
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None or plan_memory is None:
        return False
    if skill.skill_type != "probe_hidden_trigger":
        return False
    return int(plan_memory.excluded_cluster_cooldowns.get(cluster_key, 0)) > 0


def _distance_from_last_failed_cluster(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> Optional[float]:
    if plan_memory is None or not plan_memory.recent_failed_cluster_keys:
        return None
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None:
        return None
    return float(cluster_distance(cluster_key, plan_memory.recent_failed_cluster_keys[-1]) or 0.0)


def _same_row_sweep_penalty(skill: SkillSpecV1, blackboard: Optional[BlackboardStateV2], skills_by_id: Dict[str, SkillSpecV1], plan_memory: Optional[PlanMemoryStateV1]) -> float:
    if blackboard is None or skill.skill_type != "probe_hidden_trigger":
        return 0.0
    sx = skill.target_x_this_round
    sy = skill.target_y_this_round
    if sx is None or sy is None:
        return 0.0
    same_row_failures = 0
    same_col_failures = 0
    for other in skills_by_id.values():
        if other.skill_type != "probe_hidden_trigger":
            continue
        if other.latest_termination_reason_this_round not in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"} and other.latest_prior_round_termination_reason not in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"}:
            continue
        if other.target_x_this_round is None or other.target_y_this_round is None:
            continue
        if int(other.target_y_this_round) == int(sy):
            same_row_failures += 2 if other.latest_termination_reason_this_round in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"} else 1
        if int(other.target_x_this_round) == int(sx):
            same_col_failures += 2 if other.latest_termination_reason_this_round in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"} else 1
    cluster_key = cluster_key_for_skill(skill)
    _, _, qy = cluster_components(cluster_key)
    if plan_memory is not None and qy is not None:
        current_area = cluster_components(cluster_key)[0]
        for band in plan_memory.failed_row_bands_this_round:
            parts = band.split("|row|")
            if len(parts) != 2 or parts[0] != current_area:
                continue
            try:
                failed_qy = int(parts[1])
            except Exception:
                continue
            if abs(failed_qy - qy) <= 1:
                same_row_failures += 2
    return min(0.45, 0.1 * float(max(same_row_failures, same_col_failures)))


def _probe_cluster_penalty(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> float:
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None or plan_memory is None:
        return 0.0
    penalty = 0.0
    if cluster_key in set(plan_memory.recent_failed_cluster_keys):
        penalty += 0.45
    row = cluster_ledger_map(plan_memory).get(cluster_key)
    if row is not None:
        penalty += min(0.4, 0.08 * float(row.contact_no_effect_count))
        if row.locally_exhausted:
            penalty += 0.6
    cooldown = int(plan_memory.excluded_cluster_cooldowns.get(cluster_key, 0))
    if cooldown > 0:
        penalty += 0.5 + 0.1 * float(cooldown)
    return penalty


def _neighbor_cluster_penalties(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> Tuple[float, float, int]:
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None or plan_memory is None:
        return 0.0, 0.0, 0
    current_penalty = 0.0
    historical_penalty = 0.0
    current_neighbor_count = 0
    for other_key in plan_memory.failed_cluster_keys_this_round:
        distance = cluster_distance(cluster_key, other_key)
        if distance is None or distance == 0:
            continue
        if distance == 1:
            current_penalty += 0.28
            current_neighbor_count += 1
        elif distance == 2:
            current_penalty += 0.14
            current_neighbor_count += 1
    for other_key in plan_memory.recent_failed_cluster_keys:
        distance = cluster_distance(cluster_key, other_key)
        if distance is None or distance == 0:
            continue
        if distance == 1:
            historical_penalty += 0.2
        elif distance == 2:
            historical_penalty += 0.12
    return min(0.6, current_penalty), min(0.45, historical_penalty), current_neighbor_count


def _post_jump_exclusion_flag(skill: SkillSpecV1, plan_memory: Optional[PlanMemoryStateV1]) -> bool:
    cluster_key = cluster_key_for_skill(skill)
    if cluster_key is None or plan_memory is None or not plan_memory.last_distant_jump_target_cluster:
        return False
    distance = cluster_distance(cluster_key, plan_memory.last_distant_jump_target_cluster)
    if distance is not None and distance <= 1:
        return False
    if int(plan_memory.excluded_cluster_cooldowns.get(cluster_key, 0)) > 0:
        return True
    row = cluster_ledger_map(plan_memory).get(cluster_key)
    return bool(row is not None and row.contact_no_effect_count >= 2)


def _avatar_centroid(blackboard: Optional[BlackboardStateV2]) -> Optional[Tuple[float, float]]:
    if blackboard is None or not blackboard.avatar_track_table:
        return None
    return tuple(float(v) for v in blackboard.avatar_track_table[0].centroid)


def _probe_neighbor_contact_penalty(
    skill: SkillSpecV1,
    target_ref: Optional[str],
    blackboard: Optional[BlackboardStateV2],
    skills_by_id: Dict[str, SkillSpecV1],
) -> float:
    if blackboard is None or not target_ref or not target_ref.startswith("trigger_zone:"):
        return 0.0
    center = _zone_centroid(blackboard, target_ref)
    if center is None:
        return 0.0
    target_zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    penalty = 0.0
    for other in skills_by_id.values():
        if other.skill_type != "probe_hidden_trigger":
            continue
        if other.latest_termination_reason_this_round not in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"}:
            continue
        other_ref = _extract_target_ref(other, _extract_subgoal_id(other))
        if not other_ref:
            continue
        relation = _probe_locality_relation(blackboard, target_ref, other_ref)
        if relation == "exact":
            penalty += 0.55
        elif relation == "immediate":
            penalty += 0.4
        elif relation == "cluster":
            penalty += 0.18
    return min(0.75, penalty)


def _probe_historical_contact_penalty(
    skill: SkillSpecV1,
    target_ref: Optional[str],
    blackboard: Optional[BlackboardStateV2],
    skills_by_id: Dict[str, SkillSpecV1],
) -> float:
    if blackboard is None or skill.skill_type != "probe_hidden_trigger" or not target_ref or not target_ref.startswith("trigger_zone:"):
        return 0.0
    center = _zone_centroid(blackboard, target_ref)
    if center is None:
        return 0.0
    penalty = 0.0
    for other in skills_by_id.values():
        if other.skill_type != "probe_hidden_trigger":
            continue
        if other.historical_contact_no_effect_count <= 0 and other.latest_prior_round_termination_reason not in {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"}:
            continue
        other_ref = _extract_target_ref(other, _extract_subgoal_id(other))
        if not other_ref:
            continue
        weight = min(0.3, 0.05 * float(max(1, other.historical_contact_no_effect_count)))
        relation = _probe_locality_relation(blackboard, target_ref, other_ref)
        if relation == "exact":
            penalty += 0.18 + weight
        elif relation == "immediate":
            penalty += 0.12 + weight
        elif relation == "cluster":
            penalty += 0.06 + 0.5 * weight
    return min(0.45, penalty)


def _probe_local_null_count(
    skill: SkillSpecV1,
    target_ref: Optional[str],
    blackboard: Optional[BlackboardStateV2],
    skills_by_id: Dict[str, SkillSpecV1],
) -> int:
    if blackboard is None or skill.skill_type != "probe_hidden_trigger" or not target_ref or not target_ref.startswith("trigger_zone:"):
        return 0
    center = _zone_centroid(blackboard, target_ref)
    if center is None:
        return 0
    target_zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    count = 0
    for other in skills_by_id.values():
        if other.skill_type != "probe_hidden_trigger":
            continue
        if other.latest_termination_reason_this_round != "contact_no_effect":
            continue
        other_ref = _extract_target_ref(other, _extract_subgoal_id(other))
        if not other_ref:
            continue
        if other_ref == target_ref:
            count += 2
            continue
        relation = _probe_locality_relation(blackboard, target_ref, other_ref)
        if relation == "immediate":
            count += 2
        elif relation == "cluster":
            count += 1
    return count


def _probe_current_round_neighborhood_strength(
    skill: SkillSpecV1,
    target_ref: Optional[str],
    blackboard: Optional[BlackboardStateV2],
    skills_by_id: Dict[str, SkillSpecV1],
) -> float:
    if blackboard is None or skill.skill_type != "probe_hidden_trigger" or not target_ref or not target_ref.startswith("trigger_zone:"):
        return 0.0
    center = _zone_centroid(blackboard, target_ref)
    if center is None:
        return 0.0
    target_zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    strength = 0.0
    for other in skills_by_id.values():
        if other.skill_type != "probe_hidden_trigger" or other.latest_termination_reason_this_round != "contact_no_effect":
            continue
        other_ref = _extract_target_ref(other, _extract_subgoal_id(other))
        if not other_ref:
            continue
        if other_ref == target_ref:
            strength += 3.0
            continue
        relation = _probe_locality_relation(blackboard, target_ref, other_ref)
        if relation == "exact":
            strength += 3.0
        elif relation == "immediate":
            strength += 2.5
        elif relation == "cluster":
            strength += 0.75
    return strength


def _probe_historical_neighborhood_strength(
    skill: SkillSpecV1,
    target_ref: Optional[str],
    blackboard: Optional[BlackboardStateV2],
    skills_by_id: Dict[str, SkillSpecV1],
) -> float:
    if blackboard is None or skill.skill_type != "probe_hidden_trigger" or not target_ref or not target_ref.startswith("trigger_zone:"):
        return 0.0
    center = _zone_centroid(blackboard, target_ref)
    if center is None:
        return 0.0
    target_zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    strength = 0.0
    for other in skills_by_id.values():
        if other.skill_type != "probe_hidden_trigger":
            continue
        if other.historical_contact_no_effect_count <= 0 and other.latest_prior_round_termination_reason != "contact_no_effect":
            continue
        other_ref = _extract_target_ref(other, _extract_subgoal_id(other))
        if not other_ref:
            continue
        weight = min(2.0, 0.5 * float(max(1, other.historical_contact_no_effect_count)))
        relation = _probe_locality_relation(blackboard, target_ref, other_ref)
        if relation == "exact":
            strength += 1.5 + weight
        elif relation == "immediate":
            strength += 1.0 + 0.5 * weight
        elif relation == "cluster":
            strength += 0.35 + 0.2 * weight
    return strength


def _probe_same_round_neighbor(entry: Dict[str, object], ranked_entries: List[Dict[str, object]], blackboard: Optional[BlackboardStateV2]) -> bool:
    skill = entry["skill"]
    target_ref = entry.get("effective_target")
    if blackboard is None or skill.skill_type != "probe_hidden_trigger" or not isinstance(target_ref, str):
        return False
    for other in ranked_entries:
        if other is entry:
            continue
        other_skill = other["skill"]
        if other_skill.skill_type != "probe_hidden_trigger":
            continue
        if other_skill.latest_termination_reason_this_round != "contact_no_effect":
            continue
        other_ref = other.get("effective_target")
        if not isinstance(other_ref, str):
            continue
        if _probe_locality_relation(blackboard, target_ref, other_ref) in {"exact", "immediate", "cluster"}:
            return True
    return False


def _probe_same_round_excluded(
    entry: Dict[str, object],
    all_entries: List[Dict[str, object]],
    blackboard: Optional[BlackboardStateV2],
    skills_by_id: Dict[str, SkillSpecV1],
) -> bool:
    skill = entry["skill"]
    target_ref = entry.get("effective_target")
    if blackboard is None or skill.skill_type != "probe_hidden_trigger" or not isinstance(target_ref, str):
        return False
    local_null_count = _probe_local_null_count(skill, target_ref, blackboard, skills_by_id)
    current_strength = _probe_current_round_neighborhood_strength(skill, target_ref, blackboard, skills_by_id)
    if local_null_count <= 0 and current_strength <= 0.0:
        return False
    exact_hit = any(
        other.skill_type == "probe_hidden_trigger"
        and other.latest_termination_reason_this_round == "contact_no_effect"
        and _extract_target_ref(other, _extract_subgoal_id(other)) == target_ref
        for other in skills_by_id.values()
    )
    if exact_hit:
        return True
    distant_probe_exists = any(
        other_entry["skill"].skill_type == "probe_hidden_trigger"
        and other_entry.get("has_geometry")
        and not _probe_same_round_neighbor(other_entry, all_entries, blackboard)
        for other_entry in all_entries
    )
    if not distant_probe_exists:
        return False
    if current_strength >= 3.5 or local_null_count >= 2:
        return True
    return current_strength >= 0.75


def _probe_neighborhood_exhausted(
    skill: SkillSpecV1,
    target_ref: Optional[str],
    blackboard: Optional[BlackboardStateV2],
    skills_by_id: Dict[str, SkillSpecV1],
) -> bool:
    if blackboard is None or skill.skill_type != "probe_hidden_trigger" or not isinstance(target_ref, str):
        return False
    if skill.expected_effect_node_ids or skill.success_count > 0:
        return False
    local_null_count = _probe_local_null_count(skill, target_ref, blackboard, skills_by_id)
    current_strength = _probe_current_round_neighborhood_strength(skill, target_ref, blackboard, skills_by_id)
    return local_null_count >= 2 or current_strength >= 4.5


def _probe_novelty_bonus(
    skill: SkillSpecV1,
    target_ref: Optional[str],
    blackboard: Optional[BlackboardStateV2],
) -> float:
    if blackboard is None or skill.skill_type != "probe_hidden_trigger" or not target_ref or not target_ref.startswith("trigger_zone:"):
        return 0.0
    zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    if zone is None:
        return 0.0
    nearby_count = 0
    center = _zone_centroid(blackboard, target_ref)
    if center is not None:
        for other in blackboard.trigger_zone_table:
            if other.trigger_zone_id == target_ref:
                continue
            other_center = _zone_centroid(blackboard, other.trigger_zone_id)
            if other_center is None:
                continue
            distance = abs(center[0] - other_center[0]) + abs(center[1] - other_center[1])
            if distance <= 4.0:
                nearby_count += 1
    novelty = max(0.0, 0.18 - 0.03 * float(nearby_count))
    if zone.activation_count + zone.null_count + zone.contradiction_count == 0:
        novelty += 0.05
    return novelty


def _probe_distance_signal(target_ref: Optional[str], blackboard: Optional[BlackboardStateV2]) -> float:
    if blackboard is None or not target_ref or not target_ref.startswith("trigger_zone:"):
        return 0.0
    avatar = _avatar_centroid(blackboard)
    target = _zone_centroid(blackboard, target_ref)
    if avatar is None or target is None:
        return 0.0
    distance = abs(avatar[0] - target[0]) + abs(avatar[1] - target[1])
    return max(0.0, 0.2 - 0.01 * distance)


def _probe_access_signal(target_ref: Optional[str], blackboard: Optional[BlackboardStateV2]) -> float:
    if blackboard is None or not target_ref or not target_ref.startswith("trigger_zone:"):
        return 0.0
    zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    if zone is None:
        return 0.0
    signal = 0.0
    if zone.area_id is not None and blackboard.area_table and zone.area_id == blackboard.area_table[0].area_id:
        signal += 0.05
    cell_count = len(zone.cells or [])
    if cell_count > 0:
        if cell_count == 1:
            signal += 0.01
        elif cell_count <= 4:
            signal += 0.035
        elif cell_count <= 9:
            signal += 0.025
        else:
            signal += 0.01
    if zone.bbox is not None:
        width = zone.bbox.width()
        height = zone.bbox.height()
        aspect_span = max(width, height)
        if aspect_span <= 2:
            signal += 0.025
        elif aspect_span <= 4:
            signal += 0.015
    traversal_evidence = zone.entry_count + zone.dwell_count + zone.crossing_count
    if traversal_evidence > 0:
        signal += min(0.04, 0.01 * float(traversal_evidence))
    return signal


def _probe_contact_history_bonus(skill: SkillSpecV1) -> float:
    if skill.skill_type != "probe_hidden_trigger":
        return 0.0
    if "contact_no_effect" in skill.failure_mode_labels and skill.total_attempt_count > 0:
        return 0.03
    return 0.0


def _estimated_cost(skill: SkillSpecV1, target_ref: Optional[str], blackboard: Optional[BlackboardStateV2]) -> float:
    if skill.skill_type != "probe_hidden_trigger" or skill.total_attempt_count > 0 or blackboard is None or not target_ref:
        return skill.average_duration_steps
    avatar = _avatar_centroid(blackboard)
    target = _zone_centroid(blackboard, target_ref)
    zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    distance_cost = 0.0
    if avatar is not None and target is not None:
        distance_cost = 0.05 * (abs(avatar[0] - target[0]) + abs(avatar[1] - target[1]))
    size_adjust = 0.0
    if zone is not None:
        size_adjust = max(0.0, 0.2 - 0.02 * float(len(zone.cells or [])))
    return max(1.0, 2.0 + distance_cost + size_adjust)


def _estimated_goal_progress(
    skill: SkillSpecV1,
    target_ref: Optional[str],
    blackboard: Optional[BlackboardStateV2],
    skills_by_id: Dict[str, SkillSpecV1],
    raw_score: float,
) -> float:
    if skill.skill_type != "probe_hidden_trigger" or blackboard is None or not target_ref or not target_ref.startswith("trigger_zone:"):
        return raw_score
    zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    if zone is None:
        return raw_score
    progress = 0.18
    progress += min(0.2, 0.3 * float(zone.hidden_trigger_confidence))
    progress += _probe_distance_signal(target_ref, blackboard)
    progress += _probe_novelty_bonus(skill, target_ref, blackboard)
    progress += _probe_contact_history_bonus(skill)
    progress -= _probe_neighbor_contact_penalty(skill, target_ref, blackboard, skills_by_id)
    return progress


def _extract_target_ref(skill: SkillSpecV1, subgoal_id: Optional[str]) -> Optional[str]:
    refs = list(skill.precondition_ids)
    if subgoal_id:
        refs.append(subgoal_id)
    refs.append(skill.skill_id)
    for ref in refs:
        if ref.startswith("poi:") or ref.startswith("trigger_zone:"):
            return ref
        if ref.startswith("subgoal:reach:poi:"):
            return ref[len("subgoal:reach:") :]
        if "poi:" in ref:
            return ref[ref.index("poi:") :]
        if "trigger_zone:" in ref:
            return ref[ref.index("trigger_zone:") :]
    return None


def _extract_subgoal_id(skill: SkillSpecV1) -> Optional[str]:
    for ref in skill.precondition_ids:
        if ref.startswith("subgoal:"):
            return ref
    return None


def _recently_invalidated(blackboard: BlackboardStateV2, target_ref: Optional[str]) -> bool:
    if not target_ref:
        return False
    for record in reversed(list(blackboard.decision_history)[-10:]):
        if record.target_invalidated and (record.selected_target_poi_id == target_ref or record.selected_trigger_zone_id == target_ref):
            return True
    return False


def _has_geometry(blackboard: BlackboardStateV2, target_ref: str) -> bool:
    if target_ref.startswith("poi:"):
        poi = next((row for row in blackboard.poi_table if row.poi_id == target_ref), None)
        if poi is None:
            return False
        return poi.bbox is not None
    if target_ref.startswith("trigger_zone:"):
        zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
        if zone is None:
            return False
        return zone.bbox is not None or bool(zone.cells)
    return False


def _has_verification_evidence(skill: SkillSpecV1, subgoal_id: Optional[str], blackboard: Optional[BlackboardStateV2], target_ref: Optional[str]) -> bool:
    if skill.skill_type != "verify_mechanic":
        return True
    refs = list(skill.precondition_ids)
    if subgoal_id:
        refs.append(subgoal_id)
    if any(ref.startswith("trigger_zone:") for ref in refs):
        return True
    if any(ref.startswith("subgoal:") and not ref.startswith("subgoal:reach:") for ref in refs):
        return True
    if blackboard is None or not target_ref:
        return False
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
    return False


def _has_concrete_probe_binding(skill: SkillSpecV1, target_ref: Optional[str], blackboard: Optional[BlackboardStateV2]) -> bool:
    if skill.skill_type != "probe_hidden_trigger":
        return True
    if not target_ref or blackboard is None:
        return False
    if not target_ref.startswith("trigger_zone:"):
        return False
    return _has_geometry(blackboard, target_ref)


def _validation_reason(skill: SkillSpecV1, subgoal_id: Optional[str], blackboard: Optional[BlackboardStateV2], target_ref: Optional[str]) -> Optional[str]:
    target_backed_skill_types = {
        "go_to_region",
        "contact_poi",
        "probe_hidden_trigger",
        "dwell_on_region",
        "perform_action_at_region",
        "cross_transition",
        "verify_mechanic",
        "return_to_anchor",
    }
    if target_ref is None and skill.skill_type in target_backed_skill_types:
        return "missing_target"
    if blackboard is None:
        return None
    if skill.skill_type == "probe_hidden_trigger":
        if target_ref is None or not target_ref.startswith("trigger_zone:"):
            return "missing_target"
        zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
        if zone is None:
            return "missing_target"
        if not _has_geometry(blackboard, target_ref):
            return "missing_geometry"
        if _recently_invalidated(blackboard, target_ref):
            return "recently_invalidated"
        return None
    if target_ref.startswith("poi:"):
        poi = next((row for row in blackboard.poi_table if row.poi_id == target_ref), None)
        if poi is None:
            return "missing_target"
        reasons = set(poi.rejection_reasons) | set(poi.demotion_reasons)
        if any("stale" in reason for reason in reasons):
            return "stale_target"
        if poi.confidence < 0.2 or poi.object_class == "hud_like" or any("invalid" in reason for reason in reasons):
            return "invalid_target"
        if not _has_geometry(blackboard, target_ref):
            return "missing_geometry"
        reach = next((row for row in blackboard.reachability_table if row.poi_id == target_ref), None)
        if reach is not None and reach.status in {"blocked", "unreachable"}:
            return "unreachable_now"
        if _recently_invalidated(blackboard, target_ref):
            return "recently_invalidated"
        return None
    if target_ref.startswith("trigger_zone:"):
        zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
        if zone is None:
            return "missing_target"
        if not _has_geometry(blackboard, target_ref):
            return "missing_geometry"
        if _recently_invalidated(blackboard, target_ref):
            return "recently_invalidated"
        return None
    if subgoal_id and subgoal_id.startswith("subgoal:reach:poi:"):
        resolved_ref = subgoal_id[len("subgoal:reach:") :]
        return _validation_reason(skill, subgoal_id, blackboard, resolved_ref)
    return "invalid_target"


def _counter_bucket(reason: Optional[str]) -> str:
    if reason is None:
        return "candidate_kept"
    if reason in {"missing_target", "missing_geometry"}:
        return "candidate_rejected_missing_binding"
    if reason in {"unreachable_now", "recently_invalidated"}:
        return "candidate_rejected_unreachable"
    return "candidate_rejected_invalid"


def _movement_like(skill: SkillSpecV1, subgoal_id: Optional[str], target_ref: Optional[str]) -> bool:
    if skill.skill_type == "go_to_region":
        return bool(target_ref)
    if skill.skill_type == "verify_mechanic":
        return bool(target_ref and (subgoal_id or "").startswith("subgoal:reach:"))
    return False


def _dedupe_frontier(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    keep: List[Dict[str, object]] = []
    for entry in entries:
        effective_target = entry.get("effective_target")
        if not isinstance(effective_target, str) or not effective_target:
            keep.append(entry)
            continue
        skill = entry["skill"]
        if skill.skill_type not in {"go_to_region", "verify_mechanic", "probe_hidden_trigger"}:
            keep.append(entry)
            continue
        grouped.setdefault(f"{skill.skill_type}:{effective_target}" if skill.skill_type == "probe_hidden_trigger" else effective_target, []).append(entry)
    for _, group in grouped.items():
        if len(group) == 1:
            keep.append(group[0])
            continue
        group.sort(
            key=lambda entry: (
                0 if entry["skill"].skill_type == "go_to_region" else (1 if entry["skill"].skill_type == "probe_hidden_trigger" else 2),
                0 if entry.get("has_geometry") else 1,
                -float(entry["node"].estimated_success),
                -float(entry["node"].estimated_information_gain),
                float(entry["node"].estimated_cost),
            )
        )
        keep.append(group[0])
    return keep


def _selection_allowed(entry: Dict[str, object], blackboard: Optional[BlackboardStateV2]) -> bool:
    node = entry["node"]
    skill = entry["skill"]
    target_ref = entry.get("effective_target")
    if node.blocked:
        return False
    if node.excluded_by_cooldown or node.cluster_exhausted_flag:
        return False
    if float(node.estimated_success) <= 0.0:
        return False
    if entry.get("validation_failed"):
        return False
    if entry.get("target_required") and not target_ref:
        return False
    if entry.get("recently_invalidated"):
        return False
    if skill.skill_type == "probe_hidden_trigger" and not entry.get("has_geometry"):
        return False
    if skill.skill_type == "probe_hidden_trigger" and not (target_ref and isinstance(target_ref, str) and target_ref.startswith("trigger_zone:")):
        return False
    if skill.skill_type in {"go_to_region", "contact_poi", "verify_mechanic", "return_to_anchor"} and not target_ref:
        return False
    if skill.skill_type in {"probe_hidden_trigger", "dwell_on_region", "perform_action_at_region", "cross_transition"} and not (
        target_ref and isinstance(target_ref, str) and target_ref.startswith("trigger_zone:")
    ):
        return False
    return True


def _apply_recovery_mode_movement_filter(entries: List[Dict[str, object]], blackboard: Optional[BlackboardStateV2]) -> List[Dict[str, object]]:
    if not entries:
        return entries
    recovery_mode = any(_is_recovery_candidate_source(entry["node"].candidate_source) for entry in entries)
    if not recovery_mode:
        return entries
    untouched_cross_area = [
        entry
        for entry in entries
        if entry["skill"].skill_type == "go_to_region"
        and entry["skill"].total_attempt_count == 0
        and entry["skill"].failure_count == 0
        and float(entry["node"].final_score_breakdown.get("cross_area_bonus", 0.0)) > 0.0
    ]
    if not untouched_cross_area:
        return entries
    filtered: List[Dict[str, object]] = [
        entry
        for entry in entries
        if not (
            entry["skill"].skill_type == "go_to_region"
            and _movement_cluster_locally_exhausted(entry["skill"], entry.get("plan_memory"))
        )
    ]
    if not filtered:
        return entries
    surviving_movement = [entry for entry in filtered if entry["skill"].skill_type == "go_to_region"]
    if not surviving_movement:
        movement_alternatives = [
            entry for entry in entries
            if entry["skill"].skill_type == "go_to_region"
            and not _movement_cluster_locally_exhausted(entry["skill"], entry.get("plan_memory"))
        ]
        if movement_alternatives:
            best = sorted(
                movement_alternatives,
                key=lambda entry: (
                    _movement_tie_break(entry),
                    -float(entry["node"].estimated_success),
                    -float(entry["node"].estimated_goal_progress),
                ),
                reverse=True,
            )[0]
            filtered.append(best)
    return filtered


def _recovery_movement_entries(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    movement = [
        entry for entry in entries
        if entry["skill"].skill_type == "go_to_region"
        and entry.get("effective_target")
        and not entry.get("recently_invalidated")
    ]
    if not movement:
        return []
    fresh_cross_area = [
        entry for entry in movement
        if entry["skill"].total_attempt_count == 0
        and entry["skill"].failure_count == 0
        and float(entry["node"].final_score_breakdown.get("cross_area_bonus", 0.0)) > 0.0
    ]
    if fresh_cross_area:
        movement = fresh_cross_area + [entry for entry in movement if entry not in fresh_cross_area]
    filtered = [
        entry for entry in movement
        if not _movement_cluster_locally_exhausted(entry["skill"], entry.get("plan_memory"))
    ]
    return filtered if filtered else movement


def _best_fresh_nonlocal_movement_entry(
    entries: List[Dict[str, object]],
    plan_memory: Optional[PlanMemoryStateV1],
    belief: Optional[PlannerBeliefStateV1] = None,
) -> Optional[Dict[str, object]]:
    failed_clusters = set(plan_memory.recent_failed_movement_cluster_keys) if plan_memory is not None else set()
    failed_areas = set(plan_memory.failed_movement_area_ids_this_round) if plan_memory is not None else set()
    reachable_frontier = set(getattr(belief, "reachable_frontier_ids", []) or [])
    current_area_id = (
        plan_memory.failed_movement_area_ids_this_round[-1]
        if plan_memory is not None and plan_memory.failed_movement_area_ids_this_round
        else getattr(belief, "current_area_id", None)
    )
    candidates = []
    for entry in entries:
        skill = entry["skill"]
        if skill.skill_type != "go_to_region" or not entry.get("effective_target"):
            continue
        if reachable_frontier and entry.get("effective_target") not in reachable_frontier:
            continue
        if entry.get("recently_invalidated"):
            continue
        if _movement_cluster_locally_exhausted(skill, entry.get("plan_memory")):
            continue
        cluster_key = entry["node"].movement_cluster_key or cluster_key_for_skill(skill)
        area_id, _, _ = cluster_components(cluster_key)
        if current_area_id is not None and area_id == current_area_id:
            continue
        candidates.append(
            (
                float(area_id is not None and area_id != current_area_id if current_area_id is not None else 1.0),
                float(skill.total_attempt_count == 0 and skill.failure_count == 0),
                float(cluster_key not in failed_clusters if cluster_key else 1.0),
                float(area_id not in failed_areas if area_id is not None else 1.0),
                float(entry["node"].final_score_breakdown.get("cross_area_bonus", 0.0)),
                float(entry["node"].estimated_success),
                entry,
            )
        )
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][-1]


def _movement_candidate_source_label(
    skill: SkillSpecV1,
    target_ref: Optional[str],
    belief: PlannerBeliefStateV1,
    candidate_source_label: str,
) -> str:
    if skill.skill_type != "go_to_region":
        return candidate_source_label
    reachable_frontier = set(getattr(belief, "reachable_frontier_ids", []) or [])
    cluster_key = cluster_key_for_skill(skill)
    area_id, _, _ = cluster_components(cluster_key)
    current_area_id = getattr(belief, "current_area_id", None)
    if target_ref and target_ref in reachable_frontier:
        if current_area_id is not None and area_id == current_area_id:
            return "current-area local frontier"
        return "non-local reachable frontier"
    return "stale skill inventory only"


def _is_recovery_candidate_source(candidate_source: Optional[str]) -> bool:
    return candidate_source in {
        "regenerated_inventory",
        "current-area local frontier",
        "non-local reachable frontier",
        "stale skill inventory only",
    }


def _repeated_failed_movement(skill: SkillSpecV1) -> bool:
    if skill.skill_type != "go_to_region":
        return False
    if skill.total_attempt_count < 3 or skill.success_count != 0:
        return False
    labels = set(skill.failure_mode_labels)
    return bool(labels) and labels.issubset({"blocked", "route_stall"})


def _same_round_probe_excluded(skill: SkillSpecV1) -> bool:
    if skill.skill_type != "probe_hidden_trigger":
        return False
    if skill.latest_termination_reason_this_round in {"blocked", "route_stall", "invalid_target"}:
        return True
    return skill.repeated_contact_no_effect_count_this_round >= 2


def _weak_zero_attempt_movement(skill: SkillSpecV1) -> bool:
    return skill.skill_type == "go_to_region" and skill.total_attempt_count == 0


def _high_info_probe(entry: Dict[str, object]) -> bool:
    node = entry["node"]
    skill = entry["skill"]
    return bool(
        skill.skill_type == "probe_hidden_trigger"
        and entry.get("has_geometry")
        and float(node.estimated_information_gain) >= 0.5
    )


def _consecutive_failed_skill_type_penalty(skill: SkillSpecV1, blackboard: Optional[BlackboardStateV2]) -> float:
    if blackboard is None or len(blackboard.decision_history) < 2:
        return 0.0
    recent = list(blackboard.decision_history)[-2:]
    recent_modes = [record.mode for record in recent]
    failed = [record.outcome_summary not in {"success"} for record in recent]
    if not all(failed):
        return 0.0
    if skill.skill_type == "go_to_region" and all(mode == "poi_approach" for mode in recent_modes):
        return 0.45
    if skill.skill_type == "probe_hidden_trigger" and all(mode in {"step_on_region", "dwell_on_region"} for mode in recent_modes):
        return 0.25
    return 0.0


def _final_rank_key(entry: Dict[str, object]) -> Tuple[float, float, float, float, float, float]:
    node = entry["node"]
    skill = entry["skill"]
    blocked_failure_count = sum(1 for label in skill.failure_mode_labels if label in {"route_stall", "blocked"})
    return (
        float(node.estimated_information_gain),
        float(node.estimated_success),
        float(node.estimated_goal_progress),
        float(skill.success_rate),
        -float(skill.failure_count),
        -float(blocked_failure_count),
    )


def _diversify_probe_clusters(entries: List[Dict[str, object]], minimum_clusters: int = 3) -> List[Dict[str, object]]:
    probe_entries = [entry for entry in entries if entry.get("skill").skill_type == "probe_hidden_trigger" and entry.get("has_geometry")]
    if len(probe_entries) <= 1:
        return entries
    used_clusters = set()
    diversified: List[Dict[str, object]] = []
    overflow: List[Dict[str, object]] = []
    distinct_clusters = {entry["node"].candidate_cluster_key for entry in probe_entries if entry["node"].candidate_cluster_key}
    target_cluster_count = min(max(1, minimum_clusters), len(distinct_clusters))
    for entry in entries:
        if entry.get("skill").skill_type != "probe_hidden_trigger" or not entry.get("has_geometry"):
            diversified.append(entry)
            continue
        cluster_key = entry["node"].candidate_cluster_key
        if not cluster_key:
            diversified.append(entry)
            continue
        if (
            cluster_key not in used_clusters
            and all((cluster_distance(cluster_key, used) or 99) > 1 for used in used_clusters)
            and len(used_clusters) < target_cluster_count
        ):
            used_clusters.add(cluster_key)
            diversified.append(entry)
        else:
            overflow.append(entry)
    return diversified + overflow


def _block_reason_codes(node: PlanNodeV1, entry: Dict[str, object]) -> List[str]:
    reasons: List[str] = []
    if node.blocked_by_exact_cluster_exhaustion:
        reasons.append("exact_cluster_exhaustion")
    if node.blocked_by_neighbor_cooldown:
        reasons.append("neighbor_cooldown")
    if node.blocked_by_post_jump_exclusion:
        reasons.append("post_jump_exclusion")
    if node.blocked_by_unreachable:
        reasons.append("unreachable")
    if node.excluded_by_cooldown and not node.blocked_by_neighbor_cooldown:
        reasons.append("cooldown_excluded")
    if node.neighbor_of_failed_cluster and node.blocked:
        reasons.append("failed_cluster_neighbor")
    return reasons


def _blocking_reason_details(node: PlanNodeV1) -> Dict[str, object]:
    return {
        "candidate_cluster_key": node.candidate_cluster_key,
        "excluded_by_cooldown": bool(node.excluded_by_cooldown),
        "blocked_by_neighbor_cooldown": bool(node.blocked_by_neighbor_cooldown),
        "blocked_by_post_jump_exclusion": bool(node.blocked_by_post_jump_exclusion),
        "blocked_by_exact_cluster_exhaustion": bool(node.blocked_by_exact_cluster_exhaustion),
        "blocked_by_unreachable": bool(node.blocked_by_unreachable),
        "neighbor_of_failed_cluster": bool(node.neighbor_of_failed_cluster),
        "row_band_id": node.row_band_id,
    }


def _finalize_export_entry(
    entry: Dict[str, object],
    pre_filter_rank_position: int,
    post_filter_rank_position: Optional[int],
    surviving_unblocked_candidate_count: int,
    selected_after_rerank: bool = False,
) -> Dict[str, object]:
    node = entry["node"]
    blocking_reason_codes = _block_reason_codes(node, entry) if node.blocked else []
    post_filter_survived = bool(post_filter_rank_position is not None and not node.blocked)
    rank_removed_reason = None
    if not node.blocked and not post_filter_survived:
        breakdown = dict(node.final_score_breakdown or {})
        if any(
            float(breakdown.get(key, 0.0)) > 0.0
            for key in (
                "movement_cluster_retry_penalty",
                "same_area_movement_retry_penalty",
                "failed_movement_area_retry_penalty",
                "prior_contact_no_reach_penalty",
                "prior_route_stall_penalty",
                "same_target_repeat_penalty",
                "same_target_no_progress_penalty",
            )
        ):
            rank_removed_reason = "dominated_by_retry_penalty"
        elif any(
            float(breakdown.get(key, 0.0)) > 0.0
            for key in ("cluster_penalty", "neighbor_penalty", "historical_neighbor_penalty", "same_row_penalty")
        ):
            rank_removed_reason = "lower_rank_after_memory_penalty"
        else:
            rank_removed_reason = "rerank_removed"
        blocking_reason_codes = [rank_removed_reason]
    if node.blocked and not blocking_reason_codes:
        print(f"[v2][planner][warning] blocked node without reason: {node.plan_node_id}")
        blocking_reason_codes = ["unknown_block_reason"]
    if not node.blocked and post_filter_survived:
        blocking_reason_codes = []
    if node.blocked:
        post_filter_rank_position = None
    updated_node = PlanNodeV1(
        **{
            **node.to_dict(),
            "blocking_reason_codes": blocking_reason_codes,
            "surviving_unblocked_candidate_count": surviving_unblocked_candidate_count,
            "pre_filter_survived": True,
            "hard_filter_applied": True,
            "post_filter_survived": post_filter_survived,
            "rank_removed_reason": (
                None if post_filter_survived else (blocking_reason_codes[0] if blocking_reason_codes else rank_removed_reason)
            ),
            "pre_filter_rank_position": pre_filter_rank_position,
            "post_filter_rank_position": post_filter_rank_position,
            "selected_after_rerank": selected_after_rerank,
            "candidate_final_status": (
                "hard_blocked"
                if node.blocked
                else ("survived_not_selected" if post_filter_survived else "rerank_removed")
            ),
            "blocking_reason_details": _blocking_reason_details(node),
        }
    )
    entry["node"] = updated_node
    return entry


def _apply_post_filters(
    entries: List[Dict[str, object]],
    blackboard: Optional[BlackboardStateV2],
    skills_by_id: Dict[str, SkillSpecV1],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    pre_sorted = list(entries)
    pre_sorted.sort(
        key=lambda entry: (
            _final_rank_key(entry),
            _movement_tie_break(entry) if entry.get("skill").skill_type == "go_to_region" else (0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        reverse=True,
    )
    hard_filtered = [entry for entry in pre_sorted if _selection_allowed(entry, blackboard)]
    hard_filtered = _apply_recovery_mode_movement_filter(hard_filtered, blackboard)
    filtered = _diversify_probe_clusters(hard_filtered)
    post_rank_by_node_id = {
        entry["node"].plan_node_id: idx for idx, entry in enumerate(hard_filtered, start=1)
    }
    surviving_unblocked_candidate_count = len(filtered)
    exported: List[Dict[str, object]] = []
    for idx, entry in enumerate(pre_sorted, start=1):
        exported.append(
            _finalize_export_entry(
                entry,
                pre_filter_rank_position=idx,
                post_filter_rank_position=post_rank_by_node_id.get(entry["node"].plan_node_id),
                surviving_unblocked_candidate_count=surviving_unblocked_candidate_count,
            )
        )
    return exported, filtered


def _assert_finalized_export(entries: List[Dict[str, object]]) -> None:
    non_root_nodes = [entry["node"] for entry in entries if entry["node"].plan_node_id != "plan_node:root"]
    if not any(node.pre_filter_rank_position is not None for node in non_root_nodes):
        raise RuntimeError(f"{PLANNER_MODULE_PATH}:{PLANNER_BUILD_ID}: missing pre_filter_rank_position on non-root nodes")
    for node in non_root_nodes:
        if node.blocked and not node.blocking_reason_codes:
            raise RuntimeError(f"{PLANNER_MODULE_PATH}:{PLANNER_BUILD_ID}: blocked node missing blocking_reason_codes")
        if not node.blocked and node.surviving_unblocked_candidate_count <= 0:
            raise RuntimeError(f"{PLANNER_MODULE_PATH}:{PLANNER_BUILD_ID}: unblocked node missing surviving_unblocked_candidate_count")


def _final_rerank_score(node: PlanNodeV1) -> float:
    breakdown = dict(node.final_score_breakdown or {})
    return float(breakdown.get("symbolic_score", 0.0)) + 0.25 * float(breakdown.get("learned_score", 0.0))


def _sorted_surviving_unblocked(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    survivors = [
        entry for entry in entries
        if entry["node"].post_filter_rank_position is not None and not entry["node"].blocked
    ]
    survivors.sort(
        key=lambda entry: (
            int(entry["node"].post_filter_rank_position or 10**9),
            -_final_rerank_score(entry["node"]),
        )
    )
    recovery_mode = any(_is_recovery_candidate_source(entry["node"].candidate_source) for entry in survivors)
    if recovery_mode:
        top_k = min(5, len(survivors))
        top_slice = survivors[:top_k]
        has_top_movement = any(
            entry["skill"].skill_type == "go_to_region"
            and entry["node"].candidate_source == "non-local reachable frontier"
            for entry in top_slice
        )
        if not has_top_movement:
            probe_risk_in_top = any(
                entry["skill"].skill_type == "probe_hidden_trigger"
                and (
                    float(entry["node"].final_score_breakdown.get("probe_route_risk_penalty", 0.0)) > 0.0
                    or float(entry["node"].final_score_breakdown.get("probe_row_band_retry_penalty", 0.0)) > 0.0
                    or float(entry["node"].final_score_breakdown.get("historical_neighbor_penalty", 0.0)) > 0.0
                )
                for entry in top_slice
            )
            candidate = next(
                (
                    entry for entry in survivors
                    if entry["skill"].skill_type == "go_to_region"
                    and entry["node"].candidate_source == "non-local reachable frontier"
                    and (
                        float(entry["node"].final_score_breakdown.get("recovery_nonlocal_movement_bias", 0.0)) > 0.0
                        or probe_risk_in_top
                    )
                ),
                None,
            )
            if candidate is not None:
                node = candidate["node"]
                candidate["node"] = PlanNodeV1(
                    **{
                        **node.to_dict(),
                        "final_score_breakdown": {
                            **node.final_score_breakdown,
                            "recovery_movement_reservation_bonus": 1.0,
                        },
                    }
                )
                survivors = [entry for entry in survivors if entry is not candidate]
                insert_at = min(top_k - 1, len(survivors))
                survivors.insert(insert_at, candidate)
    for idx, entry in enumerate(survivors, start=1):
        node = entry["node"]
        entry["node"] = PlanNodeV1(
            **{
                **node.to_dict(),
                "post_filter_rank_position": idx,
                "post_filter_survived": True,
            }
        )
    return survivors


def _probe_geometry_key(entry: Dict[str, object], blackboard: Optional[BlackboardStateV2]) -> Tuple[float, float, float, float, float]:
    skill = entry["skill"]
    target_ref = entry.get("effective_target")
    if blackboard is None or not isinstance(target_ref, str) or not target_ref.startswith("trigger_zone:"):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == target_ref), None)
    if zone is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    cell_count = float(len(zone.cells or []))
    return (
        float(zone.hidden_trigger_confidence),
        -float(zone.contradiction_count),
        -float(zone.null_count),
        cell_count,
        -float(skill.executions_this_round),
    )


def _movement_tie_break(entry: Dict[str, object]) -> Tuple[float, ...]:
    node = entry["node"]
    skill = entry["skill"]
    plan_memory = entry.get("plan_memory")
    blackboard = entry.get("blackboard")
    blocked_failure_count = sum(1 for label in skill.failure_mode_labels if label in {"route_stall", "blocked"})
    route_family_penalty = _generalized_route_family_penalty(skill, plan_memory)
    return (
        _untouched_movement_bonus(skill) * (0.0 if route_family_penalty > 0.0 else 1.0),
        _movement_cluster_novelty_bonus(skill, plan_memory) * (0.0 if route_family_penalty > 0.0 else 1.0),
        _cross_area_bonus(skill, blackboard, plan_memory) * (0.0 if route_family_penalty > 0.0 else 1.0),
        _frontier_novelty_bonus(skill, plan_memory) * (0.0 if route_family_penalty > 0.0 else 1.0),
        -_failed_movement_area_retry_penalty(skill, plan_memory),
        -_movement_row_band_retry_penalty(skill, plan_memory),
        -route_family_penalty,
        -_same_target_no_progress_penalty(skill),
        -_same_target_repeat_penalty(skill),
        -_prior_contact_no_reach_penalty(skill),
        -_prior_route_stall_penalty(skill),
        float(not node.recent_failed_movement_cluster_match),
        float(skill.total_attempt_count == 0),
        float(skill.failure_count == 0),
        -float(node.final_score_breakdown.get("movement_cluster_retry_penalty", 0.0)),
        -float(node.final_score_breakdown.get("same_area_movement_retry_penalty", 0.0)),
        float(node.estimated_goal_progress) - float(blocked_failure_count),
    )


def plan_best_first(
    belief: PlannerBeliefStateV1,
    skills: List[SkillSpecV1],
    beam_width: int = 5,
    learned_score_map: Dict[str, float] | None = None,
    blackboard: Optional[BlackboardStateV2] = None,
    plan_memory: Optional[PlanMemoryStateV1] = None,
    candidate_source_label: str = "reused_shortlist",
) -> tuple[List[PlanNodeV1], PlanResultV1 | None]:
    nodes: List[PlanNodeV1] = []
    root = PlanNodeV1("v2.3.2", "plan_node:root", None, 0, None, None, 0.0, 0.0, 0.0, 0.0, False, "root")
    nodes.append(root)
    frontier_entries: List[Dict[str, object]] = []
    visited = set()
    skills_by_id = {skill.skill_id: skill for skill in skills if skill.skill_id}
    debug_counts = {
        "candidate_rejected_invalid": 0,
        "candidate_rejected_missing_binding": 0,
        "candidate_rejected_unreachable": 0,
        "candidate_kept": 0,
    }
    for idx, skill in enumerate(skills):
        skill = _canonical_skill(skill, skills_by_id)
        subgoal_id = _extract_subgoal_id(skill)
        target_ref = _extract_target_ref(skill, subgoal_id)
        if skill.skill_type == "verify_mechanic" and not _has_verification_evidence(skill, subgoal_id, blackboard, target_ref):
            continue
        if skill.skill_type == "probe_hidden_trigger" and not _has_concrete_probe_binding(skill, target_ref, blackboard):
            debug_counts["candidate_rejected_missing_binding"] += 1
            continue
        revisit_penalty = 0.25 if skill.skill_id in belief.plan_memory_refs else 0.0
        contradiction_penalty = 0.5 if skill.failure_mode_labels else 0.0
        symbolic_score = _score(skill, revisit_penalty, contradiction_penalty)
        learned_score = 0.0 if learned_score_map is None else float(learned_score_map.get(skill.skill_id, 0.0))
        score = symbolic_score + 0.25 * learned_score - _consecutive_failed_skill_type_penalty(skill, blackboard)
        score -= _probe_neighbor_contact_penalty(skill, target_ref, blackboard, skills_by_id)
        invalidation_reason = _validation_reason(skill, subgoal_id, blackboard, target_ref)
        if invalidation_reason is not None:
            debug_counts[_counter_bucket(invalidation_reason)] += 1
            continue
        debug_counts["candidate_kept"] += 1
        estimated_success = _estimated_success(skill, target_ref, blackboard)
        cluster_penalty = _probe_cluster_penalty(skill, plan_memory) if skill.skill_type == "probe_hidden_trigger" else 0.0
        neighbor_penalty, historical_neighbor_penalty, current_round_failed_neighbor_count = _neighbor_cluster_penalties(skill, plan_memory) if skill.skill_type == "probe_hidden_trigger" else (0.0, 0.0, 0)
        same_row_penalty = _same_row_sweep_penalty(skill, blackboard, skills_by_id, plan_memory) if skill.skill_type == "probe_hidden_trigger" else 0.0
        distance_from_last_failed_cluster = _distance_from_last_failed_cluster(skill, plan_memory) if skill.skill_type == "probe_hidden_trigger" else None
        row_band_id = None
        cluster_key = cluster_key_for_skill(skill)
        cluster_area, _, cluster_qy = cluster_components(cluster_key)
        if cluster_area is not None and cluster_qy is not None:
            row_band_id = f"{cluster_area}|row|{cluster_qy}"
        distance_bonus = 0.0
        if skill.skill_type == "probe_hidden_trigger" and distance_from_last_failed_cluster is not None and any(
            row.latest_termination_reason_this_round == "contact_no_effect" for row in skills_by_id.values()
        ):
            if distance_from_last_failed_cluster > 0.0:
                distance_bonus = min(0.24, 0.05 * float(distance_from_last_failed_cluster))
        post_jump_exclusion_flag = _post_jump_exclusion_flag(skill, plan_memory) if skill.skill_type == "probe_hidden_trigger" else False
        excluded_by_cooldown = _cooldown_applies(skill, plan_memory)
        blocked_by_exact_cluster_exhaustion = _cluster_exhausted(skill, plan_memory)
        blocked_by_neighbor_cooldown = (excluded_by_cooldown and not blocked_by_exact_cluster_exhaustion) or _probe_neighborhood_exhausted(skill, target_ref, blackboard, skills_by_id)
        blocked_by_post_jump_exclusion = post_jump_exclusion_flag
        blocked_by_unreachable = bool(entry.get("recently_invalidated")) if False else False
        if skill.skill_type == "probe_hidden_trigger":
            local_null_count = _probe_local_null_count(skill, target_ref, blackboard, skills_by_id)
            current_neighborhood = _probe_current_round_neighborhood_strength(skill, target_ref, blackboard, skills_by_id)
            historical_neighborhood = _probe_historical_neighborhood_strength(skill, target_ref, blackboard, skills_by_id)
            estimated_success = max(
                0.01,
                min(
                    1.0,
                    estimated_success
                    + _probe_contact_history_bonus(skill)
                    + distance_bonus
                    - _probe_neighbor_contact_penalty(skill, target_ref, blackboard, skills_by_id)
                    - _probe_historical_contact_penalty(skill, target_ref, blackboard, skills_by_id)
                    - cluster_penalty
                    - neighbor_penalty
                    - historical_neighbor_penalty
                    - same_row_penalty
                    - (0.45 if post_jump_exclusion_flag else 0.0)
                    - min(0.35, 0.09 * float(local_null_count))
                    - min(0.35, 0.08 * current_neighborhood)
                    - min(0.2, 0.04 * historical_neighborhood),
                ),
            )
        estimated_information_gain = _estimated_information_gain(skill, target_ref, blackboard)
        if skill.skill_type == "probe_hidden_trigger":
            local_null_count = _probe_local_null_count(skill, target_ref, blackboard, skills_by_id)
            current_neighborhood = _probe_current_round_neighborhood_strength(skill, target_ref, blackboard, skills_by_id)
            historical_neighborhood = _probe_historical_neighborhood_strength(skill, target_ref, blackboard, skills_by_id)
            estimated_information_gain = max(
                0.05,
                min(
                    1.0,
                    estimated_information_gain
                    + _probe_novelty_bonus(skill, target_ref, blackboard)
                    + _probe_distance_signal(target_ref, blackboard)
                    + 0.5 * distance_bonus
                    - _probe_neighbor_contact_penalty(skill, target_ref, blackboard, skills_by_id)
                    - _probe_historical_contact_penalty(skill, target_ref, blackboard, skills_by_id),
                ),
            )
            estimated_information_gain = max(
                0.05,
                estimated_information_gain
                - 0.6 * cluster_penalty
                - 0.5 * neighbor_penalty
                - 0.4 * historical_neighbor_penalty
                - 0.5 * same_row_penalty
                - min(0.25, 0.06 * float(local_null_count))
                - min(0.25, 0.05 * current_neighborhood)
                - min(0.12, 0.025 * historical_neighborhood),
            )
        estimated_cost = _estimated_cost(skill, target_ref, blackboard)
        route_risk_penalty = _go_to_region_route_risk_penalty(skill) if skill.skill_type == "go_to_region" else 0.0
        movement_cluster_retry_penalty = _movement_cluster_retry_penalty(skill, plan_memory) if skill.skill_type == "go_to_region" else 0.0
        same_area_movement_retry_penalty = _same_area_family_retry_penalty(skill, blackboard, plan_memory) if skill.skill_type == "go_to_region" else 0.0
        failed_movement_area_retry_penalty = _failed_movement_area_retry_penalty(skill, plan_memory) if skill.skill_type == "go_to_region" else 0.0
        movement_row_band_retry_penalty = _movement_row_band_retry_penalty(skill, plan_memory) if skill.skill_type == "go_to_region" else 0.0
        route_family_penalty = _generalized_route_family_penalty(skill, plan_memory) if skill.skill_type == "go_to_region" else 0.0
        movement_cluster_novelty_bonus = _movement_cluster_novelty_bonus(skill, plan_memory) if skill.skill_type == "go_to_region" else 0.0
        frontier_novelty_bonus = _frontier_novelty_bonus(skill, plan_memory) if skill.skill_type == "go_to_region" else 0.0
        cross_area_bonus = _cross_area_bonus(skill, blackboard, plan_memory) if skill.skill_type == "go_to_region" else 0.0
        untouched_movement_bonus = _untouched_movement_bonus(skill)
        probe_target_retry_penalty = _probe_target_retry_penalty(skill)
        probe_cluster_memory_penalty = _probe_cluster_memory_penalty(skill, plan_memory) if skill.skill_type == "probe_hidden_trigger" else 0.0
        probe_row_band_retry_penalty = _probe_row_band_retry_penalty(skill, plan_memory) if skill.skill_type == "probe_hidden_trigger" else 0.0
        probe_route_risk_penalty = _probe_route_risk_penalty(skill, plan_memory) if skill.skill_type == "probe_hidden_trigger" else 0.0
        probe_history_penalty = _probe_history_penalty(skill) if skill.skill_type == "probe_hidden_trigger" else 0.0
        prior_contact_no_reach_penalty = (
            _prior_contact_no_reach_penalty(skill)
            if skill.skill_type == "go_to_region"
            else min(0.45, 0.5 * probe_history_penalty + 0.25 * probe_cluster_memory_penalty)
        )
        prior_route_stall_penalty = _prior_route_stall_penalty(skill)
        same_target_repeat_penalty = _same_target_repeat_penalty(skill)
        same_target_no_progress_penalty = _same_target_no_progress_penalty(skill)
        same_area_family_retry_penalty = same_area_movement_retry_penalty
        movement_candidate_source = _movement_candidate_source_label(skill, target_ref, belief, candidate_source_label)
        movement_default_risk_penalty = _movement_default_risk_penalty(skill) if skill.skill_type == "go_to_region" else 0.0
        movement_route_family_unproven_bonus = _movement_route_family_unproven_bonus(skill, plan_memory) if skill.skill_type == "go_to_region" else 0.0
        recovery_nonlocal_movement_bias = _recovery_nonlocal_movement_bias(
            skill,
            movement_candidate_source,
            plan_memory,
            frontier_entries,
        ) if skill.skill_type == "go_to_region" else 0.0
        if skill.skill_type == "go_to_region":
            score += recovery_nonlocal_movement_bias + movement_route_family_unproven_bonus
        if skill.skill_type == "probe_hidden_trigger":
            score -= probe_history_penalty + 0.7 * probe_cluster_memory_penalty + 0.5 * probe_route_risk_penalty + 0.35 * probe_row_band_retry_penalty
        estimated_goal_progress = _estimated_goal_progress(skill, target_ref, blackboard, skills_by_id, score)
        if skill.skill_type == "go_to_region":
            bonus_damp = 0.0 if route_family_penalty > 0.0 else 1.0
            estimated_success = max(
                0.01,
                estimated_success
                - prior_contact_no_reach_penalty
                - prior_route_stall_penalty
                - same_target_repeat_penalty
                - same_target_no_progress_penalty
                - 0.5 * failed_movement_area_retry_penalty
                - 0.4 * movement_cluster_retry_penalty,
            )
            estimated_success = max(0.01, estimated_success - movement_default_risk_penalty + recovery_nonlocal_movement_bias)
            movement_cluster_novelty_bonus *= bonus_damp
            frontier_novelty_bonus *= bonus_damp
            cross_area_bonus *= bonus_damp
            untouched_movement_bonus *= bonus_damp
        if skill.skill_type == "probe_hidden_trigger":
            estimated_success = max(
                0.01,
                estimated_success
                - probe_target_retry_penalty
                - 0.9 * cluster_penalty
                - 0.9 * probe_cluster_memory_penalty
                - 0.85 * neighbor_penalty
                - 0.6 * historical_neighbor_penalty
                - 0.75 * probe_row_band_retry_penalty
                - 1.2 * probe_route_risk_penalty,
            )
            estimated_success = max(0.01, estimated_success - probe_history_penalty)
            estimated_information_gain = max(
                0.05,
                estimated_information_gain
                - 0.35 * probe_target_retry_penalty
                - 0.55 * probe_row_band_retry_penalty
                - 0.45 * probe_route_risk_penalty
                - 0.2 * historical_neighbor_penalty,
            )
        node = PlanNodeV1(
            "v2.3.2",
            f"plan_node:{idx:03d}",
            root.plan_node_id,
            1,
            subgoal_id,
            skill.skill_id,
            estimated_cost,
            estimated_success,
            estimated_information_gain,
            estimated_goal_progress,
            bool(blocked_by_exact_cluster_exhaustion or blocked_by_neighbor_cooldown or blocked_by_post_jump_exclusion),
            skill.skill_type,
            candidate_cluster_key=cluster_key,
            movement_cluster_key=cluster_key if skill.skill_type == "go_to_region" else None,
            cluster_failure_count=_cluster_failure_count(skill, plan_memory),
            cluster_exhausted_flag=_cluster_exhausted(skill, plan_memory),
            same_row_penalty=same_row_penalty,
            post_jump_exclusion_flag=post_jump_exclusion_flag,
            distance_from_last_failed_cluster=distance_from_last_failed_cluster,
            final_score_breakdown={
                "symbolic_score": round(symbolic_score, 6),
                "learned_score": round(learned_score, 6),
                "route_risk_penalty": round(route_risk_penalty, 6),
                "movement_cluster_retry_penalty": round(movement_cluster_retry_penalty, 6),
                "same_area_movement_retry_penalty": round(same_area_movement_retry_penalty, 6),
                "failed_movement_area_retry_penalty": round(failed_movement_area_retry_penalty, 6),
                "movement_row_band_retry_penalty": round(movement_row_band_retry_penalty, 6),
                "route_family_penalty": round(route_family_penalty, 6),
                "movement_cluster_novelty_bonus": round(movement_cluster_novelty_bonus, 6),
                "frontier_novelty_bonus": round(frontier_novelty_bonus, 6),
                "cross_area_bonus": round(cross_area_bonus, 6),
                "untouched_movement_bonus": round(untouched_movement_bonus, 6),
                "prior_contact_no_reach_penalty": round(prior_contact_no_reach_penalty, 6),
                "prior_route_stall_penalty": round(prior_route_stall_penalty, 6),
                "same_target_repeat_penalty": round(same_target_repeat_penalty, 6),
                "same_target_no_progress_penalty": round(same_target_no_progress_penalty, 6),
                "movement_default_risk_penalty": round(movement_default_risk_penalty, 6),
                "recovery_nonlocal_movement_bias": round(recovery_nonlocal_movement_bias, 6),
                "recovery_movement_reservation_bonus": 0.0,
                "movement_route_family_unproven_bonus": round(movement_route_family_unproven_bonus, 6),
                "probe_target_retry_penalty": round(probe_target_retry_penalty, 6),
                "probe_cluster_retry_penalty": round(cluster_penalty + probe_cluster_memory_penalty, 6),
                "probe_row_band_retry_penalty": round(probe_row_band_retry_penalty, 6),
                "probe_route_risk_penalty": round(probe_route_risk_penalty, 6),
                "probe_history_penalty": round(probe_history_penalty, 6),
                "same_area_family_retry_penalty": round(same_area_family_retry_penalty, 6),
                "cluster_penalty": round(cluster_penalty, 6),
                "same_row_penalty": round(same_row_penalty, 6),
                "distance_bonus": round(distance_bonus, 6),
                "neighbor_penalty": round(neighbor_penalty, 6),
                "historical_neighbor_penalty": round(historical_neighbor_penalty, 6),
                "excluded_by_cooldown": 1.0 if excluded_by_cooldown else 0.0,
                "row_band_id": float(cluster_qy if cluster_qy is not None else -1),
                "current_round_failed_neighbor_count": float(current_round_failed_neighbor_count),
                "current_round_failed_row_count": float(sum(1 for band in (plan_memory.failed_row_bands_this_round if plan_memory is not None else []) if band == row_band_id)),
            },
            excluded_by_cooldown=excluded_by_cooldown,
            neighbor_of_failed_cluster=bool(current_round_failed_neighbor_count > 0),
            row_band_id=row_band_id,
            current_round_failed_neighbor_count=current_round_failed_neighbor_count,
            current_round_failed_row_count=sum(1 for band in (plan_memory.failed_row_bands_this_round if plan_memory is not None else []) if band == row_band_id),
            recent_failed_movement_cluster_match=bool(skill.skill_type == "go_to_region" and cluster_key in set(plan_memory.recent_failed_movement_cluster_keys if plan_memory is not None else [])),
            blocked_by_exact_cluster_exhaustion=blocked_by_exact_cluster_exhaustion,
            blocked_by_neighbor_cooldown=blocked_by_neighbor_cooldown,
            blocked_by_post_jump_exclusion=blocked_by_post_jump_exclusion,
            blocked_by_unreachable=blocked_by_unreachable,
            candidate_source=movement_candidate_source,
        )
        nodes.append(node)
        frontier_entries.append(
            {
                "node": node,
                "skill": skill,
                "effective_target": target_ref,
                "movement_like": _movement_like(skill, subgoal_id, target_ref),
                "has_geometry": bool(target_ref and blackboard is not None and _has_geometry(blackboard, target_ref)),
                "plan_memory": plan_memory,
                "blackboard": blackboard,
                "validation_failed": False,
                "target_required": skill.skill_type in {
                    "go_to_region",
                    "contact_poi",
                    "probe_hidden_trigger",
                    "dwell_on_region",
                    "perform_action_at_region",
                    "cross_transition",
                    "verify_mechanic",
                    "return_to_anchor",
                },
                "recently_invalidated": bool(target_ref and blackboard is not None and _recently_invalidated(blackboard, target_ref)),
            }
        )
    deduped_entries = _dedupe_frontier(frontier_entries)
    has_fresh_movement = any(
        entry.get("skill").skill_type == "go_to_region" and entry.get("skill").total_attempt_count == 0
        for entry in deduped_entries
    )
    has_concrete_probe = any(
        entry.get("skill").skill_type == "probe_hidden_trigger" and entry.get("skill").total_attempt_count == 0 and entry.get("has_geometry")
        for entry in deduped_entries
    )
    if has_fresh_movement or has_concrete_probe:
        deduped_entries = [
            entry for entry in deduped_entries
            if not _repeated_failed_movement(entry.get("skill"))
        ]
    export_entries, deduped_entries = _apply_post_filters(deduped_entries, blackboard, skills_by_id)
    if candidate_source_label == "regenerated_inventory":
        movement_entries = [entry for entry in export_entries if entry["skill"].skill_type == "go_to_region"]
        if movement_entries and not any(entry["node"].candidate_source == "non-local reachable frontier" for entry in movement_entries):
            recovered_entry = _best_fresh_nonlocal_movement_entry(frontier_entries, plan_memory, belief)
            if recovered_entry is not None:
                movement_export_entries, movement_filtered = _apply_post_filters([recovered_entry], blackboard, skills_by_id)
                if movement_filtered:
                    export_entries = [entry for entry in export_entries if entry["skill"].skill_type != "go_to_region"] + movement_export_entries
                    deduped_entries = [entry for entry in deduped_entries if entry["skill"].skill_type != "go_to_region"] + movement_filtered
        has_surviving_movement = any(
            entry["skill"].skill_type == "go_to_region"
            and entry["node"].post_filter_rank_position is not None
            and not entry["node"].blocked
            for entry in export_entries
        )
        if not has_surviving_movement:
            recovered_entry = _best_fresh_nonlocal_movement_entry(frontier_entries, plan_memory, belief)
            if recovered_entry is not None:
                movement_export_entries, movement_filtered = _apply_post_filters([recovered_entry], blackboard, skills_by_id)
                if movement_filtered:
                    export_entries.extend(movement_export_entries)
                    deduped_entries.extend(movement_filtered)
    _assert_finalized_export(export_entries)
    if candidate_source_label == "regenerated_inventory":
        exported_skill_ids = {entry["skill"].skill_id for entry in export_entries}
        exported_nonlocal_movement = any(
            entry["skill"].skill_type == "go_to_region"
            and entry["node"].candidate_source == "non-local reachable frontier"
            for entry in export_entries
        )
        if any(
            skill.skill_type == "go_to_region"
            and skill.active
            and skill.total_attempt_count == 0
            and skill.failure_count == 0
            and not _movement_cluster_locally_exhausted(skill, plan_memory)
            for skill in skills
        ) and not any(
            entry["skill"].skill_type == "go_to_region"
            for entry in export_entries
        ):
            raise RuntimeError(f"{PLANNER_MODULE_PATH}:{PLANNER_BUILD_ID}: recovery mode omitted all fresh movement candidates")
        omitted_fresh_movement = [
            skill.skill_id
            for skill in skills
            if skill.skill_type == "go_to_region" and skill.active and skill.total_attempt_count == 0 and skill.failure_count == 0 and skill.skill_id not in exported_skill_ids
        ]
        if omitted_fresh_movement and not exported_nonlocal_movement:
            print(f"[v2][planner][warning] recovery_mode_omitted_fresh_movement truncation_reason=post_filter ids={','.join(omitted_fresh_movement[:12])}")
        surviving_movement_clusters = {
            entry["node"].movement_cluster_key
            for entry in deduped_entries
            if entry["skill"].skill_type == "go_to_region" and entry["node"].movement_cluster_key
        }
        available_fresh_clusters = {
            cluster_key_for_skill(skill)
            for skill in skills
            if skill.skill_type == "go_to_region" and skill.active and skill.total_attempt_count == 0 and skill.failure_count == 0 and cluster_key_for_skill(skill)
        }
        if len(surviving_movement_clusters) == 1 and len(available_fresh_clusters - surviving_movement_clusters) > 0:
            print(
                f"[v2][planner][error] recovery_mode_single_movement_cluster surviving={list(surviving_movement_clusters)} omitted_clusters={list(sorted(available_fresh_clusters - surviving_movement_clusters))[:8]}"
            )
    recent_navigation_failure = bool(
        blackboard is not None
        and any(record.mode == "poi_approach" and record.outcome_summary in {"contact_no_reach", "route_stall"} for record in list(blackboard.decision_history)[-2:])
    )
    if recent_navigation_failure:
        fresh_pois = [
            entry for entry in deduped_entries
            if entry["skill"].skill_type == "go_to_region"
            and entry["skill"].total_attempt_count == 0
            and entry["skill"].failure_count == 0
        ]
        failed_pois = [
            entry for entry in deduped_entries
            if entry["skill"].skill_type == "go_to_region"
            and entry["skill"].failure_count > 0
        ]
        distant_probes = [
            entry for entry in deduped_entries
            if entry["skill"].skill_type == "probe_hidden_trigger"
            and not entry["node"].excluded_by_cooldown
            and not entry["node"].cluster_exhausted_flag
        ]
        if fresh_pois:
            keep_ids = {id(entry) for entry in fresh_pois}
            deduped_entries = [entry for entry in deduped_entries if id(entry) in keep_ids] + [entry for entry in deduped_entries if id(entry) not in keep_ids]
        elif failed_pois and distant_probes:
            probe_ids = {id(entry) for entry in distant_probes}
            deduped_entries = [entry for entry in deduped_entries if id(entry) in probe_ids] + [entry for entry in deduped_entries if id(entry) not in probe_ids]
    ranked_nodes: List[PlanNodeV1] = [entry["node"] for entry in _sorted_surviving_unblocked(export_entries)]
    selected = ranked_nodes[0] if ranked_nodes else None
    if selected is None:
        regenerated_source_entries = _dedupe_frontier(frontier_entries)
        for entry in regenerated_source_entries:
            node = entry["node"]
            entry["node"] = PlanNodeV1(
                **{
                    **node.to_dict(),
                    "candidate_source": _movement_candidate_source_label(
                        entry["skill"],
                        entry.get("effective_target"),
                        belief,
                        "regenerated_inventory",
                    ),
                }
            )
        regenerated_export_entries, _ = _apply_post_filters(regenerated_source_entries, blackboard, skills_by_id)
        if not any(entry["node"].post_filter_rank_position is not None and not entry["node"].blocked and entry["skill"].skill_type == "go_to_region" for entry in regenerated_export_entries):
            recovered_entry = _best_fresh_nonlocal_movement_entry(regenerated_source_entries, plan_memory, belief)
            if recovered_entry is not None:
                movement_export_entries, movement_filtered = _apply_post_filters([recovered_entry], blackboard, skills_by_id)
                if movement_filtered:
                    regenerated_export_entries.extend(movement_export_entries)
        _assert_finalized_export(regenerated_export_entries)
        export_entries = regenerated_export_entries
        ranked_nodes = [entry["node"] for entry in _sorted_surviving_unblocked(regenerated_export_entries)]
        selected = ranked_nodes[0] if ranked_nodes else None
        if selected is None:
            regenerated_movement_entries = _recovery_movement_entries(regenerated_source_entries)
            if regenerated_movement_entries:
                movement_export_entries, _ = _apply_post_filters(regenerated_movement_entries, blackboard, skills_by_id)
                _assert_finalized_export(movement_export_entries)
                movement_ranked_nodes = [entry["node"] for entry in _sorted_surviving_unblocked(movement_export_entries)]
                if movement_ranked_nodes:
                    export_entries = movement_export_entries
                    ranked_nodes = movement_ranked_nodes
                    selected = movement_ranked_nodes[0]
    if selected is None:
        surviving_count = len([entry for entry in export_entries if not entry["node"].blocked and entry["node"].post_filter_rank_position is not None])
        reachable_frontier = set(getattr(belief, "reachable_frontier_ids", []) or [])
        current_area_id = getattr(belief, "current_area_id", None)
        has_recovery_movement = any(
            skill.skill_type == "go_to_region"
            and skill.active
            and (not reachable_frontier or any(ref in reachable_frontier for ref in skill.precondition_ids))
            and not _movement_cluster_locally_exhausted(skill, plan_memory)
            for skill in skills
        )
        has_nonlocal_recovery_movement = any(
            skill.skill_type == "go_to_region"
            and skill.active
            and (not reachable_frontier or any(ref in reachable_frontier for ref in skill.precondition_ids))
            and not _movement_cluster_locally_exhausted(skill, plan_memory)
            and cluster_components(cluster_key_for_skill(skill))[0] is not None
            and cluster_components(cluster_key_for_skill(skill))[0] != current_area_id
            for skill in skills
        )
        if not has_recovery_movement:
            has_recovery_movement = any(skill.skill_type == "go_to_region" and skill.active for skill in skills)
        root = PlanNodeV1(
            **{
                **root.to_dict(),
                "blocking_reason_codes": [],
                "pre_filter_survived": False,
                "hard_filter_applied": False,
                "post_filter_survived": False,
                "rank_removed_reason": (
                    "recovery_local_only_reinjection"
                    if has_nonlocal_recovery_movement
                    else ("recovery_no_fresh_movement_generated" if has_recovery_movement else None)
                ),
                "pre_filter_rank_position": None,
                "post_filter_rank_position": None,
                "surviving_unblocked_candidate_count": surviving_count,
                "selected_after_rerank": False,
                "candidate_final_status": None,
                "blocking_reason_details": {},
            }
        )
        finalized_nodes = [root] + [entry["node"] for entry in export_entries]
        first_non_root = next((node for node in finalized_nodes if node.plan_node_id != "plan_node:root"), None)
        if first_non_root is not None and first_non_root.pre_filter_rank_position is None:
            raise RuntimeError(f"{PLANNER_MODULE_PATH}:{PLANNER_BUILD_ID}: returning unfinalized nodes")
        return finalized_nodes, None
    ranked_ids = [node.plan_node_id for node in ranked_nodes]
    if selected.plan_node_id not in ranked_ids:
        ranked_ids = [selected.plan_node_id] + [node_id for node_id in ranked_ids if node_id != selected.plan_node_id]
    surviving_count = len([entry for entry in export_entries if not entry["node"].blocked and entry["node"].post_filter_rank_position is not None])
    better_ranked = [
        node for node in ranked_nodes
        if node.post_filter_rank_position is not None
        and selected.post_filter_rank_position is not None
        and node.post_filter_rank_position < selected.post_filter_rank_position
    ]
    if better_ranked:
        raise RuntimeError(f"{PLANNER_MODULE_PATH}:{PLANNER_BUILD_ID}: selected node violates post-filter rank order")
    for entry in export_entries:
        node = entry["node"]
        entry["node"] = PlanNodeV1(
            **{
                **node.to_dict(),
                "selected_after_rerank": bool(node.plan_node_id == selected.plan_node_id),
                "candidate_final_status": "selected" if node.plan_node_id == selected.plan_node_id else ("survived_not_selected" if node.post_filter_survived else node.candidate_final_status),
            }
        )
    finalized_ranked_nodes = [entry["node"] for entry in _sorted_surviving_unblocked(export_entries)]
    if finalized_ranked_nodes:
        selected = finalized_ranked_nodes[0]
        ranked_ids = [node.plan_node_id for node in finalized_ranked_nodes]
        for entry in export_entries:
            node = entry["node"]
            entry["node"] = PlanNodeV1(
                **{
                    **node.to_dict(),
                    "selected_after_rerank": bool(node.plan_node_id == selected.plan_node_id),
                    "candidate_final_status": "selected" if node.plan_node_id == selected.plan_node_id else ("survived_not_selected" if node.post_filter_survived else node.candidate_final_status),
                }
            )
    root = PlanNodeV1(
        **{
            **root.to_dict(),
            "blocking_reason_codes": [],
            "pre_filter_survived": False,
            "hard_filter_applied": False,
            "post_filter_survived": False,
            "rank_removed_reason": None,
            "pre_filter_rank_position": None,
            "post_filter_rank_position": None,
            "surviving_unblocked_candidate_count": surviving_count,
            "selected_after_rerank": False,
            "candidate_final_status": None,
            "blocking_reason_details": {},
        }
    )
    result = PlanResultV1(
        schema_version="v2.3.2",
        plan_id="plan:latest",
        root_plan_node_id=root.plan_node_id,
        selected_plan_node_id=selected.plan_node_id,
        selected_skill_id=selected.skill_id,
        selected_subgoal_id=selected.subgoal_id,
        planner_reason="best_first_symbolic_plus_learned" if learned_score_map else "best_first_symbolic_option",
        alternative_plan_node_ids=ranked_ids,
    )
    finalized_nodes = [root] + [entry["node"] for entry in export_entries]
    first_non_root = next((node for node in finalized_nodes if node.plan_node_id != "plan_node:root"), None)
    if first_non_root is not None and first_non_root.pre_filter_rank_position is None:
        raise RuntimeError(f"{PLANNER_MODULE_PATH}:{PLANNER_BUILD_ID}: returning unfinalized nodes")
    return finalized_nodes, result
