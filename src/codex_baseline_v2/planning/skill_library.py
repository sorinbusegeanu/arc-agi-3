from __future__ import annotations

import json
import os
import re
import tempfile
from typing import List, Optional, Tuple

from codex_baseline_v2.shared.plan_records import SkillExecutionRecordV1, SkillSpecV1
from codex_baseline_v2.shared.storage import StoragePathsV2


_KEEP_EXISTING = object()


def _atomic_write(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="v232_", suffix=".json", dir=os.path.dirname(path))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(tmp_path, path)


def _rebuild_skill(
    skill: SkillSpecV1,
    *,
    success_rate: float,
    failure_mode_labels: List[str],
    source_trace_ids: List[str],
    average_duration_steps: float,
    total_attempt_count: int,
    success_count: int,
    failure_count: int,
    active: Optional[bool] = None,
    executions_this_round: object = _KEEP_EXISTING,
    latest_termination_reason_this_round: object = _KEEP_EXISTING,
    repeated_contact_no_effect_count_this_round: object = _KEEP_EXISTING,
    target_x_this_round: object = _KEEP_EXISTING,
    target_y_this_round: object = _KEEP_EXISTING,
    historical_contact_no_effect_count: object = _KEEP_EXISTING,
    latest_prior_round_termination_reason: object = _KEEP_EXISTING,
) -> SkillSpecV1:
    return SkillSpecV1(
        schema_version=skill.schema_version,
        skill_id=skill.skill_id,
        skill_type=skill.skill_type,
        parameter_names=list(skill.parameter_names),
        precondition_ids=list(skill.precondition_ids),
        expected_effect_node_ids=list(skill.expected_effect_node_ids),
        average_duration_steps=average_duration_steps,
        success_rate=success_rate,
        failure_mode_labels=failure_mode_labels,
        source_trace_ids=source_trace_ids,
        total_attempt_count=total_attempt_count,
        success_count=success_count,
        failure_count=failure_count,
        active=skill.active if active is None else active,
        executions_this_round=skill.executions_this_round if executions_this_round is _KEEP_EXISTING else int(executions_this_round),
        latest_termination_reason_this_round=(
            skill.latest_termination_reason_this_round
            if latest_termination_reason_this_round is _KEEP_EXISTING
            else latest_termination_reason_this_round
        ),
        repeated_contact_no_effect_count_this_round=(
            skill.repeated_contact_no_effect_count_this_round
            if repeated_contact_no_effect_count_this_round is _KEEP_EXISTING
            else int(repeated_contact_no_effect_count_this_round)
        ),
        target_x_this_round=skill.target_x_this_round if target_x_this_round is _KEEP_EXISTING else target_x_this_round,
        target_y_this_round=skill.target_y_this_round if target_y_this_round is _KEEP_EXISTING else target_y_this_round,
        historical_contact_no_effect_count=(
            skill.historical_contact_no_effect_count
            if historical_contact_no_effect_count is _KEEP_EXISTING
            else int(historical_contact_no_effect_count)
        ),
        latest_prior_round_termination_reason=(
            skill.latest_prior_round_termination_reason
            if latest_prior_round_termination_reason is _KEEP_EXISTING
            else latest_prior_round_termination_reason
        ),
    )


def merge_skill_specs(existing: SkillSpecV1, incoming: SkillSpecV1) -> SkillSpecV1:
    return SkillSpecV1(
        schema_version=existing.schema_version,
        skill_id=existing.skill_id,
        skill_type=existing.skill_type,
        parameter_names=list(existing.parameter_names or incoming.parameter_names),
        precondition_ids=sorted(set(existing.precondition_ids) | set(incoming.precondition_ids)),
        expected_effect_node_ids=sorted(set(existing.expected_effect_node_ids) | set(incoming.expected_effect_node_ids)),
        average_duration_steps=existing.average_duration_steps,
        success_rate=existing.success_rate,
        failure_mode_labels=_ordered_unique(list(existing.failure_mode_labels) + list(incoming.failure_mode_labels)),
        source_trace_ids=_ordered_unique(list(existing.source_trace_ids) + list(incoming.source_trace_ids)),
        total_attempt_count=existing.total_attempt_count,
        success_count=existing.success_count,
        failure_count=existing.failure_count,
        active=bool(existing.active or incoming.active),
        executions_this_round=existing.executions_this_round,
        latest_termination_reason_this_round=existing.latest_termination_reason_this_round,
        repeated_contact_no_effect_count_this_round=existing.repeated_contact_no_effect_count_this_round,
        target_x_this_round=existing.target_x_this_round,
        target_y_this_round=existing.target_y_this_round,
        historical_contact_no_effect_count=existing.historical_contact_no_effect_count,
        latest_prior_round_termination_reason=existing.latest_prior_round_termination_reason,
    )


def _target_required(skill: SkillSpecV1) -> bool:
    return skill.skill_type in {
        "go_to_region",
        "contact_poi",
        "probe_hidden_trigger",
        "dwell_on_region",
        "perform_action_at_region",
        "cross_transition",
        "verify_mechanic",
        "return_to_anchor",
    }


def _ordered_unique(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _is_failure_reason(reason: str) -> bool:
    return reason in {"invalid_target", "blocked", "route_stall", "no_progress", "unreachable_now", "contact_no_effect", "contact_no_reach", "contact_reached_boundary_without_route_progress"}


def _canonical_trace_ids(values: List[str]) -> List[str]:
    return _ordered_unique([value for value in values if isinstance(value, str) and value.startswith("skill_execution:")])


_ROUND_RE = re.compile(r"round(\d+)")


def _execution_round_id(execution: SkillExecutionRecordV1) -> Optional[int]:
    match = _ROUND_RE.search(execution.execution_id or "")
    if match is None:
        return None
    return int(match.group(1))


def _target_xy(skill: SkillSpecV1) -> tuple[Optional[int], Optional[int]]:
    for ref in skill.precondition_ids:
        if not isinstance(ref, str) or not ref.startswith("trigger_zone:"):
            continue
        parts = ref.split(":")
        if len(parts) < 2:
            continue
        try:
            return int(parts[-2]), int(parts[-1])
        except (TypeError, ValueError):
            return None, None
    return None, None


def _refresh_skill_stats(skills: List[SkillSpecV1], executions: List[SkillExecutionRecordV1]) -> tuple[List[SkillSpecV1], List[dict]]:
    execution_by_skill = {}
    execution_rounds = [_execution_round_id(execution) for execution in executions]
    current_round_id = max((round_id for round_id in execution_rounds if round_id is not None), default=None)
    for execution in executions:
        execution_by_skill.setdefault(execution.skill_id, []).append(execution)
    refreshed: List[SkillSpecV1] = []
    payload_rows: List[dict] = []
    for skill in skills:
        rows = execution_by_skill.get(skill.skill_id, [])
        total_attempt_count = len(rows)
        success_count = sum(1 for row in rows if row.success)
        failure_count = sum(1 for row in rows if _is_failure_reason(row.termination_reason))
        success_rate = float(success_count) / float(total_attempt_count) if total_attempt_count > 0 else 0.0
        average_duration_steps = skill.average_duration_steps
        if rows:
            durations = [max(1, int(row.end_step) - int(row.start_step) + 1) for row in rows]
            average_duration_steps = float(sum(durations)) / float(len(durations))
        failure_mode_labels = sorted({row.termination_reason for row in rows if row.termination_reason and row.termination_reason != "success"})
        source_trace_ids = _canonical_trace_ids(list(skill.source_trace_ids))
        for row in rows:
            source_trace_ids = _canonical_trace_ids(source_trace_ids + [row.execution_id])
        current_round_rows = []
        if current_round_id is not None:
            current_round_rows = [row for row in rows if _execution_round_id(row) == current_round_id]
        prior_round_rows = [row for row in rows if _execution_round_id(row) is not None and _execution_round_id(row) != current_round_id]
        executions_this_round = len(current_round_rows)
        latest_termination_reason_this_round = current_round_rows[-1].termination_reason if current_round_rows else None
        repeated_contact_no_effect_count_this_round = 0
        for row in reversed(current_round_rows):
            if row.termination_reason in {"contact_no_effect", "contact_reached_boundary_without_route_progress"}:
                repeated_contact_no_effect_count_this_round += 1
            else:
                break
        historical_contact_no_effect_count = sum(
            1 for row in prior_round_rows
            if row.termination_reason in {"contact_no_effect", "contact_reached_boundary_without_route_progress"}
        )
        latest_prior_round_termination_reason = prior_round_rows[-1].termination_reason if prior_round_rows else None
        target_x_this_round = None
        target_y_this_round = None
        if skill.skill_type == "probe_hidden_trigger":
            target_x_this_round, target_y_this_round = _target_xy(skill)
        active = skill.active
        if skill.skill_type == "probe_hidden_trigger" and not any(ref.startswith("trigger_zone:") for ref in skill.precondition_ids):
            active = False
        if _target_required(skill) and not skill.precondition_ids:
            active = False
        consecutive_invalid_target = 0
        for row in reversed(rows):
            if row.termination_reason == "invalid_target":
                consecutive_invalid_target += 1
            else:
                break
        if total_attempt_count > 0 and success_rate == 0.0 and rows and all(row.termination_reason == "invalid_target" for row in rows):
            active = False
        if consecutive_invalid_target >= 2:
            active = False
        updated = _rebuild_skill(
            skill,
            success_rate=success_rate,
            failure_mode_labels=failure_mode_labels,
            source_trace_ids=source_trace_ids,
            average_duration_steps=average_duration_steps,
            total_attempt_count=total_attempt_count,
            success_count=success_count,
            failure_count=failure_count,
            active=active,
            executions_this_round=executions_this_round,
            latest_termination_reason_this_round=latest_termination_reason_this_round,
            repeated_contact_no_effect_count_this_round=repeated_contact_no_effect_count_this_round,
            target_x_this_round=target_x_this_round,
            target_y_this_round=target_y_this_round,
            historical_contact_no_effect_count=historical_contact_no_effect_count,
            latest_prior_round_termination_reason=latest_prior_round_termination_reason,
        )
        refreshed.append(updated)
        payload = updated.to_dict()
        payload["total_attempt_count"] = total_attempt_count
        payload["success_count"] = success_count
        payload["failure_count"] = failure_count
        payload_rows.append(payload)
    return refreshed, payload_rows


def _reconcile_from_executions(skills: List[SkillSpecV1], executions: List[SkillExecutionRecordV1]) -> tuple[List[SkillSpecV1], List[dict]]:
    refreshed_skills, payload_rows = _refresh_skill_stats(skills, executions)
    execution_ids_by_skill = {}
    for execution in executions:
        execution_ids_by_skill.setdefault(execution.skill_id, []).append(execution.execution_id)
    reconciled_skills: List[SkillSpecV1] = []
    reconciled_rows: List[dict] = []
    for skill, payload in zip(refreshed_skills, payload_rows):
        matching_ids = execution_ids_by_skill.get(skill.skill_id, [])
        if matching_ids and (skill.total_attempt_count == 0 or not skill.source_trace_ids):
            repaired = _rebuild_skill(
                skill,
                success_rate=skill.success_rate,
                failure_mode_labels=list(skill.failure_mode_labels),
                source_trace_ids=_canonical_trace_ids(list(skill.source_trace_ids) + matching_ids),
                average_duration_steps=skill.average_duration_steps,
                total_attempt_count=max(skill.total_attempt_count, len(matching_ids)),
                success_count=skill.success_count,
                failure_count=skill.failure_count,
                active=skill.active,
                executions_this_round=skill.executions_this_round,
                latest_termination_reason_this_round=skill.latest_termination_reason_this_round,
                repeated_contact_no_effect_count_this_round=skill.repeated_contact_no_effect_count_this_round,
                target_x_this_round=skill.target_x_this_round,
                target_y_this_round=skill.target_y_this_round,
                historical_contact_no_effect_count=skill.historical_contact_no_effect_count,
                latest_prior_round_termination_reason=skill.latest_prior_round_termination_reason,
            )
            skill = repaired
            payload = repaired.to_dict()
            payload["total_attempt_count"] = repaired.total_attempt_count
            payload["success_count"] = repaired.success_count
            payload["failure_count"] = repaired.failure_count
        reconciled_skills.append(skill)
        reconciled_rows.append(payload)
    return reconciled_skills, reconciled_rows


def reconcile_skill_library(skills: List[SkillSpecV1], executions: List[SkillExecutionRecordV1]) -> List[SkillSpecV1]:
    reconciled_skills, _ = _reconcile_from_executions(skills, executions)
    return reconciled_skills


def load_skill_library(storage: StoragePathsV2, game_id: str) -> Tuple[List[SkillSpecV1], List[SkillExecutionRecordV1]]:
    game_root = storage.game_root(game_id)
    skills_path = os.path.join(game_root, "skills.json")
    execs_path = os.path.join(game_root, "skill_executions.json")
    skills = []
    executions = []
    if os.path.exists(skills_path):
        with open(skills_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        skills = [SkillSpecV1.from_dict(v) for v in payload.get("skills", [])]
    if os.path.exists(execs_path):
        with open(execs_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        executions = [SkillExecutionRecordV1.from_dict(v) for v in payload.get("skill_executions", [])]
    skills, _ = _reconcile_from_executions(skills, executions)
    return skills, executions


def save_skill_library(storage: StoragePathsV2, game_id: str, skills: List[SkillSpecV1], executions: List[SkillExecutionRecordV1]) -> List[SkillSpecV1]:
    game_root = storage.game_root(game_id)
    refreshed_skills, payload_rows = _reconcile_from_executions(skills, executions)
    _atomic_write(os.path.join(game_root, "skill_executions.json"), {"schema_version": "v2.3.2", "skill_executions": [row.to_dict() for row in executions]})
    _atomic_write(os.path.join(game_root, "skills.json"), {"schema_version": "v2.3.2", "skills": payload_rows})
    return refreshed_skills


def _target_binding(skill: SkillSpecV1) -> str:
    return next((ref for ref in skill.precondition_ids if ref.startswith("poi:") or ref.startswith("trigger_zone:")), skill.skill_id)


def _failed_only_sort_key(skill: SkillSpecV1) -> tuple[float, float, float]:
    route_stall_recent = 1.0 if "route_stall" in skill.failure_mode_labels else 0.0
    blocked_recent = 1.0 if "blocked" in skill.failure_mode_labels else 0.0
    return (float(skill.failure_count), route_stall_recent, blocked_recent)


def candidate_skills(skills: List[SkillSpecV1], belief_state_subgoal_ids: List[str]) -> List[SkillSpecV1]:
    allowed = set(belief_state_subgoal_ids)
    recovery_failure_reasons = {"contact_no_effect", "contact_no_reach", "route_stall", "blocked", "invalid_target", "no_progress", "unreachable_now"}
    recovery_mode = any(skill.latest_termination_reason_this_round in recovery_failure_reasons for skill in skills)
    attempted_successful: List[SkillSpecV1] = []
    attempted_failed_only: List[SkillSpecV1] = []
    zero_attempt_probe: List[SkillSpecV1] = []
    zero_attempt_other: List[SkillSpecV1] = []
    zero_attempt_cap_by_type = {
        "go_to_region": 999999 if recovery_mode else 4,
        "probe_hidden_trigger": 999999 if recovery_mode else 4,
        "verify_mechanic": 999999 if recovery_mode else 2,
    }
    reserved_probe_slots = 0 if recovery_mode else 2
    failed_only_cap_by_type = {
        "go_to_region": 999999 if recovery_mode else 1,
        "probe_hidden_trigger": 999999 if recovery_mode else 1,
        "verify_mechanic": 999999 if recovery_mode else 1,
    }
    seen_zero_attempt_by_binding = set()
    kept_zero_attempt_by_type = {}
    seen_failed_only_by_binding = set()
    kept_failed_only_by_type = {}
    for skill in skills:
        if not skill.active:
            continue
        if skill.precondition_ids and not any(ref in allowed or ref.startswith("subgoal:") or ref.startswith("poi:") or ref.startswith("trigger_zone:") for ref in skill.precondition_ids):
            continue
        has_attempts = skill.total_attempt_count > 0
        if has_attempts and skill.success_count > 0:
            attempted_successful.append(skill)
            continue
        if has_attempts:
            attempted_failed_only.append(skill)
            continue
        if skill.skill_type == "probe_hidden_trigger":
            zero_attempt_probe.append(skill)
        else:
            zero_attempt_other.append(skill)
    attempted_successful.sort(key=lambda skill: (-skill.success_count, -skill.success_rate, skill.failure_count, skill.average_duration_steps))
    attempted_failed_only.sort(key=_failed_only_sort_key)
    out: List[SkillSpecV1] = list(attempted_successful)
    probe_slots_used = 0
    for skill in zero_attempt_probe + zero_attempt_other:
        target_ref = _target_binding(skill)
        if skill.skill_type == "go_to_region" and target_ref:
            binding_key = ("go_to_region", target_ref)
        elif skill.skill_type == "probe_hidden_trigger" and target_ref and target_ref.startswith("trigger_zone:"):
            binding_key = ("probe_hidden_trigger", target_ref)
        else:
            binding_key = (skill.skill_type, target_ref or skill.skill_id)
        if binding_key in seen_zero_attempt_by_binding:
            continue
        type_count = kept_zero_attempt_by_type.get(skill.skill_type, 0)
        if type_count >= zero_attempt_cap_by_type.get(skill.skill_type, 3):
            continue
        if skill.skill_type != "probe_hidden_trigger" and probe_slots_used < reserved_probe_slots and zero_attempt_probe:
            remaining_probe = any(
                ("probe_hidden_trigger", _target_binding(probe_skill)) not in seen_zero_attempt_by_binding
                and kept_zero_attempt_by_type.get("probe_hidden_trigger", 0) < zero_attempt_cap_by_type.get("probe_hidden_trigger", 0)
                for probe_skill in zero_attempt_probe
            )
            if remaining_probe:
                continue
        seen_zero_attempt_by_binding.add(binding_key)
        kept_zero_attempt_by_type[skill.skill_type] = type_count + 1
        if skill.skill_type == "probe_hidden_trigger":
            probe_slots_used += 1
        out.append(skill)
    for skill in attempted_failed_only:
        target_ref = _target_binding(skill)
        binding_key = (skill.skill_type, target_ref)
        if binding_key in seen_failed_only_by_binding:
            continue
        type_count = kept_failed_only_by_type.get(skill.skill_type, 0)
        if type_count >= failed_only_cap_by_type.get(skill.skill_type, 1):
            continue
        seen_failed_only_by_binding.add(binding_key)
        kept_failed_only_by_type[skill.skill_type] = type_count + 1
        out.append(skill)
    return out
