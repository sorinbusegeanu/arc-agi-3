from __future__ import annotations

from typing import Dict, List, Tuple

from codex_baseline_v2.shared.latent_state import active_latent_states
from codex_baseline_v2.shared.plan_records import PlannerBeliefStateV1, SkillSpecV1
from codex_baseline_v2.shared.schemas import BlackboardStateV2
from codex_baseline_v2.planning.plan_memory import (
    PlanMemoryStateV1,
    cluster_ledger_map,
    cluster_key_for_skill,
    cluster_components,
    cluster_key_from_coords,
    movement_cluster_ledger_map,
)


def _skill_target_ref(skill: SkillSpecV1) -> str:
    for ref in skill.precondition_ids:
        if ref.startswith("poi:") or ref.startswith("trigger_zone:"):
            return ref
    return skill.skill_id


def _target_area_id_from_ref(target_ref: str) -> str | None:
    parts = target_ref.split(":")
    if len(parts) <= 2:
        return None
    return ":".join(parts[:-2])


def _target_xy_from_ref(target_ref: str) -> Tuple[int | None, int | None]:
    parts = target_ref.split(":")
    try:
        return int(parts[-2]), int(parts[-1])
    except Exception:
        return None, None


def _probe_local_relation(ax: int, ay: int, bx: int, by: int) -> str:
    if ax == bx and ay == by:
        return "exact"
    distance = abs(ax - bx) + abs(ay - by)
    if distance <= 2:
        return "immediate"
    if (ax == bx or ay == by) and distance <= 8:
        return "immediate"
    if distance <= 5:
        return "immediate"
    if distance <= 10 or ax == bx or ay == by:
        return "cluster"
    return "distant"


def _diversify_movement_clusters(skill_ids: List[str], skills_by_id: dict[str, SkillSpecV1], minimum_clusters: int = 4) -> List[str]:
    diversified: List[str] = []
    overflow: List[str] = []
    used_clusters = set()
    distinct_clusters = {
        cluster_key_for_skill(skills_by_id[skill_id])
        for skill_id in skill_ids
        if skill_id in skills_by_id and cluster_key_for_skill(skills_by_id[skill_id]) is not None
    }
    target_cluster_count = min(max(1, minimum_clusters), len(distinct_clusters))
    for skill_id in skill_ids:
        skill = skills_by_id.get(skill_id)
        cluster_key = cluster_key_for_skill(skill) if skill is not None else None
        if cluster_key is None:
            diversified.append(skill_id)
            continue
        if cluster_key not in used_clusters and len(used_clusters) < target_cluster_count:
            used_clusters.add(cluster_key)
            diversified.append(skill_id)
        else:
            overflow.append(skill_id)
    return diversified + overflow


def _movement_sort_key(skill: SkillSpecV1) -> tuple[float, float, float, str]:
    cluster_key = cluster_key_for_skill(skill) or ""
    area_prefix = cluster_key.rsplit("|", 2)[0] if "|" in cluster_key else ""
    return (
        float(skill.total_attempt_count == 0 and skill.failure_count == 0),
        -float(skill.failure_count),
        -float(skill.total_attempt_count),
        f"{area_prefix}:{skill.skill_id}",
    )


def _fresh_movement_recovery_sort_key(skill: SkillSpecV1, plan_memory: PlanMemoryStateV1 | None) -> tuple[float, float, float, str]:
    cluster_key = cluster_key_for_skill(skill) or ""
    area_id, _, _ = cluster_components(cluster_key)
    failed_clusters = set(plan_memory.recent_failed_movement_cluster_keys) if plan_memory is not None else set()
    failed_areas = set(plan_memory.failed_movement_area_ids_this_round) if plan_memory is not None else set()
    return (
        float(cluster_key not in failed_clusters),
        float(area_id not in failed_areas if area_id is not None else 0.0),
        float(skill.total_attempt_count == 0 and skill.failure_count == 0),
        skill.skill_id,
    )


def _nonlocal_reachable_movement_recovery_sort_key(
    skill: SkillSpecV1,
    plan_memory: PlanMemoryStateV1 | None,
    current_area_id: str | None,
) -> tuple[float, float, float, float, str]:
    cluster_key = cluster_key_for_skill(skill) or ""
    area_id, _, _ = cluster_components(cluster_key)
    failed_clusters = set(plan_memory.recent_failed_movement_cluster_keys) if plan_memory is not None else set()
    failed_areas = set(plan_memory.failed_movement_area_ids_this_round) if plan_memory is not None else set()
    return (
        float(area_id is not None and area_id != current_area_id),
        float(cluster_key not in failed_clusters),
        float(area_id not in failed_areas if area_id is not None else 0.0),
        float(skill.total_attempt_count == 0 and skill.failure_count == 0),
        skill.skill_id,
    )


def _dedup_candidate_subgoals(blackboard: BlackboardStateV2) -> List[str]:
    if blackboard.dependency_graph is None:
        return []
    kept: List[str] = []
    seen_targets = set()
    for subgoal in blackboard.dependency_graph.subgoals:
        if subgoal.status not in {"candidate", "enabled", "blocked"}:
            continue
        if subgoal.subgoal_id.startswith("subgoal:reach:poi:"):
            target_ref = subgoal.subgoal_id[len("subgoal:reach:") :]
            if target_ref in seen_targets:
                continue
            seen_targets.add(target_ref)
            kept.append(subgoal.subgoal_id)
            continue
        if subgoal.subgoal_id.startswith("subgoal:enable_route:latent:"):
            continue
        kept.append(subgoal.subgoal_id)
    concrete = [subgoal_id for subgoal_id in kept if ":poi:" in subgoal_id or "trigger_zone:" in subgoal_id]
    return concrete[:8]


def _filtered_candidate_skill_ids(
    skills: List[SkillSpecV1],
    plan_memory: PlanMemoryStateV1 | None = None,
    force_full_inventory: bool = False,
    reachable_frontier_ids: List[str] | None = None,
) -> List[str]:
    concrete = []
    abstract = []
    seen_bindings = set()
    skills_by_id = {skill.skill_id: skill for skill in skills}
    for skill in skills:
        if not skill.active:
            continue
        binding = _skill_target_ref(skill)
        row = (skill.skill_id, skill.total_attempt_count > 0, binding.startswith("poi:") or binding.startswith("trigger_zone:"))
        if row[2]:
            dedup_key = (skill.skill_type, binding)
            if dedup_key in seen_bindings:
                continue
            seen_bindings.add(dedup_key)
            concrete.append(skill.skill_id)
        else:
            abstract.append(skill.skill_id)
    local_probe_failures = any(
        skill.skill_type == "probe_hidden_trigger" and skill.latest_termination_reason_this_round == "contact_no_effect"
        for skill in skills
    )
    failure_reasons = {"contact_no_effect", "contact_no_reach", "route_stall", "blocked", "invalid_target", "no_progress", "unreachable_now"}
    recovery_mode = (
        force_full_inventory
        or any(skill.latest_termination_reason_this_round in failure_reasons for skill in skills)
        or bool(
            plan_memory is not None
            and (
                plan_memory.failed_cluster_keys_this_round
                or plan_memory.failed_movement_area_ids_this_round
                or plan_memory.recent_failed_movement_cluster_keys
                or plan_memory.excluded_cluster_cooldowns
            )
        )
    )
    current_area_id = None
    if plan_memory is not None and plan_memory.failed_movement_area_ids_this_round:
        current_area_id = plan_memory.failed_movement_area_ids_this_round[-1]
    widen_after_probe_failure = bool(plan_memory is not None and plan_memory.failed_cluster_keys_this_round)
    failed_probe_coords = [
        (skill.target_x_this_round, skill.target_y_this_round)
        for skill in skills
        if skill.skill_type == "probe_hidden_trigger"
        and skill.latest_termination_reason_this_round == "contact_no_effect"
        and skill.target_x_this_round is not None
        and skill.target_y_this_round is not None
    ]
    historical_probe_coords = [
        (skill.target_x_this_round, skill.target_y_this_round, int(skill.historical_contact_no_effect_count))
        for skill in skills
        if skill.skill_type == "probe_hidden_trigger"
        and skill.target_x_this_round is not None
        and skill.target_y_this_round is not None
        and skill.historical_contact_no_effect_count > 0
    ]
    cluster_ledger = cluster_ledger_map(plan_memory)
    probe_rows = []
    for skill_id in [skill_id for skill_id in concrete if ":probe_hidden_trigger:" in skill_id]:
        parts = skill_id.split(":")
        try:
            sx, sy = int(parts[-2]), int(parts[-1])
        except (TypeError, ValueError):
            probe_rows.append({"skill_id": skill_id, "sx": None, "sy": None, "local_hits": 0, "historical_hits": 0, "novelty": 999})
            continue
        local_hits = 0
        exact_hit = False
        nearest_failed_distance = 9999
        for fx, fy in failed_probe_coords:
            distance = abs(int(sx) - int(fx)) + abs(int(sy) - int(fy))
            nearest_failed_distance = min(nearest_failed_distance, distance)
            relation = _probe_local_relation(int(sx), int(sy), int(fx), int(fy))
            if relation == "exact":
                exact_hit = True
                break
            if relation == "immediate":
                local_hits += 2
            elif relation == "cluster":
                local_hits += 1
        if exact_hit:
            continue
        if local_hits >= 2:
            continue
        historical_hits = 0
        for fx, fy, count in historical_probe_coords:
            relation = _probe_local_relation(int(sx), int(sy), int(fx), int(fy))
            if relation in {"immediate", "cluster"}:
                historical_hits += count
        if historical_hits >= 3:
            continue
        cluster_key = cluster_key_for_skill(skills_by_id[skill_id]) if skill_id in skills_by_id else None
        if cluster_key is not None:
            cluster_entry = cluster_ledger.get(cluster_key)
            if plan_memory is not None and skills_by_id[skill_id].skill_type == "probe_hidden_trigger" and int(plan_memory.excluded_cluster_cooldowns.get(cluster_key, 0)) > 0:
                continue
            if cluster_entry is not None and cluster_entry.locally_exhausted:
                continue
        novelty = nearest_failed_distance if nearest_failed_distance != 9999 else 999
        probe_rows.append({"skill_id": skill_id, "sx": int(sx), "sy": int(sy), "local_hits": local_hits, "historical_hits": historical_hits, "novelty": novelty})
    if local_probe_failures and not widen_after_probe_failure:
        has_distant_probe = False
        for row in probe_rows:
            sx = row["sx"]
            sy = row["sy"]
            if sx is None or sy is None:
                continue
            if all(_probe_local_relation(sx, sy, int(fx), int(fy)) == "distant" for fx, fy in failed_probe_coords):
                has_distant_probe = True
                break
        if has_distant_probe:
            probe_rows = [
                row for row in probe_rows
                if row["sx"] is None or all(
                    _probe_local_relation(int(row["sx"]), int(row["sy"]), int(fx), int(fy)) == "distant"
                    for fx, fy in failed_probe_coords
                )
            ]
    probe_cap = len(probe_rows) if recovery_mode else (8 if widen_after_probe_failure else (2 if local_probe_failures else 4))
    probe_rows.sort(key=lambda row: (row["local_hits"], row["historical_hits"], -int(row["novelty"]), row["skill_id"]))
    probe = [row["skill_id"] for row in probe_rows[:probe_cap]]
    reachable_set = set(reachable_frontier_ids or [])
    movement_rows = []
    for skill_id in [skill_id for skill_id in concrete if ":go_to_region:" in skill_id]:
        skill = skills_by_id.get(skill_id)
        if skill is None:
            continue
        target_ref = _skill_target_ref(skill)
        if recovery_mode:
            if reachable_set and target_ref not in reachable_set:
                continue
            cluster_key = cluster_key_for_skill(skill)
            ledger_row = cluster_ledger.get(cluster_key or "")
            area_id, _, _ = cluster_components(cluster_key)
            if ledger_row is not None and ledger_row.locally_exhausted:
                continue
            movement_rows.append(skill_id)
        else:
            movement_rows.append(skill_id)
    if recovery_mode:
        nonlocal_movement_rows = []
        for skill_id in movement_rows:
            cluster_key = cluster_key_for_skill(skills_by_id[skill_id])
            area_id, _, _ = cluster_components(cluster_key)
            if current_area_id is not None and area_id != current_area_id:
                nonlocal_movement_rows.append(skill_id)
        if nonlocal_movement_rows:
            movement_rows = nonlocal_movement_rows
        movement_rows.sort(
            key=lambda skill_id: _nonlocal_reachable_movement_recovery_sort_key(
                skills_by_id[skill_id],
                plan_memory,
                current_area_id,
            ),
            reverse=True,
        )
    else:
        movement_rows.sort(key=lambda skill_id: _movement_sort_key(skills_by_id[skill_id]), reverse=True)
    movement_cap = len(movement_rows) if recovery_mode else 4
    other_cap = len([skill_id for skill_id in concrete if skill_id not in set(probe + movement_rows)]) if recovery_mode else 2
    abstract_cap = 0 if recovery_mode else 2
    movement = _diversify_movement_clusters(movement_rows[:movement_cap], skills_by_id, minimum_clusters=6 if recovery_mode else 4)
    other_concrete = [skill_id for skill_id in concrete if skill_id not in set(probe + movement)][:other_cap]
    if recovery_mode:
        result = probe + movement + other_concrete
        returned = set(result)
        movement_ledger = cluster_ledger_map(plan_memory)
        fresh_reachable = []
        for skill in skills:
            if skill.skill_type != "go_to_region" or not skill.active:
                continue
            if skill.total_attempt_count != 0 or skill.failure_count != 0:
                continue
            if reachable_set and _skill_target_ref(skill) not in reachable_set:
                continue
            if skill.skill_id in returned:
                continue
            cluster_key = cluster_key_for_skill(skill)
            ledger_row = movement_ledger.get(cluster_key or "")
            if ledger_row is not None and ledger_row.locally_exhausted:
                continue
            area_id, _, _ = cluster_components(cluster_key)
            if current_area_id is not None and area_id == current_area_id:
                continue
            fresh_reachable.append(skill)
        if fresh_reachable:
            fresh_reachable.sort(
                key=lambda skill: _nonlocal_reachable_movement_recovery_sort_key(
                    skill,
                    plan_memory,
                    current_area_id,
                ),
                reverse=True,
            )
            result.extend(skill.skill_id for skill in fresh_reachable if skill.skill_id not in set(result))
        returned = set(result)
        omitted_fresh = [
            (skill.skill_id, _skill_target_ref(skill))
            for skill in skills
            if skill.skill_type == "go_to_region"
            and skill.active
            and skill.total_attempt_count == 0
            and skill.failure_count == 0
            and ((_skill_target_ref(skill) in reachable_set) if reachable_set else True)
            and skill.skill_id not in returned
        ]
        if omitted_fresh:
            payload = ",".join(f"{skill_id}:{target_ref}" for skill_id, target_ref in omitted_fresh[:12])
            print(f"[v2][planner][warning] recovery_mode_omitted_fresh_pois omission_reason=movement_filter ids={payload}")
        return result
    total_cap = 12
    return (probe + movement + other_concrete + abstract[:abstract_cap])[:total_cap]


def materialize_recovery_movement_skills(
    skills: List[SkillSpecV1],
    reachable_frontier_ids: List[str],
    plan_memory: PlanMemoryStateV1 | None = None,
) -> tuple[List[SkillSpecV1], Dict[str, int]]:
    failure_reasons = {"contact_no_effect", "contact_no_reach", "route_stall", "blocked", "invalid_target", "no_progress", "unreachable_now"}
    recovery_mode = any(skill.latest_termination_reason_this_round in failure_reasons for skill in skills) or bool(
        plan_memory is not None
        and (
            plan_memory.failed_cluster_keys_this_round
            or plan_memory.failed_movement_area_ids_this_round
            or plan_memory.recent_failed_movement_cluster_keys
            or plan_memory.excluded_cluster_cooldowns
        )
    )
    if not recovery_mode:
        return skills, {
            "total_reachable_non_local_movement_frontier_count": 0,
            "total_non_local_movement_candidates_materialized": 0,
            "total_excluded_for_exhausted_cluster": 0,
            "total_excluded_for_missing_skill_materialization": 0,
        }
    existing_by_target = {
        _skill_target_ref(skill): skill
        for skill in skills
        if skill.skill_type == "go_to_region" and skill.active
    }
    failed_clusters = set(plan_memory.recent_failed_movement_cluster_keys) if plan_memory is not None else set()
    movement_ledger = movement_cluster_ledger_map(plan_memory)
    failed_areas = set(plan_memory.failed_movement_area_ids_this_round) if plan_memory is not None else set()
    exhausted_local_area = next(iter(failed_areas), None)
    augmented = list(skills)
    total_reachable_non_local = 0
    total_materialized = 0
    total_exhausted = 0
    total_missing = 0
    for target_ref in reachable_frontier_ids:
        if not isinstance(target_ref, str) or not target_ref.startswith("poi:"):
            continue
        area_id = _target_area_id_from_ref(target_ref)
        x, y = _target_xy_from_ref(target_ref)
        if x is None or y is None:
            total_missing += 1
            continue
        if exhausted_local_area is not None and area_id == exhausted_local_area:
            continue
        total_reachable_non_local += 1
        cluster_key = cluster_key_from_coords(area_id, x, y)
        ledger_row = movement_ledger.get(cluster_key or "")
        if cluster_key in failed_clusters or (ledger_row is not None and ledger_row.locally_exhausted):
            total_exhausted += 1
            continue
        if target_ref in existing_by_target:
            continue
        synthetic = SkillSpecV1(
            schema_version="v2.3.2",
            skill_id=f"skill:go_to_region:{target_ref}",
            skill_type="go_to_region",
            parameter_names=[],
            precondition_ids=[target_ref],
            expected_effect_node_ids=[],
            average_duration_steps=0.0,
            success_rate=0.0,
            failure_mode_labels=[],
            source_trace_ids=[],
            total_attempt_count=0,
            success_count=0,
            failure_count=0,
            active=True,
            target_x_this_round=x,
            target_y_this_round=y,
        )
        augmented.append(synthetic)
        existing_by_target[target_ref] = synthetic
        total_materialized += 1
    return augmented, {
        "total_reachable_non_local_movement_frontier_count": total_reachable_non_local,
        "total_non_local_movement_candidates_materialized": total_materialized,
        "total_excluded_for_exhausted_cluster": total_exhausted,
        "total_excluded_for_missing_skill_materialization": total_missing,
    }


def build_planner_belief_state(
    blackboard: BlackboardStateV2,
    skills: List[SkillSpecV1],
    candidate_skills: List[SkillSpecV1] | None = None,
    plan_memory_refs: List[str] | None = None,
    plan_memory: PlanMemoryStateV1 | None = None,
    force_full_inventory: bool = False,
    recovery_reconstruction_debug: Dict[str, int] | None = None,
) -> PlannerBeliefStateV1:
    active = active_latent_states(blackboard.latent_states, min_confidence=0.6)
    uncertain = [state for state in blackboard.latent_states if state.current_value is None or state.confidence < 0.6]
    candidate_subgoals = _dedup_candidate_subgoals(blackboard)
    reachable_frontier_ids = [record.poi_id for record in blackboard.reachability_table if record.status in {"reachable", "reachable_now", "uncertain"}]
    recovery_debug = recovery_reconstruction_debug or {
        "total_reachable_non_local_movement_frontier_count": 0,
        "total_non_local_movement_candidates_materialized": 0,
        "total_excluded_for_exhausted_cluster": 0,
        "total_excluded_for_missing_skill_materialization": 0,
    }
    return PlannerBeliefStateV1(
        schema_version="v2.3.2",
        current_area_id=blackboard.area_table[0].area_id if blackboard.area_table else None,
        current_avatar_track_id=blackboard.avatar_track_table[0].track_id if blackboard.avatar_track_table else None,
        candidate_subgoal_ids=candidate_subgoals,
        active_latent_state_ids=[state.latent_state_id for state in active],
        uncertain_latent_state_ids=[state.latent_state_id for state in uncertain],
        reachable_frontier_ids=reachable_frontier_ids,
        candidate_skill_ids=_filtered_candidate_skill_ids(
            candidate_skills if candidate_skills is not None else skills,
            plan_memory=plan_memory,
            force_full_inventory=force_full_inventory,
            reachable_frontier_ids=reachable_frontier_ids,
        ),
        plan_memory_refs=list(plan_memory_refs or []),
        recovery_reconstruction_debug=recovery_debug,
    )
