from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

from codex_baseline_v2.analyst.analyst import analyze_episodes, analyze_episodes_parallel
from codex_baseline_v2.controller.controller import select_instruction
from codex_baseline_v2.executor.option_executor import execute_option, instruction_from_skill, skill_record_from_execution
from codex_baseline_v2.executor.online_executor import run_offline_local_execution, run_online_executions_parallel
from codex_baseline_v2.learning.ranking_inference import rank_mechanics, rank_options
from codex_baseline_v2.memory.store import (
    append_round_report,
    load_blackboard_typed,
    save_blackboard,
    save_event_table,
    save_interventions,
    save_mechanic_hypotheses,
    save_navigation_graph,
    save_world_model_archive,
)
from codex_baseline_v2.memory.graph_store import save_graph_state
from codex_baseline_v2.planning.hierarchical_planner import PLANNER_BUILD_ID, PLANNER_MODULE_PATH, plan_best_first
from codex_baseline_v2.planning.plan_memory import load_plan_memory, plan_memory_refs as build_plan_memory_refs, reconcile_plan_memory, save_plan_memory
from codex_baseline_v2.planning.planner_state_builder import build_planner_belief_state, materialize_recovery_movement_skills
from codex_baseline_v2.planning.skill_inducer import induce_skills
from codex_baseline_v2.planning.skill_library import candidate_skills, load_skill_library, reconcile_skill_library, save_skill_library
from codex_baseline_v2.runtime.environment_session import EnvironmentSessionV2
from codex_baseline_v2.runtime.session_manager import SessionManagerV2
from codex_baseline_v2.runtime.trajectory_collector import CollectionConfigV2, TrajectoryCollectorV2
from codex_baseline_v2.shared.config import V2Config
from codex_baseline_v2.shared.logging_utils import log_event
from codex_baseline_v2.shared.metrics import RoundMetricsV2, compute_round_metrics
from codex_baseline_v2.shared.plan_records import PlanResultV1
from codex_baseline_v2.shared.learning_records import OptionRankingRecordV1
from codex_baseline_v2.shared.storage import StoragePathsV2
from codex_baseline_v2.shared.schemas import BlackboardStateV2, ControllerInstructionV2, DecisionRecordV2, ExecutorOutcomeV2, SCHEMA_VERSION, TrajectoryEpisodeV2
from codex_baseline_v2.storage.sqlite_intermediates import SQLiteIntermediateStoreV2, sqlite_db_path_for_round
from codex_baseline_v2.trajectory_analysis.analyzer import analyze_trajectories
from codex_baseline_v2.trajectory_analysis.avatar_hypothesis_debug import export_avatar_debug


def _log_stage_start(game_id: str, round_id: int, stage: str) -> float:
    started_at = time.perf_counter()
    print(f"[v2] stage_start game_id={game_id} round_id={round_id} stage={stage}", flush=True)
    return started_at


def _log_stage_stop(game_id: str, round_id: int, stage: str, started_at: float, stage_timings: Dict[str, float]) -> None:
    duration = time.perf_counter() - started_at
    stage_timings[stage] = duration
    print(f"[v2] stage_stop game_id={game_id} round_id={round_id} stage={stage} duration_s={round(duration, 3)}", flush=True)


def export_postrun_heatmaps(cfg: V2Config, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    storage = StoragePathsV2(cfg.memory.storage_dir)
    session_dir = storage.game_root(cfg.game_id)
    return {
        "session_dir": session_dir,
        "session_state_or_artifacts": {
            "session_dir": session_dir,
            "storage_root": cfg.memory.storage_dir,
            "game_id": cfg.game_id,
            "summary": summary or {},
        },
    }


def _extract_tag(text: Optional[str], key: str) -> Optional[str]:
    if not text:
        return None
    prefix = key + "="
    for token in text.split():
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _export_debug_table(storage: StoragePathsV2, game_id: str, round_id: int, filename: str, payload: object) -> None:
    path = os.path.join(storage.category_path(game_id, round_id, "exports"), filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def _save_plan_memory(storage: StoragePathsV2, game_id: str, belief, nodes, result, plan_memory_state=None, failure_payload: Optional[Dict[str, Any]] = None, current_round_id: Optional[int] = None) -> None:
    game_root = storage.game_root(game_id)
    plans_path = os.path.join(game_root, "plans.json")
    failures_path = os.path.join(game_root, "plan_failures.json")
    _validate_finalized_plan_nodes(nodes)
    family_debug: Dict[str, Dict[str, float]] = {}
    if nodes:
        for family in ("probe_hidden_trigger", "go_to_region"):
            family_nodes = [node for node in nodes if getattr(node, "skill_id", None) and getattr(node, "notes", None) == family]
            if not family_nodes:
                continue
            best_score = max(
                (
                    float(getattr(node, "final_score_breakdown", {}).get("symbolic_score", 0.0))
                    + 0.25 * float(getattr(node, "final_score_breakdown", {}).get("learned_score", 0.0))
                    for node in family_nodes
                ),
                default=0.0,
            )
            blocked_count = sum(1 for node in family_nodes if getattr(node, "blocked", False))
            survivors = sum(1 for node in family_nodes if getattr(node, "post_filter_rank_position", None) is not None and not getattr(node, "blocked", False))
            avg_retry_penalty = sum(
                float(getattr(node, "final_score_breakdown", {}).get("probe_route_risk_penalty" if family == "probe_hidden_trigger" else "route_risk_penalty", 0.0))
                + float(getattr(node, "final_score_breakdown", {}).get("probe_cluster_retry_penalty" if family == "probe_hidden_trigger" else "movement_cluster_retry_penalty", 0.0))
                for node in family_nodes
            ) / float(max(1, len(family_nodes)))
            family_debug[family] = {
                "candidate_count": float(len(family_nodes)),
                "blocked_count": float(blocked_count),
                "survivor_count": float(survivors),
                "best_rerank_score": round(best_score, 6),
                "average_retry_penalty": round(avg_retry_penalty, 6),
                "selected": 1.0 if result is not None and any(getattr(node, "plan_node_id", "") == getattr(result, "selected_plan_node_id", "") for node in family_nodes) else 0.0,
            }
    payload = {
        "schema_version": "v2.3.2",
        "planner_module_path": PLANNER_MODULE_PATH,
        "planner_build_id": PLANNER_BUILD_ID,
        "export_finalized": True,
        "belief_state": belief.to_dict() if belief is not None else None,
        "plan_nodes": [node.to_dict() for node in (nodes or [])],
        "plan_result": result.to_dict() if result is not None else None,
        "plan_memory_state": plan_memory_state.to_dict() if plan_memory_state is not None else None,
        "family_rerank_debug": family_debug,
    }
    with open(plans_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    _verify_plan_artifact(plans_path)
    failures = []
    if os.path.exists(failures_path):
        with open(failures_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        failures = list(loaded.get("failures", []))
    target_round = current_round_id if current_round_id is not None else (failure_payload.get("round_id") if failure_payload is not None else None)
    if target_round is not None:
        failures = [row for row in failures if int(row.get("round_id", -1)) != int(target_round)]
    if failure_payload is not None:
        failures.append(failure_payload)
    with open(failures_path, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": "v2.3.2", "failures": failures}, handle, sort_keys=True)


def _save_plan_pass_debug(storage: StoragePathsV2, game_id: str, round_id: int, episode_idx: int, belief, nodes, result, outcome) -> None:
    path = os.path.join(storage.category_path(game_id, round_id, "exports"), f"plan_pass_{episode_idx:03d}.json")
    _validate_finalized_plan_nodes(nodes)
    payload = {
        "schema_version": "v2.4.0",
        "planner_module_path": PLANNER_MODULE_PATH,
        "planner_build_id": PLANNER_BUILD_ID,
        "export_finalized": True,
        "episode_idx": episode_idx,
        "belief_state": belief.to_dict() if belief is not None else None,
        "plan_nodes": [node.to_dict() for node in (nodes or [])],
        "plan_result": result.to_dict() if result is not None else None,
        "execution_outcome": outcome.to_dict() if outcome is not None else None,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def _validate_finalized_plan_nodes(nodes) -> None:
    if nodes is None:
        return
    non_root = [node for node in nodes if getattr(node, "plan_node_id", "") != "plan_node:root"]
    if not non_root:
        return
    if not any(node.pre_filter_rank_position is not None for node in non_root):
        raise RuntimeError("artifact-integrity-error: planner returned non-finalized plan nodes")
    for node in non_root:
        if node.blocked and not node.blocking_reason_codes:
            raise RuntimeError("artifact-integrity-error: blocked plan node missing blocking_reason_codes")
        if not node.blocked and node.surviving_unblocked_candidate_count <= 0:
            raise RuntimeError("artifact-integrity-error: unblocked plan node missing surviving_unblocked_candidate_count")


def _verify_plan_artifact(path: str) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    nodes = list(payload.get("plan_nodes", []))
    non_root = [node for node in nodes if node.get("plan_node_id") != "plan_node:root"]
    if not any(node.get("pre_filter_rank_position") is not None for node in non_root):
        raise RuntimeError("artifact-integrity-error: persisted plans.json missing finalized export fields")


def _save_ranking_records(
    storage: StoragePathsV2,
    game_id: str,
    option_record: Any,
    option_scores: Dict[str, float],
    mechanic_record: Any,
    mechanic_scores: Dict[str, float],
    mechanic_signal_count: int = 0,
) -> None:
    game_root = storage.game_root(game_id)
    option_applied = bool(option_record is not None and option_scores)
    mechanic_applied = bool(mechanic_record is not None and mechanic_scores)
    with open(os.path.join(game_root, "option_rankings.json"), "w", encoding="utf-8") as handle:
        option_state = "inactive"
        if option_record is not None and option_scores:
            option_state = "applied" if option_record.model_score_map_ref != "live_reconstruction" else "reconstructed"
        json.dump(
            {
                "schema_version": "v2.3.4",
                "state": option_state,
                "applied": option_applied,
                "ranking_record": option_record.to_dict() if option_record is not None else None,
                "score_map": option_scores if option_record is not None else {},
            },
            handle,
            sort_keys=True,
        )
    with open(os.path.join(game_root, "mechanic_rankings.json"), "w", encoding="utf-8") as handle:
        mechanic_state = "inactive"
        mechanic_suppression_reasons: Dict[str, str] = {}
        selected_mechanic_id = mechanic_record.selected_mechanic_id if mechanic_record is not None else ""
        if mechanic_record is not None and mechanic_scores:
            if mechanic_record.model_score_map_ref == "flat_suppressed":
                if mechanic_signal_count > 0 and mechanic_scores:
                    mechanic_state = "reconstructed"
                    selected_mechanic_id = max(mechanic_scores, key=mechanic_scores.get)
                else:
                    mechanic_state = "suppressed"
                    mechanic_suppression_reasons = {mechanic_id: "flat_score_cluster" for mechanic_id in mechanic_record.candidate_mechanic_ids}
            else:
                mechanic_state = "applied" if mechanic_record.model_score_map_ref != "live_reconstruction" else "reconstructed"
        json.dump(
            {
                "schema_version": "v2.3.4",
                "state": mechanic_state,
                "applied": mechanic_applied,
                "ranking_record": (
                    {
                        **mechanic_record.to_dict(),
                        "selected_mechanic_id": selected_mechanic_id,
                    }
                    if mechanic_record is not None else None
                ),
                "score_map": mechanic_scores if mechanic_record is not None else {},
                "suppression_reasons": mechanic_suppression_reasons,
            },
            handle,
            sort_keys=True,
        )


def _finalize_option_ranking_record(option_record: Any, option_scores: Dict[str, float], plan_nodes, plan_result) -> tuple[Any, Dict[str, float]]:
    if option_record is None:
        return None, {}
    final_candidate_skill_ids: List[str] = []
    seen = set()
    for node in plan_nodes or []:
        if getattr(node, "plan_node_id", "") == "plan_node:root":
            continue
        skill_id = getattr(node, "skill_id", None)
        if skill_id and skill_id not in seen:
            seen.add(skill_id)
            final_candidate_skill_ids.append(skill_id)
    selected_skill_id = str(plan_result.selected_skill_id or "") if plan_result is not None else ""
    if selected_skill_id and selected_skill_id not in seen:
        final_candidate_skill_ids.append(selected_skill_id)
    filtered_scores = {skill_id: float(option_scores.get(skill_id, 0.0)) for skill_id in final_candidate_skill_ids}
    return (
        OptionRankingRecordV1(
            schema_version=option_record.schema_version,
            ranking_id=option_record.ranking_id,
            planner_state_ref=option_record.planner_state_ref,
            candidate_skill_ids=final_candidate_skill_ids,
            selected_skill_id=selected_skill_id,
            model_score_map_ref=option_record.model_score_map_ref,
            fallback_used=option_record.fallback_used,
        ),
        filtered_scores,
    )


def _analysis_sqlite_store(cfg: V2Config, game_id: str, round_id: int):
    if getattr(cfg.storage, "backend", "files") != "sqlite":
        return None
    return SQLiteIntermediateStoreV2(sqlite_db_path_for_round(cfg.memory.storage_dir, game_id, round_id))


def _movement_planning_debug(belief, plan_nodes, plan_result=None) -> Dict[str, object]:
    nodes = [node for node in (plan_nodes or []) if getattr(node, "skill_id", None) and "go_to_region" in str(getattr(node, "skill_id", ""))]
    cluster_counts: Dict[str, int] = {}
    reachable_counts: Dict[str, int] = {}
    survivor_counts: Dict[str, int] = {}
    suppression_reasons: Dict[str, List[str]] = {}
    top_losing_cluster_keys: List[str] = []
    failed_area_ids: List[str] = []
    for node in nodes:
        cluster_key = getattr(node, "movement_cluster_key", None) or getattr(node, "candidate_cluster_key", None) or "unknown"
        cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
        if getattr(node, "candidate_source", None):
            reachable_counts[cluster_key] = reachable_counts.get(cluster_key, 0) + 1
        if getattr(node, "post_filter_rank_position", None) is not None and not getattr(node, "blocked", False):
            survivor_counts[cluster_key] = survivor_counts.get(cluster_key, 0) + 1
        if getattr(node, "blocking_reason_codes", None):
            suppression_reasons.setdefault(cluster_key, [])
            for code in getattr(node, "blocking_reason_codes", []):
                if code not in suppression_reasons[cluster_key]:
                    suppression_reasons[cluster_key].append(code)
    selected_plan_node_id = getattr(plan_result, "selected_plan_node_id", None)
    for node in sorted(nodes, key=lambda row: (row.post_filter_rank_position is None, row.post_filter_rank_position or 10**9)):
        if getattr(node, "plan_node_id", None) == selected_plan_node_id:
            continue
        cluster_key = getattr(node, "movement_cluster_key", None) or getattr(node, "candidate_cluster_key", None)
        if cluster_key and cluster_key not in top_losing_cluster_keys:
            top_losing_cluster_keys.append(cluster_key)
            area_id = cluster_key.rsplit("|", 2)[0] if "|" in cluster_key else None
            if area_id and area_id not in failed_area_ids:
                failed_area_ids.append(area_id)
        if len(top_losing_cluster_keys) >= 5:
            break
    return {
        "total_movement_candidates": len(nodes),
        "reachable_frontier_count": len(getattr(belief, "reachable_frontier_ids", []) or []) if belief is not None else 0,
        "per_cluster_candidate_count": cluster_counts,
        "per_cluster_survivor_count": survivor_counts,
        "per_cluster_reachable_count": reachable_counts,
        "per_cluster_suppression_reasons": suppression_reasons,
        "top_losing_cluster_keys": top_losing_cluster_keys,
        "failed_area_ids": failed_area_ids,
    }


def _promote_movement_planning_debug(plan_memory_state):
    from codex_baseline_v2.planning.plan_memory import ClusterLedgerEntryV1
    from codex_baseline_v2.planning.plan_memory import cluster_components
    debug = dict(getattr(plan_memory_state, "movement_planning_debug", {}) or {})
    if not debug:
        return plan_memory_state
    candidate_counts = dict(debug.get("per_cluster_candidate_count", {}))
    reachable_counts = dict(debug.get("per_cluster_reachable_count", {}))
    survivor_counts = dict(debug.get("per_cluster_survivor_count", {}))
    suppression = {str(k): list(v) for k, v in dict(debug.get("per_cluster_suppression_reasons", {})).items()}
    ledger_rows = []
    existing = {row.cluster_key: row for row in getattr(plan_memory_state, "movement_cluster_ledger", [])}
    for cluster_key in sorted(set(existing) | set(candidate_counts) | set(reachable_counts) | set(survivor_counts)):
        row = existing.get(cluster_key)
        if row is None:
            area_id, qx, qy = cluster_components(cluster_key)
            centroid_x = float(qx * 8 + 4) if qx is not None else None
            centroid_y = float(qy * 8 + 4) if qy is not None else None
            row = ClusterLedgerEntryV1(
                cluster_key=cluster_key,
                area_id=area_id,
                quantized_x=qx,
                quantized_y=qy,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
            )
        ledger_rows.append(
            ClusterLedgerEntryV1(
                cluster_key=row.cluster_key,
                area_id=row.area_id,
                quantized_x=row.quantized_x,
                quantized_y=row.quantized_y,
                centroid_x=row.centroid_x,
                centroid_y=row.centroid_y,
                failure_count=row.failure_count,
                success_count=row.success_count,
                contact_no_effect_count=row.contact_no_effect_count,
                last_failed_round=row.last_failed_round,
                repeated_no_effect_streak=row.repeated_no_effect_streak,
                locally_exhausted=row.locally_exhausted,
                candidate_count=int(candidate_counts.get(cluster_key, row.candidate_count)),
                reachable_count=int(reachable_counts.get(cluster_key, row.reachable_count)),
                survivor_count=int(survivor_counts.get(cluster_key, row.survivor_count)),
                retry_count=int(row.failure_count),
                latest_failure_reason=row.latest_failure_reason,
                suppression_reasons=list(suppression.get(cluster_key, row.suppression_reasons)),
            )
        )
    recent_failed = list(getattr(plan_memory_state, "recent_failed_movement_cluster_keys", []) or [])
    for cluster_key in debug.get("top_losing_cluster_keys", []) or []:
        if cluster_key and cluster_key not in recent_failed:
            recent_failed.append(cluster_key)
    failed_areas = list(getattr(plan_memory_state, "failed_movement_area_ids_this_round", []) or [])
    for area_id in debug.get("failed_area_ids", []) or []:
        if area_id and area_id not in failed_areas:
            failed_areas.append(area_id)
    return replace(
        plan_memory_state,
        movement_cluster_ledger=ledger_rows,
        recent_failed_movement_cluster_keys=recent_failed[-12:],
        failed_movement_area_ids_this_round=failed_areas,
    )


def _instruction_has_executable_target(instruction: ControllerInstructionV2) -> bool:
    if instruction.mode in {"random_probe", "unguided_probe", "discriminating_probe"}:
        return True
    trigger_zone_id = _extract_tag(instruction.rationale or "", "trigger_zone_id")
    if trigger_zone_id and instruction.target_geometry is not None:
        return True
    if instruction.target_geometry is not None:
        return True
    if instruction.target_poi_id:
        return True
    return False


def _instruction_trigger_zone_id(instruction: ControllerInstructionV2) -> Optional[str]:
    return _extract_tag(instruction.rationale or "", "trigger_zone_id")


def _skill_target_ref(skill) -> Optional[str]:
    for ref in getattr(skill, "precondition_ids", []) or []:
        if isinstance(ref, str) and (ref.startswith("poi:") or ref.startswith("trigger_zone:")):
            return ref
    return None


def _skill_invalidated_this_round(skill, blackboard: Optional[BlackboardStateV2], round_id: int) -> bool:
    if blackboard is None:
        return False
    target_ref = _skill_target_ref(skill)
    if not target_ref:
        return False
    if isinstance(blackboard.metadata, dict) and target_ref in set(blackboard.metadata.get("round_local_invalidated_targets", [])):
        return True
    for record in reversed(list(blackboard.decision_history)):
        if int(getattr(record, "round_id", -1)) != int(round_id):
            continue
        if record.target_invalidated and (record.selected_target_poi_id == target_ref or record.selected_trigger_zone_id == target_ref):
            return True
        if record.outcome_summary == "invalid_target_pre_execution" and (record.selected_target_poi_id == target_ref or record.selected_trigger_zone_id == target_ref):
            return True
    return False


def _choose_structured_fallback_skill(skills, plan_memory_state, blackboard: Optional[BlackboardStateV2] = None, round_id: Optional[int] = None):
    if blackboard is not None and round_id is not None:
        skills = [skill for skill in skills if not _skill_invalidated_this_round(skill, blackboard, int(round_id))]
    failed_clusters = set(getattr(plan_memory_state, "failed_cluster_keys_this_round", []) if plan_memory_state is not None else [])
    repeated_trigger_failures = sum(1 for skill in skills if skill.skill_type == "probe_hidden_trigger" and skill.latest_termination_reason_this_round == "contact_no_effect")
    repeated_navigation_failures = sum(1 for skill in skills if skill.skill_type == "go_to_region" and skill.latest_termination_reason_this_round in {"route_stall", "contact_no_reach", "blocked"})
    if repeated_navigation_failures > repeated_trigger_failures:
        for skill in skills:
            if skill.skill_type == "probe_hidden_trigger":
                from codex_baseline_v2.planning.plan_memory import cluster_key_for_skill, cluster_distance
                cluster_key = cluster_key_for_skill(skill)
                if not cluster_key:
                    continue
                if cluster_key in failed_clusters:
                    continue
                if any((cluster_distance(cluster_key, failed) or 99) > 1 for failed in failed_clusters) or not failed_clusters:
                    return "distant_trigger_probe", skill
    for skill in skills:
        if skill.skill_type == "go_to_region" and skill.active and "route_stall" not in skill.failure_mode_labels and "contact_no_reach" not in skill.failure_mode_labels:
            return "navigation_poi", skill
    for skill in skills:
        if skill.skill_type == "probe_hidden_trigger":
            from codex_baseline_v2.planning.plan_memory import cluster_key_for_skill, cluster_distance
            cluster_key = cluster_key_for_skill(skill)
            if not cluster_key:
                continue
            if cluster_key in failed_clusters:
                continue
            if any((cluster_distance(cluster_key, failed) or 99) > 1 for failed in failed_clusters) or not failed_clusters:
                return "distant_trigger_probe", skill
    for skill in skills:
        if skill.skill_type == "go_to_region" and skill.active:
            return "navigation_poi", skill
    for skill in skills:
        if skill.skill_type in {"cross_transition", "return_to_anchor"} and skill.active:
            return "cross_area_exploration", skill
    return None, None


def _surviving_plan_skill_ids(plan_nodes) -> set[str]:
    out = set()
    for node in plan_nodes or []:
        if getattr(node, "plan_node_id", "") == "plan_node:root":
            continue
        if getattr(node, "blocked", False):
            continue
        if getattr(node, "post_filter_rank_position", None) is None:
            continue
        skill_id = getattr(node, "skill_id", None)
        if skill_id:
            out.add(skill_id)
    return out


def _measurable_progress(outcome: Optional[ExecutorOutcomeV2]) -> bool:
    if outcome is None:
        return False
    if outcome.reached:
        return True
    progress = list(getattr(outcome, "target_progress", []) or [])
    if progress and max(progress) != min(progress):
        return True
    return any(
        record.distance_decreased
        or record.reached
        or bool(record.event_ids)
        or bool(record.topology_delta_id)
        or record.local_change_magnitude > 0.0
        for record in getattr(outcome, "consequence_records", [])
    )


def _bind_instruction_to_blackboard(instruction: ControllerInstructionV2, blackboard: BlackboardStateV2) -> ControllerInstructionV2:
    if instruction.target_poi_id and instruction.target_geometry is None:
        poi = next((row for row in blackboard.poi_table if row.poi_id == instruction.target_poi_id), None)
        if poi is not None and poi.bbox is not None:
            instruction = replace(instruction, target_geometry=poi.bbox)
    trigger_zone_id = _instruction_trigger_zone_id(instruction)
    if trigger_zone_id and instruction.target_geometry is None:
        zone = next((row for row in blackboard.trigger_zone_table if row.trigger_zone_id == trigger_zone_id), None)
        if zone is not None:
            bbox = zone.bbox
            if bbox is None and zone.cells:
                xs = [cell[0] for cell in zone.cells]
                ys = [cell[1] for cell in zone.cells]
                from codex_baseline_v2.shared.utils import BBox
                bbox = BBox(min(xs), min(ys), max(xs), max(ys))
            if bbox is not None:
                instruction = replace(instruction, target_type="trigger_zone", target_geometry=bbox, target_region=bbox)
    return instruction


def _invalid_pre_execution_outcome(game_id: str, instruction: ControllerInstructionV2) -> tuple[ExecutorOutcomeV2, TrajectoryEpisodeV2]:
    episode_id = f"exec_round{instruction.round_id:03d}_invalid"
    outcome = ExecutorOutcomeV2(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        round_id=instruction.round_id,
        instruction_id=instruction.instruction_id,
        instruction_mode=instruction.mode,
        target_poi_id=instruction.target_poi_id,
        target_type=instruction.target_type,
        target_geometry=instruction.target_geometry,
        target_source_round=instruction.target_source_round,
        actions=[],
        target_progress=[],
        reached=False,
        contact=False,
        blocked=False,
        outcome_summary="invalid_target_pre_execution",
        consequence_records=[],
    )
    episode = TrajectoryEpisodeV2(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        episode_id=episode_id,
        steps=[],
        done=False,
        win=False,
        seed=None,
        metadata={"mode": instruction.mode, "instruction": instruction.to_dict(), "diagnostic": "invalid_target_pre_execution"},
    )
    return outcome, episode


def _decision_target_invalidated(instruction: Optional[ControllerInstructionV2], outcome: Optional[ExecutorOutcomeV2]) -> bool:
    if instruction is None or outcome is None:
        return False
    trigger_zone_id = _instruction_trigger_zone_id(instruction)
    if trigger_zone_id and instruction.target_geometry is not None:
        if outcome.outcome_summary == "contact_reached_boundary_without_route_progress":
            return True
        return False
    flat_progress = bool(outcome.target_progress) and max(outcome.target_progress) == min(outcome.target_progress)
    if outcome.outcome_summary in {"contact_reached_boundary_without_route_progress", "route_stall"} and instruction.mode == "poi_approach":
        return True
    if outcome.outcome_summary == "contact_no_reach" and instruction.mode == "poi_approach" and flat_progress:
        return True
    return bool(outcome.outcome_summary in {"invalid_target_pre_execution", "invalid_target"} and not outcome.actions)


def _outcome_score(outcome: Any) -> tuple[float, float, float, float]:
    final_distance = float(outcome.target_progress[-1]) if getattr(outcome, "target_progress", None) else 1e9
    progress_events = sum(
        1.0 for record in getattr(outcome, "consequence_records", []) if record.reached or record.contact or record.distance_decreased
    )
    return (
        1.0 if getattr(outcome, "reached", False) else 0.0,
        1.0 if getattr(outcome, "contact", False) else 0.0,
        progress_events,
        -final_distance,
    )


def _consistent_round_metrics(metrics: RoundMetricsV2, outcomes: List[ExecutorOutcomeV2], controller_modes: List[str]) -> RoundMetricsV2:
    if not outcomes or len(controller_modes) != 1:
        return metrics
    reached = any(outcome.reached for outcome in outcomes)
    contact = any(outcome.contact for outcome in outcomes)
    blocked = any(outcome.blocked for outcome in outcomes)
    route_success_rate = metrics.route_success_rate
    contact_success_rate = metrics.contact_success_rate
    if not reached:
        route_success_rate = 0.0
    if contact and not reached:
        contact_success_rate = max(contact_success_rate, 1.0 / float(max(1, len(controller_modes))))
    if blocked and not reached:
        route_success_rate = 0.0
    return RoundMetricsV2(
        unique_states=metrics.unique_states,
        unique_pois=metrics.unique_pois,
        route_success_rate=route_success_rate,
        useful_change_rate=metrics.useful_change_rate,
        contact_success_rate=contact_success_rate,
    )


def _round_execution_aggregate(outcomes: List[ExecutorOutcomeV2]) -> Dict[str, object]:
    if not outcomes:
        return {
            "directed_episode_count": 0,
            "last_episode_outcome_summary": None,
            "outcome_counts": {},
            "contact_count": 0,
            "reach_count": 0,
            "blocked_count": 0,
        }
    outcome_counts: Dict[str, int] = {}
    for outcome in outcomes:
        outcome_counts[outcome.outcome_summary] = outcome_counts.get(outcome.outcome_summary, 0) + 1
    return {
        "directed_episode_count": len(outcomes),
        "last_episode_outcome_summary": outcomes[-1].outcome_summary,
        "outcome_counts": outcome_counts,
        "contact_count": sum(1 for outcome in outcomes if outcome.contact),
        "reach_count": sum(1 for outcome in outcomes if outcome.reached),
        "blocked_count": sum(1 for outcome in outcomes if outcome.blocked),
    }


def run_autonomous_rounds(cfg: V2Config, env_factory: Any, env_factory_path: Optional[str] = None, workers: int = 1) -> Dict[str, Any]:
    storage = StoragePathsV2(cfg.memory.storage_dir)
    session_mgr = SessionManagerV2(cfg.memory.storage_dir)
    state = session_mgr.init_or_resume(cfg.game_id, resume_if_exists=bool(cfg.runtime.resume_if_exists))

    max_rounds = int(cfg.runtime.max_rounds) if cfg.runtime else cfg.rounds
    effective_workers = max(1, min(int(workers), int(max(cfg.collection.initial_probe_episodes, cfg.collection.directed_probe_episodes))))
    last_report = None
    total_wins = 0
    max_steps_per_game = 0
    episodes_seen = 0
    shared_env = None
    shared_session: Optional[EnvironmentSessionV2] = None
    worker_pool = None
    if effective_workers <= 1:
        shared_env = env_factory()
        shared_session = EnvironmentSessionV2(shared_env, cfg.game_id)
    else:
        worker_pool = mp.get_context("spawn").Pool(processes=effective_workers)
    try:
        for round_id in range(state.round_id, max_rounds):
            print(f"[v2] round_start game_id={cfg.game_id} round_id={round_id}", flush=True)
            round_t0 = time.perf_counter()
            stage_timings: Dict[str, float] = {}
            plan_result = None
            option_record = None
            mechanic_record = None
            option_scores = {}
            mechanic_scores = {}
            storage.ensure_round_dirs(cfg.game_id, round_id)
            session = shared_session
            if session is None:
                env = env_factory()
                session = EnvironmentSessionV2(env, cfg.game_id)
            collector = TrajectoryCollectorV2(
                storage,
                CollectionConfigV2(
                    episodes=cfg.collection.initial_probe_episodes if round_id == 0 else cfg.collection.directed_probe_episodes,
                    max_steps_per_episode=cfg.collection.max_steps_per_episode,
                    max_steps_per_instruction=cfg.collection.max_steps_per_instruction,
                    seed=cfg.collection.seed,
                    action_repeat_limit=cfg.collection.action_repeat_limit,
                    keep_invalid_steps_for_debug=cfg.debug.keep_invalid_steps_for_debug,
                    write_raw_copy=bool(cfg.storage.keep_raw_env_payloads),
                    keep_observations_in_artifacts=bool(cfg.storage.keep_raw_frames),
                    keep_raw_info_in_artifacts=bool(cfg.storage.keep_raw_env_payloads),
                    keep_observation_summaries_in_artifacts=bool(cfg.storage.keep_raw_frames),
                    storage_backend=getattr(cfg.storage, "backend", "files"),
                ),
            )
            if round_id == 0:
                t0 = _log_stage_start(cfg.game_id, round_id, "collection")
                analysis_stats: Dict[str, object] = {}
                if effective_workers > 1 and env_factory_path and cfg.env.env_id and cfg.env.env_root:
                    episodes = collector.collect_round_parallel(
                        env_factory_path=env_factory_path,
                        env_id=cfg.env.env_id,
                        env_root=cfg.env.env_root,
                        mode="random_probe",
                        instruction=None,
                        round_id=round_id,
                        workers=effective_workers,
                        pool=worker_pool,
                    )
                    preanalyzed = None
                else:
                    episodes = collector.collect_round(session, "random_probe", None, round_id)
                    preanalyzed = None
                _log_stage_stop(cfg.game_id, round_id, "collection", t0, stage_timings)
                t0 = _log_stage_start(cfg.game_id, round_id, "episode_analysis")
                analyzed = analyze_episodes_parallel(
                    episodes,
                    cfg.analyst,
                    workers=effective_workers,
                    pool=worker_pool,
                    tuning_cfg=cfg.trajectory_analysis,
                    stats_out=analysis_stats,
                )
                _log_stage_stop(cfg.game_id, round_id, "episode_analysis", t0, stage_timings)
                t0 = _log_stage_start(cfg.game_id, round_id, "trajectory_analysis")
                blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=round_id, workers=effective_workers, pool=worker_pool)
                _log_stage_stop(cfg.game_id, round_id, "trajectory_analysis", t0, stage_timings)
                t0 = _log_stage_start(cfg.game_id, round_id, "artifact_write")
                collector.write_artifacts(cfg.game_id, round_id, analyzed)
                sqlite_store = _analysis_sqlite_store(cfg, cfg.game_id, round_id)
                if sqlite_store is not None:
                    sqlite_store.write_stage_stats(cfg.game_id, round_id, "episode_analysis", analysis_stats)
                if cfg.debug.export_avatar_candidates:
                    paths = storage.ensure_round_dirs(cfg.game_id, round_id)
                    with open(f"{paths['analyst_outputs']}/avatar_candidates.json", "w", encoding="utf-8") as handle:
                        handle.write(json.dumps(export_avatar_debug(analyzed), sort_keys=True))
                _log_stage_stop(cfg.game_id, round_id, "artifact_write", t0, stage_timings)
                controller_modes = ["random_probe"]
                target_progress = None
                instruction = None
                outcome = None
            else:
                t0 = _log_stage_start(cfg.game_id, round_id, "planning")
                prior_blackboard = load_blackboard_typed(storage, cfg.game_id)
                if prior_blackboard is None:
                    raise RuntimeError("Missing blackboard for directed round")
                skills, skill_executions = load_skill_library(storage, cfg.game_id)
                skills = induce_skills(prior_blackboard, existing=skills)
                skills = reconcile_skill_library(skills, skill_executions)
                plan_memory_state = reconcile_plan_memory(skills, skill_executions, load_plan_memory(storage, cfg.game_id))
                save_plan_memory(storage, cfg.game_id, plan_memory_state)
                directed_count = max(1, int(cfg.collection.directed_probe_episodes))
                _log_stage_stop(cfg.game_id, round_id, "planning", t0, stage_timings)
                t0 = _log_stage_start(cfg.game_id, round_id, "directed_execution")
                outcomes = []
                episodes = []
                round_consequence_records = []
                per_episode_passes: List[Dict[str, Any]] = []
                controller_modes = []
                target_progress = None
                instruction = None
                outcome = None
                belief = None
                plan_nodes = []
                controller_fallback_used = False
                force_regenerate_inventory = False
                for ep_idx in range(directed_count):
                    prior_blackboard = load_blackboard_typed(storage, cfg.game_id) or prior_blackboard
                    skills, skill_executions = load_skill_library(storage, cfg.game_id)
                    skills = induce_skills(prior_blackboard, existing=skills)
                    skills = reconcile_skill_library(skills, skill_executions)
                    plan_memory_state = reconcile_plan_memory(skills, skill_executions, load_plan_memory(storage, cfg.game_id))
                    plan_memory_state = replace(
                        plan_memory_state,
                        movement_planning_debug={},
                    )
                    save_plan_memory(storage, cfg.game_id, plan_memory_state)
                    memory_refs = build_plan_memory_refs(plan_memory_state)
                    planning_skills, recovery_reconstruction_debug = materialize_recovery_movement_skills(
                        skills,
                        [record.poi_id for record in prior_blackboard.reachability_table if record.status in {"reachable", "reachable_now", "uncertain"}],
                        plan_memory_state,
                    )
                    belief = build_planner_belief_state(
                        prior_blackboard,
                        planning_skills,
                        plan_memory_refs=memory_refs,
                        plan_memory=plan_memory_state,
                        force_full_inventory=force_regenerate_inventory,
                        recovery_reconstruction_debug=recovery_reconstruction_debug,
                    )
                    recovery_mode_active = bool(
                        force_regenerate_inventory
                        or plan_memory_state.failed_cluster_keys_this_round
                        or plan_memory_state.failed_movement_area_ids_this_round
                        or plan_memory_state.recent_failed_movement_cluster_keys
                        or plan_memory_state.excluded_cluster_cooldowns
                    )
                    if recovery_mode_active:
                        allowed_skill_ids = set(belief.candidate_skill_ids)
                        candidate_skill_rows = [skill for skill in planning_skills if skill.skill_id in allowed_skill_ids]
                    else:
                        candidate_skill_rows = candidate_skills(planning_skills, belief.candidate_subgoal_ids)
                    belief = build_planner_belief_state(
                        prior_blackboard,
                        planning_skills,
                        candidate_skills=candidate_skill_rows,
                        plan_memory_refs=memory_refs,
                        plan_memory=plan_memory_state,
                        force_full_inventory=force_regenerate_inventory,
                        recovery_reconstruction_debug=recovery_reconstruction_debug,
                    )
                    option_scores, option_record = rank_options(storage, cfg.game_id, belief, candidate_skill_rows)
                    candidate_mechanic_ids = [node.node_id for node in prior_blackboard.mechanic_graph.nodes] if prior_blackboard.mechanic_graph is not None else []
                    mechanic_scores, mechanic_record = rank_mechanics(storage, cfg.game_id, candidate_mechanic_ids)
                    plan_nodes, plan_result = plan_best_first(
                        belief,
                        candidate_skill_rows,
                        learned_score_map=option_scores if option_record is not None and option_scores else None,
                        blackboard=prior_blackboard,
                        plan_memory=plan_memory_state,
                        candidate_source_label="regenerated_inventory" if force_regenerate_inventory else "reused_shortlist",
                    )
                    selected_skill = next((skill for skill in planning_skills if plan_result is not None and skill.skill_id == plan_result.selected_skill_id), None)
                    if selected_skill is not None and plan_result is not None:
                        controller_fallback_used = False
                        instruction = instruction_from_skill(selected_skill, plan_result, prior_blackboard.game_id, prior_blackboard.round_id + 1)
                        instruction = _bind_instruction_to_blackboard(instruction, prior_blackboard)
                        if not _instruction_has_executable_target(instruction):
                            outcome, episode = _invalid_pre_execution_outcome(prior_blackboard.game_id, instruction)
                            skill_records = []
                            invalidated_targets = list(prior_blackboard.metadata.get("round_local_invalidated_targets", [])) if isinstance(prior_blackboard.metadata, dict) else []
                            invalidated_target = instruction.target_poi_id or _instruction_trigger_zone_id(instruction)
                            if invalidated_target and invalidated_target not in invalidated_targets:
                                invalidated_targets.append(invalidated_target)
                            prior_blackboard = replace(
                                prior_blackboard,
                                metadata={
                                    **prior_blackboard.metadata,
                                    "round_local_invalidated_targets": invalidated_targets,
                                },
                            )
                            _save_plan_memory(
                                storage,
                                cfg.game_id,
                                belief,
                                plan_nodes,
                                plan_result,
                                plan_memory_state,
                                failure_payload={"round_id": round_id, "reason": "invalid_target_pre_execution", "selected_skill_id": selected_skill.skill_id},
                                current_round_id=round_id,
                            )
                        else:
                            outcome, episode = run_offline_local_execution(
                                session,
                                instruction,
                                prior_blackboard,
                                cfg.executor,
                                episode_suffix=f"_ep{ep_idx:05d}",
                            )
                            skill_records = [skill_record_from_execution(selected_skill, instruction, outcome, episode, execution_suffix=f"ep{ep_idx:05d}")]
                            skill_executions = list(skill_executions) + skill_records
                        if selected_skill.skill_id not in {row.skill_id for row in skills}:
                            skills = list(skills) + [selected_skill]
                        skills = save_skill_library(storage, cfg.game_id, skills, skill_executions)
                        plan_memory_state = reconcile_plan_memory(skills, skill_executions, plan_memory_state)
                        plan_memory_state = replace(
                            plan_memory_state,
                            movement_planning_debug=_movement_planning_debug(belief, plan_nodes, plan_result),
                        )
                        plan_memory_state = _promote_movement_planning_debug(plan_memory_state)
                        save_plan_memory(storage, cfg.game_id, plan_memory_state)
                        finalized_option_record, finalized_option_scores = _finalize_option_ranking_record(option_record, option_scores, plan_nodes, plan_result)
                        _save_ranking_records(
                            storage,
                            cfg.game_id,
                            finalized_option_record,
                            finalized_option_scores,
                            mechanic_record,
                            mechanic_scores,
                            mechanic_signal_count=len(getattr(outcome, "consequence_records", []) or []),
                        )
                        _save_plan_memory(storage, cfg.game_id, belief, plan_nodes, plan_result, plan_memory_state, current_round_id=round_id)
                        prior_blackboard = replace(
                            prior_blackboard,
                            metadata={
                                **prior_blackboard.metadata,
                                "round_local_probe_outcomes": list(prior_blackboard.metadata.get("round_local_probe_outcomes", []))
                                + [
                                    {
                                        "round_id": round_id,
                                        "skill_id": selected_skill.skill_id,
                                        "trigger_zone_id": _instruction_trigger_zone_id(instruction),
                                        "outcome_summary": outcome.outcome_summary,
                                    }
                                ],
                            },
                        )
                        force_regenerate_inventory = not _measurable_progress(outcome)
                    else:
                        controller_fallback_used = True
                        surviving_skill_ids = _surviving_plan_skill_ids(plan_nodes)
                        fallback_pool = [skill for skill in candidate_skill_rows if skill.skill_id in surviving_skill_ids]
                        fallback_stage, fallback_skill = _choose_structured_fallback_skill(
                            fallback_pool,
                            plan_memory_state,
                            prior_blackboard,
                            round_id,
                        )
                        if fallback_skill is not None:
                            plan_result = None
                            selected_skill = fallback_skill
                            controller_fallback_used = False
                            synthetic_plan = PlanResultV1(
                                schema_version="v2.3.2",
                                plan_id="plan:fallback",
                                root_plan_node_id="plan_node:root",
                                selected_plan_node_id="",
                                selected_skill_id=fallback_skill.skill_id,
                                selected_subgoal_id=None,
                                planner_reason=fallback_stage or "planner_fallback",
                                alternative_plan_node_ids=[],
                            )
                            instruction = instruction_from_skill(fallback_skill, synthetic_plan, prior_blackboard.game_id, prior_blackboard.round_id + 1)
                        else:
                            instruction = select_instruction(prior_blackboard, cfg.controller, cfg.scoring, round_id)
                        instruction = _bind_instruction_to_blackboard(instruction, prior_blackboard)
                        if not _instruction_has_executable_target(instruction):
                            outcome, episode = _invalid_pre_execution_outcome(prior_blackboard.game_id, instruction)
                            invalidated_targets = list(prior_blackboard.metadata.get("round_local_invalidated_targets", [])) if isinstance(prior_blackboard.metadata, dict) else []
                            invalidated_target = instruction.target_poi_id or _instruction_trigger_zone_id(instruction)
                            if invalidated_target and invalidated_target not in invalidated_targets:
                                invalidated_targets.append(invalidated_target)
                            prior_blackboard = replace(
                                prior_blackboard,
                                metadata={
                                    **prior_blackboard.metadata,
                                    "round_local_invalidated_targets": invalidated_targets,
                                },
                            )
                        else:
                            outcome, episode = run_offline_local_execution(
                                session,
                                instruction,
                                prior_blackboard,
                                cfg.executor,
                                episode_suffix=f"_ep{ep_idx:05d}",
                            )
                        _save_plan_memory(
                            storage,
                            cfg.game_id,
                            belief,
                            plan_nodes,
                            plan_result,
                            failure_payload={"round_id": round_id, "reason": fallback_stage or ("recovery_no_fresh_movement_generated" if belief is not None and belief.reachable_frontier_ids else "planner_fallback"), "selected_skill_id": getattr(fallback_skill, "skill_id", None)}
                            if plan_result is None and controller_fallback_used
                            else None,
                            plan_memory_state=plan_memory_state,
                            current_round_id=round_id,
                        )
                        finalized_option_record, finalized_option_scores = _finalize_option_ranking_record(option_record, option_scores, plan_nodes, plan_result)
                        _save_ranking_records(
                            storage,
                            cfg.game_id,
                            finalized_option_record,
                            finalized_option_scores,
                            mechanic_record,
                            mechanic_scores,
                            mechanic_signal_count=len(getattr(outcome, "consequence_records", []) or []),
                        )
                        skills = save_skill_library(storage, cfg.game_id, skills, skill_executions)
                        plan_memory_state = replace(
                            plan_memory_state,
                            movement_planning_debug=_movement_planning_debug(belief, plan_nodes, plan_result),
                        )
                        plan_memory_state = _promote_movement_planning_debug(plan_memory_state)
                        save_plan_memory(storage, cfg.game_id, plan_memory_state)
                        force_regenerate_inventory = bool(outcome is not None and outcome.outcome_summary in {"route_stall", "contact_no_effect", "contact_no_reach", "blocked", "invalid_target", "unreachable_now"})
                    outcomes.append(outcome)
                    round_consequence_records.extend(list(outcome.consequence_records))
                    episodes.append(episode)
                    controller_modes.append(instruction.mode)
                    target_progress = outcome.target_progress
                    per_episode_passes.append(
                        {
                            "episode_idx": ep_idx,
                            "selected_skill_id": plan_result.selected_skill_id if plan_result is not None else None,
                            "selected_plan_node_id": plan_result.selected_plan_node_id if plan_result is not None else None,
                            "instruction_id": instruction.instruction_id if instruction is not None else None,
                            "outcome_summary": outcome.outcome_summary if outcome is not None else None,
                            "execution_outcome": (
                                {
                                    "outcome_summary": outcome.outcome_summary,
                                    "blocked": outcome.blocked,
                                    "contact": outcome.contact,
                                    "reached": outcome.reached,
                                    "target_progress": list(outcome.target_progress),
                                    "diagnostics": list(getattr(outcome, "diagnostics", []) or []),
                                }
                                if outcome is not None else None
                            ),
                        }
                    )
                    _save_plan_pass_debug(storage, cfg.game_id, round_id, ep_idx, belief, plan_nodes, plan_result, outcome)
                    if outcome.outcome_summary in {"invalid_target", "unreachable_now"}:
                        break
                    if outcome.outcome_summary in {"route_stall", "contact_no_effect", "contact_no_reach", "blocked"}:
                        force_regenerate_inventory = True
                _log_stage_stop(cfg.game_id, round_id, "directed_execution", t0, stage_timings)
                t0 = _log_stage_start(cfg.game_id, round_id, "episode_analysis")
                analysis_stats: Dict[str, object] = {}
                analyzed = analyze_episodes_parallel(
                    episodes,
                    cfg.analyst,
                    workers=effective_workers,
                    pool=worker_pool,
                    tuning_cfg=cfg.trajectory_analysis,
                    stats_out=analysis_stats,
                )
                _log_stage_stop(cfg.game_id, round_id, "episode_analysis", t0, stage_timings)
                t0 = _log_stage_start(cfg.game_id, round_id, "trajectory_analysis")
                blackboard = analyze_trajectories(
                    analyzed,
                    cfg.trajectory_analysis,
                    round_id=round_id,
                    prior_blackboard=prior_blackboard,
                    workers=effective_workers,
                    pool=worker_pool,
                )
                _log_stage_stop(cfg.game_id, round_id, "trajectory_analysis", t0, stage_timings)
                t0 = _log_stage_start(cfg.game_id, round_id, "artifact_write")
                blackboard = replace(
                    blackboard,
                    consequence_table=list(blackboard.consequence_table) + [record for row_outcome in outcomes for record in row_outcome.consequence_records],
                )
                collector.write_artifacts(cfg.game_id, round_id, analyzed)
                sqlite_store = _analysis_sqlite_store(cfg, cfg.game_id, round_id)
                if sqlite_store is not None:
                    sqlite_store.write_stage_stats(cfg.game_id, round_id, "episode_analysis", analysis_stats)
                if cfg.debug.export_avatar_candidates:
                    paths = storage.ensure_round_dirs(cfg.game_id, round_id)
                    with open(f"{paths['analyst_outputs']}/avatar_candidates.json", "w", encoding="utf-8") as handle:
                        handle.write(json.dumps(export_avatar_debug(analyzed), sort_keys=True))
                _log_stage_stop(cfg.game_id, round_id, "artifact_write", t0, stage_timings)
                if per_episode_passes:
                    _export_debug_table(storage, cfg.game_id, round_id, "directed_episode_plan_passes.json", per_episode_passes)
            total_wins += sum(1 for episode in episodes if episode.win)
            episodes_seen += len(episodes)
            round_wins = sum(1 for episode in episodes if episode.win)
            round_max_steps = max((len(episode.steps) for episode in episodes), default=0)
            for episode in episodes:
                max_steps_per_game = max(max_steps_per_game, len(episode.steps))
            reach_lookup = {r.poi_id: r.status for r in blackboard.reachability_table}
            episode_ids = {ep.episode_id for ep in analyzed}
            round_consequences = [
                record
                for record in blackboard.consequence_table
                if record.episode_id in episode_ids or (instruction is not None and record.instruction_id == instruction.instruction_id)
            ]
            if round_id > 0:
                round_consequences = list(round_consequence_records)
            metrics = compute_round_metrics(
                analyzed,
                blackboard.poi_table,
                reach_lookup,
                round_consequences,
                controller_modes,
                len(blackboard.avatar_hypotheses),
                target_progress=target_progress,
                blackboard=blackboard,
                executor_outcomes=outcomes if round_id > 0 else None,
            )
            if round_id > 0:
                metrics = _consistent_round_metrics(metrics, outcomes, controller_modes)
            diagnostics = []
            # Historical diagnostics based on removed metrics are intentionally commented out:
            # if metrics.states_observed > 0 and metrics.unique_states == 0:
            #     diagnostics.append("state_hash_missing")
            # if metrics.candidate_avatar_count == 0:
            #     diagnostics.append("no_avatar_candidates")
            # if metrics.candidate_poi_count > 0 and metrics.reachable_poi_count == 0:
            #     diagnostics.append("no_reachable_pois")
            # targeted_run = instruction.mode not in {"random_probe", "unguided_probe"} if round_id > 0 else False
            # if targeted_run and metrics.target_progress_mean == 0:
            #     diagnostics.append("no_target_progress")
            # if targeted_run and metrics.route_success_rate == 0:
            #     diagnostics.append("no_route_success")
            if round_id > 0 and instruction.mode not in {"random_probe", "unguided_probe"} and metrics.route_success_rate == 0:
                diagnostics.append("no_route_success")
            instruction_history = list(blackboard.metadata.get("instruction_history", [])) if isinstance(blackboard.metadata, dict) else []
            if round_id > 0:
                instruction_history.append(
                    {
                        "round_id": round_id,
                        "mode": instruction.mode,
                        "instruction_id": instruction.instruction_id,
                        "target_poi_id": instruction.target_poi_id,
                        "outcome": "progress" if any(c.distance_decreased or c.reached or c.contact for c in outcome.consequence_records) else "no_progress",
                    }
                )
            outcome_history = list(blackboard.metadata.get("executor_outcome_history", [])) if isinstance(blackboard.metadata, dict) else []
            if round_id > 0:
                outcome_history.append(outcome.to_dict())
            linked_events = [event for event in blackboard.event_table if event.episode_id in episode_ids]
            linked_event_ids = [event.event_id for event in linked_events]
            linked_interventions = [record for record in blackboard.intervention_table if record.round_id == round_id]
            linked_intervention_ids = [record.instruction_id for record in linked_interventions]
            linked_causal_links = [
                link for link in blackboard.cause_effect_table if link.intervention_id in linked_intervention_ids or link.effect_event_id in linked_event_ids
            ]
            promoted_mechanics = [
                mech for mech in blackboard.mechanic_hypotheses if mech.status == "promoted" and any(event_id in linked_event_ids for event_id in mech.support_event_ids)
            ]
            pre_execution_invalidation = bool(
                outcome is not None and outcome.outcome_summary in {"invalid_target", "invalid_target_pre_execution"} and not outcome.actions
            )
            selected_trigger_zone_id = _instruction_trigger_zone_id(instruction) if instruction is not None else None
            decision_record = DecisionRecordV2(
                schema_version=SCHEMA_VERSION,
                game_id=cfg.game_id,
                round_id=round_id,
                instruction_id=instruction.instruction_id if instruction is not None else f"round{round_id:03d}:random_probe",
                selected_target_poi_id=instruction.target_poi_id if instruction is not None else None,
                selected_area_id=next((poi.area_id for poi in blackboard.poi_table if poi.poi_id == instruction.target_poi_id), None) if instruction is not None else None,
                mode=controller_modes[0] if controller_modes else "unknown",
                rationale_codes=[code for code in ((instruction.rationale.split() if instruction is not None and instruction.rationale else [])) if code][:6],
                ranked_candidate_ids=(
                    ([plan_result.selected_plan_node_id] + [node_id for node_id in plan_result.alternative_plan_node_ids if node_id != plan_result.selected_plan_node_id])[:5]
                    if plan_result is not None and plan_result.selected_plan_node_id
                    else list(instruction.ranked_alternatives) if instruction is not None else []
                ),
                outcome_summary="invalid_target_pre_execution" if pre_execution_invalidation else (outcome.outcome_summary if outcome is not None else "initial_probe"),
                progress_score=(max(target_progress) if target_progress else 0.0) if target_progress is not None else None,
                target_invalidated=_decision_target_invalidated(instruction, outcome),
                selected_skill_id=(
                    plan_result.selected_skill_id
                    if plan_result is not None and plan_result.selected_skill_id
                    else (selected_skill.skill_id if round_id > 0 and 'selected_skill' in locals() and selected_skill is not None else None)
                ),
                selected_plan_node_id=(
                    plan_result.selected_plan_node_id
                    if plan_result is not None and plan_result.selected_plan_node_id
                    else None
                ),
                selected_trigger_zone_id=selected_trigger_zone_id,
                selected_chain_id=_extract_tag(instruction.rationale if instruction is not None else None, "chain_id"),
                selected_hidden_hypothesis_id=_extract_tag(instruction.rationale if instruction is not None else None, "hidden_hypothesis_id"),
            )
            blackboard = replace(
                blackboard,
                decision_history=list(blackboard.decision_history) + [decision_record],
                metadata={
                    **blackboard.metadata,
                    "diagnostics": diagnostics,
                    "metrics": metrics.to_dict(),
                    "instruction_history": instruction_history[-20:],
                    "executor_outcome_history": outcome_history[-10:],
                    "round_mode": controller_modes[0] if controller_modes else "unknown",
                    "learned_option_ranking_applied": bool(round_id > 0 and option_record is not None and option_scores),
                    "learned_mechanic_ranking_applied": bool(round_id > 0 and mechanic_record is not None and mechanic_scores),
                },
            )
            save_blackboard(cfg.memory, storage, blackboard)
            save_graph_state(storage, blackboard)
            if cfg.memory.persist_world_model:
                save_world_model_archive(storage, blackboard)
            if cfg.memory.persist_events:
                save_event_table(storage, blackboard)
            if cfg.memory.persist_navigation_graph:
                save_navigation_graph(storage, blackboard)
            if cfg.memory.persist_interventions:
                save_interventions(storage, blackboard)
            if cfg.memory.persist_world_model:
                save_mechanic_hypotheses(storage, blackboard)
            if cfg.debug.export_trigger_zones:
                _export_debug_table(storage, cfg.game_id, round_id, "trigger_zones.json", [row.to_dict() for row in blackboard.trigger_zone_table])
            if cfg.debug.export_event_graph:
                _export_debug_table(storage, cfg.game_id, round_id, "event_graph.json", [row.to_dict() for row in blackboard.event_edge_table])
            if cfg.debug.export_event_sequence_patterns:
                _export_debug_table(storage, cfg.game_id, round_id, "event_sequence_patterns.json", [row.to_dict() for row in blackboard.event_sequence_patterns])
            if cfg.debug.export_hidden_trigger_hypotheses:
                _export_debug_table(storage, cfg.game_id, round_id, "hidden_trigger_hypotheses.json", [row.to_dict() for row in blackboard.hidden_trigger_hypotheses])
            if cfg.debug.export_causal_chain_hypotheses:
                _export_debug_table(storage, cfg.game_id, round_id, "causal_chain_hypotheses.json", [row.to_dict() for row in blackboard.causal_chain_hypotheses])
            if cfg.debug.export_counterfactual_traces:
                _export_debug_table(storage, cfg.game_id, round_id, "counterfactual_traces.json", [row.to_dict() for row in blackboard.counterfactual_traces])
            report = {
                "round_id": round_id,
                "game_id": cfg.game_id,
                "poi_count": len(blackboard.poi_table),
                "metrics": metrics.to_dict(),
                "stage_timings": {
                    **{k: round(v, 6) for k, v in stage_timings.items()},
                    "round_total": round(time.perf_counter() - round_t0, 6),
                    "analysis_substages": (blackboard.metadata.get("analysis_stage_timings", {}) if isinstance(blackboard.metadata, dict) else {}),
                },
                "diagnostics": diagnostics,
                "invalid_target_link_count": sum(1 for c in blackboard.consequence_table if c.consequence_class == "invalid_target_link"),
                "wins": round_wins,
                "max_steps_per_game": round_max_steps,
                "decision": decision_record.to_dict(),
                "execution_outcome_scope": "last_directed_episode" if round_id > 0 else "round_probe_collection",
                "execution_outcome": (
                    {
                        **outcome.to_dict(),
                        "outcome_summary": "invalid_target_pre_execution",
                        "blocked": False,
                    }
                    if pre_execution_invalidation
                    else (outcome.to_dict() if outcome is not None else None)
                ),
                "round_execution_aggregate": _round_execution_aggregate(outcomes if round_id > 0 else []),
                "linked_event_ids": linked_event_ids,
                "linked_causal_link_ids": [link.link_id for link in linked_causal_links],
                "promoted_mechanic_hypothesis_ids": [mech.hypothesis_id for mech in promoted_mechanics],
            }
            append_round_report(storage, cfg.game_id, round_id, report)
            log_event(cfg.logging.log_dir, "v2_round_complete", report)
            print(
                f"[v2] round_complete game_id={cfg.game_id} round_id={round_id} poi_count={len(blackboard.poi_table)} wins={round_wins} max_steps_per_game={round_max_steps} total_s={round(time.perf_counter() - round_t0, 3)}",
                flush=True,
            )
            last_report = report

        if last_report is not None:
            summary = {
                **last_report,
                "wins": total_wins,
                "max_steps_per_game": max_steps_per_game,
                "episodes_seen": episodes_seen,
            }
            session_mgr.export_summary(cfg.game_id, summary)
            return summary
    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()
        if shared_env is not None and hasattr(shared_env, "close"):
            try:
                shared_env.close()
            except Exception:
                pass
    return {
        "game_id": cfg.game_id,
        "wins": total_wins,
        "max_steps_per_game": max_steps_per_game,
        "episodes_seen": episodes_seen,
    }
