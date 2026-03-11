from __future__ import annotations

import importlib
import multiprocessing as mp
from multiprocessing.pool import Pool
from typing import List, Optional, Tuple

from codex_baseline_v2.executor.route_planner import build_route_context, plan_route
from codex_baseline_v2.runtime.environment_session import EnvironmentSessionV2
from codex_baseline_v2.runtime.trajectory_policy import PolicyStateV2, TrajectoryPolicyV2
from codex_baseline_v2.shared.config import ExecutorConfigV2
from codex_baseline_v2.shared.metrics import normalize_consequence_class
from codex_baseline_v2.shared.state_identity import canonical_state_identity
from codex_baseline_v2.shared.schemas import (
    BlackboardStateV2,
    ConsequenceRecordV2,
    ControllerInstructionV2,
    ExecutorOutcomeV2,
    ObjectRecordV2,
    SCHEMA_VERSION,
    TrajectoryEpisodeV2,
    TrajectoryStepV2,
)
from codex_baseline_v2.shared.utils import BBox, bbox_from_points, connected_components_per_color, grid_diff


def _distance_to_bbox(point: Tuple[int, int], bbox: BBox) -> float:
    x, y = point
    if bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2:
        return 0.0
    dx = max(bbox.x1 - x, 0, x - bbox.x2)
    dy = max(bbox.y1 - y, 0, y - bbox.y2)
    return float(dx + dy)


def _touches(point_bbox: BBox, target_bbox: BBox) -> bool:
    return not (
        point_bbox.x2 < target_bbox.x1 - 1
        or point_bbox.x1 > target_bbox.x2 + 1
        or point_bbox.y2 < target_bbox.y1 - 1
        or point_bbox.y1 > target_bbox.y2 + 1
    )


def _inside_bbox(point: Optional[Tuple[int, int]], target_bbox: Optional[BBox]) -> bool:
    if point is None or target_bbox is None:
        return False
    return target_bbox.x1 <= point[0] <= target_bbox.x2 and target_bbox.y1 <= point[1] <= target_bbox.y2


def _extract_tag(rationale: str, key: str) -> Optional[str]:
    prefix = key + "="
    for token in rationale.split():
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _abort_execution(
    session: EnvironmentSessionV2,
    instruction: ControllerInstructionV2,
    episode_id: str,
    reason: str,
) -> tuple[ExecutorOutcomeV2, TrajectoryEpisodeV2]:
    outcome = ExecutorOutcomeV2(
        schema_version=SCHEMA_VERSION,
        game_id=session.game_id,
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
        blocked=True,
        outcome_summary=reason,
        consequence_records=[],
    )
    episode = TrajectoryEpisodeV2(
        schema_version=SCHEMA_VERSION,
        game_id=session.game_id,
        episode_id=episode_id,
        steps=[],
        done=False,
        win=False,
        seed=None,
        metadata={"mode": instruction.mode, "instruction": instruction.to_dict(), "diagnostic": reason},
    )
    return outcome, episode


def _target_recently_invalidated(blackboard: BlackboardStateV2, instruction: ControllerInstructionV2) -> bool:
    target_ref = instruction.target_poi_id or _extract_tag(instruction.rationale or "", "trigger_zone_id")
    if not target_ref:
        return False
    for record in reversed(list(blackboard.decision_history)[-10:]):
        if record.target_invalidated and (record.selected_target_poi_id == target_ref or record.selected_trigger_zone_id == target_ref):
            return True
    return False


def _route_unreachable(plan) -> bool:
    return plan is not None and plan.blocked and plan.next_subgoal is None


def _has_repeated_action_pattern(actions: List[object]) -> bool:
    action_ids = [getattr(action, "action_id", None) for action in actions if getattr(action, "action_id", None) is not None]
    if len(action_ids) < 4:
        return False
    if len(action_ids) >= 4 and len(set(action_ids[-4:])) == 1:
        return True
    if len(action_ids) >= 6 and action_ids[-6:-3] == action_ids[-3:]:
        return True
    if len(action_ids) >= 4 and action_ids[-4:-2] == action_ids[-2:]:
        return True
    return False


def _has_repeated_state_cycle(state_hashes: List[Optional[str]]) -> bool:
    hashes = [value for value in state_hashes if value]
    if len(hashes) < 4:
        return False
    if len(hashes) >= 4 and hashes[-4:-2] == hashes[-2:]:
        return True
    if len(hashes) >= 6 and hashes[-6:-3] == hashes[-3:]:
        return True
    if len(hashes) >= 4 and len(set(hashes[-4:])) == 1:
        return True
    return False


def _infer_live_avatar(observation: Optional[List[List[int]]], blackboard, last_position: Optional[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], Optional[BBox]]:
    if observation is None:
        return last_position, None
    comps = connected_components_per_color(observation)
    candidates: List[Tuple[float, Tuple[int, int], BBox]] = []
    for hypothesis in blackboard.avatar_hypotheses:
        color = hypothesis.color
        for comp in comps.get(color, []):
            bbox = bbox_from_points(comp)
            if bbox is None:
                continue
            centroid = (int(round(bbox.centroid()[0])), int(round(bbox.centroid()[1])))
            if last_position is not None:
                dist = abs(centroid[0] - last_position[0]) + abs(centroid[1] - last_position[1])
            else:
                dist = abs(centroid[0] - int(round(hypothesis.centroid[0]))) + abs(centroid[1] - int(round(hypothesis.centroid[1])))
            candidates.append((float(dist), centroid, bbox))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        _, centroid, bbox = candidates[0]
        return centroid, bbox
    if last_position is not None:
        return last_position, BBox(last_position[0], last_position[1], last_position[0], last_position[1])
    return None, None


def run_online_execution(
    session: EnvironmentSessionV2,
    instruction: ControllerInstructionV2,
    blackboard,
    cfg: ExecutorConfigV2,
    episode_suffix: str = "",
) -> tuple[ExecutorOutcomeV2, TrajectoryEpisodeV2]:
    episode_id = f"exec_round{instruction.round_id:03d}{episode_suffix}"
    policy = TrajectoryPolicyV2()
    state = PolicyStateV2(
        round_id=instruction.round_id,
        action_semantics_table=list(blackboard.action_semantics_table),
        action_context_table=list(blackboard.action_context_table),
    )
    obs = session.reset()
    steps: List[TrajectoryStepV2] = []
    actions = []
    progress: List[float] = []
    reached = False
    contact = False
    blocked = False
    target_poi = None
    stall_count = 0
    missed_avatar_steps = 0
    target_bbox = instruction.target_geometry
    if _target_recently_invalidated(blackboard, instruction):
        return _abort_execution(session, instruction, episode_id, "invalid_target")
    trigger_zone_id = _extract_tag(instruction.rationale or "", "trigger_zone_id")
    if instruction.mode not in {"random_probe", "unguided_probe"} and not instruction.target_poi_id and not trigger_zone_id and target_bbox is None:
        return _abort_execution(session, instruction, episode_id, "invalid_target")
    if instruction.target_poi_id:
        for poi in blackboard.poi_table:
            if poi.poi_id == instruction.target_poi_id:
                target_poi = poi
                break
    route_context = build_route_context(blackboard, target_poi) if target_poi is not None else None
    if instruction.target_poi_id and target_poi is None:
        return _abort_execution(session, instruction, episode_id, "invalid_target")
    if instruction.mode == "poi_approach" and (target_bbox is None or target_poi is None or route_context is None):
        return _abort_execution(session, instruction, episode_id, "invalid_target")

    prev_distance: Optional[float] = None
    best_distance: Optional[float] = None
    last_live_avatar: Optional[Tuple[int, int]] = None
    last_live_bbox: Optional[BBox] = None
    prev_obs = obs
    state_hash_history: List[Optional[str]] = []
    no_progress_steps = 0
    no_route_improvement_steps = 0
    loop_stop_reason: Optional[str] = None
    for step_idx in range(cfg.max_steps):
        avail = session.available_actions()
        live_avatar, live_bbox = _infer_live_avatar(obs, blackboard, last_live_avatar)
        if live_avatar is not None:
            last_live_avatar = live_avatar
            last_live_bbox = live_bbox
            missed_avatar_steps = 0
        else:
            missed_avatar_steps += 1
            if missed_avatar_steps > 2:
                last_live_avatar = None
                last_live_bbox = None
        state.current_position = last_live_avatar
        state.step_idx = step_idx
        state.action_context_key = f"{instruction.mode}:{instruction.target_poi_id or 'none'}"

        if target_poi is not None and last_live_avatar is not None:
            plan = plan_route(
                blackboard,
                target_poi,
                (int(last_live_avatar[0]), int(last_live_avatar[1])),
                prev_distance=prev_distance,
                route_context=route_context,
            )
            if _route_unreachable(plan):
                blocked = False
                loop_stop_reason = "unreachable_now"
                break
            if plan.progress_valid and plan.next_subgoal is not None:
                state.target_centroid = plan.next_subgoal
                action = policy.instructed_action(instruction, plan.next_subgoal, avail, state)
            else:
                action = policy.fallback_action(avail, state)
            current_distance = plan.distance_prev
        elif target_bbox is not None and last_live_avatar is not None:
            goal = (int(round(target_bbox.centroid()[0])), int(round(target_bbox.centroid()[1])))
            plan = None
            current_distance = _distance_to_bbox(last_live_avatar, target_bbox)
            if instruction.mode == "counterfactual_avoid_contact" and current_distance is not None and current_distance <= max(1.0, cfg.target_reach_distance + 1.0):
                action = policy.fallback_action(avail, state)
            elif instruction.mode == "action_in_region" and _inside_bbox(last_live_avatar, target_bbox):
                action = policy.fallback_action(avail, state)
            else:
                action = policy.instructed_action(instruction, goal, avail, state)
        else:
            plan = None
            action = policy.unguided_probe(avail)
            current_distance = _distance_to_bbox(last_live_avatar, target_bbox) if last_live_avatar is not None and target_bbox is not None else None

        result = session.step(action)
        new_avatar, new_bbox = _infer_live_avatar(result.observation, blackboard, last_live_avatar)
        if new_avatar is not None:
            policy.observe_transition(state, last_live_avatar, new_avatar, action)
            last_live_avatar = new_avatar
            last_live_bbox = new_bbox
            missed_avatar_steps = 0
        else:
            missed_avatar_steps += 1
            if missed_avatar_steps > 2:
                last_live_avatar = None
                last_live_bbox = None
        new_distance = _distance_to_bbox(last_live_avatar, target_bbox) if last_live_avatar is not None and target_bbox is not None else None
        if new_distance is not None:
            progress.append(new_distance)
            prev_distance = new_distance
            if best_distance is None or new_distance < best_distance - 1e-6:
                best_distance = new_distance
                no_progress_steps = 0
            else:
                no_progress_steps += 1
            if new_distance <= cfg.target_reach_distance:
                reached = True
        else:
            no_progress_steps += 1
        if instruction.mode in {"step_on_region", "dwell_on_region", "action_in_region", "cross_boundary_edge"} and _inside_bbox(last_live_avatar, target_bbox):
            contact = True
        elif last_live_bbox is not None and target_bbox is not None and _touches(last_live_bbox, target_bbox):
            contact = True
        actions.append(action)
        pre_state = canonical_state_identity(obs, include_payload=False)
        post_state = canonical_state_identity(result.observation, include_payload=False)
        state_hash_history.append(post_state.get("state_hash"))
        local_change = 0.0
        global_change = 0.0
        if prev_obs is not None and result.observation is not None:
            changed, points = grid_diff(prev_obs, result.observation)
            global_change = float(changed)
            if target_bbox is not None:
                local_change = float(sum(1 for (x, y) in points if target_bbox.x1 <= x <= target_bbox.x2 and target_bbox.y1 <= y <= target_bbox.y2))
        steps.append(
            TrajectoryStepV2(
                schema_version=SCHEMA_VERSION,
                game_id=session.game_id,
                episode_id=episode_id,
                step_idx=step_idx,
                action=action,
                pre_state_hash=pre_state.get("state_hash"),
                post_state_hash=post_state.get("state_hash"),
                state_hash_valid=bool(pre_state.get("valid") and post_state.get("valid")),
                instruction_id=instruction.instruction_id,
                target_poi_id=instruction.target_poi_id,
                target_type=instruction.target_type,
                target_geometry=instruction.target_geometry,
                target_source_round=instruction.target_source_round,
                reward=result.reward,
                done=result.done,
                observation=obs,
                area_id=target_poi.area_id if target_poi is not None else None,
                avatar_track_id=blackboard.avatar_track_table[0].track_id if blackboard.avatar_track_table else None,
                predicted_avatar_centroid=tuple(plan.next_subgoal) if plan is not None and plan.next_subgoal is not None else None,
                actual_avatar_centroid=tuple(last_live_avatar) if last_live_avatar is not None else None,
                event_ids=[],
                intervention_id=f"intervention:{instruction.instruction_id}",
                action_context_key=state.action_context_key,
                observation_summary=None,
                info={
                    "available_actions": result.available_actions,
                    "step_info": result.info,
                    "progress": {
                        "distance_prev": current_distance,
                        "distance_new": new_distance,
                        "distance_delta": None if current_distance is None or new_distance is None else current_distance - new_distance,
                        "progress_valid": bool(plan.progress_valid) if plan is not None else False,
                        "progress_reason": plan.progress_reason if plan is not None else "untargeted",
                        "fallback_mode": plan.fallback_mode if plan is not None else None,
                        "contact": contact,
                        "live_avatar_position": list(last_live_avatar) if last_live_avatar is not None else None,
                        "local_change_magnitude": local_change,
                        "global_change_magnitude": global_change,
                    },
                    "collection_mode": instruction.mode,
                    "route_edge_ids": list(plan.route_edge_ids) if plan is not None else [],
                    "probe_mode": instruction.mode,
                    "target_trigger_zone_id": _extract_tag(instruction.rationale, "trigger_zone_id"),
                    "state_signature_version": pre_state.get("state_signature_version"),
                },
            )
        )
        prev_obs = result.observation
        obs = result.observation
        if result.done:
            break
        route_improved = bool(plan is not None and plan.progress_valid and plan.distance_delta is not None and plan.distance_delta > 0.0)
        if route_improved:
            stall_count = 0
            no_route_improvement_steps = 0
        else:
            stall_count += 1
            if not contact:
                no_route_improvement_steps += 1
        if _has_repeated_action_pattern(actions):
            blocked = False
            loop_stop_reason = "route_stall"
            break
        if _has_repeated_state_cycle(state_hash_history):
            blocked = False
            loop_stop_reason = "route_stall"
            break
        if no_progress_steps >= max(2, cfg.blocked_repeat_limit):
            blocked = False
            loop_stop_reason = "no_progress"
            break
        if no_route_improvement_steps >= max(2, cfg.blocked_repeat_limit):
            blocked = False
            loop_stop_reason = "route_stall"
            break
        if stall_count >= cfg.blocked_repeat_limit:
            blocked = True
            loop_stop_reason = "blocked"
            break

    consequences: List[ConsequenceRecordV2] = []
    consequence_target_id = instruction.target_poi_id or trigger_zone_id
    if consequence_target_id and target_bbox is not None and (steps or contact or reached):
        initial_distance = progress[0] if progress else None
        final_distance = progress[-1] if progress else None
        total_local_change = float(sum(step.info.get("progress", {}).get("local_change_magnitude", 0.0) for step in steps))
        total_global_change = float(sum(step.info.get("progress", {}).get("global_change_magnitude", 0.0) for step in steps))
        meaningful_change_threshold = max(1.0, float(cfg.target_reach_distance))
        if steps and steps[-1].done:
            consequence_class = "terminal_like"
        elif reached:
            consequence_class = "progress_like"
        elif contact:
            if total_local_change > 0.0:
                consequence_class = "local_change"
            elif total_global_change >= meaningful_change_threshold:
                consequence_class = "global_change"
            else:
                consequence_class = "no_change"
        elif total_local_change >= meaningful_change_threshold:
            consequence_class = "local_change"
        elif total_global_change >= meaningful_change_threshold:
            consequence_class = "global_change"
        else:
            consequence_class = "no_change"
        consequences.append(
            ConsequenceRecordV2(
                schema_version=SCHEMA_VERSION,
                game_id=session.game_id,
                poi_id=consequence_target_id,
                round_id=instruction.round_id,
                episode_id=episode_id,
                instruction_id=instruction.instruction_id,
                target_poi_id=instruction.target_poi_id,
                distance_decreased=bool(initial_distance is not None and final_distance is not None and final_distance < initial_distance),
                reached=reached,
                contact=contact,
                local_change_magnitude=total_local_change,
                global_change_magnitude=total_global_change,
                reward_delta=float(sum(step.reward for step in steps)) if steps else None,
                terminal_flag_changed=bool(steps and steps[-1].done),
                object_change_summary="trigger_zone_contact" if trigger_zone_id and contact else "online_execution",
                followup_poi_ids=[],
                consequence_class=normalize_consequence_class(consequence_class),
            )
        )
    outcome_summary = loop_stop_reason or "offline_local_execution"
    no_effect_contact = bool(
        contact
        and not reached
        and not any(record.event_ids or record.cause_effect_link_ids or record.topology_delta_id for record in consequences)
        and all(record.local_change_magnitude <= 0.0 for record in consequences)
    )
    if no_effect_contact and trigger_zone_id and outcome_summary in {"blocked", "route_stall", "no_progress", "offline_local_execution"}:
        blocked = False
        outcome_summary = "contact_no_effect"
    elif contact and not reached and outcome_summary in {"blocked", "route_stall", "no_progress"}:
        blocked = False
    flat_progress = bool(progress) and max(progress) == min(progress)
    no_route_success = not any(record.distance_decreased or record.reached for record in consequences)
    if contact and not reached and flat_progress and no_route_success:
        outcome_summary = "contact_reached_boundary_without_route_progress"
    elif contact and not reached and outcome_summary in {"blocked", "route_stall", "no_progress"}:
        outcome_summary = "contact_no_reach"
    negative_planning_feedback = bool(outcome_summary == "contact_no_effect")
    if outcome_summary in {"route_stall", "contact_no_reach", "contact_reached_boundary_without_route_progress"} and not reached and flat_progress:
        negative_planning_feedback = True
    outcome = ExecutorOutcomeV2(
        schema_version=SCHEMA_VERSION,
        game_id=session.game_id,
        round_id=instruction.round_id,
        instruction_id=instruction.instruction_id,
        instruction_mode=instruction.mode,
        target_poi_id=instruction.target_poi_id,
        target_type=instruction.target_type,
        target_geometry=instruction.target_geometry,
        target_source_round=instruction.target_source_round,
        actions=actions,
        target_progress=progress,
        reached=reached,
        contact=contact,
        blocked=blocked,
        outcome_summary=outcome_summary,
        consequence_records=consequences,
        negative_planning_feedback=negative_planning_feedback,
    )
    status = session.progress_status()
    episode = TrajectoryEpisodeV2(
        schema_version=SCHEMA_VERSION,
        game_id=session.game_id,
        episode_id=episode_id,
        steps=steps,
        done=bool(steps and steps[-1].done),
        win=bool(status.get("win", False)),
        seed=None,
        metadata={
            "mode": instruction.mode,
            "instruction": instruction.to_dict(),
            "collection_mode": instruction.mode,
            "intervention_id": f"intervention:{instruction.instruction_id}",
            "probe_mode": instruction.mode,
            "intent_class": "hidden_trigger_probe" if instruction.target_type == "trigger_zone" else "causal_chain_verification" if instruction.target_type == "causal_chain" else "counterfactual_disambiguation_probe" if instruction.target_type == "counterfactual" else "poi_interaction_probe",
            "target_trigger_zone_id": _extract_tag(instruction.rationale, "trigger_zone_id"),
            "win": bool(status.get("win", False)),
            "execution_mode": "offline_local",
        },
    )
    return outcome, episode


def run_offline_local_execution(
    session: EnvironmentSessionV2,
    instruction: ControllerInstructionV2,
    blackboard,
    cfg: ExecutorConfigV2,
    episode_suffix: str = "",
) -> tuple[ExecutorOutcomeV2, TrajectoryEpisodeV2]:
    return run_online_execution(session, instruction, blackboard, cfg, episode_suffix=episode_suffix)


def run_online_executions_parallel(
    env_factory_path: str,
    env_id: str,
    env_root: str,
    instruction: ControllerInstructionV2,
    blackboard: BlackboardStateV2,
    cfg: ExecutorConfigV2,
    episodes: int,
    workers: int,
    pool: Optional[Pool] = None,
) -> List[tuple[ExecutorOutcomeV2, TrajectoryEpisodeV2]]:
    payloads = [
        (
            env_factory_path,
            env_id,
            env_root,
            instruction.to_dict(),
            blackboard.to_dict(),
            cfg,
            ep_idx,
        )
        for ep_idx in range(max(0, episodes))
    ]
    if pool is not None:
        return pool.map(_run_online_execution_worker, payloads)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as local_pool:
        return local_pool.map(_run_online_execution_worker, payloads)


def _run_online_execution_worker(payload) -> tuple[ExecutorOutcomeV2, TrajectoryEpisodeV2]:
    env_factory_path, env_id, env_root, instruction_payload, blackboard_payload, cfg, ep_idx = payload
    module_name, func_name = env_factory_path.rsplit(":", 1)
    mod = importlib.import_module(module_name)
    env_factory = getattr(mod, func_name)
    try:
        env = env_factory(env_id=env_id, env_root=env_root)
    except TypeError:
        env = env_factory()
    session = EnvironmentSessionV2(env, env_id)
    instruction = ControllerInstructionV2.from_dict(instruction_payload)
    blackboard = BlackboardStateV2.from_dict(blackboard_payload)
    try:
        return run_online_execution(session, instruction, blackboard, cfg, episode_suffix=f"_ep{ep_idx:05d}")
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
