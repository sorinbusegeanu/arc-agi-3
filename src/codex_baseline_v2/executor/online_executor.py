from __future__ import annotations

from typing import List, Optional, Tuple

from codex_baseline_v2.executor.route_planner import plan_route
from codex_baseline_v2.runtime.environment_session import EnvironmentSessionV2
from codex_baseline_v2.runtime.trajectory_policy import PolicyStateV2, TrajectoryPolicyV2
from codex_baseline_v2.shared.config import ExecutorConfigV2
from codex_baseline_v2.shared.state_identity import canonical_state_identity
from codex_baseline_v2.shared.schemas import (
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
) -> tuple[ExecutorOutcomeV2, TrajectoryEpisodeV2]:
    policy = TrajectoryPolicyV2()
    state = PolicyStateV2(round_id=instruction.round_id)
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
    if instruction.mode not in {"random_probe", "unguided_probe"} and not instruction.target_poi_id:
        outcome = ExecutorOutcomeV2(
            schema_version=SCHEMA_VERSION,
            game_id=session.game_id,
            round_id=instruction.round_id,
            instruction_id=instruction.instruction_id,
            instruction_mode=instruction.mode,
            target_poi_id=None,
            target_type=instruction.target_type,
            target_geometry=instruction.target_geometry,
            target_source_round=instruction.target_source_round,
            actions=[],
            target_progress=[],
            reached=False,
            contact=False,
            blocked=True,
            outcome_summary="missing_target_reference",
            consequence_records=[],
        )
        episode = TrajectoryEpisodeV2(
            schema_version=SCHEMA_VERSION,
            game_id=session.game_id,
            episode_id=f"exec_round{instruction.round_id:03d}",
            steps=[],
            done=False,
            win=False,
            seed=None,
            metadata={"mode": instruction.mode, "instruction": instruction.to_dict(), "diagnostic": "missing_target_reference"},
        )
        return outcome, episode
    if instruction.target_poi_id:
        for poi in blackboard.poi_table:
            if poi.poi_id == instruction.target_poi_id:
                target_poi = poi
                break
    if instruction.target_poi_id and target_poi is None:
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
            outcome_summary="missing_target_reference",
            consequence_records=[],
        )
        episode = TrajectoryEpisodeV2(
            schema_version=SCHEMA_VERSION,
            game_id=session.game_id,
            episode_id=f"exec_round{instruction.round_id:03d}",
            steps=[],
            done=False,
            win=False,
            seed=None,
            metadata={"mode": instruction.mode, "instruction": instruction.to_dict(), "diagnostic": "missing_target_reference"},
        )
        return outcome, episode

    prev_distance: Optional[float] = None
    last_live_avatar: Optional[Tuple[int, int]] = None
    last_live_bbox: Optional[BBox] = None
    prev_obs = obs
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

        if target_poi is not None and last_live_avatar is not None:
            plan = plan_route(
                blackboard,
                target_poi,
                (int(last_live_avatar[0]), int(last_live_avatar[1])),
                prev_distance=prev_distance,
            )
            if plan.progress_valid and plan.next_subgoal is not None:
                state.target_centroid = plan.next_subgoal
                action = policy.instructed_action(instruction, plan.next_subgoal, avail, state)
            else:
                action = policy.fallback_action(avail, state)
            current_distance = plan.distance_prev
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
            if new_distance <= cfg.target_reach_distance:
                reached = True
        if last_live_bbox is not None and target_bbox is not None and _touches(last_live_bbox, target_bbox):
            contact = True
        if plan is not None and plan.progress_valid and (plan.distance_delta is None or plan.distance_delta <= 0.0):
            stall_count += 1
        else:
            stall_count = 0
        actions.append(action)
        pre_state = canonical_state_identity(obs, include_payload=False)
        post_state = canonical_state_identity(result.observation, include_payload=False)
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
                episode_id=f"exec_round{instruction.round_id:03d}",
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
                        "live_avatar_position": list(last_live_avatar) if last_live_avatar is not None else None,
                        "local_change_magnitude": local_change,
                        "global_change_magnitude": global_change,
                    },
                    "collection_mode": instruction.mode,
                    "state_signature_version": pre_state.get("state_signature_version"),
                },
            )
        )
        prev_obs = result.observation
        obs = result.observation
        if result.done:
            break
        if stall_count >= cfg.blocked_repeat_limit:
            blocked = True
            break

    consequences: List[ConsequenceRecordV2] = []
    if instruction.target_poi_id and target_bbox is not None:
        initial_distance = progress[0] if progress else None
        final_distance = progress[-1] if progress else None
        consequences.append(
            ConsequenceRecordV2(
                schema_version=SCHEMA_VERSION,
                game_id=session.game_id,
                poi_id=instruction.target_poi_id,
                round_id=instruction.round_id,
                episode_id=f"exec_round{instruction.round_id:03d}",
                instruction_id=instruction.instruction_id,
                target_poi_id=instruction.target_poi_id,
                distance_decreased=bool(initial_distance is not None and final_distance is not None and final_distance < initial_distance),
                reached=reached,
                contact=contact,
                local_change_magnitude=float(sum(step.info.get("progress", {}).get("local_change_magnitude", 0.0) for step in steps)),
                global_change_magnitude=float(sum(step.info.get("progress", {}).get("global_change_magnitude", 0.0) for step in steps)),
                reward_delta=float(sum(step.reward for step in steps)) if steps else None,
                terminal_flag_changed=bool(steps and steps[-1].done),
                object_change_summary="online_execution",
                followup_poi_ids=[],
                consequence_class="terminal_like" if steps and steps[-1].done else "progress_like" if reached or contact else "no_change",
            )
        )
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
        outcome_summary="online_execution",
        consequence_records=consequences,
    )
    episode = TrajectoryEpisodeV2(
        schema_version=SCHEMA_VERSION,
        game_id=session.game_id,
        episode_id=f"exec_round{instruction.round_id:03d}",
        steps=steps,
        done=bool(steps and steps[-1].done),
        win=False,
        seed=None,
        metadata={"mode": instruction.mode, "instruction": instruction.to_dict(), "collection_mode": instruction.mode},
    )
    return outcome, episode
