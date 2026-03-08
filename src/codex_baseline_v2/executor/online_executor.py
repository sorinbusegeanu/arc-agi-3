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
    SCHEMA_VERSION,
    TrajectoryEpisodeV2,
    TrajectoryStepV2,
)


def run_online_execution(
    session: EnvironmentSessionV2,
    instruction: ControllerInstructionV2,
    blackboard,
    cfg: ExecutorConfigV2,
) -> tuple[ExecutorOutcomeV2, TrajectoryEpisodeV2]:
    policy = TrajectoryPolicyV2()
    state = PolicyStateV2()
    obs = session.reset()
    steps: List[TrajectoryStepV2] = []
    actions = []
    progress: List[float] = []
    reached = False
    contact = False
    blocked = False
    stall_count = 0
    target_poi = None
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
    for step_idx in range(cfg.max_steps):
        avail = session.available_actions()
        if target_poi is not None:
            avatar_center = None
            if blackboard.avatar_hypotheses:
                avatar_center = blackboard.avatar_hypotheses[0].centroid
            if avatar_center is None:
                avatar_center = target_poi.centroid
            plan = plan_route(
                blackboard,
                target_poi,
                (int(avatar_center[0]), int(avatar_center[1])),
                prev_distance=prev_distance,
            )
            if plan.next_subgoal is None:
                stall_count += 1
                action = policy.fallback_action(avail)
            else:
                action = policy.instructed_action(instruction, plan.next_subgoal, avail, state)
            if plan.distance_estimate is not None and plan.progress_valid:
                progress.append(plan.distance_estimate)
                prev_distance = plan.distance_estimate
            if plan.distance_estimate is not None and plan.distance_estimate <= cfg.target_reach_distance and plan.progress_valid:
                reached = True
                contact = True
        else:
            action = policy.unguided_probe(avail)
        result = session.step(action)
        pre_state = canonical_state_identity(obs, include_payload=False)
        post_state = canonical_state_identity(result.observation, include_payload=False)
        actions.append(action)
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
                        "distance_prev": plan.distance_prev if target_poi is not None else None,
                        "distance_new": plan.distance_estimate if target_poi is not None else None,
                        "distance_delta": plan.distance_delta if target_poi is not None else None,
                        "progress_valid": plan.progress_valid if target_poi is not None else False,
                        "progress_reason": plan.progress_reason if target_poi is not None else "untargeted",
                        "fallback_mode": plan.fallback_mode if target_poi is not None else None,
                    },
                    "state_signature_version": pre_state.get("state_signature_version"),
                },
            )
        )
        obs = result.observation
        if result.done:
            break
        if stall_count >= cfg.blocked_repeat_limit:
            blocked = True
            break
    consequences: List[ConsequenceRecordV2] = []
    if instruction.target_poi_id:
        consequences.append(
            ConsequenceRecordV2(
                schema_version=SCHEMA_VERSION,
                game_id=session.game_id,
                poi_id=instruction.target_poi_id,
                round_id=instruction.round_id,
                episode_id=f"exec_round{instruction.round_id:03d}",
                instruction_id=instruction.instruction_id,
                target_poi_id=instruction.target_poi_id,
                distance_decreased=bool(progress),
                reached=reached,
                contact=contact,
                local_change_magnitude=0.0,
                global_change_magnitude=0.0,
                reward_delta=None,
                terminal_flag_changed=False,
                object_change_summary="online_execution",
                followup_poi_ids=[],
                consequence_class="progress_like" if reached else "no_change",
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
        metadata={"mode": instruction.mode, "instruction": instruction.to_dict()},
    )
    return outcome, episode
