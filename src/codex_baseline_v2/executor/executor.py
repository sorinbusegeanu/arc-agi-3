from __future__ import annotations

from typing import List, Optional, Tuple

from codex_baseline_v2.shared.config import ExecutorConfigV2
from codex_baseline_v2.shared.schemas import (
    ConsequenceRecordV2,
    ControllerInstructionV2,
    ExecutorOutcomeV2,
    SCHEMA_VERSION,
    TrajectoryEpisodeV2,
)
from codex_baseline_v2.shared.utils import BBox, bbox_distance


def _target_centroid(target_region: Optional[BBox]) -> Optional[Tuple[float, float]]:
    if target_region is None:
        return None
    return target_region.centroid()


def _compute_progress(centroid: Tuple[float, float], target: Tuple[float, float]) -> float:
    dx = centroid[0] - target[0]
    dy = centroid[1] - target[1]
    return (dx * dx + dy * dy) ** 0.5


def execute_instruction_offline(
    episodes: List[TrajectoryEpisodeV2],
    instruction: ControllerInstructionV2,
    cfg: ExecutorConfigV2,
) -> ExecutorOutcomeV2:
    target = _target_centroid(instruction.target_region)
    actions = []
    progress_values: List[float] = []
    reached = False
    contact = False
    blocked = False
    consequences: List[ConsequenceRecordV2] = []
    initial_distance: Optional[float] = None
    last_distance: Optional[float] = None

    for episode in episodes:
        for step in episode.steps:
            actions.append(step.action)
            if target and step.observation_summary:
                avatar_candidates = step.observation_summary.avatar_candidates
                if avatar_candidates:
                    centroid = avatar_candidates[0].centroid
                elif step.observation_summary.objects:
                    centroid = step.observation_summary.objects[0].centroid
                else:
                    centroid = target
                distance = _compute_progress(centroid, target)
                progress_values.append(distance)
                if initial_distance is None:
                    initial_distance = distance
                last_distance = distance
                if distance <= cfg.target_reach_distance:
                    reached = True
                    contact = True
            if len(actions) >= cfg.max_steps:
                break
        if len(actions) >= cfg.max_steps:
            break

    if not actions or (progress_values and all(v >= progress_values[0] for v in progress_values[1:])):
        blocked = True

    if instruction.target_poi_id and target:
        episode_id = episodes[0].episode_id if episodes else f"exec_round{instruction.round_id:03d}"
        consequences.append(
            ConsequenceRecordV2(
                schema_version=SCHEMA_VERSION,
                game_id=instruction.game_id,
                poi_id=instruction.target_poi_id,
                round_id=instruction.round_id,
                episode_id=episode_id,
                instruction_id=instruction.instruction_id,
                target_poi_id=instruction.target_poi_id,
                distance_decreased=bool(initial_distance is not None and last_distance is not None and last_distance < initial_distance),
                reached=reached,
                contact=contact,
                local_change_magnitude=0.0,
                global_change_magnitude=0.0,
                reward_delta=None,
                terminal_flag_changed=False,
                object_change_summary="offline_replay",
                followup_poi_ids=[],
                consequence_class="no_change" if not reached else "progress_like",
            )
        )

    return ExecutorOutcomeV2(
        schema_version=SCHEMA_VERSION,
        game_id=instruction.game_id,
        round_id=instruction.round_id,
        instruction_id=instruction.instruction_id,
        instruction_mode=instruction.mode,
        target_poi_id=instruction.target_poi_id,
        target_type=instruction.target_type,
        target_geometry=instruction.target_geometry,
        target_source_round=instruction.target_source_round,
        actions=actions,
        target_progress=progress_values,
        reached=reached,
        contact=contact,
        blocked=blocked,
        outcome_summary="offline_execution",
        consequence_records=consequences,
    )
