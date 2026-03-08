from __future__ import annotations

from typing import List, Optional, Tuple

from codex_baseline_v2.shared.schemas import ConsequenceRecordV2, ControllerInstructionV2, SCHEMA_VERSION, TrajectoryEpisodeV2
from codex_baseline_v2.shared.utils import BBox, grid_diff


def _distance_to_bbox(point: Tuple[int, int], bbox: BBox) -> float:
    cx, cy = point
    if bbox.x1 <= cx <= bbox.x2 and bbox.y1 <= cy <= bbox.y2:
        return 0.0
    dx = max(bbox.x1 - cx, 0, cx - bbox.x2)
    dy = max(bbox.y1 - cy, 0, cy - bbox.y2)
    return float(dx + dy)


def link_outcomes(
    episodes: List[TrajectoryEpisodeV2],
    instruction: Optional[ControllerInstructionV2],
    target_bbox: Optional[BBox],
    round_id: int,
) -> List[ConsequenceRecordV2]:
    consequences: List[ConsequenceRecordV2] = []
    targeted = instruction is not None
    if instruction is None or target_bbox is None:
        for episode in episodes:
            for step in episode.steps:
                consequences.append(
                    ConsequenceRecordV2(
                        schema_version=SCHEMA_VERSION,
                        game_id=episode.game_id,
                        poi_id=instruction.target_poi_id if instruction else "unknown",
                        round_id=round_id,
                        episode_id=episode.episode_id,
                        instruction_id=instruction.instruction_id if instruction else None,
                        target_poi_id=instruction.target_poi_id if instruction else None,
                        distance_decreased=False,
                        reached=False,
                        contact=False,
                        local_change_magnitude=0.0,
                        global_change_magnitude=0.0,
                        reward_delta=None,
                        terminal_flag_changed=bool(step.done),
                        object_change_summary="invalid_target_link",
                        followup_poi_ids=[],
                        consequence_class="invalid_target_link" if targeted else "no_progress",
                    )
                )
        return consequences
    for episode in episodes:
        prev_obs = None
        prev_distance = None
        for step in episode.steps:
            obs = step.observation
            if obs is None:
                continue
            local_change = 0.0
            global_change = 0.0
            if prev_obs is not None:
                changed, points = grid_diff(prev_obs, obs)
                global_change = float(changed)
                local_change = float(sum(1 for (x, y) in points if target_bbox.x1 <= x <= target_bbox.x2 and target_bbox.y1 <= y <= target_bbox.y2))
            prev_obs = obs
            avatar_point = None
            if step.observation_summary and step.observation_summary.avatar_candidates:
                avatar_point = step.observation_summary.avatar_candidates[0].centroid
            if avatar_point is None:
                avatar_point = target_bbox.centroid()
            distance = _distance_to_bbox((int(round(avatar_point[0])), int(round(avatar_point[1]))), target_bbox)
            distance_reduced = False
            if prev_distance is not None and distance < prev_distance:
                distance_reduced = True
            prev_distance = distance
            reached = distance <= 0.0
            contact = distance <= 1.0
            if reached:
                outcome_class = "target_reached"
            elif contact:
                outcome_class = "target_contact"
            elif distance_reduced:
                outcome_class = "distance_reduced"
            elif local_change > 0 and global_change == 0:
                outcome_class = "local_effect_only"
            elif global_change > 0 and local_change == 0:
                outcome_class = "global_effect_only"
            elif global_change == 0 and local_change == 0:
                outcome_class = "no_visible_effect"
            else:
                outcome_class = "no_progress"
            consequences.append(
                ConsequenceRecordV2(
                    schema_version=SCHEMA_VERSION,
                    game_id=episode.game_id,
                    poi_id=instruction.target_poi_id or "unknown",
                    round_id=round_id,
                    episode_id=episode.episode_id,
                    instruction_id=instruction.instruction_id,
                    target_poi_id=instruction.target_poi_id,
                    distance_decreased=distance_reduced,
                    reached=reached,
                    contact=contact,
                    local_change_magnitude=local_change,
                    global_change_magnitude=global_change,
                    reward_delta=None,
                    terminal_flag_changed=bool(step.done),
                    object_change_summary="target_outcome",
                    followup_poi_ids=[],
                    consequence_class=outcome_class,
                )
            )
    return consequences
