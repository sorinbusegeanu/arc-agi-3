from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import HiddenTriggerConfigV2
from codex_baseline_v2.shared.schemas import ChangeEventV2, InterventionRecordV2, SpatialInterventionCellV2, TrajectoryEpisodeV2, SCHEMA_VERSION


def update_spatial_intervention_field(
    existing_field: List[SpatialInterventionCellV2],
    episodes: List[TrajectoryEpisodeV2],
    interventions: List[InterventionRecordV2],
    events: List[ChangeEventV2],
    cfg: HiddenTriggerConfigV2,
) -> list[SpatialInterventionCellV2]:
    merged: Dict[Tuple[Optional[str], Tuple[int, int]], SpatialInterventionCellV2] = {
        (cell.area_id, tuple(cell.cell)): cell for cell in existing_field
    }
    event_by_episode = defaultdict(list)
    for event in events:
        event_by_episode[event.episode_id].append(event)
    max_dwell_window = max(cfg.dwell_step_thresholds)
    max_round_id = max((record.round_id for record in interventions), default=0)
    for episode in episodes:
        episode_events = sorted(event_by_episode.get(episode.episode_id, []), key=lambda event: event.start_step_idx)
        event_starts = [event.start_step_idx for event in episode_events]
        transition_suffix = [0] * (len(episode_events) + 1)
        for idx in range(len(episode_events) - 1, -1, -1):
            transition_suffix[idx] = transition_suffix[idx + 1] + int(episode_events[idx].locality == "cross_area_transition")
        prev_cell = None
        for step in episode.steps:
            cell = None
            if step.actual_avatar_centroid is not None:
                cell = (int(round(step.actual_avatar_centroid[0])), int(round(step.actual_avatar_centroid[1])))
            elif step.predicted_avatar_centroid is not None:
                cell = (int(round(step.predicted_avatar_centroid[0])), int(round(step.predicted_avatar_centroid[1])))
            if cell is None:
                continue
            key = (step.area_id, cell)
            prev = merged.get(
                key,
                SpatialInterventionCellV2(SCHEMA_VERSION, episode.game_id, step.area_id, cell, 0, 0, 0, [], 0, 0, 0, 0, 0.0, 0),
            )
            action_counts = defaultdict(int, {int(a): int(b) for a, b in prev.action_counts})
            if step.action.action_id is not None:
                action_counts[int(step.action.action_id)] += 1
            step_events = step.event_ids
            right_bound = bisect_right(event_starts, step.step_idx + max_dwell_window)
            current_bound = bisect_right(event_starts, step.step_idx)
            delayed = max(0, right_bound - current_bound)
            transitions = transition_suffix[current_bound]
            null_increment = 1 if step.intervention_id and not step_events else 0
            score = prev.hidden_trigger_score + 0.25 * len(step_events) + 0.1 * delayed - cfg.null_penalty * float(null_increment)
            merged[key] = SpatialInterventionCellV2(
                schema_version=SCHEMA_VERSION,
                game_id=episode.game_id,
                area_id=step.area_id,
                cell=cell,
                visit_count=prev.visit_count + 1,
                dwell_count=prev.dwell_count + int(prev_cell == cell),
                crossing_count=prev.crossing_count + int(prev_cell is not None and prev_cell != cell),
                action_counts=sorted(action_counts.items()),
                post_event_count=prev.post_event_count + len(step_events),
                post_delayed_event_count=prev.post_delayed_event_count + delayed,
                post_transition_event_count=prev.post_transition_event_count + transitions,
                null_probe_count=prev.null_probe_count + null_increment,
                hidden_trigger_score=max(0.0, score),
                last_updated_round=max(prev.last_updated_round, max_round_id),
            )
            prev_cell = cell
    return list(merged.values())
