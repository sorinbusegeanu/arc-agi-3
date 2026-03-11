from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import EventExtractionConfigV2
from codex_baseline_v2.shared.schemas import (
    AreaStateV2,
    AvatarTrackHypothesisV2,
    ChangeEventV2,
    EventRegionDeltaV2,
    ObjectStateDeltaV2,
    ObservationSummaryV2,
    SCHEMA_VERSION,
    TrajectoryEpisodeV2,
    TrajectoryStepV2,
)
from codex_baseline_v2.shared.utils import BBox, bbox_distance, bbox_from_points, bbox_iou, grid_diff


def _step_position(step: TrajectoryStepV2) -> Optional[Tuple[float, float]]:
    return step.actual_avatar_centroid or step.predicted_avatar_centroid


def _diff_payload(prev_obs: Optional[List[List[int]]], obs: Optional[List[List[int]]]) -> Tuple[int, List[Tuple[int, int]], float]:
    if prev_obs is None or obs is None or not obs or not obs[0]:
        return (0, [], 0.0)
    count, points = grid_diff(prev_obs, obs)
    ratio = float(count) / float(max(1, len(obs) * len(obs[0])))
    return count, points, ratio


def _classify_locality(
    step: TrajectoryStepV2,
    bbox: BBox,
    area_changed: bool,
    cfg: EventExtractionConfigV2,
) -> str:
    if area_changed:
        return "cross_area_transition"
    avatar = _step_position(step)
    if avatar is None:
        return "unknown"
    avatar_bbox = BBox(int(round(avatar[0])), int(round(avatar[1])), int(round(avatar[0])), int(round(avatar[1])))
    dist = bbox_distance(avatar_bbox, bbox)
    if step.observation_summary and step.observation_summary.hud_region_candidates:
        hud_overlap = max((bbox_iou(region, bbox) for region in step.observation_summary.hud_region_candidates), default=0.0)
        if hud_overlap >= 0.5:
            return "hud_only"
    if dist <= cfg.local_radius:
        return "local"
    if dist >= cfg.remote_min_distance:
        return "remote_same_area"
    return "global_same_area"


def _classify_type(step: TrajectoryStepV2, ratio: float, locality: str, cfg: EventExtractionConfigV2) -> str:
    if step.done:
        return "terminal_like"
    if abs(float(step.reward)) > 0.0:
        return "reward_like"
    if locality == "cross_area_transition" or ratio >= cfg.transition_frame_change_ratio:
        return "transition"
    if step.event_ids and ratio > cfg.min_region_change_ratio:
        return "mixed"
    if ratio > 0.0 and _step_position(step) is not None:
        return "movement_change" if locality == "local" else "object_state_change"
    return "object_state_change"


def _region_delta(points: List[Tuple[int, int]], bbox: BBox, ratio: float) -> EventRegionDeltaV2:
    return EventRegionDeltaV2(
        schema_version=SCHEMA_VERSION,
        region_role="changed_region",
        bbox=bbox,
        pixel_change_ratio=ratio,
        object_births=0,
        object_deaths=0,
        object_moves=1 if points else 0,
        object_state_changes=1 if ratio > 0.0 else 0,
    )


def _summary_objects(step: TrajectoryStepV2) -> List:
    summary = step.observation_summary
    if summary is None:
        return []
    return list(summary.objects) + list(summary.avatar_candidates)


def _object_deltas(game_id: str, event_id: str, prev_step: TrajectoryStepV2, step: TrajectoryStepV2) -> List[ObjectStateDeltaV2]:
    previous = _summary_objects(prev_step)
    current = _summary_objects(step)
    out: List[ObjectStateDeltaV2] = []
    used_current: set[str] = set()
    for pre in previous:
        best = None
        best_iou = 0.0
        for post in current:
            if post.object_id in used_current:
                continue
            score = bbox_iou(pre.bbox, post.bbox)
            if score > best_iou:
                best_iou = score
                best = post
        if best is None or best_iou < 0.2:
            out.append(
                ObjectStateDeltaV2(
                    schema_version=SCHEMA_VERSION,
                    game_id=game_id,
                    event_id=event_id,
                    pre_object_id=pre.object_id,
                    post_object_id=None,
                    delta_type="disappear",
                    pre_bbox=pre.bbox,
                    post_bbox=None,
                    pre_palette=[pre.color],
                    post_palette=[],
                    confidence=0.5,
                )
            )
            continue
        used_current.add(best.object_id)
        delta_type = "move"
        if pre.color != best.color:
            delta_type = "state_change"
        elif pre.bbox != best.bbox:
            delta_type = "move"
        out.append(
            ObjectStateDeltaV2(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                event_id=event_id,
                pre_object_id=pre.object_id,
                post_object_id=best.object_id,
                delta_type=delta_type,
                pre_bbox=pre.bbox,
                post_bbox=best.bbox,
                pre_palette=[pre.color],
                post_palette=[best.color],
                confidence=max(0.3, best_iou),
            )
        )
    for post in current:
        if post.object_id in used_current:
            continue
        out.append(
            ObjectStateDeltaV2(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                event_id=event_id,
                pre_object_id=None,
                post_object_id=post.object_id,
                delta_type="appear",
                pre_bbox=None,
                post_bbox=post.bbox,
                pre_palette=[],
                post_palette=[post.color],
                confidence=0.5,
            )
        )
    return out[:32]


def extract_change_events_from_episodes(
    episodes: List[TrajectoryEpisodeV2],
    cfg: EventExtractionConfigV2,
    area_table: List[AreaStateV2],
    avatar_tracks: List[AvatarTrackHypothesisV2],
    prior_events: Optional[List[ChangeEventV2]] = None,
) -> list[ChangeEventV2]:
    events: List[ChangeEventV2] = list(prior_events or [])
    existing_ids = {row.event_id for row in events}
    del area_table, avatar_tracks

    for episode in episodes:
        prev_step: Optional[TrajectoryStepV2] = None
        open_event: Optional[Dict[str, object]] = None
        gap = 0
        for step in episode.steps:
            if prev_step is None:
                prev_step = step
                continue
            changed, points, ratio = _diff_payload(prev_step.observation, step.observation)
            area_changed = prev_step.area_id is not None and step.area_id is not None and prev_step.area_id != step.area_id
            should_emit = ratio >= cfg.min_region_change_ratio or abs(float(step.reward)) > 0.0 or step.done or area_changed
            if should_emit:
                bbox = bbox_from_points(points) or (step.target_geometry if step.target_geometry is not None else BBox(0, 0, 0, 0))
                locality = _classify_locality(step, bbox, area_changed, cfg)
                if open_event is None:
                    open_event = {
                        "start": prev_step.step_idx,
                        "peak": step.step_idx,
                        "end": step.step_idx,
                        "peak_ratio": ratio,
                        "points": list(points),
                        "bbox": bbox,
                        "step": step,
                        "prev_step": prev_step,
                    }
                else:
                    open_event["end"] = step.step_idx
                    open_event["points"] = list(open_event["points"]) + list(points)
                    open_event["bbox"] = bbox_from_points(list(open_event["points"])) or bbox
                    if ratio >= float(open_event["peak_ratio"]):
                        open_event["peak"] = step.step_idx
                        open_event["peak_ratio"] = ratio
                        open_event["step"] = step
                        open_event["prev_step"] = prev_step
                open_event["locality"] = locality
                gap = 0
            else:
                gap += 1
            if open_event is not None and (gap > cfg.event_merge_gap or step is episode.steps[-1]):
                peak_step = open_event["step"]
                prev_for_peak = open_event["prev_step"]
                event_ratio = float(open_event["peak_ratio"])
                bbox = open_event["bbox"]
                locality = str(open_event.get("locality", "unknown"))
                event_type = _classify_type(peak_step, event_ratio, locality, cfg)
                event_id = "event:%s:%s:%s" % (episode.episode_id, open_event["start"], open_event["end"])
                if event_id not in existing_ids:
                    events.append(
                        ChangeEventV2(
                            schema_version=SCHEMA_VERSION,
                            game_id=episode.game_id,
                            event_id=event_id,
                            episode_id=episode.episode_id,
                            start_step_idx=int(open_event["start"]),
                            peak_step_idx=int(open_event["peak"]),
                            end_step_idx=int(open_event["end"]),
                            event_type=event_type,
                            locality=locality,
                            trigger_context=peak_step.action_context_key or "unknown",
                            trigger_instruction_id=peak_step.instruction_id,
                            trigger_target_poi_id=peak_step.target_poi_id,
                            trigger_area_id=peak_step.area_id,
                            pre_area_id=prev_for_peak.area_id,
                            post_area_id=peak_step.area_id,
                            region_deltas=[_region_delta(list(open_event["points"]), bbox, event_ratio)],
                            object_state_deltas=_object_deltas(episode.game_id, event_id, prev_for_peak, peak_step),
                            reward_delta=peak_step.reward,
                            terminal_flag_changed=bool(peak_step.done),
                            confidence=min(1.0, 0.4 + event_ratio),
                        )
                    )
                    existing_ids.add(event_id)
                open_event = None
                gap = 0
            prev_step = step
    return events
