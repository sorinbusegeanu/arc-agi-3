from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import TrajectoryAnalysisConfigV2
from codex_baseline_v2.shared.schemas import (
    BlackboardStateV2,
    CandidatePOIV2,
    ConsequenceRecordV2,
    ObjectRecordV2,
    SCHEMA_VERSION,
    TrajectoryEpisodeV2,
)
from codex_baseline_v2.shared.utils import BBox, bbox_distance, bbox_iou, grid_diff, merge_bboxes
from codex_baseline_v2.trajectory_analysis.reachability import classify_reachability


def _grid_dims(episodes: List[TrajectoryEpisodeV2], prior_blackboard: Optional[BlackboardStateV2]) -> Tuple[int, int]:
    for episode in episodes:
        for step in episode.steps:
            if step.observation is None:
                continue
            height = len(step.observation)
            width = len(step.observation[0]) if height else 0
            return width, height
    if prior_blackboard and prior_blackboard.traversable_map:
        return (
            int(prior_blackboard.traversable_map.get("width", 0)),
            int(prior_blackboard.traversable_map.get("height", 0)),
        )
    return 0, 0


def _poi_key(poi: CandidatePOIV2) -> Tuple[int, int, int, int, str]:
    return (poi.bbox.x1, poi.bbox.y1, poi.bbox.x2, poi.bbox.y2, poi.source_type)


def _merge_pois_exact(pois: List[CandidatePOIV2], min_persistence: int) -> List[CandidatePOIV2]:
    grouped: Dict[Tuple[int, int, int, int, str], List[CandidatePOIV2]] = defaultdict(list)
    for poi in pois:
        grouped[_poi_key(poi)].append(poi)
    merged: List[CandidatePOIV2] = []
    for key, items in grouped.items():
        if len(items) < min_persistence:
            continue
        bbox = merge_bboxes([p.bbox for p in items])
        if bbox is None:
            continue
        centroid = bbox.centroid()
        confidence = min(1.0, sum(p.confidence for p in items) / float(len(items)))
        merged.append(
            CandidatePOIV2(
                schema_version=SCHEMA_VERSION,
                poi_id=f"poi:{items[0].game_id}:{key[0]}:{key[1]}:{key[2]}:{key[3]}:{key[4]}",
                game_id=items[0].game_id,
                source_type=items[0].source_type,
                bbox=bbox,
                centroid=centroid,
                object_class=items[0].object_class,
                reachable_now="uncertain",
                confidence=confidence,
                expected_information_gain=max(p.expected_information_gain for p in items),
                expected_interaction_type=items[0].expected_interaction_type,
                evidence_count=sum(p.evidence_count for p in items),
                first_seen_ref=items[0].first_seen_ref,
                last_seen_ref=items[-1].last_seen_ref,
                type_confidence=sum(p.type_confidence for p in items) / float(len(items)),
                utility_confidence=sum(p.utility_confidence for p in items) / float(len(items)),
                rejection_reasons=sorted({r for p in items for r in p.rejection_reasons}),
                demotion_reasons=sorted({r for p in items for r in p.demotion_reasons}),
            )
        )
    return merged


def _stable_palette_union(prior: List[int], current: List[int]) -> List[int]:
    seen = set()
    merged: List[int] = []
    for color in list(prior) + list(current):
        if color in seen:
            continue
        seen.add(color)
        merged.append(int(color))
    return merged


def _poi_match_score(prior: CandidatePOIV2, current: CandidatePOIV2) -> Optional[float]:
    if prior.poi_id == current.poi_id:
        return 10.0
    if prior.source_type != current.source_type:
        return None
    if prior.object_class != current.object_class:
        return None
    iou = bbox_iou(prior.bbox, current.bbox)
    centroid_dist = bbox_distance(prior.bbox, current.bbox)
    if iou <= 0.0 and centroid_dist > 3.0:
        return None
    return float(iou * 5.0 - centroid_dist * 0.1)


def _merge_poi_pair(prior: CandidatePOIV2, current: CandidatePOIV2) -> CandidatePOIV2:
    bbox = merge_bboxes([prior.bbox, current.bbox]) or current.bbox
    return CandidatePOIV2(
        schema_version=current.schema_version,
        poi_id=prior.poi_id,
        game_id=prior.game_id,
        source_type=prior.source_type,
        bbox=bbox,
        centroid=bbox.centroid(),
        object_class=prior.object_class,
        reachable_now=current.reachable_now if current.reachable_now != "uncertain" else prior.reachable_now,
        confidence=min(1.0, max(prior.confidence, current.confidence) + 0.05 * max(1, current.evidence_count)),
        expected_information_gain=max(prior.expected_information_gain, current.expected_information_gain),
        expected_interaction_type=current.expected_interaction_type if current.expected_interaction_type != "unknown" else prior.expected_interaction_type,
        evidence_count=int(prior.evidence_count + current.evidence_count),
        first_seen_ref=prior.first_seen_ref or current.first_seen_ref,
        last_seen_ref=current.last_seen_ref or prior.last_seen_ref,
        type_confidence=min(1.0, (prior.type_confidence + current.type_confidence) / 2.0 + 0.05),
        utility_confidence=min(1.0, (prior.utility_confidence + current.utility_confidence) / 2.0 + 0.05),
        rejection_reasons=sorted(set(prior.rejection_reasons) | set(current.rejection_reasons)),
        demotion_reasons=sorted(set(prior.demotion_reasons) | set(current.demotion_reasons)),
    )


def _merge_poi_tables(prior_pois: List[CandidatePOIV2], current_pois: List[CandidatePOIV2]) -> List[CandidatePOIV2]:
    merged = list(prior_pois)
    used_prior: set[int] = set()
    for current in current_pois:
        best_idx = None
        best_score = None
        for idx, prior in enumerate(prior_pois):
            if idx in used_prior:
                continue
            score = _poi_match_score(prior, current)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            merged.append(current)
            continue
        used_prior.add(best_idx)
        merged[best_idx] = _merge_poi_pair(prior_pois[best_idx], current)
    return merged


def _object_match_score(prior: ObjectRecordV2, current: ObjectRecordV2) -> Optional[float]:
    if prior.object_class != current.object_class or prior.color != current.color:
        return None
    iou = bbox_iou(prior.bbox, current.bbox)
    centroid_dist = bbox_distance(prior.bbox, current.bbox)
    if iou <= 0.0 and centroid_dist > 3.0:
        return None
    return float(iou * 5.0 - centroid_dist * 0.1)


def _merge_object_pair(prior: ObjectRecordV2, current: ObjectRecordV2) -> ObjectRecordV2:
    bbox = merge_bboxes([prior.bbox, current.bbox]) or current.bbox
    return ObjectRecordV2(
        schema_version=current.schema_version,
        object_id=prior.object_id,
        game_id=prior.game_id,
        episode_id=current.episode_id,
        bbox=bbox,
        centroid=bbox.centroid(),
        color=prior.color,
        area=max(prior.area, current.area),
        aspect_ratio=float(bbox.width()) / float(max(1, bbox.height())),
        object_class=prior.object_class,
        confidence=min(1.0, max(prior.confidence, current.confidence) + 0.05),
        evidence_refs=sorted(set(prior.evidence_refs) | set(current.evidence_refs)),
        first_seen_ref=prior.first_seen_ref or current.first_seen_ref,
        last_seen_ref=current.last_seen_ref or prior.last_seen_ref,
    )


def _merge_avatar_hypotheses(prior_objects: List[ObjectRecordV2], current_objects: List[ObjectRecordV2]) -> List[ObjectRecordV2]:
    merged = list(prior_objects)
    used_prior: set[int] = set()
    for current in current_objects:
        best_idx = None
        best_score = None
        for idx, prior in enumerate(prior_objects):
            if idx in used_prior:
                continue
            score = _object_match_score(prior, current)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            merged.append(current)
            continue
        used_prior.add(best_idx)
        merged[best_idx] = _merge_object_pair(prior_objects[best_idx], current)
    merged.sort(key=lambda o: o.confidence, reverse=True)
    return merged


def _merge_traversable_map(
    prior_map: Optional[Dict[str, object]],
    current_points: Dict[Tuple[int, int], int],
    width: int,
    height: int,
) -> Dict[str, object]:
    visits: Dict[Tuple[int, int], int] = {}
    if prior_map:
        for point in prior_map.get("points", []):
            coord = (int(point["x"]), int(point["y"]))
            visits[coord] = visits.get(coord, 0) + int(point.get("visits", 1))
    for coord, count in current_points.items():
        visits[coord] = visits.get(coord, 0) + int(count)
    if prior_map:
        width = max(width, int(prior_map.get("width", 0)))
        height = max(height, int(prior_map.get("height", 0)))
    return {
        "width": int(width),
        "height": int(height),
        "points": [{"x": int(x), "y": int(y), "visits": int(v)} for (x, y), v in sorted(visits.items())],
    }


def _distance_to_bbox(point: Tuple[int, int], bbox: BBox) -> float:
    x, y = point
    if bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2:
        return 0.0
    dx = max(bbox.x1 - x, 0, x - bbox.x2)
    dy = max(bbox.y1 - y, 0, y - bbox.y2)
    return float(dx + dy)


def _derive_target_consequences(episodes: List[TrajectoryEpisodeV2], round_id: int) -> List[ConsequenceRecordV2]:
    consequences: List[ConsequenceRecordV2] = []
    for episode in episodes:
        prev_obs = None
        prev_distance = None
        for step in episode.steps:
            if not (step.instruction_id or step.target_poi_id):
                prev_obs = step.observation
                continue
            target_bbox = step.target_geometry
            if target_bbox is None:
                continue
            local_change = 0.0
            global_change = 0.0
            if prev_obs is not None and step.observation is not None:
                changed, points = grid_diff(prev_obs, step.observation)
                global_change = float(changed)
                local_change = float(sum(1 for (x, y) in points if target_bbox.x1 <= x <= target_bbox.x2 and target_bbox.y1 <= y <= target_bbox.y2))
            avatar_point = None
            if step.observation_summary and step.observation_summary.avatar_candidates:
                avatar_point = step.observation_summary.avatar_candidates[0].centroid
            elif step.observation_summary and step.observation_summary.objects:
                avatar_point = step.observation_summary.objects[0].centroid
            if avatar_point is None:
                avatar_point = target_bbox.centroid()
            distance = _distance_to_bbox((int(round(avatar_point[0])), int(round(avatar_point[1]))), target_bbox)
            distance_decreased = prev_distance is not None and distance < prev_distance
            reached = distance <= 0.0
            contact = distance <= 1.0
            if step.done:
                consequence_class = "terminal_like"
            elif reached or contact:
                consequence_class = "progress_like"
            elif global_change > 0.0:
                consequence_class = "global_change"
            elif local_change > 0.0:
                consequence_class = "local_change"
            elif distance_decreased:
                consequence_class = "progress_like"
            else:
                consequence_class = "no_change"
            consequences.append(
                ConsequenceRecordV2(
                    schema_version=SCHEMA_VERSION,
                    game_id=episode.game_id,
                    poi_id=step.target_poi_id or "unknown",
                    round_id=round_id,
                    episode_id=episode.episode_id,
                    instruction_id=step.instruction_id,
                    target_poi_id=step.target_poi_id,
                    distance_decreased=bool(distance_decreased),
                    reached=bool(reached),
                    contact=bool(contact),
                    local_change_magnitude=local_change,
                    global_change_magnitude=global_change,
                    reward_delta=step.reward,
                    terminal_flag_changed=bool(step.done),
                    object_change_summary="target_execution",
                    followup_poi_ids=[],
                    consequence_class=consequence_class,
                )
            )
            prev_obs = step.observation
            prev_distance = distance
    return consequences


def analyze_trajectories(
    episodes: List[TrajectoryEpisodeV2],
    cfg: TrajectoryAnalysisConfigV2,
    round_id: int,
    prior_blackboard: Optional[BlackboardStateV2] = None,
) -> BlackboardStateV2:
    palette: List[int] = []
    current_candidate_pois: List[CandidatePOIV2] = []
    current_avatar_hypotheses: List[ObjectRecordV2] = []
    traversable_points: Dict[Tuple[int, int], int] = {}

    for episode in episodes:
        for step in episode.steps:
            summary = step.observation_summary
            if summary is None:
                continue
            palette = _stable_palette_union(palette, summary.palette)
            current_candidate_pois.extend(summary.candidate_pois)
            current_avatar_hypotheses.extend(summary.avatar_candidates)
            for avatar in summary.avatar_candidates:
                cx, cy = avatar.centroid
                pt = (int(round(cx)), int(round(cy)))
                traversable_points[pt] = traversable_points.get(pt, 0) + 1

    current_merged_pois = _merge_pois_exact(current_candidate_pois, cfg.min_poi_persistence)
    consequences = _derive_target_consequences(episodes, round_id)

    if prior_blackboard is not None:
        palette = _stable_palette_union(prior_blackboard.palette, palette)
        merged_pois = _merge_poi_tables(prior_blackboard.poi_table, current_merged_pois)
        avatar_hypotheses = _merge_avatar_hypotheses(prior_blackboard.avatar_hypotheses, current_avatar_hypotheses)
        width, height = _grid_dims(episodes, prior_blackboard)
        traversable_map = _merge_traversable_map(prior_blackboard.traversable_map, traversable_points, width, height)
        consequence_table = list(prior_blackboard.consequence_table) + consequences
        unresolved_hypotheses = sorted(set(prior_blackboard.unresolved_hypotheses) | ({"avatar_identity"} if len(avatar_hypotheses) > 1 else set()))
        falsified_hypotheses = list(prior_blackboard.falsified_hypotheses)
        metadata = dict(prior_blackboard.metadata)
        game_id = prior_blackboard.game_id
    else:
        merged_pois = current_merged_pois
        avatar_hypotheses = current_avatar_hypotheses
        width, height = _grid_dims(episodes, prior_blackboard)
        traversable_map = _merge_traversable_map(None, traversable_points, width, height)
        consequence_table = consequences
        unresolved_hypotheses = ["avatar_identity"] if len(avatar_hypotheses) > 1 else []
        falsified_hypotheses = []
        metadata = {}
        game_id = episodes[0].game_id if episodes else "unknown_game"

    reachability_table = classify_reachability(merged_pois, [o.centroid for o in avatar_hypotheses], traversable_map)
    reachability_lookup = {r.poi_id: r.status for r in reachability_table}
    merged_pois = [
        CandidatePOIV2(
            schema_version=poi.schema_version,
            poi_id=poi.poi_id,
            game_id=poi.game_id,
            source_type=poi.source_type,
            bbox=poi.bbox,
            centroid=poi.centroid,
            object_class=poi.object_class,
            reachable_now=reachability_lookup.get(poi.poi_id, poi.reachable_now),
            confidence=poi.confidence,
            expected_information_gain=poi.expected_information_gain,
            expected_interaction_type=poi.expected_interaction_type,
            evidence_count=poi.evidence_count,
            first_seen_ref=poi.first_seen_ref,
            last_seen_ref=poi.last_seen_ref,
            type_confidence=poi.type_confidence,
            utility_confidence=poi.utility_confidence,
            rejection_reasons=poi.rejection_reasons,
            demotion_reasons=poi.demotion_reasons,
        )
        for poi in merged_pois
    ]

    return BlackboardStateV2(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        round_id=round_id,
        palette=palette,
        poi_table=merged_pois,
        reachability_table=reachability_table,
        consequence_table=consequence_table,
        avatar_hypotheses=avatar_hypotheses,
        traversable_map=traversable_map,
        unresolved_hypotheses=unresolved_hypotheses,
        falsified_hypotheses=falsified_hypotheses,
        metadata=metadata,
    )
