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

from codex_baseline_v2.shared.utils import BBox, bbox_iou, merge_bboxes
from codex_baseline_v2.trajectory_analysis.reachability import classify_reachability


def _grid_dims(episodes: List[TrajectoryEpisodeV2]) -> Tuple[int, int]:
    for episode in episodes:
        for step in episode.steps:
            if step.observation is None:
                continue
            height = len(step.observation)
            width = len(step.observation[0]) if height else 0
            return width, height
    return 0, 0


def _poi_key(poi: CandidatePOIV2) -> Tuple[int, int, int, int, str]:
    return (poi.bbox.x1, poi.bbox.y1, poi.bbox.x2, poi.bbox.y2, poi.source_type)


def _merge_pois(pois: List[CandidatePOIV2], min_persistence: int) -> List[CandidatePOIV2]:
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


def analyze_trajectories(
    episodes: List[TrajectoryEpisodeV2],
    cfg: TrajectoryAnalysisConfigV2,
    round_id: int,
) -> BlackboardStateV2:
    palette = []
    candidate_pois: List[CandidatePOIV2] = []
    avatar_hypotheses: List[ObjectRecordV2] = []
    consequences: List[ConsequenceRecordV2] = []
    traversable_points: Dict[Tuple[int, int], int] = {}

    for episode in episodes:
        for step in episode.steps:
            summary = step.observation_summary
            if summary is None:
                continue
            palette = summary.palette
            candidate_pois.extend(summary.candidate_pois)
            avatar_hypotheses.extend(summary.avatar_candidates)
            for avatar in summary.avatar_candidates:
                cx, cy = avatar.centroid
                pt = (int(round(cx)), int(round(cy)))
                traversable_points[pt] = traversable_points.get(pt, 0) + 1
            for poi in summary.candidate_pois:
                for region in summary.active_regions:
                    if bbox_iou(poi.bbox, region) > 0.0:
                        consequences.append(
                            ConsequenceRecordV2(
                                schema_version=SCHEMA_VERSION,
                                game_id=episode.game_id,
                                poi_id=poi.poi_id,
                                round_id=round_id,
                                episode_id=episode.episode_id,
                                instruction_id=None,
                                target_poi_id=poi.poi_id,
                                distance_decreased=True,
                                reached=False,
                                contact=False,
                                local_change_magnitude=0.5,
                                global_change_magnitude=0.2,
                                reward_delta=None,
                                terminal_flag_changed=bool(step.done),
                                object_change_summary="active_region_overlap",
                                followup_poi_ids=[],
                                consequence_class="local_change" if not step.done else "terminal_like",
                            )
                        )

    merged_pois = _merge_pois(candidate_pois, cfg.min_poi_persistence)
    width, height = _grid_dims(episodes)
    traversable_map = {
        "width": int(width),
        "height": int(height),
        "points": [{"x": int(x), "y": int(y), "visits": int(v)} for (x, y), v in traversable_points.items()],
    }
    avatar_centroids = [o.centroid for o in avatar_hypotheses]
    reachability_table = classify_reachability(merged_pois, avatar_centroids, traversable_map)

    return BlackboardStateV2(
        schema_version=SCHEMA_VERSION,
        game_id=episodes[0].game_id if episodes else "unknown_game",
        round_id=round_id,
        palette=palette,
        poi_table=merged_pois,
        reachability_table=reachability_table,
        consequence_table=consequences,
        avatar_hypotheses=avatar_hypotheses,
        traversable_map=traversable_map,
        unresolved_hypotheses=["avatar_identity"] if len(avatar_hypotheses) > 1 else [],
        falsified_hypotheses=[],
        metadata={},
    )
