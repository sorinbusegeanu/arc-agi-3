from __future__ import annotations

from typing import List, Tuple

from codex_baseline_v2.shared.config import AnalystConfigV2
from codex_baseline_v2.shared.schemas import CandidatePOIV2, ObjectRecordV2, SCHEMA_VERSION
from codex_baseline_v2.shared.utils import bbox_from_points, bbox_iou, merge_bboxes


def _is_border_elongated(obj: ObjectRecordV2, grid_width: int, grid_height: int) -> bool:
    touches_border = obj.bbox.x1 == 0 or obj.bbox.y1 == 0 or obj.bbox.x2 >= grid_width - 1 or obj.bbox.y2 >= grid_height - 1
    elongated = obj.aspect_ratio >= 4.0 or obj.aspect_ratio <= 0.25
    return touches_border and elongated


def _merge_overlapping_elongated(objs: List[ObjectRecordV2]) -> List[ObjectRecordV2]:
    if not objs:
        return objs
    used = set()
    merged: List[ObjectRecordV2] = []
    for idx, obj in enumerate(objs):
        if idx in used:
            continue
        overlaps = [obj]
        for jdx, other in enumerate(objs):
            if jdx == idx or jdx in used:
                continue
            if obj.color != other.color:
                continue
            if bbox_iou(obj.bbox, other.bbox) > 0.2:
                overlaps.append(other)
                used.add(jdx)
        if len(overlaps) == 1:
            merged.append(obj)
            continue
        merged_bbox = merge_bboxes([o.bbox for o in overlaps])
        if merged_bbox is None:
            merged.append(obj)
            continue
        merged.append(
            ObjectRecordV2(
                schema_version=SCHEMA_VERSION,
                object_id=f"merged:{obj.object_id}",
                game_id=obj.game_id,
                episode_id=obj.episode_id,
                bbox=merged_bbox,
                centroid=merged_bbox.centroid(),
                color=obj.color,
                area=sum(o.area for o in overlaps),
                aspect_ratio=float(merged_bbox.width()) / float(max(1, merged_bbox.height())),
                object_class=obj.object_class,
                confidence=max(o.confidence for o in overlaps),
                evidence_refs=obj.evidence_refs,
                first_seen_ref=obj.first_seen_ref,
                last_seen_ref=obj.last_seen_ref,
            )
        )
    return merged


def mine_pois(
    objects: List[ObjectRecordV2],
    bg_colors: List[int],
    motion_points: List[Tuple[int, int]],
    cfg: AnalystConfigV2,
    episode_id: str,
    step_idx: int,
    grid_width: int,
    grid_height: int,
) -> List[CandidatePOIV2]:
    poi_list: List[CandidatePOIV2] = []
    elongated = [o for o in objects if o.aspect_ratio >= 3.0 or o.aspect_ratio <= 0.33]
    merged_elongated = _merge_overlapping_elongated(elongated)
    obj_pool = [o for o in objects if o not in elongated] + merged_elongated
    motion_set = set(motion_points)
    emitted: List[CandidatePOIV2] = []
    for obj in obj_pool:
        if obj.color in bg_colors:
            continue
        rejection_reasons: List[str] = []
        demotion_reasons: List[str] = []
        base_confidence = 0.45
        type_confidence = 0.5
        utility_confidence = 0.5
        if _is_border_elongated(obj, grid_width, grid_height):
            rejection_reasons.append("border_elongated")
            base_confidence *= 0.4
            type_confidence *= 0.5
        if obj.object_class == "hud_like":
            rejection_reasons.append("likely_hud")
            base_confidence *= 0.3
            type_confidence *= 0.4
        if obj.area <= 2:
            rejection_reasons.append("tiny_fragment")
            base_confidence *= 0.4
            utility_confidence *= 0.5
        overlap_motion = any((x, y) in motion_set for x in range(obj.bbox.x1, obj.bbox.x2 + 1) for y in range(obj.bbox.y1, obj.bbox.y2 + 1))
        if not overlap_motion:
            demotion_reasons.append("static_decorative")
            utility_confidence *= 0.7
            base_confidence *= 0.8
        for existing in emitted:
            if existing.object_class == obj.object_class and existing.bbox and bbox_iou(existing.bbox, obj.bbox) > 0.6:
                demotion_reasons.append("duplicate_overlap")
                base_confidence *= 0.6
                utility_confidence *= 0.7
                break
        candidate = CandidatePOIV2(
            schema_version=SCHEMA_VERSION,
            poi_id=f"poi:{episode_id}:{step_idx}:{obj.object_id}",
            game_id=obj.game_id,
            source_type="color_component",
            bbox=obj.bbox,
            centroid=obj.centroid,
            object_class=obj.object_class,
            reachable_now="uncertain",
            confidence=min(1.0, max(0.05, base_confidence)),
            expected_information_gain=min(1.0, utility_confidence + 0.1),
            expected_interaction_type="unknown",
            evidence_count=1,
            first_seen_ref=f"{episode_id}:{step_idx}",
            last_seen_ref=f"{episode_id}:{step_idx}",
            type_confidence=type_confidence,
            utility_confidence=utility_confidence,
            rejection_reasons=rejection_reasons,
            demotion_reasons=demotion_reasons,
        )
        poi_list.append(candidate)
        emitted.append(candidate)
    if motion_points:
        bbox = bbox_from_points(motion_points)
        if bbox is not None:
            poi_list.append(
                CandidatePOIV2(
                    schema_version=SCHEMA_VERSION,
                    poi_id=f"poi_motion:{episode_id}:{step_idx}",
                    game_id=objects[0].game_id if objects else "unknown_game",
                    source_type="motion_hotspot",
                    bbox=bbox,
                    centroid=bbox.centroid(),
                    object_class="unknown",
                    reachable_now="uncertain",
                    confidence=0.5,
                    expected_information_gain=0.6,
                    expected_interaction_type="probe",
                    evidence_count=1,
                    first_seen_ref=f"{episode_id}:{step_idx}",
                    last_seen_ref=f"{episode_id}:{step_idx}",
                    type_confidence=0.5,
                    utility_confidence=0.6,
                    rejection_reasons=[],
                    demotion_reasons=[],
                )
            )
    return poi_list
