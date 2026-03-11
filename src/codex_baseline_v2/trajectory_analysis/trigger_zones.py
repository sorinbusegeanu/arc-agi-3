from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import HiddenTriggerConfigV2
from codex_baseline_v2.shared.schemas import BlackboardStateV2, TrajectoryEpisodeV2, TriggerZoneV2, SCHEMA_VERSION
from codex_baseline_v2.shared.utils import BBox


def _bbox_from_cells(cells: List[Tuple[int, int]]) -> Optional[BBox]:
    if not cells:
        return None
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def build_trigger_zone_candidates(episodes: List[TrajectoryEpisodeV2], blackboard: Optional[BlackboardStateV2], cfg: HiddenTriggerConfigV2) -> list[TriggerZoneV2]:
    game_id = blackboard.game_id if blackboard is not None else (episodes[0].game_id if episodes else "unknown_game")
    existing_zone_ids = {zone.trigger_zone_id for zone in (blackboard.trigger_zone_table if blackboard is not None else [])}
    zone_cells: Dict[str, set[Tuple[int, int]]] = defaultdict(set)
    zone_meta: Dict[str, Dict[str, object]] = {}
    for poi in (blackboard.poi_table if blackboard is not None else []):
        zone_id = f"trigger_zone:poi:{poi.poi_id}"
        if zone_id in existing_zone_ids:
            continue
        cells = [
            (x, y)
            for x in range(poi.bbox.x1, poi.bbox.x2 + 1)
            for y in range(poi.bbox.y1, poi.bbox.y2 + 1)
        ]
        zone_cells[zone_id].update(cells)
        zone_meta[zone_id] = {
            "area_id": poi.area_id,
            "source_kind": "visible_poi",
            "condition_type": "contact_or_overlap",
            "anchor_poi_id": poi.poi_id,
            "evidence_refs": [poi.first_seen_ref] if poi.first_seen_ref else [],
        }
    field_counts: Dict[Tuple[Optional[str], Tuple[int, int]], Dict[str, int]] = defaultdict(lambda: {"visit": 0, "dwell": 0, "cross": 0, "activation": 0, "null": 0})
    per_action: Dict[Tuple[Optional[str], Tuple[int, int]], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for episode in episodes:
        prev_cell = None
        streak = 0
        for step in episode.steps:
            cell = None
            if step.actual_avatar_centroid is not None:
                cell = (int(round(step.actual_avatar_centroid[0])), int(round(step.actual_avatar_centroid[1])))
            elif step.predicted_avatar_centroid is not None:
                cell = (int(round(step.predicted_avatar_centroid[0])), int(round(step.predicted_avatar_centroid[1])))
            if cell is None:
                continue
            key = (step.area_id, cell)
            field_counts[key]["visit"] += 1
            if step.action.action_id is not None:
                per_action[key][int(step.action.action_id)] += 1
            if prev_cell == cell:
                streak += 1
                field_counts[key]["dwell"] += 1
            else:
                streak = 1
            if prev_cell is not None and prev_cell != cell:
                field_counts[key]["cross"] += 1
            if step.event_ids:
                field_counts[key]["activation"] += len(step.event_ids)
            elif step.intervention_id:
                field_counts[key]["null"] += 1
            prev_cell = cell
    zones: List[TriggerZoneV2] = []
    for (area_id, cell), counts in field_counts.items():
        if counts["visit"] < cfg.min_region_visits_for_candidate and counts["cross"] < cfg.min_boundary_crossings_for_candidate:
            continue
        zone_id = f"trigger_zone:cell:{area_id or 'none'}:{cell[0]}:{cell[1]}"
        cells = [cell]
        source_kind = "floor_region" if counts["dwell"] >= 1 else "boundary_edge" if counts["cross"] >= cfg.min_boundary_crossings_for_candidate else "action_region"
        condition_type = "dwell" if counts["dwell"] >= max(cfg.dwell_step_thresholds) else "cross" if counts["cross"] >= cfg.min_boundary_crossings_for_candidate else "step_on"
        zones.append(
            TriggerZoneV2(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                trigger_zone_id=zone_id,
                area_id=area_id,
                source_kind=source_kind,
                condition_type=condition_type,
                cells=cells,
                bbox=_bbox_from_cells(cells),
                anchor_poi_id=None,
                entry_count=counts["visit"],
                dwell_count=counts["dwell"],
                crossing_count=counts["cross"],
                per_action_counts=sorted((action_id, count) for action_id, count in per_action[(area_id, cell)].items()),
                activation_count=counts["activation"],
                null_count=counts["null"],
                contradiction_count=max(0, counts["null"] - counts["activation"]),
                last_triggered_round=blackboard.round_id if blackboard is not None else None,
                hidden_trigger_confidence=min(1.0, (counts["activation"] + 0.5 * counts["cross"]) / float(max(1, counts["visit"] + counts["null"]))),
                evidence_refs=[],
            )
        )
    for zone_id, cells in zone_cells.items():
        meta = zone_meta[zone_id]
        zones.append(
            TriggerZoneV2(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                trigger_zone_id=zone_id,
                area_id=meta["area_id"],  # type: ignore[index]
                source_kind=str(meta["source_kind"]),
                condition_type=str(meta["condition_type"]),
                cells=sorted(cells),
                bbox=_bbox_from_cells(sorted(cells)),
                anchor_poi_id=meta["anchor_poi_id"],  # type: ignore[index]
                entry_count=0,
                dwell_count=0,
                crossing_count=0,
                per_action_counts=[],
                activation_count=0,
                null_count=0,
                contradiction_count=0,
                last_triggered_round=blackboard.round_id if blackboard is not None else None,
                hidden_trigger_confidence=0.2,
                evidence_refs=list(meta["evidence_refs"]),  # type: ignore[arg-type]
            )
        )
    return zones


def merge_trigger_zones(existing_zones: List[TriggerZoneV2], new_zones: List[TriggerZoneV2], cfg: HiddenTriggerConfigV2) -> list[TriggerZoneV2]:
    merged: Dict[str, TriggerZoneV2] = {zone.trigger_zone_id: zone for zone in existing_zones}
    for zone in new_zones:
        prev = merged.get(zone.trigger_zone_id)
        if prev is None:
            merged[zone.trigger_zone_id] = zone
            continue
        action_counts = defaultdict(int)
        for action_id, count in prev.per_action_counts + zone.per_action_counts:
            action_counts[int(action_id)] += int(count)
        cells = sorted(set(prev.cells) | set(zone.cells))
        merged[zone.trigger_zone_id] = TriggerZoneV2(
            schema_version=SCHEMA_VERSION,
            game_id=zone.game_id,
            trigger_zone_id=zone.trigger_zone_id,
            area_id=zone.area_id or prev.area_id,
            source_kind=zone.source_kind if zone.source_kind != "unknown_hidden" else prev.source_kind,
            condition_type=zone.condition_type if zone.condition_type != "unknown" else prev.condition_type,
            cells=cells,
            bbox=_bbox_from_cells(cells),
            anchor_poi_id=zone.anchor_poi_id or prev.anchor_poi_id,
            entry_count=prev.entry_count + zone.entry_count,
            dwell_count=prev.dwell_count + zone.dwell_count,
            crossing_count=prev.crossing_count + zone.crossing_count,
            per_action_counts=sorted(action_counts.items()),
            activation_count=prev.activation_count + zone.activation_count,
            null_count=prev.null_count + zone.null_count,
            contradiction_count=prev.contradiction_count + zone.contradiction_count,
            last_triggered_round=zone.last_triggered_round if zone.last_triggered_round is not None else prev.last_triggered_round,
            hidden_trigger_confidence=max(prev.hidden_trigger_confidence, zone.hidden_trigger_confidence),
            evidence_refs=sorted(set(prev.evidence_refs) | set(zone.evidence_refs)),
        )
    return list(merged.values())
