from __future__ import annotations

from collections import Counter

from v3_1.analysis.adapters_env import normalize_observation
from v3_1.analysis.mechanic_graph_extraction import extract_mechanic_graph_delta
from v3_1.analysis.pattern_identity import patch_crop, stable_descriptor, stable_pattern_id
from v3_1.analysis.area_assignment import assign_area
from v3_1.analysis.avatar_tracking import track_avatar
from v3_1.analysis.motion_analysis import summarize_motion
from v3_1.analysis.observation_summary import summarize_observation
from v3_1.analysis.poi_detection import detect_pois
from v3_1.contracts.messages import AnalyzedEpisode, BlackboardDelta, RawEpisode
from v3_1.execution.env_factory import normalize_action_lookup
from v3_1.mechanics.hypothesis_orchestrator import orchestrate_hypotheses
from v3_1.utils.ids import stable_digest


def _degenerate_avatar_path(avatar_tracking: dict) -> bool:
    cells = []
    for row in avatar_tracking.get("per_step", []):
        centroid = row.get("main_centroid")
        if not isinstance(centroid, list) or len(centroid) != 2:
            continue
        cells.append((int(float(centroid[0])), int(float(centroid[1]))))
    return len(set(cells)) <= 1


def _fallback_avatar_tracking(step_summaries: list[dict], avatar_tracking: dict) -> dict:
    if not _degenerate_avatar_path(avatar_tracking):
        return avatar_tracking
    previous = None
    patched = []
    for row, summary in zip(avatar_tracking.get("per_step", []), step_summaries):
        regions = list(summary.get("active_regions", []))
        chosen = None
        if regions:
            regions.sort(
                key=lambda region: (
                    abs(float(region.get("area", 0))),
                    abs(float(region.get("centroid", [0.0, 0.0])[0]) - float(previous[0])) + abs(float(region.get("centroid", [0.0, 0.0])[1]) - float(previous[1])) if previous is not None else 0.0,
                )
            )
            chosen = list(regions[0].get("centroid", [])) if isinstance(regions[0].get("centroid"), list) else None
        if chosen is None:
            chosen = row.get("main_centroid")
        if chosen is None and previous is not None:
            chosen = list(previous)
        if isinstance(chosen, list) and len(chosen) == 2:
            previous = [float(chosen[0]), float(chosen[1])]
        patched.append({**row, "main_centroid": list(previous) if previous is not None else None})
    return {**avatar_tracking, "per_step": patched, "tracking_source": "change_region_fallback"}


def _topology_action_key(step_row: dict) -> str:
    action_family = str(step_row.get("action_family") or "").strip().lower()
    if action_family and action_family != "unknown":
        return action_family
    action_name = str(step_row.get("action_name") or "").strip().lower()
    if action_name:
        return action_name
    return "unknown"


def _topology_from_steps(step_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str, str, str]] = Counter()
    node_cells: dict[str, list[int]] = {}
    edge_evidence_refs: dict[tuple[str, str, str, str], set[str]] = {}
    previous_node_id = None
    for step in step_rows:
        cell = step.get("avatar_cell")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            previous_node_id = None
            continue
        cell = [int(float(cell[0])), int(float(cell[1]))]
        node_id = f"cell:{cell[0]}:{cell[1]}"
        node_counts[node_id] += 1
        node_cells[node_id] = cell
        if previous_node_id is not None:
            action_key = _topology_action_key(step)
            transition_type = "move" if str(step.get("action_family") or "").strip().lower() == "move" else "action"
            edge_key = (previous_node_id, node_id, action_key, transition_type)
            edge_counts[edge_key] += 1
            edge_evidence_refs.setdefault(edge_key, set()).add(f"step:{int(step.get('step_idx', 0) or 0)}")
        previous_node_id = node_id
    nodes = [{"node_id": node_id, "cell": cell, "visits": count} for node_id, count in node_counts.items() for cell in [node_cells[node_id]]]
    edges = [
        {
            "edge_id": f"{src}->{dst}:{action_key}:{transition_type}",
            "src": src,
            "dst": dst,
            "action_key": action_key,
            "transition_type": transition_type,
            "success_count": count,
            "blocked_count": 0,
            "uncertain_count": 0,
            "evidence_refs": sorted(edge_evidence_refs.get((src, dst, action_key, transition_type), set())),
        }
        for (src, dst, action_key, transition_type), count in edge_counts.items()
    ]
    nodes.sort(key=lambda row: row["node_id"])
    edges.sort(key=lambda row: row["edge_id"])
    return nodes, edges


def _consequences(raw_episode: RawEpisode, motion: dict, step_summaries: list[dict], step_rows: list[dict]) -> list[dict]:
    rows = []
    by_step = {row.get("step_idx"): row for row in step_rows}
    for step_idx, movement_row in enumerate(motion["movement_rows"]):
        if movement_row["local_change_area"] <= 0 and not raw_episode.steps[step_idx].done and not raw_episode.steps[step_idx].reward:
            continue
        step = by_step.get(step_idx, {})
        rows.append(
            {
                "consequence_id": f"consequence:{raw_episode.episode_id}:{step_idx}",
                "step_idx": step_idx,
                "action": step.get("action_name"),
                "action_id": step.get("action_id"),
                "action_name": step.get("action_name"),
                "action_family": step.get("action_family", "unknown"),
                "reward": raw_episode.steps[step_idx].reward,
                "done": raw_episode.steps[step_idx].done,
                "local_change_area": movement_row["local_change_area"],
                "blocked": movement_row["blocked"],
                "action_effect_near_avatar": movement_row["action_effect_near_avatar"],
                "evidence_count": max(1, len(step_summaries[step_idx].get("change_regions", []))),
                "evidence_refs": [f"{raw_episode.episode_id}:{step_idx}"],
            }
        )
    return rows


def _validate_analysis_mode(analysis_mode: str) -> str:
    normalized = str(analysis_mode or "").strip().lower()
    if normalized not in {"probe", "directed_outcome"}:
        raise ValueError(f"invalid analysis_mode: {analysis_mode!r}")
    return normalized


def _analysis_priorities(*, analysis_mode: str) -> dict:
    if analysis_mode == "probe":
        return {
            "entity_recall_weight": 1.0,
            "poi_recall_weight": 1.0,
            "trigger_suspicion_weight": 0.75,
            "topology_growth_weight": 1.0,
            "outcome_attribution_weight": 0.2,
            "route_progress_weight": 0.25,
            "localized_effect_weight": 0.2,
        }
    return {
        "entity_recall_weight": 0.65,
        "poi_recall_weight": 0.55,
        "trigger_suspicion_weight": 0.45,
        "topology_growth_weight": 0.5,
        "outcome_attribution_weight": 1.0,
        "route_progress_weight": 1.0,
        "localized_effect_weight": 1.0,
    }


def _mode_select_pois(pois: list[dict], *, analysis_mode: str) -> list[dict]:
    rows = [dict(row) for row in pois]
    if analysis_mode == "probe":
        rows.sort(
            key=lambda row: (
                -float(row.get("novelty", 0.0)),
                -float(row.get("confidence", 0.0)),
                -int(row.get("observations", 0) or 0),
                str(row.get("poi_id", "")),
            )
        )
        return rows
    filtered = []
    for row in rows:
        interaction_attempts = int(row.get("interaction_attempts", 0) or 0)
        effect_score = float(row.get("interaction_effect_score", 0.0) or 0.0)
        if interaction_attempts > 0 or effect_score > 0.0 or float(row.get("distance_score", 0.0) or 0.0) >= 0.2:
            filtered.append(row)
    filtered.sort(
        key=lambda row: (
            -float(row.get("interaction_effect_score", 0.0)),
            -float(row.get("distance_score", 0.0)),
            -float(row.get("confidence", 0.0)),
            str(row.get("poi_id", "")),
        )
    )
    return filtered or rows


def _mode_select_consequences(rows: list[dict], *, analysis_mode: str) -> list[dict]:
    payload = [dict(row) for row in rows]
    if analysis_mode == "probe":
        return payload
    selected = []
    for row in payload:
        if bool(row.get("done")) or float(row.get("reward", 0.0) or 0.0) != 0.0:
            selected.append(row)
            continue
        if bool(row.get("action_effect_near_avatar")) or int(row.get("local_change_area", 0) or 0) > 0:
            selected.append(row)
    return selected or payload


def _mode_select_topology(nodes: list[dict], edges: list[dict], *, analysis_mode: str) -> tuple[list[dict], list[dict]]:
    if analysis_mode == "probe":
        return [dict(row) for row in nodes], [dict(row) for row in edges]
    directed_edges = []
    for row in edges:
        payload = dict(row)
        if str(payload.get("transition_type") or "") == "move" or int(payload.get("success_count", 0) or 0) > 0:
            directed_edges.append(payload)
    node_ids = {str(row.get("src")) for row in directed_edges} | {str(row.get("dst")) for row in directed_edges}
    directed_nodes = [dict(row) for row in nodes if str(row.get("node_id")) in node_ids]
    return directed_nodes or [dict(row) for row in nodes], directed_edges or [dict(row) for row in edges]


def _extract_trigger_zones(*, step_summaries: list[dict], step_rows: list[dict], analysis_mode: str) -> list[dict]:
    rows: list[dict] = []

    def _bbox_list(payload: object) -> list[int] | None:
        if isinstance(payload, list) and len(payload) == 4:
            return [int(value) for value in payload]
        if isinstance(payload, dict):
            return [
                int(payload.get("x1", 0) or 0),
                int(payload.get("y1", 0) or 0),
                int(payload.get("x2", 0) or 0),
                int(payload.get("y2", 0) or 0),
            ]
        return None

    def _bbox_area(bbox: list[int]) -> int:
        return max(0, (int(bbox[2]) - int(bbox[0]) + 1) * (int(bbox[3]) - int(bbox[1]) + 1))

    def _bbox_overlap(left: list[int], right: list[int]) -> int:
        x1 = max(int(left[0]), int(right[0]))
        y1 = max(int(left[1]), int(right[1]))
        x2 = min(int(left[2]), int(right[2]))
        y2 = min(int(left[3]), int(right[3]))
        if x2 < x1 or y2 < y1:
            return 0
        return (x2 - x1 + 1) * (y2 - y1 + 1)

    if analysis_mode == "probe":
        for summary, step in zip(step_summaries, step_rows):
            structure_candidates = [dict(row) for row in list(summary.get("structure_candidates", []) or [])]
            for region_index, region in enumerate(list(summary.get("change_regions", []) or [])):
                bbox = _bbox_list(region.get("bbox"))
                if not bbox or len(bbox) != 4:
                    continue
                region_area = int(region.get("area", 0) or 0)
                if region_area < 2 or region_area > 36:
                    continue
                best_structure = {}
                best_overlap = 0
                for candidate in structure_candidates:
                    candidate_bbox = _bbox_list(candidate.get("bbox"))
                    if not candidate_bbox:
                        continue
                    overlap = _bbox_overlap(bbox, candidate_bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_structure = candidate
                overlap_ratio = best_overlap / float(max(1, _bbox_area(bbox)))
                structure_backed = bool(best_structure) and overlap_ratio >= 0.35
                stable_zone_id = f"trigger:probe_region:{stable_digest((step.get('area_id'), tuple(bbox)))}"
                structure_entity_id = str(
                    best_structure.get("stable_entity_id_hint")
                    or best_structure.get("matched_prior_id")
                    or ""
                )
                if structure_entity_id.startswith("object:"):
                    structure_entity_id = ""
                rows.append(
                    {
                        "trigger_id": stable_zone_id,
                        "entity_id": structure_entity_id if structure_backed else None,
                        "area_id": step.get("area_id"),
                        "zone_bbox": bbox,
                        "supporting_steps": [int(step.get("step_idx", 0) or 0)],
                        "confidence": min(
                            0.9,
                            0.18
                            + (0.015 * region_area)
                            + (0.22 * float(best_structure.get("identity_confidence", 0.0) or 0.0) if structure_backed else 0.0),
                        ),
                        "evidence_refs": [f"{step.get('step_idx', 0)}:{region_index}"],
                        "trigger_kind": "structure_overlap_region" if structure_backed else "suspicious_region",
                        "region_backed": not structure_backed,
                        "trigger_evidence_class": "object_backed_candidate" if structure_backed else "region_suspicion",
                    }
                )
        return rows
    for summary, step in zip(step_summaries, step_rows):
        telemetry = dict(step.get("telemetry", {}) or {})
        effect_region = dict(telemetry.get("effect_region", {}) or {})
        bbox = _bbox_list(effect_region.get("bbox"))
        if not bbox or len(bbox) != 4:
            target_coordinates = step.get("target_coordinates")
            if isinstance(target_coordinates, (list, tuple)) and len(target_coordinates) == 2:
                x, y = int(target_coordinates[0]), int(target_coordinates[1])
                bbox = [x, y, x, y]
        if not bbox or len(bbox) != 4:
            continue
        stable_zone_id = f"trigger:region:{stable_digest((step.get('area_id'), tuple(bbox)))}"
        rows.append(
            {
                "trigger_id": stable_zone_id,
                "entity_id": step.get("target_entity_id"),
                "area_id": step.get("area_id"),
                "zone_bbox": bbox,
                "supporting_steps": [int(step.get("step_idx", 0) or 0)],
                "confidence": min(1.0, 0.35 + (0.01 * int(step.get("changed_cells", 0) or 0))),
                "evidence_refs": [f"{step.get('step_idx', 0)}:directed"],
                "trigger_kind": "localized_attribution",
                "effect_changed_cells": int(step.get("changed_cells", 0) or 0),
                "region_backed": True,
                "trigger_evidence_class": "localized_effect_region",
            }
        )
    return rows


def _poi_direct_evidence_fields(poi: dict) -> list[str]:
    fields: list[str] = []
    if poi.get("bbox"):
        fields.append("bbox")
    if poi.get("centroid"):
        fields.append("centroid")
    if poi.get("observations") is not None:
        fields.append("observations")
    if poi.get("signature"):
        fields.append("signature")
    return fields


def _poi_is_directly_observed(*, poi: dict, analysis_mode: str) -> bool:
    if bool(poi.get("rejected")):
        return False
    if not _poi_direct_evidence_fields(poi):
        return False
    if analysis_mode == "probe":
        return True
    return bool(
        int(poi.get("interaction_attempts", 0) or 0) > 0
        or float(poi.get("interaction_effect_score", 0.0) or 0.0) > 0.0
        or int(poi.get("observations", 0) or 0) > 0
    )


def _consequence_direct_evidence_fields(row: dict) -> list[str]:
    fields: list[str] = []
    if row.get("action_name"):
        fields.append("action_name")
    if row.get("action_family"):
        fields.append("action_family")
    if row.get("evidence_refs"):
        fields.append("evidence_refs")
    if row.get("reward") not in {None, ""}:
        fields.append("reward")
    if row.get("done") is not None:
        fields.append("done")
    if int(row.get("local_change_area", 0) or 0) > 0:
        fields.append("local_change_area")
    return fields


def _consequence_is_directly_observed(*, row: dict, analysis_mode: str) -> bool:
    if not row.get("evidence_refs"):
        return False
    if analysis_mode == "probe":
        return bool(
            str(row.get("action_family") or "") == "move"
            and int(row.get("local_change_area", 0) or 0) > 0
        )
    if analysis_mode != "directed_outcome":
        return False
    return bool(
        row.get("done") is not None
        or row.get("reward") not in {None, ""}
        or int(row.get("local_change_area", 0) or 0) > 0
    )


def _topology_node_direct_evidence_fields(row: dict) -> list[str]:
    fields: list[str] = []
    if row.get("node_id"):
        fields.append("node_id")
    if row.get("cell"):
        fields.append("cell")
    if row.get("visits") is not None:
        fields.append("visits")
    return fields


def _topology_edge_direct_evidence_fields(row: dict) -> list[str]:
    fields: list[str] = []
    for key in ("edge_id", "src", "dst", "action_key", "evidence_refs"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            continue
        if (not isinstance(value, (list, tuple, dict))) or len(value) > 0:
            fields.append(key)
    return fields


def _analysis_objective(*, analysis_mode: str, row_kind: str) -> str:
    if analysis_mode == "probe":
        return "discovery" if row_kind in {"areas", "entities", "topology_nodes", "topology_edges"} else "broad_trigger_suspicion"
    return "terminal_attribution" if row_kind in {"consequences", "entities"} else "route_progress_attribution"


def _stamp_row(
    row: dict,
    *,
    analysis_mode: str,
    row_kind: str,
    direct_evidence_present: bool,
    direct_evidence_fields: list[str],
    contradiction_flag: bool = False,
    observation_support_span: tuple[int, int] | None = None,
) -> dict:
    payload = dict(row)
    payload["direct_evidence_present"] = bool(direct_evidence_present)
    payload["direct_evidence_fields"] = list(direct_evidence_fields)
    payload["contradiction_flag"] = bool(contradiction_flag)
    payload["observation_support_span"] = list(observation_support_span or (0, 0))
    payload["analysis_objective"] = _analysis_objective(analysis_mode=analysis_mode, row_kind=row_kind)
    return payload


def _annotate_pattern_descriptors(pois: list[dict], *, step_summaries: list[dict], normalized_observations: list[list[list[int]]]) -> list[dict]:
    rows = []
    for poi in list(pois or []):
        payload = dict(poi)
        bbox_payload = payload.get("bbox")
        bbox = None
        if isinstance(bbox_payload, list) and len(bbox_payload) == 4:
            bbox = [int(value) for value in bbox_payload]
        elif isinstance(bbox_payload, dict):
            bbox = [
                int(bbox_payload.get("x1", 0) or 0),
                int(bbox_payload.get("y1", 0) or 0),
                int(bbox_payload.get("x2", 0) or 0),
                int(bbox_payload.get("y2", 0) or 0),
            ]
        observation = normalized_observations[-1] if normalized_observations else []
        if bbox and observation:
            patch = patch_crop(observation, bbox)
            payload["pattern_descriptor"] = stable_descriptor(patch)
            payload["pattern_id"] = stable_pattern_id(patch) if patch else None
        else:
            payload["pattern_descriptor"] = {}
            payload["pattern_id"] = None
        rows.append(payload)
    return rows


def _attach_identity_to_pois(pois: list[dict], *, step_summaries: list[dict]) -> list[dict]:
    rows = []
    signature_objects: dict[str, list[dict]] = {}
    for summary in list(step_summaries or []):
        for obj in list(summary.get("objects", []) or []):
            signature = str(obj.get("signature") or "")
            if signature:
                signature_objects.setdefault(signature, []).append(dict(obj))
    for poi in list(pois or []):
        payload = dict(poi)
        matched_objects = list(signature_objects.get(str(payload.get("signature") or ""), []))
        best = max(
            matched_objects,
            key=lambda row: (
                float(row.get("identity_confidence", 0.0) or 0.0),
                -abs(float(row.get("centroid", [0.0, 0.0])[0]) - float(payload.get("centroid", [0.0, 0.0])[0])) if isinstance(row.get("centroid"), (list, tuple)) and isinstance(payload.get("centroid"), (list, tuple)) else 0.0,
            ),
            default={},
        )
        payload["identity_confidence"] = float(best.get("identity_confidence", 0.0) or 0.0)
        payload["identity_status"] = str(best.get("identity_status") or "unknown")
        payload["candidate_prior_ids"] = list(best.get("candidate_prior_ids", []) or [])
        if best.get("matched_prior_id"):
            payload["matched_prior_id"] = best.get("matched_prior_id")
        rows.append(payload)
    return rows


def _attach_target_effects_to_pois(pois: list[dict], *, step_rows: list[dict], blackboard_snapshot: dict | None) -> list[dict]:
    blackboard_entities = dict((blackboard_snapshot or {}).get("state", {}).get("entities", {})) if isinstance((blackboard_snapshot or {}).get("state"), dict) else dict((blackboard_snapshot or {}).get("entities", {}))
    if not pois or not step_rows:
        return [dict(poi) for poi in pois]

    attributed = [dict(poi) for poi in pois]
    by_target: dict[str, dict[str, int | float | str]] = {}
    poi_by_index = {index: row for index, row in enumerate(attributed)}

    def _match_poi_index(target_id: str) -> int | None:
        for index, poi in poi_by_index.items():
            if str(poi.get("entity_id") or "") == target_id:
                return index
            if str(poi.get("stable_entity_id_hint") or "") == target_id:
                return index
        target = dict(blackboard_entities.get(target_id, {}))
        if not target:
            return None
        target_signature = str(target.get("signature") or "")
        target_centroid = target.get("centroid")
        best_index = None
        best_score = -1.0
        for index, poi in poi_by_index.items():
            score = 0.0
            if target_signature and str(poi.get("signature") or "") == target_signature:
                score += 10.0
            poi_centroid = poi.get("centroid")
            if isinstance(target_centroid, (list, tuple)) and isinstance(poi_centroid, (list, tuple)) and len(target_centroid) == 2 and len(poi_centroid) == 2:
                distance = abs(float(target_centroid[0]) - float(poi_centroid[0])) + abs(float(target_centroid[1]) - float(poi_centroid[1]))
                score += max(0.0, 5.0 - min(5.0, distance / 4.0))
            if score > best_score:
                best_score = score
                best_index = index
        return best_index if best_score > 0.0 else None

    target_to_poi_index: dict[str, int] = {}
    for row in step_rows:
        target_id = row.get("target_entity_id")
        if not target_id:
            continue
        target_id = str(target_id)
        if target_id not in target_to_poi_index:
            match_index = _match_poi_index(target_id)
            if match_index is not None:
                target_to_poi_index[target_id] = match_index
        if target_id not in target_to_poi_index:
            continue
        family = str(row.get("action_family") or "unknown")
        changed_cells = int(row.get("changed_cells", 0) or 0)
        current = by_target.setdefault(
            target_id,
            {
                "movement_attempts": 0,
                "interact_attempts": 0,
                "click_attempts": 0,
                "movement_effect_sum": 0,
                "interact_effect_sum": 0,
                "click_effect_sum": 0,
            },
        )
        if family == "move":
            current["movement_attempts"] = int(current["movement_attempts"]) + 1
            current["movement_effect_sum"] = int(current["movement_effect_sum"]) + changed_cells
        elif family == "interact":
            current["interact_attempts"] = int(current["interact_attempts"]) + 1
            current["interact_effect_sum"] = int(current["interact_effect_sum"]) + changed_cells
        elif family == "click_at":
            current["click_attempts"] = int(current["click_attempts"]) + 1
            current["click_effect_sum"] = int(current["click_effect_sum"]) + changed_cells

    for target_id, stats in by_target.items():
        poi_index = target_to_poi_index.get(target_id)
        if poi_index is None:
            continue
        poi = attributed[poi_index]
        movement_attempts = int(stats["movement_attempts"])
        interact_attempts = int(stats["interact_attempts"])
        click_attempts = int(stats["click_attempts"])
        movement_effect_sum = int(stats["movement_effect_sum"])
        interact_effect_sum = int(stats["interact_effect_sum"])
        click_effect_sum = int(stats["click_effect_sum"])
        movement_effect_score = min(1.0, (movement_effect_sum / float(movement_attempts) / 50.0) if movement_attempts > 0 else 0.0)
        interact_effect_score = min(1.0, (interact_effect_sum / float(interact_attempts) / 50.0) if interact_attempts > 0 else 0.0)
        click_effect_score = min(1.0, (click_effect_sum / float(click_attempts) / 50.0) if click_attempts > 0 else 0.0)
        if interact_attempts > 0:
            candidate_effect_mode = "interact"
            candidate_effect_score = interact_effect_score
        elif click_attempts > 0:
            candidate_effect_mode = "click_at"
            candidate_effect_score = click_effect_score
        else:
            candidate_effect_mode = "move"
            candidate_effect_score = movement_effect_score
        poi["stable_entity_id_hint"] = target_id
        poi["movement_attempts"] = max(int(poi.get("movement_attempts", 0) or 0), movement_attempts)
        poi["interact_attempts"] = max(int(poi.get("interact_attempts", 0) or 0), interact_attempts)
        poi["click_attempts"] = max(int(poi.get("click_attempts", 0) or 0), click_attempts)
        poi["movement_effect_sum"] = max(int(poi.get("movement_effect_sum", 0) or 0), movement_effect_sum)
        poi["interact_effect_sum"] = max(int(poi.get("interact_effect_sum", 0) or 0), interact_effect_sum)
        poi["click_effect_sum"] = max(int(poi.get("click_effect_sum", 0) or 0), click_effect_sum)
        poi["movement_effect_score"] = max(float(poi.get("movement_effect_score", 0.0) or 0.0), movement_effect_score)
        poi["interact_effect_score"] = max(float(poi.get("interact_effect_score", 0.0) or 0.0), interact_effect_score)
        poi["click_effect_score"] = max(float(poi.get("click_effect_score", 0.0) or 0.0), click_effect_score)
        if float(candidate_effect_score) >= float(poi.get("candidate_effect_score", 0.0) or 0.0):
            poi["candidate_effect_mode"] = candidate_effect_mode
            poi["candidate_effect_score"] = float(candidate_effect_score)
    return attributed


def _structure_entity_id(candidate: dict, *, area_id: str | None) -> str:
    matched_prior_id = str(candidate.get("matched_prior_id") or "")
    if matched_prior_id and not matched_prior_id.startswith("object:"):
        return matched_prior_id
    stable_key = {
        "area_id": area_id or "none",
        "signature": str(candidate.get("signature") or ""),
        "kind": str(candidate.get("kind") or "structure"),
        "primary_color": int(candidate.get("primary_color", 0) or 0),
        "bbox": dict(candidate.get("bbox", {}) or {}),
    }
    return f"entity:{stable_digest(stable_key)}"


def _supplemental_structure_entities(*, step_summaries: list[dict], normalized_observations: list[list[list[int]]], area_sequence: list[dict]) -> list[dict]:
    supplemental: list[dict] = []
    aggregated: dict[str, dict] = {}
    for step_idx, summary in enumerate(list(step_summaries or [])):
        area_id = area_sequence[step_idx]["area_id"] if step_idx < len(area_sequence) else None
        for candidate in list(summary.get("structure_candidates", []) or []):
            candidate_score = float(candidate.get("score", 0.0) or 0.0)
            candidate_identity = float(candidate.get("identity_confidence", 0.0) or 0.0)
            candidate_pattern_id = str(candidate.get("pattern_id") or "")
            candidate_area = int(candidate.get("area", 0) or 0)
            if candidate_score < 0.62:
                continue
            if candidate_area <= 2 or candidate_area > 18:
                continue
            if candidate_identity < 0.55 and not candidate_pattern_id:
                continue
            bbox_payload = dict(candidate.get("bbox", {}) or {})
            bbox = {
                "x1": int(bbox_payload.get("x1", 0) or 0),
                "y1": int(bbox_payload.get("y1", 0) or 0),
                "x2": int(bbox_payload.get("x2", 0) or 0),
                "y2": int(bbox_payload.get("y2", 0) or 0),
            }
            bbox_list = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
            observation = normalized_observations[step_idx] if step_idx < len(normalized_observations) else []
            patch = patch_crop(observation, bbox_list) if observation else []
            descriptor = stable_descriptor(patch) if patch else dict(candidate.get("stable_descriptor", {}) or {})
            pattern_id = stable_pattern_id(patch) if patch else None
            entity_id = _structure_entity_id(candidate, area_id=area_id)
            existing = aggregated.get(entity_id)
            confidence = min(
                1.0,
                0.20
                + (0.32 * candidate_score)
                + (0.30 * candidate_identity)
                + (0.10 if candidate_pattern_id else 0.0),
            )
            payload = {
                "entity_id": entity_id,
                "poi_id": entity_id,
                "kind": "poi",
                "poi_class": "structure",
                "signature": str(candidate.get("signature") or ""),
                "centroid": list(candidate.get("centroid", [0.0, 0.0]) or [0.0, 0.0]),
                "bbox": bbox,
                "area": int(candidate.get("area", 0) or 0),
                "primary_color": int(candidate.get("primary_color", 0) or 0),
                "type_hints": sorted(set(list(candidate.get("type_hints", []) or []) + ["structure_candidate"])),
                "utility": float(candidate.get("score", 0.0) or 0.0),
                "novelty": 0.15,
                "confidence": confidence,
                "observations": 1,
                "demotion_reasons": [],
                "identity_confidence": float(candidate.get("identity_confidence", 0.0) or 0.0),
                "identity_status": str(candidate.get("identity_status") or "unknown"),
                "matched_prior_id": candidate.get("matched_prior_id"),
                "candidate_prior_ids": list(candidate.get("candidate_prior_ids", []) or []),
                "stable_entity_id_hint": entity_id,
                "pattern_descriptor": descriptor,
                "pattern_id": pattern_id,
                "canonical_descriptor": {
                    "signature": str(candidate.get("signature") or ""),
                    "kind": str(candidate.get("kind") or "structure"),
                    "primary_color": int(candidate.get("primary_color", 0) or 0),
                    "bbox_size": [max(1, bbox["x2"] - bbox["x1"] + 1), max(1, bbox["y2"] - bbox["y1"] + 1)],
                },
                "movement_attempts": 0,
                "interact_attempts": 0,
                "click_attempts": 0,
                "movement_effect_sum": 0,
                "interact_effect_sum": 0,
                "click_effect_sum": 0,
                "movement_effect_score": 0.0,
                "interact_effect_score": 0.0,
                "click_effect_score": 0.0,
                "candidate_effect_score": 0.0,
                "candidate_effect_mode": "move",
                "area_id": area_id,
            }
            if existing is None:
                payload["supporting_steps"] = [step_idx]
                aggregated[entity_id] = payload
                continue
            existing["confidence"] = max(float(existing.get("confidence", 0.0) or 0.0), confidence)
            existing["utility"] = max(float(existing.get("utility", 0.0) or 0.0), float(payload.get("utility", 0.0) or 0.0))
            existing["identity_confidence"] = max(float(existing.get("identity_confidence", 0.0) or 0.0), float(payload.get("identity_confidence", 0.0) or 0.0))
            existing["observations"] = int(existing.get("observations", 0) or 0) + 1
            existing["supporting_steps"] = list(dict.fromkeys(list(existing.get("supporting_steps", []) or []) + [step_idx]))
            existing["type_hints"] = sorted(set(list(existing.get("type_hints", []) or []) + list(payload.get("type_hints", []) or [])))
            if pattern_id and not existing.get("pattern_id"):
                existing["pattern_id"] = pattern_id
                existing["pattern_descriptor"] = descriptor

    supplemental = sorted(
        aggregated.values(),
        key=lambda row: (
            -float(row.get("identity_confidence", 0.0) or 0.0),
            -int(row.get("observations", 0) or 0),
            -float(row.get("confidence", 0.0) or 0.0),
            str(row.get("entity_id") or ""),
        ),
    )
    supplemental = [
        row for row in supplemental
        if (
            int(row.get("observations", 0) or 0) >= 2
            or (
                float(row.get("identity_confidence", 0.0) or 0.0) >= 0.72
                and bool(row.get("pattern_id"))
            )
        )
    ]
    return supplemental[:16]


def _merge_entity_candidates(base: list[dict], supplemental: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in list(base or []):
        payload = dict(row)
        key = str(payload.get("entity_id") or payload.get("poi_id") or payload.get("signature") or stable_digest(payload))
        merged[key] = payload
    for row in list(supplemental or []):
        payload = dict(row)
        key = str(payload.get("entity_id") or payload.get("poi_id") or payload.get("signature") or stable_digest(payload))
        existing = merged.get(key)
        if existing is None:
            merged[key] = payload
            continue
        for numeric_key in (
            "confidence",
            "utility",
            "novelty",
            "identity_confidence",
            "movement_effect_score",
            "interact_effect_score",
            "click_effect_score",
            "candidate_effect_score",
        ):
            existing[numeric_key] = max(float(existing.get(numeric_key, 0.0) or 0.0), float(payload.get(numeric_key, 0.0) or 0.0))
        for count_key in (
            "observations",
            "movement_attempts",
            "interact_attempts",
            "click_attempts",
            "movement_effect_sum",
            "interact_effect_sum",
            "click_effect_sum",
        ):
            existing[count_key] = max(int(existing.get(count_key, 0) or 0), int(payload.get(count_key, 0) or 0))
        existing["type_hints"] = sorted(set(list(existing.get("type_hints", []) or []) + list(payload.get("type_hints", []) or [])))
        if not existing.get("pattern_id") and payload.get("pattern_id"):
            existing["pattern_id"] = payload.get("pattern_id")
            existing["pattern_descriptor"] = dict(payload.get("pattern_descriptor", {}) or {})
        if not existing.get("matched_prior_id") and payload.get("matched_prior_id"):
            existing["matched_prior_id"] = payload.get("matched_prior_id")
        if not existing.get("stable_entity_id_hint") and payload.get("stable_entity_id_hint"):
            existing["stable_entity_id_hint"] = payload.get("stable_entity_id_hint")
    rows = list(merged.values())
    rows.sort(key=lambda row: (-float(row.get("confidence", 0.0) or 0.0), -float(row.get("utility", 0.0) or 0.0), str(row.get("entity_id") or "")))
    return rows


def _collapse_trigger_zones(rows: list[dict], *, analysis_mode: str) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in list(rows or []):
        trigger_id = str(row.get("trigger_id") or "")
        if not trigger_id:
            continue
        payload = dict(row)
        current = grouped.get(trigger_id)
        if current is None:
            payload["support_count"] = len(list(payload.get("supporting_steps", []) or []))
            grouped[trigger_id] = payload
            continue
        current["supporting_steps"] = sorted(set(list(current.get("supporting_steps", []) or []) + list(payload.get("supporting_steps", []) or [])))
        current["evidence_refs"] = sorted(set(list(current.get("evidence_refs", []) or []) + list(payload.get("evidence_refs", []) or [])))
        current["support_count"] = len(list(current.get("supporting_steps", []) or []))
        current["confidence"] = max(float(current.get("confidence", 0.0) or 0.0), float(payload.get("confidence", 0.0) or 0.0))
        current["effect_changed_cells"] = int(current.get("effect_changed_cells", 0) or 0) + int(payload.get("effect_changed_cells", 0) or 0)
        current["region_backed"] = bool(current.get("region_backed")) or bool(payload.get("region_backed"))
        if str(payload.get("trigger_evidence_class") or "") in {"object_backed", "object_backed_candidate"}:
            current["trigger_evidence_class"] = str(payload.get("trigger_evidence_class") or "object_backed_candidate")
        elif str(current.get("trigger_evidence_class") or "") != "object_backed":
            current["trigger_evidence_class"] = str(payload.get("trigger_evidence_class") or current.get("trigger_evidence_class") or "region_suspicion")
        if not current.get("entity_id") and payload.get("entity_id"):
            current["entity_id"] = payload.get("entity_id")
    collapsed = list(grouped.values())
    if analysis_mode == "probe":
        collapsed = [
            row for row in collapsed
            if (
                (str(row.get("trigger_evidence_class") or "") == "object_backed_candidate" and int(row.get("support_count", 0) or 0) >= 2)
                or (str(row.get("trigger_evidence_class") or "") == "region_suspicion" and int(row.get("support_count", 0) or 0) >= 3)
                or (str(row.get("trigger_evidence_class") or "") == "object_backed" and int(row.get("support_count", 0) or 0) >= 1)
            )
        ]
        cap = 12
    else:
        collapsed = [
            row for row in collapsed
            if int(row.get("effect_changed_cells", 0) or 0) > 0
            and (
                str(row.get("trigger_evidence_class") or "") != "region_suspicion"
                or int(row.get("support_count", 0) or 0) >= 2
            )
        ]
        cap = 8
    collapsed.sort(
        key=lambda row: (
            0 if str(row.get("trigger_evidence_class") or "") in {"object_backed", "object_backed_candidate"} else 1,
            -int(row.get("support_count", 0) or 0),
            -float(row.get("confidence", 0.0) or 0.0),
            -int(row.get("effect_changed_cells", 0) or 0),
            str(row.get("trigger_id") or ""),
        )
    )
    return collapsed[:cap]


def analyze_episode(
    raw_episode: RawEpisode,
    analysis_mode: str,
    blackboard_snapshot: dict | None = None,
    mechanic_graph_snapshot: dict | None = None,
    hypothesis_config: object | None = None,
    llm_adapter: object | None = None,
    hypothesis_registry_snapshot: dict | None = None,
) -> AnalyzedEpisode:
    analysis_mode = _validate_analysis_mode(analysis_mode)
    normalized_observations = [normalize_observation(step.observation) for step in raw_episode.steps]
    step_summaries: list[dict] = []
    known_areas: list[dict] = []
    area_sequence: list[dict] = []
    initial_prior_entities = []
    if blackboard_snapshot is not None:
        blackboard_state = dict((blackboard_snapshot or {}).get("state", {}) or {}) if isinstance(blackboard_snapshot, dict) else dict(getattr(blackboard_snapshot, "state", {}) or {})
        initial_prior_entities = [dict(row) for row in dict(blackboard_state.get("entities", {}) or {}).values()]
    prior_entities = list(initial_prior_entities)

    for step_idx, observation in enumerate(normalized_observations):
        previous = normalized_observations[step_idx - 1] if step_idx > 0 else None
        summary = summarize_observation(observation, previous, prior_entities=prior_entities)
        area = assign_area(summary, known_areas)
        area_sequence.append(area)
        if not any(existing["area_id"] == area["area_id"] for existing in known_areas):
            known_areas.append(area)
        enriched = dict(summary)
        enriched["step_idx"] = step_idx
        enriched["area_id"] = area["area_id"]
        step_summaries.append(enriched)
        prior_entities = [dict(row) for row in list(summary.get("objects", []) or [])]

    avatar_tracking = _fallback_avatar_tracking(step_summaries, track_avatar(step_summaries))
    motion = summarize_motion(raw_episode.steps, step_summaries, avatar_tracking)
    priorities = _analysis_priorities(analysis_mode=analysis_mode)
    step_rows = []
    for step_idx, (raw_step, summary, area) in enumerate(zip(raw_episode.steps, step_summaries, area_sequence)):
        tracking_row = next((row for row in avatar_tracking["per_step"] if row["step_idx"] == step_idx), None)
        normalized_action = normalize_action_lookup(raw_step.action, (raw_step.info or {}).get("available_actions"))
        raw_action = raw_step.action if isinstance(raw_step.action, dict) else {}
        target_coordinates = raw_action.get("coordinates") or raw_action.get("target_coordinates") or raw_action.get("click_target_coordinates")
        step_rows.append(
            {
                "step_idx": step_idx,
                "action": raw_step.action,
                "action_id": normalized_action.get("action_id"),
                "action_name": normalized_action.get("action_name"),
                "action_family": normalized_action.get("action_family", "unknown"),
                "action_type": normalized_action.get("action_name"),
                "target_entity_id": raw_step.action.get("target_entity_id") or raw_step.action.get("target") if isinstance(raw_step.action, dict) else None,
                "target_coordinates": list(target_coordinates) if isinstance(target_coordinates, (list, tuple)) else None,
                "reward": raw_step.reward,
                "done": raw_step.done,
                "area_id": area["area_id"],
                "state_hash": summary["state_identity"]["state_hash"],
                "avatar_centroid": tracking_row["main_centroid"] if tracking_row is not None else None,
                "avatar_cell": [int(float(tracking_row["main_centroid"][0])), int(float(tracking_row["main_centroid"][1]))] if tracking_row is not None and tracking_row.get("main_centroid") is not None else None,
                "object_ids": [obj["object_id"] for obj in summary["objects"]],
                "changed_cells": sum(int(region.get("area", 0)) for region in summary.get("change_regions", [])),
                "change_region_count": len(summary["change_regions"]),
                "analysis_mode": analysis_mode,
                "telemetry": dict(raw_step.info or {}),
            }
        )
    topology_nodes, topology_edges = _topology_from_steps(step_rows)
    topology_nodes, topology_edges = _mode_select_topology(topology_nodes, topology_edges, analysis_mode=analysis_mode)
    consequences = _mode_select_consequences(_consequences(raw_episode, motion, step_summaries, step_rows), analysis_mode=analysis_mode)
    detected_pois = _mode_select_pois(detect_pois(step_summaries, avatar_tracking, step_rows), analysis_mode=analysis_mode)
    supplemental_structure = _supplemental_structure_entities(
        step_summaries=step_summaries,
        normalized_observations=normalized_observations,
        area_sequence=area_sequence,
    )
    pois = _attach_target_effects_to_pois(
        _attach_identity_to_pois(
        _annotate_pattern_descriptors(
        _merge_entity_candidates(detected_pois, supplemental_structure),
        step_summaries=step_summaries,
        normalized_observations=normalized_observations,
        ),
        step_summaries=step_summaries,
        ),
        step_rows=step_rows,
        blackboard_snapshot=blackboard_snapshot,
    )
    trigger_zones = _collapse_trigger_zones(
        _extract_trigger_zones(step_summaries=step_summaries, step_rows=step_rows, analysis_mode=analysis_mode),
        analysis_mode=analysis_mode,
    )

    delta = BlackboardDelta(
        session_id=raw_episode.session_id,
        run_id=raw_episode.run_id,
        game_id=raw_episode.game_id,
        round_id=raw_episode.round_id,
        pass_id=raw_episode.pass_id,
        episode_id=raw_episode.episode_id,
        delta_id=f"delta:{raw_episode.episode_id}:{stable_digest({'steps': step_rows})}",
        areas=tuple(
            _stamp_row({
                "area_id": area["area_id"],
                "area_signature": area["area_signature"],
                "width": area["width"],
                "height": area["height"],
                "palette": area["palette"],
                "background_color": area["background_color"],
                "state_hash": area["state_hash"],
                "visit_count": 1,
                "analysis_mode": analysis_mode,
                "source_stage": "analysis",
                "source_pass_id": raw_episode.pass_id,
                "source_episode_id": raw_episode.episode_id,
                "confidence": 1.0,
                "inference_method": "direct_observation",
                "factual_observation": True,
            }, analysis_mode=analysis_mode, row_kind="areas", direct_evidence_present=True, direct_evidence_fields=["area_id", "area_signature", "state_hash"], observation_support_span=(0, max(0, len(step_summaries) - 1)))
            for area in known_areas
        ),
        entities=tuple(
            _stamp_row(
                dict(
                    poi,
                    analysis_mode=analysis_mode,
                    source_stage="analysis",
                    source_pass_id=raw_episode.pass_id,
                    source_episode_id=raw_episode.episode_id,
                    confidence=float(poi.get("confidence", 0.0)),
                    inference_method="direct_observation" if _poi_is_directly_observed(poi=poi, analysis_mode=analysis_mode) else "poi_detection",
                    factual_observation=_poi_is_directly_observed(poi=poi, analysis_mode=analysis_mode),
                    area_id=next(
                        (
                            area_sequence[index]["area_id"]
                            for index, summary in enumerate(step_summaries)
                            if any(obj["signature"] == poi["signature"] for obj in summary["objects"])
                        ),
                        area_sequence[-1]["area_id"] if area_sequence else None,
                    ),
                ),
                analysis_mode=analysis_mode,
                row_kind="entities",
                direct_evidence_present=_poi_is_directly_observed(poi=poi, analysis_mode=analysis_mode),
                direct_evidence_fields=_poi_direct_evidence_fields(poi),
                contradiction_flag=bool(poi.get("rejected")),
                observation_support_span=(0, max(0, int(poi.get("observations", 1) or 1) - 1)),
            )
            for poi in pois
        ),
        consequences=tuple(
            _stamp_row(
                dict(
                    row,
                    analysis_mode=analysis_mode,
                    source_stage="analysis",
                    source_pass_id=raw_episode.pass_id,
                    source_episode_id=raw_episode.episode_id,
                    confidence=min(1.0, 0.45 + (0.1 * float(row.get("evidence_count", 1) or 1))),
                    inference_method="direct_observation" if _consequence_is_directly_observed(row=row, analysis_mode=analysis_mode) else "consequence_reconstruction",
                    factual_observation=_consequence_is_directly_observed(row=row, analysis_mode=analysis_mode),
                ),
                analysis_mode=analysis_mode,
                row_kind="consequences",
                direct_evidence_present=_consequence_is_directly_observed(row=row, analysis_mode=analysis_mode),
                direct_evidence_fields=_consequence_direct_evidence_fields(row),
                contradiction_flag=bool(
                    row.get("blocked")
                    and not row.get("action_effect_near_avatar")
                    and int(row.get("local_change_area", 0) or 0) <= 0
                    and float(row.get("reward", 0.0) or 0.0) == 0.0
                    and not bool(row.get("done"))
                ),
                observation_support_span=(int(row.get("step_idx", 0) or 0), int(row.get("step_idx", 0) or 0)),
            )
            for row in consequences
        ),
        trigger_zones=tuple(
            _stamp_row(
                dict(
                    row,
                    analysis_mode=analysis_mode,
                    source_stage="analysis",
                    source_pass_id=raw_episode.pass_id,
                    source_episode_id=raw_episode.episode_id,
                    inference_method="trigger_zone_extraction",
                    factual_observation=bool(analysis_mode == "directed_outcome" and row.get("effect_changed_cells", 0) > 0),
                ),
                analysis_mode=analysis_mode,
                row_kind="trigger_zones",
                direct_evidence_present=bool(analysis_mode == "directed_outcome" and row.get("effect_changed_cells", 0) > 0),
                direct_evidence_fields=["trigger_id", "zone_bbox", "supporting_steps", "effect_region"] if analysis_mode == "directed_outcome" and row.get("effect_changed_cells", 0) > 0 else ["trigger_id", "zone_bbox", "supporting_steps"],
                contradiction_flag=False,
                observation_support_span=(min(list(row.get("supporting_steps", [0]) or [0])), max(list(row.get("supporting_steps", [0]) or [0]))),
            )
            for row in trigger_zones
        ),
        topology_nodes=tuple(
            _stamp_row(
                dict(
                    row,
                    analysis_mode=analysis_mode,
                    source_stage="analysis",
                    source_pass_id=raw_episode.pass_id,
                    source_episode_id=raw_episode.episode_id,
                    confidence=0.6 if analysis_mode == "probe" else 0.7,
                    inference_method="direct_observation",
                    factual_observation=True,
                ),
                analysis_mode=analysis_mode,
                row_kind="topology_nodes",
                direct_evidence_present=True,
                direct_evidence_fields=_topology_node_direct_evidence_fields(row),
                observation_support_span=(0, max(0, len(step_rows) - 1)),
            )
            for row in topology_nodes
        ),
        topology_edges=tuple(
            _stamp_row(
                dict(
                    row,
                    analysis_mode=analysis_mode,
                    source_stage="analysis",
                    source_pass_id=raw_episode.pass_id,
                    source_episode_id=raw_episode.episode_id,
                    confidence=0.6 if analysis_mode == "probe" else 0.75,
                    inference_method="direct_observation",
                    factual_observation=True,
                ),
                analysis_mode=analysis_mode,
                row_kind="topology_edges",
                direct_evidence_present=True,
                direct_evidence_fields=_topology_edge_direct_evidence_fields(row),
                observation_support_span=(0, max(0, len(step_rows) - 1)),
            )
            for row in topology_edges
        ),
        evidence=tuple(f"{raw_episode.episode_id}:{row['step_idx']}" for row in step_rows),
        material_change=bool(step_rows),
        metadata={
            "analysis_mode": analysis_mode,
            "analysis_priorities": priorities,
            "main_track_id": avatar_tracking["main_track_id"],
            "area_sequence": [area["area_id"] for area in area_sequence],
            "avatar_path": motion["avatar_path"],
            "step_rows": step_rows,
            "structure_candidates": supplemental_structure,
        },
    )

    analyzed_episode = AnalyzedEpisode(
        session_id=raw_episode.session_id,
        run_id=raw_episode.run_id,
        game_id=raw_episode.game_id,
        round_id=raw_episode.round_id,
        pass_id=raw_episode.pass_id,
        episode_id=raw_episode.episode_id,
        raw_episode_id=raw_episode.episode_id,
        summary={
            "analysis_mode": analysis_mode,
            "step_count": len(raw_episode.steps),
            "won": raw_episode.won,
            "total_reward": raw_episode.total_reward,
            "main_track_id": avatar_tracking["main_track_id"],
            "avatar_visits": motion["avatar_path"],
            "area_sequence": [area["area_id"] for area in area_sequence],
            "step_rows": step_rows,
            "structure_candidate_count": len(supplemental_structure),
            "background_colors": [summary["background"]["color"] for summary in step_summaries],
            "state_hashes": [summary["state_identity"]["state_hash"] for summary in step_summaries],
        },
        objects=tuple(
            dict(obj, step_idx=step_idx, area_id=step_summaries[step_idx]["area_id"])
            for step_idx, summary in enumerate(step_summaries)
            for obj in summary["objects"]
        ),
        avatar_tracks=tuple(avatar_tracking["tracks"]),
        points_of_interest=tuple(pois),
        areas=tuple(
            {
                "area_id": area["area_id"],
                "area_signature": area["area_signature"],
                "width": area["width"],
                "height": area["height"],
                "palette": area["palette"],
                "background_color": area["background_color"],
                "state_hash": area["state_hash"],
            }
            for area in known_areas
        ),
        motion=tuple(
            [{"motion_id": "motion:episode", "avatar_path": motion["avatar_path"], "motion_regions": motion["motion_regions"]}]
            + motion["movement_rows"]
        ),
        blackboard_deltas=(delta,),
        mechanic_graph_delta=None,
        metadata={
            "analysis_mode": analysis_mode,
            "analysis_priorities": priorities,
            "step_summaries": step_summaries,
            "avatar_tracking": avatar_tracking,
            "motion": motion,
            "structure_candidates": supplemental_structure,
        },
    )
    mechanic_graph_delta = extract_mechanic_graph_delta(
        raw_episode,
        analyzed_episode,
        current_blackboard_snapshot=blackboard_snapshot,
        current_mechanic_graph_snapshot=mechanic_graph_snapshot,
        hypothesis_config=hypothesis_config,
        llm_adapter=llm_adapter,
        hypothesis_registry_snapshot=hypothesis_registry_snapshot,
    )
    analyzed_with_graph = AnalyzedEpisode(
        **{
            **analyzed_episode.__dict__,
            "mechanic_graph_delta": mechanic_graph_delta,
        },
    )
    hypothesis_result = orchestrate_hypotheses(
        raw_episode=raw_episode,
        analyzed_episode=analyzed_with_graph,
        mechanic_graph_snapshot=mechanic_graph_snapshot,
        blackboard_snapshot=blackboard_snapshot,
        hypothesis_config=hypothesis_config,
        llm_adapter=llm_adapter,
        hypothesis_registry_snapshot=hypothesis_registry_snapshot,
    )
    return AnalyzedEpisode(
        **{
            **analyzed_with_graph.__dict__,
            "mechanic_graph_delta": mechanic_graph_delta,
            "deterministic_hypothesis_bundle": hypothesis_result.get("deterministic_bundle"),
            "llm_hypothesis_bundle": hypothesis_result.get("llm_bundle"),
            "metadata": {
                **dict(analyzed_with_graph.metadata or {}),
                "hypothesis_gating_summary": dict(hypothesis_result.get("gating_summary", {}) or {}),
                "llm_operation_summary": dict(hypothesis_result.get("llm_operation_summary", {}) or {}),
            },
        },
    )
