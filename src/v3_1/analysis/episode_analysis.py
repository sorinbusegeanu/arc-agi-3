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
from v3_1.world.entities import _match_score, _merge_bbox, _stable_entity_id


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


def _support_family_emit_inputs(classifier_truth_surface: dict | None) -> dict:
    payload = dict(classifier_truth_surface or {})
    return {
        "counterfactual_evidence_observed": payload.get("counterfactual_evidence_observed"),
        "exit_attempt_evidence_observed": payload.get("exit_attempt_evidence_observed"),
        "expected_effect_type": payload.get("expected_effect_type"),
        "expected_relation_type": payload.get("expected_relation_type") or payload.get("expected_effect_relation"),
        "expected_target_id": payload.get("expected_target_id") or payload.get("expected_effect_target_id"),
        "expected_trigger_contact_observed": payload.get("expected_trigger_contact_observed"),
        "expected_region_reached": payload.get("expected_region_reached"),
        "observed_effect_change": payload.get("observed_effect_change"),
        "observed_effect_absent": payload.get("observed_effect_absent") if "observed_effect_absent" in payload else payload.get("expected_effect_absent"),
        "attempted_boundary_contact": payload.get("attempted_boundary_contact"),
        "attempted_portal_contact": payload.get("attempted_portal_contact"),
        "attempted_terminal_affordance_contact": payload.get("attempted_terminal_affordance_contact"),
        "attempted_escape_direction": payload.get("attempted_escape_direction"),
        "exit_attempt_target_id": payload.get("exit_attempt_target_id"),
    }


def _family_debug_entry(*, classifier_flag: bool | None) -> dict:
    return {
        "classifier_flag": classifier_flag,
        "emit_attempted": False,
        "relation_resolved": False,
        "row_emitted": False,
        "suppressed": False,
        "suppression_reason_codes": [],
        "resolution_failure_reason_codes": [],
    }


def _resolve_family_target(step_rows: list[dict], emit_inputs: dict) -> tuple[str | None, str | None]:
    target_entity_id = str(emit_inputs.get("expected_target_id") or emit_inputs.get("exit_attempt_target_id") or "") or None
    target_area_id = None
    if not target_entity_id:
        for step in list(step_rows or []):
            telemetry = dict(step.get("telemetry", {}) or {})
            target_area_id = str(step.get("area_id") or telemetry.get("pre_step_area_id") or telemetry.get("post_step_area_id") or "") or None
            if target_area_id:
                break
    return target_entity_id, target_area_id


def _handle_counterfactual_support_family(*, emit_inputs: dict, step_rows: list[dict], consequences: list[dict], analysis_mode: str) -> tuple[list[dict], dict]:
    debug = _family_debug_entry(classifier_flag=emit_inputs.get("counterfactual_evidence_observed"))
    emitted: list[dict] = []
    if emit_inputs.get("counterfactual_evidence_observed") is not True:
        debug["suppressed"] = True
        debug["suppression_reason_codes"].append("classifier_false")
        return emitted, debug
    debug["emit_attempted"] = True
    target_entity_id, target_area_id = _resolve_family_target(step_rows, emit_inputs)
    if not target_entity_id and not target_area_id:
        debug["suppressed"] = True
        debug["resolution_failure_reason_codes"].append("missing_target_resolution")
        return emitted, debug
    first_step = dict(step_rows[0] or {}) if step_rows else {}
    step_idx = int(first_step.get("step_idx", 0) or 0)
    action_name = str(first_step.get("action_name") or first_step.get("action_type") or "unknown")
    action_family = str(first_step.get("action_family") or "unknown")
    evidence_ref = f"{step_idx}:counterfactual"
    debug["relation_resolved"] = True
    emitted.append(
        {
            "consequence_id": f"counterfactual:{target_entity_id or target_area_id or 'unknown'}",
            "step_idx": step_idx,
            "action": action_name,
            "action_name": action_name,
            "action_id": first_step.get("action_id"),
            "action_family": action_family,
            "target_entity_id": target_entity_id,
            "area_id": target_area_id,
            "done": False,
            "reward": 0.0,
            "local_change_area": 0,
            "blocked": bool(emit_inputs.get("observed_effect_absent")),
            "evidence_count": 1,
            "evidence_refs": [evidence_ref],
            "telemetry": dict(emit_inputs),
            "supports_counterfactual_relation": True,
            "counterfactual_support_count": 1,
            "support_family": "counterfactual",
        }
    )
    debug["row_emitted"] = True
    return emitted, debug


def _handle_exit_attempt_support_family(*, emit_inputs: dict, step_rows: list[dict], consequences: list[dict], analysis_mode: str) -> tuple[list[dict], dict]:
    debug = _family_debug_entry(classifier_flag=emit_inputs.get("exit_attempt_evidence_observed"))
    emitted: list[dict] = []
    if emit_inputs.get("exit_attempt_evidence_observed") is not True:
        debug["suppressed"] = True
        debug["suppression_reason_codes"].append("classifier_false")
        return emitted, debug
    debug["emit_attempted"] = True
    target_entity_id, target_area_id = _resolve_family_target(step_rows, emit_inputs)
    if not target_entity_id and not target_area_id and not emit_inputs.get("attempted_escape_direction"):
        debug["suppressed"] = True
        debug["resolution_failure_reason_codes"].append("missing_boundary_or_region_resolution")
        return emitted, debug
    first_step = dict(step_rows[0] or {}) if step_rows else {}
    step_idx = int(first_step.get("step_idx", 0) or 0)
    action_name = str(first_step.get("action_name") or first_step.get("action_type") or "unknown")
    action_family = str(first_step.get("action_family") or "unknown")
    evidence_ref = f"{step_idx}:exit_attempt"
    debug["relation_resolved"] = True
    emitted.append(
        {
            "consequence_id": f"exit_attempt:{target_entity_id or target_area_id or emit_inputs.get('attempted_escape_direction') or 'unknown'}",
            "step_idx": step_idx,
            "action": action_name,
            "action_name": action_name,
            "action_id": first_step.get("action_id"),
            "action_family": action_family,
            "target_entity_id": target_entity_id,
            "area_id": target_area_id,
            "done": False,
            "reward": 0.0,
            "local_change_area": 0,
            "blocked": bool(emit_inputs.get("attempted_boundary_contact")),
            "evidence_count": 1,
            "evidence_refs": [evidence_ref],
            "telemetry": dict(emit_inputs),
            "supports_exit_attempt_relation": True,
            "exit_attempt_support_count": 1,
            "support_family": "exit_attempt",
        }
    )
    debug["row_emitted"] = True
    return emitted, debug


def _annotate_directed_support(step_rows: list[dict], topology_edges: list[dict], consequences: list[dict], *, analysis_mode: str, classifier_truth_surface: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    edge_rows = [dict(row) for row in list(topology_edges or [])]
    consequence_rows = [dict(row) for row in list(consequences or [])]
    emit_inputs = _support_family_emit_inputs(classifier_truth_surface)
    has_any_truth_surface = any(value is not None for value in emit_inputs.values())
    counterfactual_rows, counterfactual_debug = _handle_counterfactual_support_family(
        emit_inputs=emit_inputs,
        step_rows=step_rows,
        consequences=consequence_rows,
        analysis_mode=analysis_mode,
    )
    exit_attempt_rows, exit_attempt_debug = _handle_exit_attempt_support_family(
        emit_inputs=emit_inputs,
        step_rows=step_rows,
        consequences=consequence_rows,
        analysis_mode=analysis_mode,
    )
    consequence_rows.extend(counterfactual_rows)
    consequence_rows.extend(exit_attempt_rows)
    emit_debug = {
        "families": {
            "counterfactual": counterfactual_debug,
            "exit_attempt": exit_attempt_debug,
        },
        "counterfactual_emit_attempt_count": int(bool(counterfactual_debug.get("emit_attempted"))),
        "counterfactual_emit_relation_resolution_failure_count": len(list(counterfactual_debug.get("resolution_failure_reason_codes", []) or [])),
        "counterfactual_emit_suppressed_count": int(bool(counterfactual_debug.get("suppressed"))),
        "exit_attempt_emit_attempt_count": int(bool(exit_attempt_debug.get("emit_attempted"))),
        "exit_attempt_emit_relation_resolution_failure_count": len(list(exit_attempt_debug.get("resolution_failure_reason_codes", []) or [])),
        "exit_attempt_emit_suppressed_count": int(bool(exit_attempt_debug.get("suppressed"))),
        "counterfactual_emit_attempt_count_directed": int(bool(counterfactual_debug.get("emit_attempted")) and analysis_mode == "directed_outcome"),
        "counterfactual_emit_attempt_count_probe": int(bool(counterfactual_debug.get("emit_attempted")) and analysis_mode != "directed_outcome"),
        "exit_attempt_emit_attempt_count_directed": int(bool(exit_attempt_debug.get("emit_attempted")) and analysis_mode == "directed_outcome"),
        "exit_attempt_emit_attempt_count_probe": int(bool(exit_attempt_debug.get("emit_attempted")) and analysis_mode != "directed_outcome"),
    }
    if has_any_truth_surface and not emit_debug["families"]:
        emit_debug["invariant_failure"] = ["family_handler_not_invoked"]
    for family_name in ("counterfactual", "exit_attempt"):
        family = emit_debug["families"][family_name]
        if family.get("classifier_flag") is True and not family.get("emit_attempted"):
            family["suppression_reason_codes"].append("classifier_true_but_no_emit_attempt")
            emit_debug.setdefault("invariant_failure", []).append(f"{family_name}:classifier_true_but_no_emit_attempt")
    return edge_rows, consequence_rows, emit_debug


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
    bbox_payload = dict(candidate.get("bbox", {}) or {})
    bbox = {
        "x1": int(bbox_payload.get("x1", 0) or 0),
        "y1": int(bbox_payload.get("y1", 0) or 0),
        "x2": int(bbox_payload.get("x2", 0) or 0),
        "y2": int(bbox_payload.get("y2", 0) or 0),
    }
    pattern_id = str(candidate.get("pattern_id") or "")
    signature = str(candidate.get("signature") or "")
    stable_key = {
        "pattern_id": pattern_id,
        "signature": signature,
        "kind": str(candidate.get("kind") or "structure"),
        "primary_color": int(candidate.get("primary_color", 0) or 0),
        "bbox": {
            "x1": int(round(bbox["x1"] / 2.0) * 2),
            "y1": int(round(bbox["y1"] / 2.0) * 2),
            "x2": int(round(bbox["x2"] / 2.0) * 2),
            "y2": int(round(bbox["y2"] / 2.0) * 2),
        },
    }
    return f"entity:{stable_digest(stable_key)}"


def _supplemental_structure_entities(*, step_summaries: list[dict], normalized_observations: list[list[list[int]]], area_sequence: list[dict]) -> list[dict]:
    supplemental: list[dict] = []
    aggregated: dict[str, dict] = {}
    spatial_groups: dict[tuple[str | None, str, int, int, int, int], str] = {}

    def _canonical_group_key(*, area_id: str | None, candidate: dict, pattern_id: str) -> tuple[str, int, int, int, int]:
        bbox_payload = dict(candidate.get("bbox", {}) or {})
        x1 = int(bbox_payload.get("x1", 0) or 0)
        y1 = int(bbox_payload.get("y1", 0) or 0)
        x2 = int(bbox_payload.get("x2", 0) or 0)
        y2 = int(bbox_payload.get("y2", 0) or 0)
        return (
            pattern_id or str(candidate.get("signature") or ""),
            int(round(x1 / 2.0) * 2),
            int(round(y1 / 2.0) * 2),
            int(round(x2 / 2.0) * 2),
            int(round(y2 / 2.0) * 2),
        )

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
            pattern_id = stable_pattern_id(patch) if patch else str(candidate.get("pattern_id") or "") or None
            entity_id = _structure_entity_id({**candidate, "pattern_id": pattern_id}, area_id=area_id)
            group_key = _canonical_group_key(area_id=area_id, candidate={**candidate, "pattern_id": pattern_id}, pattern_id=str(pattern_id or ""))
            entity_id = spatial_groups.setdefault(group_key, entity_id)
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
                "kind": "entity",
                "entity_class": "structure",
                "poi_class": "structure",
                "poi_bucket": "supplemental_structure",
                "planner_visible": False,
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
                "visit_count": 1,
            }
            if existing is None:
                payload["supporting_steps"] = [step_idx]
                aggregated[entity_id] = payload
                continue
            existing["confidence"] = max(float(existing.get("confidence", 0.0) or 0.0), confidence)
            existing["utility"] = max(float(existing.get("utility", 0.0) or 0.0), float(payload.get("utility", 0.0) or 0.0))
            existing["identity_confidence"] = max(float(existing.get("identity_confidence", 0.0) or 0.0), float(payload.get("identity_confidence", 0.0) or 0.0))
            existing["observations"] = int(existing.get("observations", 0) or 0) + 1
            existing["visit_count"] = int(existing.get("visit_count", 0) or 0) + 1
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
    def _canonical_merge_key(row: dict) -> str:
        bbox_payload = dict(row.get("bbox", {}) or {})
        bbox_key = (
            int(round(int(bbox_payload.get("x1", 0) or 0) / 2.0) * 2),
            int(round(int(bbox_payload.get("y1", 0) or 0) / 2.0) * 2),
            int(round(int(bbox_payload.get("x2", 0) or 0) / 2.0) * 2),
            int(round(int(bbox_payload.get("y2", 0) or 0) / 2.0) * 2),
        )
        stable_key = {
            "pattern_id": str(row.get("pattern_id") or ""),
            "signature": str(row.get("signature") or ""),
            "poi_class": str(row.get("poi_class") or ""),
            "primary_color": int(row.get("primary_color", 0) or 0),
            "bbox": bbox_key,
        }
        return stable_digest(stable_key)

    merged: dict[str, dict] = {}
    for row in list(base or []):
        payload = dict(row)
        key = _canonical_merge_key(payload)
        merged[key] = payload
    for row in list(supplemental or []):
        payload = dict(row)
        key = _canonical_merge_key(payload)
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


def _promote_structure_entities_to_pois(structure_entities: list[dict]) -> tuple[list[dict], dict]:
    promoted: list[dict] = []
    rejection_reason_counts: dict[str, int] = {}
    bottom_strip_children: list[dict] = []
    for row in list(structure_entities or []):
        payload = dict(row)
        observations = int(payload.get("observations", 0) or 0)
        identity_confidence = float(payload.get("identity_confidence", 0.0) or 0.0)
        pattern_id = str(payload.get("pattern_id") or "")
        effect_score = max(
            float(payload.get("interact_effect_score", 0.0) or 0.0),
            float(payload.get("click_effect_score", 0.0) or 0.0),
            float(payload.get("candidate_effect_score", 0.0) or 0.0),
        )
        if observations < 2:
            rejection_reason_counts["insufficient_structure_observations"] = rejection_reason_counts.get("insufficient_structure_observations", 0) + 1
            continue
        if identity_confidence < 0.72:
            rejection_reason_counts["identity_confidence_too_low"] = rejection_reason_counts.get("identity_confidence_too_low", 0) + 1
            continue
        if not pattern_id and effect_score < 0.2:
            rejection_reason_counts["missing_pattern_and_effect_support"] = rejection_reason_counts.get("missing_pattern_and_effect_support", 0) + 1
            continue
        bbox = dict(payload.get("bbox", {}) or {})
        bbox_width = max(1, int(bbox.get("x2", 0) or 0) - int(bbox.get("x1", 0) or 0) + 1)
        bbox_height = max(1, int(bbox.get("y2", 0) or 0) - int(bbox.get("y1", 0) or 0) + 1)
        low_visit = observations <= 3
        weak_effect = effect_score < 0.1
        bottom_strip = int(bbox.get("y1", 0) or 0) >= 50 and bbox_height <= 3
        if bottom_strip and low_visit and weak_effect:
            payload["strip_region_candidate"] = True
            bottom_strip_children.append(payload)
            continue
        payload["kind"] = "poi"
        payload["poi_id"] = str(payload.get("poi_id") or payload.get("entity_id") or "")
        payload["poi_bucket"] = "structural"
        payload["planner_visible"] = True
        payload["planner_targetable"] = False
        payload["promotion_source"] = "supplemental_structure_admission"
        promoted.append(payload)
    if bottom_strip_children:
        merged_bbox = {
            "x1": min(int(dict(row.get("bbox", {}) or {}).get("x1", 0)) for row in bottom_strip_children),
            "y1": min(int(dict(row.get("bbox", {}) or {}).get("y1", 0)) for row in bottom_strip_children),
            "x2": max(int(dict(row.get("bbox", {}) or {}).get("x2", 0)) for row in bottom_strip_children),
            "y2": max(int(dict(row.get("bbox", {}) or {}).get("y2", 0)) for row in bottom_strip_children),
        }
        merged = dict(bottom_strip_children[0])
        merged["entity_id"] = f"entity:{stable_digest({'strip_region': merged_bbox, 'area_id': merged.get('area_id')})}"
        merged["poi_id"] = merged["entity_id"]
        merged["kind"] = "poi"
        merged["poi_bucket"] = "structural"
        merged["planner_visible"] = True
        merged["planner_targetable"] = False
        merged["promotion_source"] = "supplemental_structure_admission"
        merged["bbox"] = merged_bbox
        merged["area"] = sum(int(row.get("area", 0) or 0) for row in bottom_strip_children)
        merged["observations"] = max(int(row.get("observations", 0) or 0) for row in bottom_strip_children)
        merged["visit_count"] = max(int(row.get("visit_count", 0) or 0) for row in bottom_strip_children)
        merged["merged_input_poi_ids"] = [str(row.get("entity_id") or "") for row in bottom_strip_children]
        merged["poi_source_provenance"] = ["structure_promotion"]
        promoted.append(merged)
        rejection_reason_counts["merged_bottom_strip_children"] = len(bottom_strip_children)
    return promoted, {
        "structure_entity_candidate_count": len(list(structure_entities or [])),
        "structure_entity_promoted_count": len(promoted),
        "structure_entity_rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "promoted_structure_pois": [
            {
                "poi_id": str(row.get("poi_id") or row.get("entity_id") or ""),
                "entity_id": str(row.get("entity_id") or ""),
                "bbox": dict(row.get("bbox", {}) or {}),
                "area": int(row.get("area", 0) or 0),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "identity_confidence": float(row.get("identity_confidence", 0.0) or 0.0),
                "pattern_id": row.get("pattern_id"),
                "planner_targetable": bool(row.get("planner_targetable", False)),
            }
            for row in promoted
        ],
    }


def _adopt_prior_poi_ids(pois: list[dict], blackboard_snapshot: dict | None) -> list[dict]:
    if blackboard_snapshot is None:
        return [dict(row) for row in list(pois or [])]
    blackboard_state = dict((blackboard_snapshot or {}).get("state", {}) or {}) if isinstance(blackboard_snapshot, dict) else {}
    prior_rows = [
        dict(row)
        for row in dict(blackboard_state.get("observed_entities", {}) or {}).values()
        if isinstance(row, dict) and str(row.get("kind") or "") == "poi"
    ]
    adopted: list[dict] = []
    for row in list(pois or []):
        payload = dict(row)
        best_match = None
        best_score = 0.0
        payload_sources = set(list(payload.get("poi_source_provenance", []) or []))
        for prior in prior_rows:
            prior_sources = set(list(prior.get("poi_source_provenance", []) or []))
            if payload_sources and prior_sources and payload_sources != prior_sources:
                continue
            score = _match_score(prior, payload)
            if score > best_score:
                best_score = score
                best_match = prior
        if best_match is not None and best_score >= 0.65:
            stable_id = str(best_match.get("entity_id") or best_match.get("poi_id") or "")
            if stable_id:
                payload["entity_id"] = stable_id
                payload["poi_id"] = stable_id
                payload["stable_entity_id_hint"] = stable_id
        adopted.append(payload)
    return adopted


def _bbox_area_value(bbox: dict) -> int:
    if not isinstance(bbox, dict) or not bbox:
        return 0
    return max(0, (int(bbox.get("x2", 0) or 0) - int(bbox.get("x1", 0) or 0) + 1) * (int(bbox.get("y2", 0) or 0) - int(bbox.get("y1", 0) or 0) + 1))


def _bbox_iou(left: dict, right: dict) -> float:
    if not left or not right:
        return 0.0
    x1 = max(int(left.get("x1", 0) or 0), int(right.get("x1", 0) or 0))
    y1 = max(int(left.get("y1", 0) or 0), int(right.get("y1", 0) or 0))
    x2 = min(int(left.get("x2", 0) or 0), int(right.get("x2", 0) or 0))
    y2 = min(int(left.get("y2", 0) or 0), int(right.get("y2", 0) or 0))
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1 + 1) * (y2 - y1 + 1)
    union = _bbox_area_value(left) + _bbox_area_value(right) - intersection
    return float(intersection) / float(max(1, union))


def _bbox_contains_ratio(outer: dict, inner: dict) -> float:
    if not outer or not inner:
        return 0.0
    x1 = max(int(outer.get("x1", 0) or 0), int(inner.get("x1", 0) or 0))
    y1 = max(int(outer.get("y1", 0) or 0), int(inner.get("y1", 0) or 0))
    x2 = min(int(outer.get("x2", 0) or 0), int(inner.get("x2", 0) or 0))
    y2 = min(int(outer.get("y2", 0) or 0), int(inner.get("y2", 0) or 0))
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1 + 1) * (y2 - y1 + 1)
    return float(intersection) / float(max(1, _bbox_area_value(inner)))


def _poi_semantic_label(row: dict) -> str:
    return str(
        row.get("semantic_label")
        or row.get("mechanic_role")
        or row.get("poi_class")
        or row.get("kind")
        or "unknown"
    )


def _poi_effect_profile(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("movement_effect_score", 0.0) or 0.0),
        float(row.get("interact_effect_score", 0.0) or 0.0),
        float(row.get("click_effect_score", 0.0) or 0.0),
    )


def _strongly_distinct_poi_role(left: dict, right: dict) -> bool:
    left_bbox = dict(left.get("bbox", {}) or {})
    right_bbox = dict(right.get("bbox", {}) or {})
    if _poi_semantic_label(left) != _poi_semantic_label(right):
        return True
    if bool(left.get("pattern_id")) and bool(right.get("pattern_id")) and str(left.get("pattern_id")) != str(right.get("pattern_id")):
        if _bbox_iou(left_bbox, right_bbox) < 0.55 and _bbox_contains_ratio(left_bbox, right_bbox) < 0.85 and _bbox_contains_ratio(right_bbox, left_bbox) < 0.85:
            return True
    if _poi_effect_profile(left) != _poi_effect_profile(right):
        effect_gap = sum(abs(a - b) for a, b in zip(_poi_effect_profile(left), _poi_effect_profile(right)))
        if effect_gap >= 0.35:
            return True
    if _bbox_iou(left_bbox, right_bbox) < 0.35 and _bbox_contains_ratio(left_bbox, right_bbox) < 0.8 and _bbox_contains_ratio(right_bbox, left_bbox) < 0.8:
        return True
    return False


def _central_parent_child_canonicalize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    working = [dict(row) for row in list(rows or [])]
    decision_debug: list[dict] = []
    for row in working:
        row.setdefault("poi_hierarchy_level", 0)
        row.setdefault("parent_poi_id", None)
        row.setdefault("child_poi_ids", [])
        row.setdefault("hierarchy_role", "parent_region")
    detector_candidates = [
        row for row in working
        if "detector" in set(list(row.get("poi_source_provenance", []) or []))
        and float(row.get("canonical_track_persistence", 0.0) or 0.0) >= 0.65
        and _bbox_area_value(dict(row.get("bbox", {}) or {})) >= 12
        and int(dict(row.get("bbox", {}) or {}).get("y1", 0) or 0) < 52
    ]
    consumed_ids: set[str] = set()
    final_rows: list[dict] = []
    for row in sorted(working, key=lambda item: (-_bbox_area_value(dict(item.get("bbox", {}) or {})), -float(item.get("confidence", 0.0) or 0.0), str(item.get("entity_id") or ""))):
        row_id = str(row.get("entity_id") or row.get("poi_id") or "")
        if row_id in consumed_ids:
            continue
        row_bbox = dict(row.get("bbox", {}) or {})
        row_area = _bbox_area_value(row_bbox)
        cluster = []
        for other in detector_candidates:
            other_id = str(other.get("entity_id") or other.get("poi_id") or "")
            if other_id == row_id or other_id in consumed_ids:
                continue
            other_bbox = dict(other.get("bbox", {}) or {})
            other_area = _bbox_area_value(other_bbox)
            if row_area < 80 or other_area < 80:
                continue
            high_overlap = _bbox_iou(row_bbox, other_bbox) >= 0.55
            contained = _bbox_contains_ratio(row_bbox, other_bbox) >= 0.85 or _bbox_contains_ratio(other_bbox, row_bbox) >= 0.85
            if not (high_overlap or contained):
                continue
            if _strongly_distinct_poi_role(row, other):
                continue
            cluster.append(other)
        if not cluster:
            final_rows.append(row)
            continue
        peer_group = [row] + cluster
        parent = max(
            peer_group,
            key=lambda item: (
                _bbox_area_value(dict(item.get("bbox", {}) or {})),
                float(item.get("canonical_track_persistence", 0.0) or 0.0),
                float(item.get("confidence", 0.0) or 0.0),
            ),
        )
        parent_id = str(parent.get("entity_id") or parent.get("poi_id") or "")
        retained_children: list[dict] = []
        merged_peers: list[str] = []
        input_ids: list[str] = []
        input_rows: list[dict] = []
        for peer in peer_group:
            peer_id = str(peer.get("entity_id") or peer.get("poi_id") or "")
            input_ids.append(peer_id)
            input_rows.append({"poi_id": peer_id, "bbox": dict(peer.get("bbox", {}) or {}), "provenance": list(peer.get("poi_source_provenance", []) or [])})
            if peer_id == parent_id:
                continue
            distinct_subregion = _bbox_iou(dict(parent.get("bbox", {}) or {}), dict(peer.get("bbox", {}) or {})) < 0.35 and _bbox_contains_ratio(dict(parent.get("bbox", {}) or {}), dict(peer.get("bbox", {}) or {})) < 0.8
            distinct_semantics = _strongly_distinct_poi_role(parent, peer)
            if distinct_subregion or distinct_semantics:
                child = dict(peer)
                child["parent_poi_id"] = parent_id
                child["poi_hierarchy_level"] = 1
                child["hierarchy_role"] = "functional_child"
                retained_children.append(child)
            else:
                merged_peers.append(peer_id)
                consumed_ids.add(peer_id)
        parent_payload = dict(parent)
        parent_payload["entity_id"] = parent_id
        parent_payload["poi_id"] = parent_id
        parent_payload["poi_hierarchy_level"] = 0
        parent_payload["parent_poi_id"] = None
        parent_payload["child_poi_ids"] = [str(child.get("entity_id") or child.get("poi_id") or "") for child in retained_children]
        parent_payload["hierarchy_role"] = "parent_region"
        parent_payload["merged_input_poi_ids"] = list(dict.fromkeys(list(parent_payload.get("merged_input_poi_ids", []) or []) + input_ids))
        final_rows.append(parent_payload)
        final_rows.extend(retained_children)
        for peer in peer_group:
            peer_id = str(peer.get("entity_id") or peer.get("poi_id") or "")
            if peer_id != parent_id:
                consumed_ids.add(peer_id)
        decision_debug.append(
            {
                "input_overlapping_pois": input_rows,
                "overlap_scores": [
                    {
                        "left": parent_id,
                        "right": str(peer.get("entity_id") or peer.get("poi_id") or ""),
                        "iou": _bbox_iou(dict(parent.get("bbox", {}) or {}), dict(peer.get("bbox", {}) or {})),
                        "containment": max(
                            _bbox_contains_ratio(dict(parent.get("bbox", {}) or {}), dict(peer.get("bbox", {}) or {})),
                            _bbox_contains_ratio(dict(peer.get("bbox", {}) or {}), dict(parent.get("bbox", {}) or {})),
                        ),
                    }
                    for peer in peer_group
                    if str(peer.get("entity_id") or peer.get("poi_id") or "") != parent_id
                ],
                "semantic_similarity": [
                    {
                        "left": parent_id,
                        "right": str(peer.get("entity_id") or peer.get("poi_id") or ""),
                        "same_semantic_label": _poi_semantic_label(parent) == _poi_semantic_label(peer),
                        "same_pattern_id": bool(parent.get("pattern_id")) and str(parent.get("pattern_id")) == str(peer.get("pattern_id")),
                    }
                    for peer in peer_group
                    if str(peer.get("entity_id") or peer.get("poi_id") or "") != parent_id
                ],
                "chosen_parent": parent_id,
                "retained_children": [str(child.get("entity_id") or child.get("poi_id") or "") for child in retained_children],
                "merged_peers": merged_peers,
                "reason": "high_overlap_detector_backed_cluster",
            }
        )
    collapsed_rows: list[dict] = []
    for row in sorted(final_rows, key=lambda item: (-_bbox_area_value(dict(item.get("bbox", {}) or {})), -float(item.get("confidence", 0.0) or 0.0), str(item.get("entity_id") or ""))):
        row_bbox = dict(row.get("bbox", {}) or {})
        row_area = _bbox_area_value(row_bbox)
        merged = False
        for parent in collapsed_rows:
            parent_bbox = dict(parent.get("bbox", {}) or {})
            parent_area = _bbox_area_value(parent_bbox)
            if row_area < 80 or parent_area < 80:
                continue
            if "detector" not in set(list(row.get("poi_source_provenance", []) or [])) or "detector" not in set(list(parent.get("poi_source_provenance", []) or [])):
                continue
            if float(row.get("canonical_track_persistence", 0.0) or 0.0) < 0.65 or float(parent.get("canonical_track_persistence", 0.0) or 0.0) < 0.65:
                continue
            same_band_geometry = (
                abs(int(parent_bbox.get("x1", 0) or 0) - int(row_bbox.get("x1", 0) or 0)) <= 2
                and abs(int(parent_bbox.get("x2", 0) or 0) - int(row_bbox.get("x2", 0) or 0)) <= 2
                and abs(int(parent_bbox.get("y1", 0) or 0) - int(row_bbox.get("y1", 0) or 0)) <= 3
                and abs(int(parent_bbox.get("y2", 0) or 0) - int(row_bbox.get("y2", 0) or 0)) <= 4
            )
            overlap_ok = (
                _bbox_iou(parent_bbox, row_bbox) >= 0.55
                or _bbox_contains_ratio(parent_bbox, row_bbox) >= 0.85
                or _bbox_contains_ratio(row_bbox, parent_bbox) >= 0.85
                or same_band_geometry
            )
            if not overlap_ok:
                continue
            if _strongly_distinct_poi_role(parent, row) and not same_band_geometry:
                continue
            parent["bbox"] = _merge_bbox(parent.get("bbox"), row.get("bbox"))
            parent["merged_input_poi_ids"] = list(dict.fromkeys(list(parent.get("merged_input_poi_ids", []) or []) + list(row.get("merged_input_poi_ids", []) or []) + [str(row.get("entity_id") or row.get("poi_id") or "")]))
            parent["poi_source_provenance"] = sorted(set(list(parent.get("poi_source_provenance", []) or []) + list(row.get("poi_source_provenance", []) or [])))
            parent["child_poi_ids"] = list(dict.fromkeys(list(parent.get("child_poi_ids", []) or []) + list(row.get("child_poi_ids", []) or [])))
            decision_debug.append(
                {
                    "input_overlapping_pois": [
                        {"poi_id": str(parent.get("entity_id") or parent.get("poi_id") or ""), "bbox": dict(parent_bbox), "provenance": list(parent.get("poi_source_provenance", []) or [])},
                        {"poi_id": str(row.get("entity_id") or row.get("poi_id") or ""), "bbox": dict(row_bbox), "provenance": list(row.get("poi_source_provenance", []) or [])},
                    ],
                    "overlap_scores": [
                        {
                            "left": str(parent.get("entity_id") or parent.get("poi_id") or ""),
                            "right": str(row.get("entity_id") or row.get("poi_id") or ""),
                            "iou": _bbox_iou(parent_bbox, row_bbox),
                            "containment": max(_bbox_contains_ratio(parent_bbox, row_bbox), _bbox_contains_ratio(row_bbox, parent_bbox)),
                        }
                    ],
                    "semantic_similarity": [
                        {
                            "left": str(parent.get("entity_id") or parent.get("poi_id") or ""),
                            "right": str(row.get("entity_id") or row.get("poi_id") or ""),
                            "same_pattern_id": bool(parent.get("pattern_id")) and str(parent.get("pattern_id")) == str(row.get("pattern_id")),
                            "same_semantic_label": _poi_semantic_label(parent) == _poi_semantic_label(row),
                        }
                    ],
                    "chosen_parent": str(parent.get("entity_id") or parent.get("poi_id") or ""),
                    "retained_children": list(parent.get("child_poi_ids", []) or []),
                    "merged_peers": [str(row.get("entity_id") or row.get("poi_id") or "")],
                    "reason": "final_large_overlap_parent_collapse",
                }
            )
            merged = True
            break
        if not merged:
            collapsed_rows.append(row)
    collapsed_rows.sort(key=lambda item: (int(item.get("poi_hierarchy_level", 0) or 0), -float(item.get("confidence", 0.0) or 0.0), str(item.get("entity_id") or "")))
    return collapsed_rows, decision_debug


def _collapse_same_level_central_parents(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    working = [dict(row) for row in list(rows or [])]
    debug_rows: list[dict] = []
    result: list[dict] = []
    for row in sorted(working, key=lambda item: (-_bbox_area_value(dict(item.get("bbox", {}) or {})), -float(item.get("confidence", 0.0) or 0.0), str(item.get("entity_id") or ""))):
        row_bbox = dict(row.get("bbox", {}) or {})
        merged = False
        for parent in result:
            parent_bbox = dict(parent.get("bbox", {}) or {})
            if not (
                bool(row.get("planner_targetable", False))
                and bool(parent.get("planner_targetable", False))
                and int(row.get("poi_hierarchy_level", 0) or 0) == 0
                and int(parent.get("poi_hierarchy_level", 0) or 0) == 0
                and str(row.get("hierarchy_role") or "parent_region") == "parent_region"
                and str(parent.get("hierarchy_role") or "parent_region") == "parent_region"
            ):
                continue
            if "detector" not in set(list(row.get("poi_source_provenance", []) or [])) or "detector" not in set(list(parent.get("poi_source_provenance", []) or [])):
                continue
            if float(row.get("canonical_track_persistence", 0.0) or 0.0) < 0.65 or float(parent.get("canonical_track_persistence", 0.0) or 0.0) < 0.65:
                continue
            distinct_child_coverage = bool(row.get("child_poi_ids")) or bool(parent.get("child_poi_ids"))
            if distinct_child_coverage:
                continue
            same_band_geometry = (
                abs(int(parent_bbox.get("x1", 0) or 0) - int(row_bbox.get("x1", 0) or 0)) <= 2
                and abs(int(parent_bbox.get("x2", 0) or 0) - int(row_bbox.get("x2", 0) or 0)) <= 2
                and abs(int(parent_bbox.get("y1", 0) or 0) - int(row_bbox.get("y1", 0) or 0)) <= 3
                and abs(int(parent_bbox.get("y2", 0) or 0) - int(row_bbox.get("y2", 0) or 0)) <= 4
            )
            iou = _bbox_iou(parent_bbox, row_bbox)
            contain_pr = _bbox_contains_ratio(parent_bbox, row_bbox)
            contain_rp = _bbox_contains_ratio(row_bbox, parent_bbox)
            if not (iou >= 0.55 or contain_pr >= 0.85 or contain_rp >= 0.85 or same_band_geometry):
                continue
            if _strongly_distinct_poi_role(parent, row) and not same_band_geometry:
                continue
            parent["bbox"] = _merge_bbox(parent.get("bbox"), row.get("bbox"))
            parent["merged_input_poi_ids"] = list(dict.fromkeys(list(parent.get("merged_input_poi_ids", []) or []) + list(row.get("merged_input_poi_ids", []) or []) + [str(row.get("entity_id") or row.get("poi_id") or "")]))
            debug_rows.append(
                {
                    "input_overlapping_pois": [
                        {"poi_id": str(parent.get("entity_id") or parent.get("poi_id") or ""), "bbox": dict(parent_bbox), "provenance": list(parent.get("poi_source_provenance", []) or [])},
                        {"poi_id": str(row.get("entity_id") or row.get("poi_id") or ""), "bbox": dict(row_bbox), "provenance": list(row.get("poi_source_provenance", []) or [])},
                    ],
                    "overlap_scores": [
                        {
                            "left": str(parent.get("entity_id") or parent.get("poi_id") or ""),
                            "right": str(row.get("entity_id") or row.get("poi_id") or ""),
                            "iou": iou,
                            "containment": max(contain_pr, contain_rp),
                        }
                    ],
                    "semantic_similarity": [
                        {
                            "left": str(parent.get("entity_id") or parent.get("poi_id") or ""),
                            "right": str(row.get("entity_id") or row.get("poi_id") or ""),
                            "same_pattern_id": bool(parent.get("pattern_id")) and str(parent.get("pattern_id")) == str(row.get("pattern_id")),
                            "same_semantic_label": _poi_semantic_label(parent) == _poi_semantic_label(row),
                        }
                    ],
                    "chosen_parent": str(parent.get("entity_id") or parent.get("poi_id") or ""),
                    "retained_children": list(parent.get("child_poi_ids", []) or []),
                    "merged_peers": [str(row.get("entity_id") or row.get("poi_id") or "")],
                    "reason": "same_level_central_parent_collapse",
                }
            )
            merged = True
            break
        if not merged:
            result.append(row)
    return result, debug_rows


def _collapsed_peer_ids(*decision_sets: list[dict]) -> set[str]:
    collapsed: set[str] = set()
    for decisions in decision_sets:
        for row in list(decisions or []):
            for peer_id in list(row.get("merged_peers", []) or []):
                if peer_id:
                    collapsed.add(str(peer_id))
    return collapsed


def _filter_collapsed_peers(rows: list[dict], collapsed_peer_ids: set[str]) -> list[dict]:
    if not collapsed_peer_ids:
        return [dict(row) for row in list(rows or [])]
    filtered: list[dict] = []
    for row in list(rows or []):
        row_id = str(row.get("entity_id") or row.get("poi_id") or "")
        if row_id in collapsed_peer_ids:
            continue
        filtered.append(dict(row))
    return filtered


def _assert_no_collapsed_peers(rows: list[dict], collapsed_peer_ids: set[str], *, stage: str) -> None:
    if not collapsed_peer_ids:
        return
    remaining = sorted(
        str(row.get("entity_id") or row.get("poi_id") or "")
        for row in list(rows or [])
        if str(row.get("entity_id") or row.get("poi_id") or "") in collapsed_peer_ids
    )
    if remaining:
        raise AssertionError(f"{stage}: collapsed peers still present: {remaining}")


def _cross_canonicalize_pois(detector_pois: list[dict], promoted_structure_pois: list[dict]) -> tuple[list[dict], dict]:
    merged: dict[str, dict] = {}
    collapse_debug: list[dict] = []

    def _source_label(row: dict) -> str:
        return "detector" if str(row.get("promotion_source") or "") != "supplemental_structure_admission" else "structure_promotion"

    def _merge_row(row: dict) -> None:
        incoming_row = dict(row)
        incoming_row.setdefault("poi_source_provenance", [_source_label(incoming_row)])
        incoming_row.setdefault("merged_input_poi_ids", [str(incoming_row.get("poi_id") or incoming_row.get("entity_id") or "")])
        match_id = None
        best_score = -1.0
        hinted_id = str(incoming_row.get("stable_entity_id_hint") or "")
        if hinted_id and hinted_id in merged:
            match_id = hinted_id
            best_score = 999.0
        for entity_id, candidate in merged.items():
            if match_id is not None and best_score >= 999.0:
                break
            score = _match_score(candidate, incoming_row)
            if score > best_score:
                best_score = score
                match_id = entity_id
        if match_id is None or best_score < 0.65:
            match_id = _stable_entity_id(incoming_row)
        prior = merged.get(match_id, {})
        if prior:
            prior_sources = list(prior.get("poi_source_provenance", []) or [])
            new_sources = list(incoming_row.get("poi_source_provenance", []) or [])
            collapse_type = "detector vs structure-promotion duplicate"
            if set(prior_sources) == {"detector"} and set(new_sources) == {"detector"}:
                collapse_type = "detector vs detector duplicate"
            elif set(prior_sources) == {"structure_promotion"} and set(new_sources) == {"structure_promotion"}:
                collapse_type = "structure-promotion vs structure-promotion duplicate"
            collapse_debug.append(
                {
                    "kept_entity_id": match_id,
                    "incoming_poi_id": str(incoming_row.get("poi_id") or incoming_row.get("entity_id") or ""),
                    "prior_poi_ids": list(prior.get("merged_input_poi_ids", []) or []),
                    "incoming_source": _source_label(incoming_row),
                    "prior_sources": list(prior_sources),
                    "collapse_type": collapse_type,
                    "match_score": float(best_score),
                }
            )
        payload = dict(prior)
        payload.update(incoming_row)
        payload["entity_id"] = match_id
        payload["poi_id"] = match_id
        payload["stable_entity_id"] = match_id
        payload["bbox"] = _merge_bbox(prior.get("bbox"), incoming_row.get("bbox"))
        payload["centroid"] = incoming_row.get("centroid") or prior.get("centroid")
        payload["confidence"] = max(float(prior.get("confidence", 0.0) or 0.0), float(incoming_row.get("confidence", 0.0) or 0.0))
        payload["observations"] = int(prior.get("observations", 0) or 0) + int(incoming_row.get("observations", 1) or 1)
        payload["poi_source_provenance"] = sorted(set(list(prior.get("poi_source_provenance", []) or []) + list(incoming_row.get("poi_source_provenance", []) or [])))
        payload["merged_input_poi_ids"] = list(dict.fromkeys(list(prior.get("merged_input_poi_ids", []) or []) + list(incoming_row.get("merged_input_poi_ids", []) or [])))
        payload["canonical_track_persistence"] = max(float(prior.get("canonical_track_persistence", 0.0) or 0.0), float(incoming_row.get("canonical_track_persistence", 0.0) or 0.0))
        merged[match_id] = payload

    for row in list(detector_pois or []):
        _merge_row(dict(row, poi_source_provenance=["detector"], planner_targetable=True))
    for row in list(promoted_structure_pois or []):
        _merge_row(dict(row, poi_source_provenance=["structure_promotion"], planner_targetable=bool(row.get("planner_targetable", False))))
    rows = sorted(
        merged.values(),
        key=lambda row: (-float(row.get("confidence", 0.0) or 0.0), -float(row.get("utility", 0.0) or 0.0), str(row.get("entity_id") or "")),
    )
    for row in rows:
        provenance = set(list(row.get("poi_source_provenance", []) or []))
        detector_supported = "detector" in provenance
        structure_only = provenance == {"structure_promotion"}
        row["planner_targetable"] = bool(
            detector_supported
            or (
                not structure_only
                and bool(row.get("planner_targetable", False))
            )
        )
        row.setdefault("poi_hierarchy_level", 0)
        row.setdefault("parent_poi_id", None)
        row.setdefault("child_poi_ids", [])
        row.setdefault("hierarchy_role", "parent_region")
    final_rows, hierarchy_debug = _central_parent_child_canonicalize(rows)
    return final_rows, {
        "cross_canonicalized_poi_count": len(final_rows),
        "cross_canonicalization_collapses": list(collapse_debug),
        "central_poi_hierarchy_decisions": list(hierarchy_debug),
    }


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
    classifier_truth_surface: dict | None = None,
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
    outcome_summary = dict(raw_episode.metadata.get("outcome_summary", {}) or {}) if isinstance(raw_episode.metadata, dict) else {}
    classifier_truth_surface = dict(classifier_truth_surface or outcome_summary or {})
    outcome_telemetry = {
        key: outcome_summary.get(key)
        for key in (
            "counterfactual_evidence_observed",
            "exit_attempt_evidence_observed",
            "expected_effect_type",
            "expected_relation_type",
            "expected_effect_relation",
            "expected_target_id",
            "expected_trigger_contact_observed",
            "expected_region_reached",
            "observed_effect_change",
            "observed_effect_absent",
            "expected_effect_absent",
            "attempted_boundary_contact",
            "attempted_portal_contact",
            "attempted_terminal_affordance_contact",
            "attempted_escape_direction",
            "exit_attempt_target_id",
            "weak_expectation_basis",
            "blocked",
            "trigger_contact_observed",
            "target_contact_observed",
        )
        if key in outcome_summary
    }
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
                "telemetry": {**dict(raw_step.info or {}), **outcome_telemetry},
            }
        )
    topology_nodes, topology_edges = _topology_from_steps(step_rows)
    topology_nodes, topology_edges = _mode_select_topology(topology_nodes, topology_edges, analysis_mode=analysis_mode)
    consequences = _mode_select_consequences(_consequences(raw_episode, motion, step_summaries, step_rows), analysis_mode=analysis_mode)
    topology_edges, consequences, support_family_emit_debug = _annotate_directed_support(
        step_rows,
        topology_edges,
        consequences,
        analysis_mode=analysis_mode,
        classifier_truth_surface=classifier_truth_surface,
    )
    detected_pois_raw, detector_debug = detect_pois(step_summaries, avatar_tracking, step_rows)
    detected_pois = _mode_select_pois(detected_pois_raw, analysis_mode=analysis_mode)
    supplemental_structure = _supplemental_structure_entities(
        step_summaries=step_summaries,
        normalized_observations=normalized_observations,
        area_sequence=area_sequence,
    )
    promoted_structure_pois, structure_promotion_debug = _promote_structure_entities_to_pois(supplemental_structure)
    canonical_planner_pois, cross_canonicalization_debug = _cross_canonicalize_pois(detected_pois, promoted_structure_pois)
    canonical_planner_pois = _adopt_prior_poi_ids(canonical_planner_pois, blackboard_snapshot)
    canonical_planner_pois, same_level_parent_debug = _collapse_same_level_central_parents(canonical_planner_pois)
    collapsed_peer_ids = _collapsed_peer_ids(
        list(cross_canonicalization_debug.get("central_poi_hierarchy_decisions", [])),
        list(same_level_parent_debug or []),
    )
    canonical_planner_pois = _filter_collapsed_peers(canonical_planner_pois, collapsed_peer_ids)
    _assert_no_collapsed_peers(canonical_planner_pois, collapsed_peer_ids, stage="post_collapse_canonical_planner_pois")
    planner_pois = _attach_target_effects_to_pois(
        _attach_identity_to_pois(
        _annotate_pattern_descriptors(
        canonical_planner_pois,
        step_summaries=step_summaries,
        normalized_observations=normalized_observations,
        ),
        step_summaries=step_summaries,
        ),
        step_rows=step_rows,
        blackboard_snapshot=blackboard_snapshot,
    )
    planner_pois = _filter_collapsed_peers(planner_pois, collapsed_peer_ids)
    _assert_no_collapsed_peers(planner_pois, collapsed_peer_ids, stage="post_effect_planner_pois")
    supplemental_entities = _attach_target_effects_to_pois(
        _attach_identity_to_pois(
        _annotate_pattern_descriptors(
        supplemental_structure,
        step_summaries=step_summaries,
        normalized_observations=normalized_observations,
        ),
        step_summaries=step_summaries,
        ),
        step_rows=step_rows,
        blackboard_snapshot=blackboard_snapshot,
    )
    entity_rows = _merge_entity_candidates(planner_pois, supplemental_entities)
    entity_rows = _filter_collapsed_peers(entity_rows, collapsed_peer_ids)
    _assert_no_collapsed_peers(entity_rows, collapsed_peer_ids, stage="entity_rows_before_delta")
    trigger_zones = _collapse_trigger_zones(
        _extract_trigger_zones(step_summaries=step_summaries, step_rows=step_rows, analysis_mode=analysis_mode),
        analysis_mode=analysis_mode,
    )
    poi_detection_debug = dict(detector_debug or {})
    if not poi_detection_debug:
        poi_detection_debug = {
            "raw_poi_candidates": [],
            "rejected_tiny_candidates": [],
            "merged_clusters": [],
            "final_exported_canonical_pois": [],
            "supplemental_structure_candidates": [],
        }
    poi_detection_debug["supplemental_structure_candidates"] = [
        {
            "entity_id": str(row.get("entity_id") or ""),
            "bbox": dict(row.get("bbox", {}) or {}),
            "area": int(row.get("area", 0) or 0),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "identity_confidence": float(row.get("identity_confidence", 0.0) or 0.0),
            "pattern_id": row.get("pattern_id"),
        }
        for row in list(supplemental_structure or [])
    ]
    poi_detection_debug["promoted_structure_pois"] = list(structure_promotion_debug.get("promoted_structure_pois", []))
    poi_detection_debug["structure_entity_candidate_count"] = int(structure_promotion_debug.get("structure_entity_candidate_count", 0) or 0)
    poi_detection_debug["structure_entity_promoted_count"] = int(structure_promotion_debug.get("structure_entity_promoted_count", 0) or 0)
    poi_detection_debug["structure_entity_rejection_reason_counts"] = dict(structure_promotion_debug.get("structure_entity_rejection_reason_counts", {}) or {})
    poi_detection_debug["cross_canonicalized_poi_count"] = int(cross_canonicalization_debug.get("cross_canonicalized_poi_count", 0) or 0)
    poi_detection_debug["cross_canonicalization_collapses"] = list(cross_canonicalization_debug.get("cross_canonicalization_collapses", []))
    poi_detection_debug["central_poi_hierarchy_decisions"] = list(cross_canonicalization_debug.get("central_poi_hierarchy_decisions", [])) + list(same_level_parent_debug or [])
    poi_detection_debug["collapsed_peer_ids"] = sorted(collapsed_peer_ids)
    poi_detection_debug["detector_only_poi_count"] = len(list(detected_pois or []))
    poi_detection_debug["promoted_structure_poi_count"] = len(list(promoted_structure_pois or []))
    canonical_exported_pois = [
        row
        for row in list(entity_rows or [])
        if str(row.get("kind") or "") == "poi" and bool(row.get("planner_visible", True))
    ]
    canonical_exported_pois = _filter_collapsed_peers(canonical_exported_pois, collapsed_peer_ids)
    _assert_no_collapsed_peers(canonical_exported_pois, collapsed_peer_ids, stage="final_exported_canonical_pois")
    poi_detection_debug["final_exported_canonical_pois"] = [
        {
            "poi_id": str(row.get("poi_id") or row.get("entity_id") or ""),
            "planner_visible": bool(row.get("planner_visible", True)),
            "poi_bucket": str(row.get("poi_bucket") or ""),
            "poi_class": str(row.get("poi_class") or row.get("kind") or ""),
            "bbox": dict(row.get("bbox", {}) or {}),
            "area": int(row.get("area", 0) or 0),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "poi_source_provenance": list(row.get("poi_source_provenance", []) or []),
            "merged_input_poi_ids": list(row.get("merged_input_poi_ids", []) or []),
            "planner_targetable": bool(row.get("planner_targetable", False)),
            "poi_hierarchy_level": int(row.get("poi_hierarchy_level", 0) or 0),
            "parent_poi_id": row.get("parent_poi_id"),
            "child_poi_ids": list(row.get("child_poi_ids", []) or []),
            "hierarchy_role": str(row.get("hierarchy_role") or "parent_region"),
        }
        for row in canonical_exported_pois
    ]
    poi_detection_debug["final_exported_count"] = len(canonical_exported_pois)
    poi_detection_debug["final_post_merge_observed_poi_count"] = len(canonical_exported_pois)
    poi_detection_debug["debug_source"] = "canonical_export_pipeline"

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
            for poi in entity_rows
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
                direct_evidence_fields=_consequence_direct_evidence_fields(row)
                + [field for field in ["supports_directed_outcome_relation", "supports_exit_attempt_relation", "supports_counterfactual_relation"] if bool(row.get(field, False))],
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
                direct_evidence_fields=_topology_edge_direct_evidence_fields(row)
                + [field for field in ["supports_directed_outcome_relation", "supports_exit_attempt_relation", "supports_counterfactual_relation"] if bool(row.get(field, False))],
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
            "poi_detection_debug": poi_detection_debug,
            "collapsed_poi_ids": sorted(collapsed_peer_ids),
            "support_family_emit_debug": support_family_emit_debug,
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
        points_of_interest=tuple(planner_pois),
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
            "poi_detection_debug": poi_detection_debug,
            "support_family_emit_debug": support_family_emit_debug,
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
