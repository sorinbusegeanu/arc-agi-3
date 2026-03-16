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
    if analysis_mode == "probe":
        for summary, step in zip(step_summaries, step_rows):
            for region_index, region in enumerate(list(summary.get("change_regions", []) or [])):
                bbox = list(region.get("bbox", [])) if isinstance(region.get("bbox"), list) else None
                if not bbox or len(bbox) != 4:
                    continue
                rows.append(
                    {
                        "trigger_id": f"trigger:{step.get('step_idx', 0)}:{region_index}",
                        "entity_id": None,
                        "area_id": step.get("area_id"),
                        "zone_bbox": bbox,
                        "supporting_steps": [int(step.get("step_idx", 0) or 0)],
                        "confidence": min(0.8, 0.25 + (0.02 * int(region.get("area", 0) or 0))),
                        "evidence_refs": [f"{step.get('step_idx', 0)}:{region_index}"],
                        "trigger_kind": "suspicious_region",
                    }
                )
        return rows
    for summary, step in zip(step_summaries, step_rows):
        telemetry = dict(step.get("telemetry", {}) or {})
        effect_region = dict(telemetry.get("effect_region", {}) or {})
        bbox = list(effect_region.get("bbox", [])) if isinstance(effect_region.get("bbox"), list) else None
        if not bbox or len(bbox) != 4:
            target_coordinates = step.get("target_coordinates")
            if isinstance(target_coordinates, (list, tuple)) and len(target_coordinates) == 2:
                x, y = int(target_coordinates[0]), int(target_coordinates[1])
                bbox = [x, y, x, y]
        if not bbox or len(bbox) != 4:
            continue
        rows.append(
            {
                "trigger_id": f"trigger:directed:{step.get('step_idx', 0)}",
                "entity_id": step.get("target_entity_id"),
                "area_id": step.get("area_id"),
                "zone_bbox": bbox,
                "supporting_steps": [int(step.get("step_idx", 0) or 0)],
                "confidence": min(1.0, 0.35 + (0.01 * int(step.get("changed_cells", 0) or 0))),
                "evidence_refs": [f"{step.get('step_idx', 0)}:directed"],
                "trigger_kind": "localized_attribution",
                "effect_changed_cells": int(step.get("changed_cells", 0) or 0),
            }
        )
    return rows


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
        bbox = list(payload.get("bbox", [])) if isinstance(payload.get("bbox"), list) else None
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


def _attach_target_effects_to_pois(pois: list[dict], *, step_rows: list[dict], blackboard_snapshot: dict | None) -> list[dict]:
    blackboard_entities = dict((blackboard_snapshot or {}).get("state", {}).get("entities", {})) if isinstance((blackboard_snapshot or {}).get("state"), dict) else dict((blackboard_snapshot or {}).get("entities", {}))
    if not pois or not step_rows or not blackboard_entities:
        return [dict(poi) for poi in pois]

    attributed = [dict(poi) for poi in pois]
    by_target: dict[str, dict[str, int | float | str]] = {}
    poi_by_index = {index: row for index, row in enumerate(attributed)}

    def _match_poi_index(target_id: str) -> int | None:
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

    for step_idx, observation in enumerate(normalized_observations):
        previous = normalized_observations[step_idx - 1] if step_idx > 0 else None
        summary = summarize_observation(observation, previous)
        area = assign_area(summary, known_areas)
        area_sequence.append(area)
        if not any(existing["area_id"] == area["area_id"] for existing in known_areas):
            known_areas.append(area)
        enriched = dict(summary)
        enriched["step_idx"] = step_idx
        enriched["area_id"] = area["area_id"]
        step_summaries.append(enriched)

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
    pois = _attach_target_effects_to_pois(
        _annotate_pattern_descriptors(
        _mode_select_pois(detect_pois(step_summaries, avatar_tracking, step_rows), analysis_mode=analysis_mode),
        step_summaries=step_summaries,
        normalized_observations=normalized_observations,
        ),
        step_rows=step_rows,
        blackboard_snapshot=blackboard_snapshot,
    )
    trigger_zones = _extract_trigger_zones(step_summaries=step_summaries, step_rows=step_rows, analysis_mode=analysis_mode)

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
            _stamp_row(dict(
                poi,
                analysis_mode=analysis_mode,
                source_stage="analysis",
                source_pass_id=raw_episode.pass_id,
                source_episode_id=raw_episode.episode_id,
                confidence=float(poi.get("confidence", 0.0)),
                inference_method="poi_detection",
                factual_observation=False,
                area_id=next(
                    (
                        area_sequence[index]["area_id"]
                        for index, summary in enumerate(step_summaries)
                        if any(obj["signature"] == poi["signature"] for obj in summary["objects"])
                    ),
                    area_sequence[-1]["area_id"] if area_sequence else None,
                ),
            ), analysis_mode=analysis_mode, row_kind="entities", direct_evidence_present=False, direct_evidence_fields=[], contradiction_flag=bool(poi.get("rejected")), observation_support_span=(0, max(0, int(poi.get("observations", 1) or 1) - 1)))
            for poi in pois
        ),
        consequences=tuple(
            _stamp_row(dict(
                row,
                analysis_mode=analysis_mode,
                source_stage="analysis",
                source_pass_id=raw_episode.pass_id,
                source_episode_id=raw_episode.episode_id,
                confidence=min(1.0, 0.45 + (0.1 * float(row.get("evidence_count", 1) or 1))),
                inference_method="consequence_reconstruction",
                factual_observation=False,
            ), analysis_mode=analysis_mode, row_kind="consequences", direct_evidence_present=False, direct_evidence_fields=[], contradiction_flag=bool(row.get("blocked") and not row.get("action_effect_near_avatar")), observation_support_span=(int(row.get("step_idx", 0) or 0), int(row.get("step_idx", 0) or 0)))
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
            _stamp_row(dict(
                row,
                analysis_mode=analysis_mode,
                source_stage="analysis",
                source_pass_id=raw_episode.pass_id,
                source_episode_id=raw_episode.episode_id,
                confidence=0.6 if analysis_mode == "probe" else 0.7,
                inference_method="topology_growth" if analysis_mode == "probe" else "route_progress_reconstruction",
                factual_observation=False,
            ), analysis_mode=analysis_mode, row_kind="topology_nodes", direct_evidence_present=False, direct_evidence_fields=[], observation_support_span=(0, max(0, len(step_rows) - 1)))
            for row in topology_nodes
        ),
        topology_edges=tuple(
            _stamp_row(dict(
                row,
                analysis_mode=analysis_mode,
                source_stage="analysis",
                source_pass_id=raw_episode.pass_id,
                source_episode_id=raw_episode.episode_id,
                confidence=0.6 if analysis_mode == "probe" else 0.75,
                inference_method="topology_growth" if analysis_mode == "probe" else "route_progress_reconstruction",
                factual_observation=False,
            ), analysis_mode=analysis_mode, row_kind="topology_edges", direct_evidence_present=False, direct_evidence_fields=[], observation_support_span=(0, max(0, len(step_rows) - 1)))
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
        },
    )
    mechanic_graph_delta, deterministic_hypothesis_bundle, llm_hypothesis_bundle = extract_mechanic_graph_delta(
        raw_episode,
        analyzed_episode,
        current_blackboard_snapshot=blackboard_snapshot,
        current_mechanic_graph_snapshot=mechanic_graph_snapshot,
        hypothesis_config=hypothesis_config,
        llm_adapter=llm_adapter,
        hypothesis_registry_snapshot=hypothesis_registry_snapshot,
    )
    return AnalyzedEpisode(
        **{
            **analyzed_episode.__dict__,
            "mechanic_graph_delta": mechanic_graph_delta,
            "deterministic_hypothesis_bundle": deterministic_hypothesis_bundle,
            "llm_hypothesis_bundle": llm_hypothesis_bundle,
        },
    )
