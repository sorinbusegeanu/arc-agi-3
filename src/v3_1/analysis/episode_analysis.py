from __future__ import annotations

from collections import Counter

from v3_1.analysis.adapters_env import normalize_observation
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


def analyze_episode(raw_episode: RawEpisode) -> AnalyzedEpisode:
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
            }
        )
    topology_nodes, topology_edges = _topology_from_steps(step_rows)
    consequences = _consequences(raw_episode, motion, step_summaries, step_rows)
    pois = detect_pois(step_summaries, avatar_tracking, step_rows)

    delta = BlackboardDelta(
        session_id=raw_episode.session_id,
        run_id=raw_episode.run_id,
        game_id=raw_episode.game_id,
        round_id=raw_episode.round_id,
        pass_id=raw_episode.pass_id,
        episode_id=raw_episode.episode_id,
        delta_id=f"delta:{raw_episode.episode_id}:{stable_digest({'steps': step_rows})}",
        areas=tuple(
            {
                "area_id": area["area_id"],
                "area_signature": area["area_signature"],
                "width": area["width"],
                "height": area["height"],
                "palette": area["palette"],
                "background_color": area["background_color"],
                "state_hash": area["state_hash"],
                "visit_count": 1,
            }
            for area in known_areas
        ),
        entities=tuple(
            dict(
                poi,
                area_id=next(
                    (
                        area_sequence[index]["area_id"]
                        for index, summary in enumerate(step_summaries)
                        if any(obj["signature"] == poi["signature"] for obj in summary["objects"])
                    ),
                    area_sequence[-1]["area_id"] if area_sequence else None,
                ),
            )
            for poi in pois
        ),
        consequences=tuple(consequences),
        trigger_zones=(),
        topology_nodes=tuple(topology_nodes),
        topology_edges=tuple(topology_edges),
        evidence=tuple(f"{raw_episode.episode_id}:{row['step_idx']}" for row in step_rows),
        material_change=bool(step_rows),
        metadata={
            "main_track_id": avatar_tracking["main_track_id"],
            "area_sequence": [area["area_id"] for area in area_sequence],
            "avatar_path": motion["avatar_path"],
            "step_rows": step_rows,
        },
    )

    return AnalyzedEpisode(
        session_id=raw_episode.session_id,
        run_id=raw_episode.run_id,
        game_id=raw_episode.game_id,
        round_id=raw_episode.round_id,
        pass_id=raw_episode.pass_id,
        episode_id=raw_episode.episode_id,
        raw_episode_id=raw_episode.episode_id,
        summary={
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
        metadata={
            "step_summaries": step_summaries,
            "avatar_tracking": avatar_tracking,
            "motion": motion,
        },
    )
