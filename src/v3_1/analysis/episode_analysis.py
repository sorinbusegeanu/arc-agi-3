from __future__ import annotations

from collections import Counter

from v3_1.analysis.adapters_env import normalize_observation
from v3_1.analysis.area_assignment import assign_area
from v3_1.analysis.avatar_tracking import track_avatar
from v3_1.analysis.motion_analysis import summarize_motion
from v3_1.analysis.observation_summary import summarize_observation
from v3_1.analysis.poi_detection import detect_pois
from v3_1.contracts.messages import AnalyzedEpisode, BlackboardDelta, RawEpisode
from v3_1.utils.ids import stable_digest


def _topology_from_steps(step_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    node_cells: dict[str, list[int]] = {}
    previous_node_id = None
    for step in step_rows:
        centroid = step.get("avatar_centroid")
        if centroid is None:
            previous_node_id = None
            continue
        cell = [int(float(centroid[0])), int(float(centroid[1]))]
        node_id = f"cell:{cell[0]}:{cell[1]}"
        node_counts[node_id] += 1
        node_cells[node_id] = cell
        if previous_node_id is not None:
            edge_counts[(previous_node_id, node_id)] += 1
        previous_node_id = node_id
    nodes = [{"node_id": node_id, "cell": cell, "visits": count} for node_id, count in node_counts.items() for cell in [node_cells[node_id]]]
    edges = [{"edge_id": f"{src}->{dst}", "src": src, "dst": dst, "count": count} for (src, dst), count in edge_counts.items()]
    nodes.sort(key=lambda row: row["node_id"])
    edges.sort(key=lambda row: row["edge_id"])
    return nodes, edges


def _consequences(raw_episode: RawEpisode, motion: dict, step_summaries: list[dict]) -> list[dict]:
    rows = []
    for step_idx, movement_row in enumerate(motion["movement_rows"]):
        if movement_row["local_change_area"] <= 0 and not raw_episode.steps[step_idx].done and not raw_episode.steps[step_idx].reward:
            continue
        rows.append(
            {
                "consequence_id": f"consequence:{raw_episode.episode_id}:{step_idx}",
                "step_idx": step_idx,
                "reward": raw_episode.steps[step_idx].reward,
                "done": raw_episode.steps[step_idx].done,
                "local_change_area": movement_row["local_change_area"],
                "blocked": movement_row["blocked"],
                "action_effect_near_avatar": movement_row["action_effect_near_avatar"],
                "evidence_count": max(1, len(step_summaries[step_idx].get("change_regions", []))),
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

    avatar_tracking = track_avatar(step_summaries)
    motion = summarize_motion(raw_episode.steps, step_summaries, avatar_tracking)
    pois = detect_pois(step_summaries, avatar_tracking)
    topology_nodes, topology_edges = _topology_from_steps(motion["movement_rows"])
    consequences = _consequences(raw_episode, motion, step_summaries)

    step_rows = []
    for step_idx, (raw_step, summary, area) in enumerate(zip(raw_episode.steps, step_summaries, area_sequence)):
        tracking_row = next((row for row in avatar_tracking["per_step"] if row["step_idx"] == step_idx), None)
        step_rows.append(
            {
                "step_idx": step_idx,
                "action": raw_step.action,
                "reward": raw_step.reward,
                "done": raw_step.done,
                "area_id": area["area_id"],
                "state_hash": summary["state_identity"]["state_hash"],
                "avatar_centroid": tracking_row["main_centroid"] if tracking_row is not None else None,
                "object_ids": [obj["object_id"] for obj in summary["objects"]],
                "change_region_count": len(summary["change_regions"]),
            }
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
