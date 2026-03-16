from __future__ import annotations

from collections import Counter


def episode_export_row(analyzed_episode) -> dict:
    step_rows = list(analyzed_episode.summary.get("step_rows", []) or [])
    steps = []
    for row in step_rows:
        cell = row.get("avatar_cell")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            continue
        steps.append(
            {
                "step_idx": int(row.get("step_idx", len(steps))),
                "avatar_cell": [int(cell[0]), int(cell[1])],
                "action_id": row.get("action_id"),
                "action_name": row.get("action_name"),
                "action_family": row.get("action_family", "unknown"),
                "changed_cells": int(row.get("changed_cells", 0) or 0),
                "transition_type": "move" if str(row.get("action_family") or "").strip().lower() == "move" else "action",
                "evidence_refs": [f"{analyzed_episode.episode_id}:{int(row.get('step_idx', len(steps)))}"],
            }
        )
    pois = [
        {"poi_id": str(poi.get("poi_id") or poi.get("entity_id") or ""), "centroid": list(poi.get("centroid", []))}
        for poi in analyzed_episode.points_of_interest
        if isinstance(poi.get("centroid"), (list, tuple)) and len(poi.get("centroid")) == 2
    ]
    return {"episode_id": analyzed_episode.episode_id, "steps": steps, "pois": pois}


def build_round_analysis_summary(*, round_id: int, analyzed_episodes: list, candidate_effect_mode_used: str) -> dict:
    step_rows = [
        row
        for episode in analyzed_episodes
        for row in list(getattr(episode, "summary", {}).get("step_rows", []) or [])
    ]
    normalized_action_types = [str(row.get("action_name") or "").strip().lower() for row in step_rows]
    normalized_action_families = [str(row.get("action_family") or "unknown").strip().lower() for row in step_rows]
    action_type_histogram = dict(sorted(Counter(normalized_action_types).items()))
    changed_steps_count = sum(1 for row in step_rows if int(row.get("changed_cells", 0) or 0) > 0)
    move_steps_count = sum(1 for action_family in normalized_action_families if action_family == "move")
    interact_steps_count = sum(1 for action_family in normalized_action_families if action_family == "interact")
    click_steps_count = sum(1 for action_family in normalized_action_families if action_family == "click_at")
    undo_steps_count = sum(1 for action_family in normalized_action_families if action_family == "undo")
    reset_steps_count = sum(1 for action_family in normalized_action_families if action_family == "reset")
    movement_steps_with_change = sum(
        1 for row, action_family in zip(step_rows, normalized_action_families)
        if action_family == "move" and int(row.get("changed_cells", 0) or 0) > 0
    )
    interact_steps_with_change = sum(
        1 for row, action_family in zip(step_rows, normalized_action_families)
        if action_family == "interact" and int(row.get("changed_cells", 0) or 0) > 0
    )
    click_steps_with_change = sum(
        1 for row, action_family in zip(step_rows, normalized_action_families)
        if action_family == "click_at" and int(row.get("changed_cells", 0) or 0) > 0
    )
    unknown_action_type_count = sum(1 for action_family in normalized_action_families if action_family == "unknown")
    return {
        "round_id": int(round_id),
        "step_count": len(step_rows),
        "changed_steps_count": changed_steps_count,
        "move_steps_count": move_steps_count,
        "interact_steps_count": interact_steps_count,
        "click_steps_count": click_steps_count,
        "undo_steps_count": undo_steps_count,
        "reset_steps_count": reset_steps_count,
        "movement_steps_with_change": movement_steps_with_change,
        "interact_steps_with_change": interact_steps_with_change,
        "click_steps_with_change": click_steps_with_change,
        "action_type_histogram": action_type_histogram,
        "unknown_action_type_count": unknown_action_type_count,
        "candidate_effect_mode_used": candidate_effect_mode_used,
    }


def actual_effect_mode(step_rows: list[dict], fallback: str) -> str:
    families = [str(row.get("action_family") or "unknown").strip().lower() for row in step_rows]
    if any(family == "click_at" for family in families):
        return "click_at"
    if any(family == "interact" for family in families):
        return "interact"
    if any(family == "move" for family in families):
        return "move"
    return fallback


def available_families_from_blackboard(state: dict) -> set[str]:
    families = {"move"}
    for entity in state.get("entities", {}).values():
        if float(entity.get("interact_effect_score", 0.0)) > 0.0 or int(entity.get("interact_attempts", 0) or 0) > 0:
            families.add("interact")
        if float(entity.get("click_effect_score", 0.0)) > 0.0 or int(entity.get("click_attempts", 0) or 0) > 0:
            families.add("click_at")
    for consequence in state.get("consequences", {}).values():
        family = str(consequence.get("action_family") or "").strip().lower()
        if family in {"interact", "click_at"}:
            families.add(family)
    return families


def normalize_candidate_for_export(candidate: dict, *, available_families: set[str], executed_family: str | None = None) -> dict:
    row = dict(candidate or {})
    required_family = str(row.get("required_action_family") or "unknown").lower()
    effect_family = str(row.get("effect_action_family") or required_family).lower()
    chosen_family = executed_family or effect_family or required_family
    if chosen_family not in available_families or available_families == {"move"}:
        chosen_family = "move"
    row["required_action_family"] = chosen_family
    action = dict(row.get("action", {}))
    action["type"] = chosen_family
    action["required_action_family"] = chosen_family
    row["action"] = action
    return row


def decision_export_payload(decision, *, available_families: set[str], executed_family: str) -> dict:
    payload = dict(decision.__dict__)
    payload["ranked_candidates"] = tuple(
        normalize_candidate_for_export(candidate, available_families=available_families)
        for candidate in payload.get("ranked_candidates", ())
    )
    selected_action = dict(payload.get("selected_action", {})) if isinstance(payload.get("selected_action"), dict) else None
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {}
    selected_candidate = dict(metadata.get("selected_candidate", {})) if isinstance(metadata.get("selected_candidate"), dict) else {}
    metadata["fallback_candidates"] = [
        normalize_candidate_for_export(candidate, available_families=available_families)
        for candidate in list(metadata.get("fallback_candidates", []))
    ]
    metadata["blocked_candidates"] = [
        normalize_candidate_for_export(candidate, available_families=available_families)
        for candidate in list(metadata.get("blocked_candidates", []))
    ]
    if selected_candidate:
        normalized_selected = normalize_candidate_for_export(
            selected_candidate,
            available_families=available_families,
            executed_family=executed_family,
        )
        metadata["selected_candidate"] = normalized_selected
        selected_action = dict(normalized_selected.get("action", {}))
    if selected_action is not None:
        selected_action["type"] = executed_family if executed_family in {"move", "interact", "click_at"} else str(selected_action.get("type") or "move")
        selected_action["required_action_family"] = selected_action["type"]
        payload["selected_action"] = selected_action
    payload["metadata"] = metadata
    return payload
