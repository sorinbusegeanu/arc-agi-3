from __future__ import annotations

import os
from typing import Any, Dict

from codex_baseline_v2.analyst.analyst import extract_main_sprite_pixel_visits_from_session_artifacts
from codex_baseline_v2.memory.store import (
    persist_final_avatar_visit_heatmap_artifact,
    persist_round_one_poi_heatmap_artifact,
)
from codex_baseline_v2.shared.storage import (
    get_final_avatar_visit_heatmap_path,
    get_round_one_poi_heatmap_path,
)
from codex_baseline_v2.trajectory_analysis.analyzer import extract_round_one_poi_coordinates_from_session_artifacts
from codex_baseline_v2.visualization.heatmaps import (
    build_avatar_visit_heatmap_from_coordinates,
    build_poi_heatmap_from_coordinates,
    save_heatmap_png,
)


def _game_id_from_session_inputs(session_dir: str, session_state_or_artifacts: Any, config: Any) -> str:
    if isinstance(session_state_or_artifacts, dict):
        game_id = session_state_or_artifacts.get("game_id")
        if isinstance(game_id, str) and game_id:
            return game_id
    game_id = getattr(config, "game_id", None)
    if isinstance(game_id, str) and game_id:
        return game_id
    base = os.path.basename(session_dir.rstrip("/"))
    return base[len("game_") :] if base.startswith("game_") else base


def generate_postrun_session_heatmaps(session_dir, session_state_or_artifacts, config):
    visualization_cfg = getattr(config, "visualization", None)
    if visualization_cfg is None:
        return {}
    grid_size = (
        int(getattr(visualization_cfg, "heatmap_grid_width", 64)),
        int(getattr(visualization_cfg, "heatmap_grid_height", 64)),
    )
    game_id = _game_id_from_session_inputs(session_dir, session_state_or_artifacts, config)
    generated: Dict[str, str] = {}
    if bool(getattr(visualization_cfg, "enable_round_one_poi_heatmap", True)):
        poi_coords = extract_round_one_poi_coordinates_from_session_artifacts(session_state_or_artifacts)
        poi_matrix = build_poi_heatmap_from_coordinates(poi_coords, grid_size=grid_size)
        poi_path = get_round_one_poi_heatmap_path(session_dir, game_id)
        save_heatmap_png(poi_matrix, poi_path, f"{game_id} round-1 POI heatmap", "log")
        persist_round_one_poi_heatmap_artifact(session_dir, game_id, poi_path)
        generated["round_one_poi_heatmap"] = poi_path
    if bool(getattr(visualization_cfg, "enable_final_avatar_visit_heatmap", True)):
        visit_coords = extract_main_sprite_pixel_visits_from_session_artifacts(session_state_or_artifacts)
        visit_matrix = build_avatar_visit_heatmap_from_coordinates(visit_coords, grid_size=grid_size)
        visit_path = get_final_avatar_visit_heatmap_path(session_dir, game_id)
        save_heatmap_png(visit_matrix, visit_path, f"{game_id} avatar visit heatmap", "log")
        persist_final_avatar_visit_heatmap_artifact(session_dir, game_id, visit_path)
        generated["final_avatar_visit_heatmap"] = visit_path
    return generated
