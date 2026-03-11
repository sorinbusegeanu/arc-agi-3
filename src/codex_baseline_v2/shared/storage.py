from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class StoragePathsV2:
    root: str

    def game_root(self, game_id: str) -> str:
        return os.path.join(self.root, f"game_{game_id}")

    def round_root(self, game_id: str, round_id: int) -> str:
        return os.path.join(self.game_root(game_id), f"round_{round_id:03d}")

    def category_path(self, game_id: str, round_id: int, category: str) -> str:
        return os.path.join(self.round_root(game_id, round_id), category)

    def ensure_round_dirs(self, game_id: str, round_id: int) -> Dict[str, str]:
        categories = [
            "raw_trajectories",
            "normalized_trajectories",
            "analyst_outputs",
            "blackboard_snapshots",
            "round_reports",
            "controller_decisions",
            "executor_outcomes",
            "exports",
            "logs",
        ]
        game_root = self.game_root(game_id)
        round_root = self.round_root(game_id, round_id)
        os.makedirs(game_root, exist_ok=True)
        os.makedirs(round_root, exist_ok=True)
        paths = {
            "game_root": game_root,
            "round_root": round_root,
        }
        for category in categories:
            path = self.category_path(game_id, round_id, category)
            os.makedirs(path, exist_ok=True)
            paths[category] = path
        return paths


def get_round_one_poi_heatmap_path(session_dir: str, game_id: str) -> str:
    return os.path.join(session_dir, "postrun_exports", f"{game_id}_round_one_poi_heatmap.png")


def get_final_avatar_visit_heatmap_path(session_dir: str, game_id: str) -> str:
    return os.path.join(session_dir, "postrun_exports", f"{game_id}_final_avatar_visit_heatmap.png")
