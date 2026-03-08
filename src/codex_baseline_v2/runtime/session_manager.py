from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional

from codex_baseline_v2.shared.storage import StoragePathsV2


@dataclass
class SessionStateV2:
    game_id: str
    round_id: int
    storage_root: str


class SessionManagerV2:
    def __init__(self, storage_root: str) -> None:
        self.storage = StoragePathsV2(storage_root)

    def init_or_resume(self, game_id: str, resume_if_exists: bool = True) -> SessionStateV2:
        game_root = self.storage.game_root(game_id)
        if not os.path.exists(game_root):
            os.makedirs(game_root, exist_ok=True)
            return SessionStateV2(game_id=game_id, round_id=0, storage_root=self.storage.root)
        if not resume_if_exists:
            shutil.rmtree(game_root, ignore_errors=True)
            os.makedirs(game_root, exist_ok=True)
            return SessionStateV2(game_id=game_id, round_id=0, storage_root=self.storage.root)
        # resume from latest round
        rounds = [d for d in os.listdir(game_root) if d.startswith("round_")]
        if not rounds:
            return SessionStateV2(game_id=game_id, round_id=0, storage_root=self.storage.root)
        rounds.sort()
        last_round = int(rounds[-1].split("_")[1])
        return SessionStateV2(game_id=game_id, round_id=last_round + 1, storage_root=self.storage.root)

    def write_resume_marker(self, game_id: str, round_id: int, payload: Dict[str, Any]) -> str:
        path = os.path.join(self.storage.round_root(game_id, round_id), "resume_marker.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return path

    def load_resume_marker(self, game_id: str, round_id: int) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.storage.round_root(game_id, round_id), "resume_marker.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def export_summary(self, game_id: str, payload: Dict[str, Any]) -> str:
        path = os.path.join(self.storage.game_root(game_id), "summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return path
