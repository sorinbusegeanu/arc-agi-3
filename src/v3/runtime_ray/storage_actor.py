from __future__ import annotations

import json
import os
from typing import Dict, List

from codex_baseline_v2.shared.storage import StoragePathsV2

from .messages import PersistenceRequest


class StorageActor:
    def __init__(self, root_dir: str = "runs_v3") -> None:
        self.storage = StoragePathsV2(root_dir)

    def persist(self, request: PersistenceRequest) -> Dict[str, object]:
        game_root = self.storage.game_root(request.game_id)
        if request.artifact_family == "run_scoped":
            os.makedirs(game_root, exist_ok=True)
            path = os.path.join(game_root, request.ordering_key)
        else:
            round_root = self.storage.category_path(request.game_id, request.round_id, "exports")
            os.makedirs(round_root, exist_ok=True)
            path = os.path.join(round_root, request.ordering_key)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(request.payload, handle, sort_keys=True)
        return {"path": path, "artifact_family": request.artifact_family}
