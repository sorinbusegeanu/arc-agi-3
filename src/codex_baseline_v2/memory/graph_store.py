from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from codex_baseline_v2.shared.schemas import BlackboardStateV2, DependencyGraphStateV1, LatentStateHypothesisV1, MechanicGraphStateV1
from codex_baseline_v2.shared.storage import StoragePathsV2


def _atomic_write(path: str, payload: object) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="v231_", suffix=".json", dir=os.path.dirname(path))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(tmp_path, path)
    return path


def save_graph_state(storage: StoragePathsV2, blackboard: BlackboardStateV2) -> None:
    exports = storage.category_path(blackboard.game_id, blackboard.round_id, "exports")
    _atomic_write(os.path.join(exports, "latent_states.json"), {"schema_version": "v2.3.1", "latent_states": [row.to_dict() for row in blackboard.latent_states]})
    _atomic_write(os.path.join(exports, "mechanic_graph.json"), blackboard.mechanic_graph.to_dict() if blackboard.mechanic_graph is not None else {"schema_version": "v2.3.1", "graph_id": "mechanic_graph:latest", "nodes": [], "edges": [], "updated_round": blackboard.round_id, "updated_step": 0})
    _atomic_write(os.path.join(exports, "dependency_graph.json"), blackboard.dependency_graph.to_dict() if blackboard.dependency_graph is not None else {"schema_version": "v2.3.1", "subgoals": [], "updated_round": blackboard.round_id, "updated_step": 0})


def load_latest_latent_states(storage: StoragePathsV2, game_id: str) -> list[LatentStateHypothesisV1]:
    path = os.path.join(storage.game_root(game_id), "blackboard_latest.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [LatentStateHypothesisV1.from_dict(v) for v in payload.get("latent_states", [])]


def load_latest_mechanic_graph(storage: StoragePathsV2, game_id: str) -> Optional[MechanicGraphStateV1]:
    path = os.path.join(storage.game_root(game_id), "blackboard_latest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    graph = payload.get("mechanic_graph")
    return MechanicGraphStateV1.from_dict(graph) if graph is not None else None


def load_latest_dependency_graph(storage: StoragePathsV2, game_id: str) -> Optional[DependencyGraphStateV1]:
    path = os.path.join(storage.game_root(game_id), "blackboard_latest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    graph = payload.get("dependency_graph")
    return DependencyGraphStateV1.from_dict(graph) if graph is not None else None
