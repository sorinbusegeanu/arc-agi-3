from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional

from codex_baseline_v2.shared.config import MemoryConfigV2
from codex_baseline_v2.shared.schemas import BlackboardStateV2
from codex_baseline_v2.shared.state_identity import canonical_state_identity
from codex_baseline_v2.shared.storage import StoragePathsV2


def _atomic_write(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="v2_", suffix=".json", dir=os.path.dirname(path))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(tmp_path, path)


def _write_blackboard_history(storage: StoragePathsV2, blackboard: BlackboardStateV2, payload: Dict[str, Any]) -> str:
    history_path = os.path.join(
        storage.category_path(blackboard.game_id, blackboard.round_id, "blackboard_snapshots"),
        f"blackboard_round_{blackboard.round_id:03d}.json",
    )
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    return history_path


def save_blackboard(cfg: MemoryConfigV2, storage: StoragePathsV2, blackboard: BlackboardStateV2) -> str:
    path = os.path.join(storage.game_root(blackboard.game_id), "blackboard_latest.json")
    payload = blackboard.to_dict()
    payload.setdefault("metadata", {})
    last_obs = payload["metadata"].get("last_observation")
    if isinstance(last_obs, list):
        identity = canonical_state_identity(last_obs, include_payload=False)
        payload["metadata"]["last_observation_state_hash"] = identity.get("state_hash")
        payload["metadata"]["state_signature_version"] = identity.get("state_signature_version")
        payload["metadata"]["state_hash_valid"] = bool(identity.get("valid"))
    if cfg.atomic_writes:
        _atomic_write(path, payload)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
    _write_blackboard_history(storage, blackboard, payload)
    return path


def load_blackboard_raw(storage: StoragePathsV2, game_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(storage.game_root(game_id), "blackboard_latest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_blackboard(storage: StoragePathsV2, game_id: str) -> Optional[BlackboardStateV2]:
    payload = load_blackboard_raw(storage, game_id)
    if payload is None:
        return None
    return BlackboardStateV2.from_dict(payload)


def load_blackboard_typed(storage: StoragePathsV2, game_id: str) -> Optional[BlackboardStateV2]:
    return load_blackboard(storage, game_id)


def _save_archive(storage: StoragePathsV2, game_id: str, round_id: int, filename: str, payload: Any) -> str:
    path = os.path.join(storage.category_path(game_id, round_id, "exports"), filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    return path


def save_world_model_archive(storage: StoragePathsV2, blackboard: BlackboardStateV2) -> str:
    return _save_archive(storage, blackboard.game_id, blackboard.round_id, "world_model.json", blackboard.to_dict())


def save_event_table(storage: StoragePathsV2, blackboard: BlackboardStateV2) -> str:
    return _save_archive(storage, blackboard.game_id, blackboard.round_id, "event_table.json", [v.to_dict() for v in blackboard.event_table])


def save_navigation_graph(storage: StoragePathsV2, blackboard: BlackboardStateV2) -> str:
    return _save_archive(
        storage,
        blackboard.game_id,
        blackboard.round_id,
        "navigation_graph.json",
        {"cells": [v.to_dict() for v in blackboard.navigation_cells], "edges": [v.to_dict() for v in blackboard.navigation_edges]},
    )


def save_interventions(storage: StoragePathsV2, blackboard: BlackboardStateV2) -> str:
    return _save_archive(storage, blackboard.game_id, blackboard.round_id, "interventions.json", [v.to_dict() for v in blackboard.intervention_table])


def save_mechanic_hypotheses(storage: StoragePathsV2, blackboard: BlackboardStateV2) -> str:
    return _save_archive(storage, blackboard.game_id, blackboard.round_id, "mechanic_hypotheses.json", [v.to_dict() for v in blackboard.mechanic_hypotheses])


def append_round_report(storage: StoragePathsV2, game_id: str, round_id: int, report: Dict[str, Any]) -> str:
    path = os.path.join(storage.category_path(game_id, round_id, "round_reports"), "round_report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True)
    return path


def _persist_postrun_artifact_manifest(session_dir: str, artifact_key: str, artifact_path: str, artifact_type: str) -> str:
    manifest_path = os.path.join(session_dir, "postrun_artifacts.json")
    payload: Dict[str, Any] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    payload.setdefault("artifacts", {})
    payload["artifacts"][artifact_key] = {
        "artifact_type": artifact_type,
        "path": artifact_path,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    return manifest_path


def persist_round_one_poi_heatmap_artifact(session_dir: str, game_id: str, artifact_path: str) -> str:
    return _persist_postrun_artifact_manifest(
        session_dir,
        "round_one_poi_heatmap",
        artifact_path,
        "round_one_poi_heatmap_png",
    )


def persist_final_avatar_visit_heatmap_artifact(session_dir: str, game_id: str, artifact_path: str) -> str:
    return _persist_postrun_artifact_manifest(
        session_dir,
        "final_avatar_visit_heatmap",
        artifact_path,
        "final_avatar_visit_heatmap_png",
    )
