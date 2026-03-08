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


def save_blackboard(cfg: MemoryConfigV2, storage: StoragePathsV2, blackboard: BlackboardStateV2) -> str:
    path = os.path.join(storage.game_root(blackboard.game_id), "blackboard_latest.json")
    payload = blackboard.to_dict()
    last_obs = payload.get("metadata", {}).get("last_observation")
    if isinstance(last_obs, list):
        identity = canonical_state_identity(last_obs, include_payload=False)
        payload.setdefault("metadata", {})
        payload["metadata"]["last_observation_state_hash"] = identity.get("state_hash")
        payload["metadata"]["state_signature_version"] = identity.get("state_signature_version")
        payload["metadata"]["state_hash_valid"] = bool(identity.get("valid"))
    if cfg.atomic_writes:
        _atomic_write(path, payload)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
    return path


def load_blackboard(storage: StoragePathsV2, game_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(storage.game_root(game_id), "blackboard_latest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_round_report(storage: StoragePathsV2, game_id: str, round_id: int, report: Dict[str, Any]) -> str:
    path = os.path.join(storage.category_path(game_id, round_id, "round_reports"), "round_report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True)
    return path
