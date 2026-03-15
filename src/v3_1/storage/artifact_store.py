from __future__ import annotations

from pathlib import Path

from v3_1.contracts.messages import PersistentMemoryFlushRequest, PersistentMemoryFlushResult
from v3_1.storage.persistent_memory import PersistentMemoryStore
from v3_1.storage.paths import round_root, session_root, visualization_root
from v3_1.storage.serialization import dumps


class ArtifactStore:
    def __init__(self, root_dir: str, persistent_memory_store: PersistentMemoryStore | None = None) -> None:
        self.root_dir = root_dir
        self.persistent_memory_store = persistent_memory_store

    def write_json(self, *, session_id: str, round_id: int, artifact_name: str, payload) -> str:
        root = round_root(self.root_dir, session_id, round_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / artifact_name
        encoded = dumps(payload)
        path.write_text(encoded, encoding="utf-8")
        return str(path)

    def write_bytes(self, *, session_id: str, round_id: int, artifact_name: str, payload: bytes) -> str:
        root = round_root(self.root_dir, session_id, round_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / artifact_name
        path.write_bytes(payload)
        return str(path)

    def write_session_json(self, *, session_id: str, artifact_name: str, payload) -> str:
        root = session_root(self.root_dir, session_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / artifact_name
        encoded = dumps(payload)
        path.write_text(encoded, encoding="utf-8")
        return str(path)

    def write_session_bytes(self, *, session_id: str, artifact_name: str, payload: bytes) -> str:
        root = session_root(self.root_dir, session_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / artifact_name
        path.write_bytes(payload)
        return str(path)

    def write_visualization_bytes(self, *, session_id: str, artifact_name: str, payload: bytes) -> str:
        root = visualization_root(self.root_dir, session_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / artifact_name
        path.write_bytes(payload)
        return str(path)

    def flush_persistent_memory(self, request: PersistentMemoryFlushRequest) -> PersistentMemoryFlushResult:
        if self.persistent_memory_store is None:
            raise RuntimeError("persistent memory store is not configured")
        return self.persistent_memory_store.flush(request)
