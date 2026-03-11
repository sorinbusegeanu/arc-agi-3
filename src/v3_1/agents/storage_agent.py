from __future__ import annotations

import ray

from v3_1.storage.artifact_store import ArtifactStore
from v3_1.storage.manifests import manifest_entry
from v3_1.storage.session_store import SessionStore
from v3_1.storage.sqlite_index import SQLiteIndex


@ray.remote
class StorageAgent:
    def __init__(self, *, root_dir: str, sqlite_path: str | None = None) -> None:
        self.store = ArtifactStore(root_dir)
        self.session_store = SessionStore()
        self.sqlite_index = SQLiteIndex(sqlite_path) if sqlite_path else None

    def persist(self, *, session_id: str, round_id: int, kind: str, name: str, payload) -> str:
        location = self.store.write_json(session_id=session_id, round_id=round_id, artifact_name=name, payload=payload)
        entry = manifest_entry(kind, name, location)
        self.session_store.record(entry)
        if self.sqlite_index is not None:
            self.sqlite_index.insert_manifest(kind=kind, name=name, location=location)
        return location

    def manifests(self) -> list[dict]:
        return list(self.session_store.manifests)
