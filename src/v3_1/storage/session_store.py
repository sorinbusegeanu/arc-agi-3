from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionStore:
    manifests: list[dict] = field(default_factory=list)
    memory_snapshots: list[dict] = field(default_factory=list)
    persistent_memory_flushes: list[dict] = field(default_factory=list)
    persistent_memory_db_location: str | None = None

    def record(self, entry: dict) -> None:
        self.manifests.append(entry)

    def record_memory_snapshot(self, entry: dict) -> None:
        self.memory_snapshots.append(entry)
        self.manifests.append(entry)

    def record_persistent_memory_flush(self, entry: dict) -> None:
        self.persistent_memory_flushes.append(entry)
        self.manifests.append(entry)

    def set_persistent_memory_db_location(self, location: str) -> None:
        self.persistent_memory_db_location = location
