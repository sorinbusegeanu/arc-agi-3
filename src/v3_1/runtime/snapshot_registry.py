from __future__ import annotations

from dataclasses import dataclass, field
from time import time

import ray

from v3_1.contracts.errors import SnapshotNotFoundError


@dataclass
class SnapshotRegistry:
    snapshots: dict[str, dict] = field(default_factory=dict)

    def register(self, handle: str, snapshot: object) -> str:
        ref = snapshot if isinstance(snapshot, ray.ObjectRef) else ray.put(snapshot)
        self.snapshots[handle] = {
            "ref": ref,
            "registered_at_ms": int(time() * 1000),
        }
        return handle

    def get_ref(self, handle: str):
        if handle not in self.snapshots:
            raise SnapshotNotFoundError(handle)
        return self.snapshots[handle]["ref"]

    def get(self, handle: str) -> object:
        return ray.get(self.get_ref(handle))
