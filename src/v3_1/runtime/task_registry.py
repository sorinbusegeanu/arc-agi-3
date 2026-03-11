from __future__ import annotations

from dataclasses import dataclass, field
from time import time

import ray


@dataclass
class TaskRegistry:
    tasks: dict[str, dict] = field(default_factory=dict)

    def put(self, task_id: str, payload: object) -> None:
        ref = payload if isinstance(payload, ray.ObjectRef) else ray.put(payload)
        self.tasks[task_id] = {
            "ref": ref,
            "status": "submitted",
            "submitted_at_ms": int(time() * 1000),
            "completed_at_ms": None,
        }

    def mark_completed(self, task_id: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["completed_at_ms"] = int(time() * 1000)

    def get_ref(self, task_id: str):
        row = self.tasks.get(task_id)
        return None if row is None else row["ref"]

    def get(self, task_id: str) -> object | None:
        ref = self.get_ref(task_id)
        return None if ref is None else ray.get(ref)

    def status(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)
