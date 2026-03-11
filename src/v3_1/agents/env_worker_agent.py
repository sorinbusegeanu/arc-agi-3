from __future__ import annotations

import ray

from v3_1.execution.env_worker import EnvWorker


@ray.remote
class EnvWorkerAgent:
    def __init__(self, worker_id: str, *, env_factory: str | None, env_id: str | None, env_root: str | None, seed: int | None) -> None:
        self.worker = EnvWorker.from_config(worker_id, env_factory=env_factory, env_id=env_id, env_root=env_root, seed=seed)

    def execute(self, request):
        return self.worker.run(request)
