from __future__ import annotations

from typing import Any

from . import multiprocess as _multiprocess


class _CountingStageQueue:
    def __init__(self, queue: Any, counters: Any, index: int) -> None:
        self._queue = queue
        self._counters = counters
        self._index = int(index)

    def put(self, item: Any, *args: Any, **kwargs: Any) -> Any:
        result = self._queue.put(item, *args, **kwargs)
        self._counters[self._index] = int(self._counters[self._index]) + 1
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._queue, name)


def tracked_actor_process_main(
    *,
    production_counters: Any,
    counter_index: int,
    stage_queue: Any,
    **kwargs: Any,
) -> None:
    _multiprocess.actor_process_main(
        stage_queue=_CountingStageQueue(stage_queue, production_counters, counter_index),
        **kwargs,
    )


def install_actor_production_telemetry(process_topology_cls: type) -> None:
    if getattr(process_topology_cls, "_actor_production_telemetry_installed", False):
        return

    original_init = process_topology_cls.__init__

    def topology_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # One lock-free counter per actor slot. Slots are single-writer and may be
        # reused by sequential jobs, so counters remain cumulative for the epoch.
        self._actor_production_counters = self.ctx.Array("Q", self.actors, lock=False)

    def start_actor(
        self: Any,
        *,
        index: int,
        spec: Any,
        actor_id: int,
        steps: int,
        seed: int,
        env_root: str | None,
        adapter_factory_path: str,
        alfred_backend_factory: str | None,
        run_nonce: int,
        initial_policy: Any,
        epsilon: float,
        stagnation: float = 0.0,
        policy_refresh_steps: int = 64,
        policy_refresh_ms: float = 250.0,
    ) -> None:
        process = self.actor_ctx.Process(
            target=tracked_actor_process_main,
            kwargs={
                "production_counters": self._actor_production_counters,
                "counter_index": int(index),
                "spec": spec,
                "actor_id": actor_id,
                "steps": steps,
                "seed": seed,
                "env_root": env_root,
                "initial_policy": initial_policy,
                "policy_updates": self.policy_updates[index],
                "epsilon": float(epsilon),
                "stagnation": float(stagnation),
                "policy_refresh_steps": int(policy_refresh_steps),
                "policy_refresh_ms": float(policy_refresh_ms),
                "stage_queue": self.stage_queue,
                "result_queue": self.result_queue,
                "adapter_factory_path": adapter_factory_path,
                "alfred_backend_factory": alfred_backend_factory,
                "run_nonce": int(run_nonce),
            },
            name=f"v9-actor-{actor_id}",
        )
        process.start()
        self.actor_processes.append(process)

    @property
    def produced_steps(self: Any) -> int:
        return sum(int(value) for value in self._actor_production_counters)

    process_topology_cls.__init__ = topology_init
    process_topology_cls.start_actor = start_actor
    process_topology_cls.produced_steps = produced_steps
    process_topology_cls._actor_production_telemetry_installed = True
