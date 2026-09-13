from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from random import Random
from typing import Any

from v9.cognition.action_selection import choose_action

from .multiprocess import ActionRequest, ActorDone, ProcessTopology
from .multiprocess_ingest import publish_encoded_transition


@dataclass(frozen=True, slots=True)
class ProcessActorResult:
    actor_id: int
    game_id: str
    steps: int
    positive_boundaries: int
    negative_boundaries: int
    resets: int


def run_process_jobs(
    runtime: Any,
    jobs: list[tuple[int, Any, int, int]],
    *,
    actor_limit: int,
    stage_workers: int,
    shards: int,
    queue_capacity: int,
    epsilon: float,
    env_root: str | None,
    alfred_backend_factory: str | None,
    start_method: str | None = None,
) -> list[ProcessActorResult]:
    topology = ProcessTopology(
        actors=min(int(actor_limit), len(jobs)),
        stage_workers=int(stage_workers),
        shards=int(shards),
        queue_capacity=int(queue_capacity),
        start_method=start_method,
    )
    topology.start_workers()
    pending = list(jobs)
    active: dict[int, tuple[int, Any]] = {}
    free_slots = list(range(topology.actors))
    rngs: dict[int, Random] = {}
    results: list[ProcessActorResult] = []
    run_nonce = int(runtime.watermark)
    published = 0

    def launch_available() -> None:
        while pending and free_slots:
            actor_id, spec, steps, seed = pending.pop(0)
            slot = free_slots.pop(0)
            rngs[actor_id] = Random(seed)
            topology.start_actor(
                index=slot,
                spec=spec,
                actor_id=actor_id,
                steps=steps,
                seed=seed,
                env_root=env_root,
                adapter_factory_path="v9.cli:make_adapter",
                alfred_backend_factory=alfred_backend_factory,
                run_nonce=run_nonce,
            )
            active[actor_id] = (slot, topology.actor_processes[-1])

    launch_available()
    try:
        while active:
            progressed = False

            while True:
                try:
                    request = topology.action_requests.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(request, ActionRequest):
                    continue
                slot, _ = active[request.actor_id]
                action = choose_action(
                    runtime.read_view,
                    request.actions,
                    rng=rngs[request.actor_id],
                    epsilon=float(epsilon),
                    target_environment_id=request.environment_instance_id,
                )
                topology.action_responses[slot].put((request.request_id, int(action)))
                progressed = True

            while True:
                try:
                    item = topology.publication_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, tuple) and len(item) == 4 and item[0] == "transition":
                    publish_encoded_transition(runtime, item[3])
                    published += 1
                    progressed = True

            while True:
                try:
                    done = topology.result_queue.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(done, ActorDone):
                    continue
                slot, process = active.pop(done.actor_id)
                process.join(timeout=30)
                if process.exitcode not in (0, None):
                    raise RuntimeError(f"actor process {done.actor_id} exited with code {process.exitcode}")
                free_slots.append(slot)
                free_slots.sort()
                results.append(ProcessActorResult(
                    done.actor_id,
                    done.game_id,
                    done.steps,
                    done.positive_boundaries,
                    done.negative_boundaries,
                    done.resets,
                ))
                launch_available()
                progressed = True

            for actor_id, (_, process) in tuple(active.items()):
                if not process.is_alive() and process.exitcode not in (0, None):
                    raise RuntimeError(f"actor process {actor_id} exited with code {process.exitcode}")

            if not progressed:
                time.sleep(0.001)

        topology.stop_stage_workers()
        topology.stop_shard_workers()
        shard_done = 0
        while shard_done < int(shards):
            item = topology.publication_queue.get(timeout=30)
            if isinstance(item, tuple) and item and item[0] == "transition":
                publish_encoded_transition(runtime, item[3])
                published += 1
            elif isinstance(item, tuple) and item and item[0] == "shard_done":
                shard_done += 1
        topology.join_shard_workers()

        runtime.unified_telemetry.set_gauge("actor_processes", min(int(actor_limit), len(jobs)))
        runtime.unified_telemetry.set_gauge("stage_worker_processes", int(stage_workers))
        runtime.unified_telemetry.set_gauge("shard_worker_processes", int(shards))
        runtime.unified_telemetry.set_gauge("multiprocess_transitions_published", int(published))
        return sorted(results, key=lambda row: row.actor_id)
    except BaseException:
        topology.terminate()
        raise
