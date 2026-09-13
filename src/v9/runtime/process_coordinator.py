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
    progress_interval_seconds: float = 60.0,
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
    total_steps = sum(int(row[2]) for row in jobs)
    next_progress = time.monotonic() + max(1.0, float(progress_interval_seconds))

    def _print_progress() -> None:
        metrics = runtime.metrics()
        levels = dict(metrics.get("memory_levels", {}))
        percent = (100.0 * published / total_steps) if total_steps else 100.0
        print(
            f"{time.strftime('[%H:%M]')} progress "
            f"{percent:5.1f}% "
            f"steps={published}/{total_steps} "
            f"memories={metrics.get('memories', 0)} "
            f"M0={levels.get('M0', 0)} "
            f"M1={levels.get('M1', 0)} "
            f"M2={levels.get('M2', 0)} "
            f"M3={levels.get('M3', 0)} "
            f"M4={levels.get('M4', 0)} "
            f"M5={levels.get('M5', 0)} "
            f"M6={levels.get('M6', 0)} "
            f"M7={levels.get('M7', 0)} "
            f"pred_err={float(metrics.get('prediction_error', 0.0)):.4f} "
            f"success={100.0 * float(metrics.get('success_rate', 0.0)):.1f}% "
            f"compression={float(metrics.get('compression_ratio', 0.0)):.3f}",
            flush=True,
        )

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

            now = time.monotonic()
            if now >= next_progress:
                _print_progress()
                next_progress = now + max(1.0, float(progress_interval_seconds))

            if not progressed:
                time.sleep(0.001)

        topology.signal_stage_stop()
        while any(process.is_alive() for process in topology.stage_processes):
            try:
                item = topology.publication_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if isinstance(item, tuple) and item and item[0] == "transition":
                publish_encoded_transition(runtime, item[3])
                published += 1
        topology.join_stage_workers()

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

        runtime.set_telemetry_gauge("actor_processes", min(int(actor_limit), len(jobs)))
        runtime.set_telemetry_gauge("stage_worker_processes", int(stage_workers))
        runtime.set_telemetry_gauge("shard_worker_processes", int(shards))
        runtime.set_telemetry_gauge("multiprocess_transitions_published", int(published))
        _print_progress()
        return sorted(results, key=lambda row: row.actor_id)
    except BaseException:
        topology.terminate()
        raise
