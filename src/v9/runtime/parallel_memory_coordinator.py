from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from random import Random
from typing import Any

from v9.cognition.action_selection import choose_action

from .memory_pipeline import DerivationResult, IngestionTask, PreparedIngestion
from .memory_worker_topology import MemoryWorkerTopology
from .multiprocess import ActionRequest, ActorDone, ProcessTopology
from .multiprocess_ingest import publish_transition_symbols


@dataclass(frozen=True, slots=True)
class ProcessActorResult:
    actor_id: int
    game_id: str
    steps: int
    positive_boundaries: int
    negative_boundaries: int
    resets: int


def run_parallel_memory_jobs(
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
    start_method: str | None,
    progress_interval_seconds: float,
    ingest_workers: int,
    derivation_workers: int,
    ingest_queue_capacity: int,
    derivation_queue_capacity: int,
    publication_queue_capacity: int,
) -> list[ProcessActorResult]:
    topology = ProcessTopology(
        actors=min(int(actor_limit), len(jobs)),
        stage_workers=int(stage_workers),
        shards=int(shards),
        queue_capacity=max(int(queue_capacity), int(publication_queue_capacity)),
        start_method=start_method,
    )
    topology.start_workers()
    memory = MemoryWorkerTopology(
        topology.ctx,
        ingest_workers=int(ingest_workers),
        derivation_workers=int(derivation_workers),
        ingest_queue_capacity=int(ingest_queue_capacity),
        derivation_queue_capacity=int(derivation_queue_capacity),
        result_queue_capacity=max(int(publication_queue_capacity), 1024),
    )
    memory.start()

    pending = list(jobs)
    active = {}
    free_slots = list(range(topology.actors))
    rngs = {}
    results = []
    run_nonce = int(runtime.watermark)
    watermark_cursor = int(runtime.watermark)
    total_steps = sum(int(row[2]) for row in jobs)
    started_at = time.monotonic()
    next_progress = started_at + max(1.0, float(progress_interval_seconds))

    sampled = ingested = derived = 0
    ingest_sequence = ingest_apply = 1
    derive_task_id = derive_apply = 1
    ingest_watermarks = {}
    ingest_results = {}
    derive_results = {}
    inflight = set()
    last_support = {}

    def launch() -> None:
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

    def dispatch_transition(transition: Any) -> None:
        nonlocal sampled, ingest_sequence, watermark_cursor
        sampled += 1
        sequence = ingest_sequence
        ingest_sequence += 1
        watermark_cursor += 1
        ingest_watermarks[sequence] = watermark_cursor
        memory.ingest_queue.put(IngestionTask(sequence, watermark_cursor, transition))
        watermark_cursor += len(tuple(transition.symbols))

    def schedule(signature: int) -> None:
        nonlocal derive_task_id
        signature = int(signature)
        if signature in inflight:
            return
        task = runtime.build_derivation_task(signature, task_id=derive_task_id)
        if task is None or int(task.support) <= int(last_support.get(signature, 0)):
            return
        inflight.add(signature)
        memory.derivation_queue.put(task)
        derive_task_id += 1

    def apply_ingest(prepared: PreparedIngestion) -> None:
        nonlocal ingested
        sequence = int(prepared.sequence)
        signatures = runtime.apply_prepared_ingestion(prepared)
        base = ingest_watermarks.pop(sequence)
        publish_transition_symbols(
            runtime,
            prepared.transition,
            base_watermark=base if prepared.event is not None else base - 1,
        )
        runtime.unified_telemetry.record_curriculum_event(
            step=prepared.transition.curriculum_step,
            environment_family=prepared.identity.family,
            game_scenario=prepared.transition.game_scenario,
        )
        ingested += 1
        for signature in signatures:
            schedule(int(signature))

    def apply_ready() -> None:
        nonlocal ingest_apply, derive_apply, derived
        while ingest_apply in ingest_results:
            apply_ingest(ingest_results.pop(ingest_apply))
            ingest_apply += 1
        while derive_apply in derive_results:
            result = derive_results.pop(derive_apply)
            runtime.apply_derivation_result(result)
            signature = int(result.structural_signature)
            last_support[signature] = max(int(last_support.get(signature, 0)), int(result.support))
            inflight.discard(signature)
            derived += 1
            derive_apply += 1
            schedule(signature)

    def drain_results(*, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = False
        first = True
        while True:
            try:
                item = memory.result_queue.get(timeout=timeout) if block and first else memory.result_queue.get_nowait()
            except queue.Empty:
                break
            first = False
            if item[0] == "worker_error":
                raise RuntimeError(f"{item[1]} worker task {item[2]} failed: {item[3]}")
            if item[0] == "ingest":
                ingest_results[int(item[1])] = item[2]
                progressed = True
            elif item[0] == "derivation":
                derive_results[int(item[1])] = item[2]
                progressed = True
        apply_ready()
        return progressed

    def telemetry() -> None:
        elapsed = max(1e-9, time.monotonic() - started_at)
        for key, value in memory.queue_depths().items():
            runtime.unified_telemetry.set_gauge(key, value)
        gauges = {
            "active_actor_processes": len(active),
            "active_ingest_workers": int(ingest_workers),
            "active_derivation_workers": int(derivation_workers),
            "sampled_steps": sampled,
            "ingested_steps": ingested,
            "derivations_applied": derived,
            "sampling_backlog": max(0, sampled - ingested),
            "derivation_inflight": len(inflight),
            "sampling_rate": sampled / elapsed,
            "ingestion_rate": ingested / elapsed,
            "derivation_rate": derived / elapsed,
        }
        for key, value in gauges.items():
            runtime.unified_telemetry.set_gauge(key, value)

    def progress() -> None:
        telemetry()
        metrics = runtime.metrics()
        levels = dict(metrics.get("memory_levels", {}))
        diag = dict(metrics.get("telemetry_diagnostics", {}))
        pct = 100.0 * ingested / total_steps if total_steps else 100.0
        print(
            f"v9 progress {pct:5.1f}% sampled={sampled}/{total_steps} ingested={ingested} "
            f"ingest_q={diag.get('ingest_queue_depth', -1)} derive_q={diag.get('derivation_queue_depth', -1)} "
            f"M0={levels.get('M0',0)} M1={levels.get('M1',0)} M2={levels.get('M2',0)} "
            f"M3={levels.get('M3',0)} M4={levels.get('M4',0)} M5={levels.get('M5',0)} "
            f"M6={levels.get('M6',0)} M7={levels.get('M7',0)}",
            flush=True,
        )

    launch()
    try:
        while active:
            progressed = False
            while True:
                try:
                    request = topology.action_requests.get_nowait()
                except queue.Empty:
                    break
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
                if item[0] == "transition":
                    dispatch_transition(item[3])
                    progressed = True
            progressed = drain_results() or progressed
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
                results.append(ProcessActorResult(done.actor_id, done.game_id, done.steps, done.positive_boundaries, done.negative_boundaries, done.resets))
                launch()
                progressed = True
            if time.monotonic() >= next_progress:
                progress()
                next_progress = time.monotonic() + max(1.0, float(progress_interval_seconds))
            if not progressed:
                time.sleep(0.001)

        topology.signal_stage_stop()
        while any(p.is_alive() for p in topology.stage_processes):
            try:
                item = topology.publication_queue.get(timeout=0.05)
                if item[0] == "transition":
                    dispatch_transition(item[3])
            except queue.Empty:
                pass
            drain_results()
        topology.join_stage_workers()

        topology.stop_shard_workers()
        shard_done = 0
        while shard_done < int(shards):
            item = topology.publication_queue.get(timeout=30)
            if item[0] == "transition":
                dispatch_transition(item[3])
            elif item[0] == "shard_done":
                shard_done += 1
            drain_results()
        topology.join_shard_workers()

        memory.signal_ingest_stop()
        while ingested < sampled:
            drain_results(block=True, timeout=30)
        memory.join_ingest()

        while inflight or derive_results:
            drain_results(block=bool(inflight), timeout=30)
        memory.signal_derivation_stop()
        memory.join_derivation()
        drain_results()

        for key, value in {
            "actor_processes": min(int(actor_limit), len(jobs)),
            "stage_worker_processes": int(stage_workers),
            "shard_worker_processes": int(shards),
            "ingest_worker_processes": int(ingest_workers),
            "derivation_worker_processes": int(derivation_workers),
            "multiprocess_transitions_published": int(ingested),
        }.items():
            runtime.unified_telemetry.set_gauge(key, value)
        progress()
        return sorted(results, key=lambda row: row.actor_id)
    except BaseException:
        topology.terminate()
        raise
