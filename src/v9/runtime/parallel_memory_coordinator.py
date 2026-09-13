from __future__ import annotations

import queue
import time
from dataclasses import dataclass, replace
from typing import Any

from .memory_pipeline import DerivationResult, IngestionTask, PreparedIngestion
from .memory_worker_topology import MemoryWorkerTopology
from .multiprocess import ActorDone, ProcessTopology
from .multiprocess_ingest import publish_transition_symbols


@dataclass(frozen=True, slots=True)
class ProcessActorResult:
    actor_id: int
    game_id: str
    steps: int
    positive_boundaries: int
    negative_boundaries: int
    episode_boundaries: int
    resets: int
    policy_refreshes: int = 0


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
    actor_view_refresh_steps: int = 64,
    actor_view_refresh_ms: float = 250.0,
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
    results = []
    initial_policy = runtime.actor_policy_snapshot()
    published_policy_generation = int(initial_policy.generation)
    next_policy_publish = time.monotonic() + max(0.01, float(actor_view_refresh_ms) / 1000.0)
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
    coordinator_batch_size = 256

    def launch() -> None:
        while pending and free_slots:
            actor_id, spec, steps, seed = pending.pop(0)
            slot = free_slots.pop(0)
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
                initial_policy=initial_policy,
                epsilon=float(epsilon),
                policy_refresh_steps=int(actor_view_refresh_steps),
                policy_refresh_ms=float(actor_view_refresh_ms),
            )
            active[actor_id] = (slot, topology.actor_processes[-1])

    def dispatch_transition(transition: Any) -> None:
        nonlocal sampled, ingest_sequence, watermark_cursor
        sampled += 1
        canonical_sequence = runtime.reserve_producer_sequence(
            int(transition.actor_id),
            int(transition.producer_sequence),
        )
        if canonical_sequence != int(transition.producer_sequence):
            transition = replace(transition, producer_sequence=canonical_sequence)
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
        runtime.record_curriculum_event(
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
        for _ in range(coordinator_batch_size):
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
            runtime.set_telemetry_gauge(key, value)
        gauges = {
            "active_actor_processes": len(active),
            "coordinator_action_requests": 0,
            "policy_snapshot_generation": int(published_policy_generation),
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
            runtime.set_telemetry_gauge(key, value)

    def progress() -> None:
        telemetry()
        metrics = runtime.metrics()
        levels = dict(metrics.get("memory_levels", {}))
        diag = dict(metrics.get("telemetry_diagnostics", {}))
        pct = 100.0 * ingested / total_steps if total_steps else 100.0
        print(
            f"{time.strftime('[%H:%M]')} progress {pct:5.1f}% sampled={sampled}/{total_steps} ingested={ingested} "
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
            for _ in range(coordinator_batch_size):
                try:
                    item = topology.publication_queue.get_nowait()
                except queue.Empty:
                    break
                if item[0] == "transition":
                    dispatch_transition(item[3])
                    progressed = True
            progressed = drain_results() or progressed
            for _ in range(coordinator_batch_size):
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
                results.append(ProcessActorResult(done.actor_id, done.game_id, done.steps, done.positive_boundaries, done.negative_boundaries, done.episode_boundaries, done.resets, done.policy_refreshes))
                launch()
                progressed = True
            now = time.monotonic()
            if now >= next_policy_publish:
                snapshot = runtime.actor_policy_snapshot()
                if int(snapshot.generation) > int(published_policy_generation):
                    topology.publish_policy_snapshot(
                        tuple(slot for slot, _ in active.values()),
                        snapshot,
                    )
                    published_policy_generation = int(snapshot.generation)
                    runtime.set_telemetry_gauge("policy_snapshot_generation", published_policy_generation)
                next_policy_publish = now + max(0.01, float(actor_view_refresh_ms) / 1000.0)
            if now >= next_progress:
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
            "coordinator_action_requests": 0,
            "policy_snapshot_generation": int(published_policy_generation),
            "policy_snapshot_refreshes": sum(int(row.policy_refreshes) for row in results),
        }.items():
            runtime.set_telemetry_gauge(key, value)
        progress()
        return sorted(results, key=lambda row: row.actor_id)
    except BaseException:
        memory.terminate()
        topology.terminate()
        raise
