from __future__ import annotations

import queue
import time
from dataclasses import dataclass, replace
from typing import Any

from .memory_pipeline import DerivationResult, IngestionTask, PreparedIngestion
from .memory_worker_topology import MemoryWorkerTopology
from .multiprocess import ActorDone, ActorError, ProcessTopology


_ACTOR_COMPLETION_GRACE_SECONDS = 2.0


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


def _reconcile_actor_liveness(
    active: dict[int, tuple[int, Any]],
    clean_exit_without_done: dict[int, float],
    *,
    now: float | None = None,
    grace_seconds: float = _ACTOR_COMPLETION_GRACE_SECONDS,
) -> int:
    current_time = time.monotonic() if now is None else float(now)
    missing = 0
    for actor_id, (_, process) in tuple(active.items()):
        exitcode = process.exitcode
        if exitcode is None:
            clean_exit_without_done.pop(actor_id, None)
            continue
        if exitcode != 0:
            raise RuntimeError(f"actor process {actor_id} exited before completion with code {exitcode}")
        missing += 1
        first_seen = clean_exit_without_done.setdefault(actor_id, current_time)
        if current_time - first_seen >= float(grace_seconds):
            process_name = getattr(process, "name", f"actor-{actor_id}")
            raise RuntimeError(
                f"actor {actor_id} ({process_name}) exited cleanly but no ActorDone was received "
                f"within {float(grace_seconds):.1f}s; actor completion queue protocol failed"
            )
    return missing


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
    topology = ProcessTopology(actors=min(int(actor_limit), len(jobs)), stage_workers=int(stage_workers), shards=int(shards), queue_capacity=max(int(queue_capacity), int(publication_queue_capacity)), start_method=start_method)
    topology.start_workers()
    memory = MemoryWorkerTopology(topology.ctx, ingest_workers=int(ingest_workers), derivation_workers=int(derivation_workers), ingest_queue_capacity=int(ingest_queue_capacity), derivation_queue_capacity=int(derivation_queue_capacity), result_queue_capacity=max(int(publication_queue_capacity), 1024))
    memory.start()

    pending = list(jobs)
    active: dict[int, tuple[int, Any]] = {}
    free_slots = list(range(topology.actors))
    results: list[ProcessActorResult] = []
    clean_exit_without_done: dict[int, float] = {}
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
    ingest_results: dict[int, PreparedIngestion] = {}
    derive_results: dict[int, DerivationResult] = {}
    inflight: set[int] = set()
    last_support: dict[int, int] = {}
    coordinator_batch_size = 256
    canonical_apply_seconds = 0.0
    canonical_apply_events = 0
    clean_shutdown = False

    def launch_one() -> bool:
        if not pending or not free_slots:
            return False
        actor_id, spec, steps, seed = pending.pop(0)
        slot = free_slots.pop(0)
        launch_started = time.perf_counter()
        topology.start_actor(index=slot, spec=spec, actor_id=actor_id, steps=steps, seed=seed, env_root=env_root, adapter_factory_path="v9.cli:make_adapter", alfred_backend_factory=alfred_backend_factory, run_nonce=run_nonce, initial_policy=runtime.actor_policy_snapshot(), epsilon=float(epsilon), policy_refresh_steps=int(actor_view_refresh_steps), policy_refresh_ms=float(actor_view_refresh_ms))
        active[actor_id] = (slot, topology.actor_processes[-1])
        runtime.set_telemetry_gauge("actor_launch_latency_ms", 1000.0 * (time.perf_counter() - launch_started))
        runtime.set_telemetry_gauge("actors_launched", len(topology.actor_processes))
        return True

    def dispatch_transition(transition: Any) -> None:
        nonlocal sampled, ingest_sequence, watermark_cursor
        sampled += 1
        canonical_sequence = runtime.reserve_producer_sequence(int(transition.actor_id), int(transition.producer_sequence))
        if canonical_sequence != int(transition.producer_sequence):
            transition = replace(transition, producer_sequence=canonical_sequence)
        sequence = ingest_sequence
        ingest_sequence += 1
        watermark_cursor += 1
        memory.ingest_queue.put(IngestionTask(sequence, watermark_cursor, transition))
        watermark_cursor += len(tuple(transition.symbols))

    def schedule(signature: int) -> None:
        nonlocal derive_task_id
        signature = int(signature)
        if signature in inflight:
            return
        task = runtime.build_derivation_task(signature, task_id=derive_task_id)
        if task is None:
            return
        previous_support = int(last_support.get(signature, 0))
        current_support = int(task.support)
        if current_support <= previous_support:
            return
        if previous_support >= 2 and current_support < previous_support * 2:
            return
        inflight.add(signature)
        memory.derivation_queue.put(task)
        derive_task_id += 1

    def apply_ready() -> bool:
        nonlocal ingest_apply, derive_apply, ingested, derived
        nonlocal canonical_apply_seconds, canonical_apply_events
        apply_started = time.perf_counter()
        prepared_batch: list[PreparedIngestion] = []
        while ingest_apply in ingest_results and len(prepared_batch) < coordinator_batch_size:
            prepared_batch.append(ingest_results.pop(ingest_apply))
            ingest_apply += 1
        if prepared_batch:
            batch_method = getattr(runtime, "apply_prepared_ingestion_batch", None)
            signature_rows = batch_method(prepared_batch) if callable(batch_method) else tuple(runtime.apply_prepared_ingestion(row) for row in prepared_batch)
            for prepared, signatures in zip(prepared_batch, signature_rows):
                runtime.record_curriculum_event(step=prepared.transition.curriculum_step, environment_family=prepared.identity.family, game_scenario=prepared.transition.game_scenario)
                ingested += 1
                for signature in signatures:
                    schedule(int(signature))

        derivation_batch: list[DerivationResult] = []
        while derive_apply in derive_results and len(derivation_batch) < coordinator_batch_size:
            derivation_batch.append(derive_results.pop(derive_apply))
            derive_apply += 1
        if derivation_batch:
            batch_method = getattr(runtime, "apply_derivation_results_batch", None)
            if callable(batch_method):
                batch_method(derivation_batch)
            else:
                for result in derivation_batch:
                    runtime.apply_derivation_result(result)
            for result in derivation_batch:
                signature = int(result.structural_signature)
                last_support[signature] = max(int(last_support.get(signature, 0)), int(result.support))
                inflight.discard(signature)
                derived += 1
                schedule(signature)

        applied = len(prepared_batch) + len(derivation_batch)
        if applied:
            canonical_apply_seconds += time.perf_counter() - apply_started
            canonical_apply_events += applied
        return applied > 0

    def drain_results(*, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = apply_ready()
        first = not progressed
        for _ in range(coordinator_batch_size * 2):
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
        return apply_ready() or progressed

    def drain_actor_results() -> bool:
        progressed = False
        for _ in range(coordinator_batch_size):
            try:
                done = topology.result_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(done, ActorError):
                raise RuntimeError(f"actor {done.actor_id} ({done.game_id}) failed: {done.message}\n{done.traceback_text}")
            if not isinstance(done, ActorDone):
                continue
            entry = active.pop(done.actor_id, None)
            if entry is None:
                continue
            slot, _ = entry
            clean_exit_without_done.pop(done.actor_id, None)
            free_slots.append(slot)
            free_slots.sort()
            results.append(ProcessActorResult(done.actor_id, done.game_id, done.steps, done.positive_boundaries, done.negative_boundaries, done.episode_boundaries, done.resets, done.policy_refreshes))
            progressed = True
        return progressed

    def telemetry() -> None:
        elapsed = max(1e-9, time.monotonic() - started_at)
        for key, value in memory.queue_depths().items():
            runtime.set_telemetry_gauge(key, value)
        gauges = {"active_actor_processes": len(active), "coordinator_action_requests": 0, "policy_snapshot_generation": int(published_policy_generation), "active_ingest_workers": int(ingest_workers), "active_derivation_workers": int(derivation_workers), "sampled_steps": sampled, "ingested_steps": ingested, "derivations_applied": derived, "sampling_backlog": max(0, sampled - ingested), "derivation_inflight": len(inflight), "sampling_rate": sampled / elapsed, "ingestion_rate": ingested / elapsed, "derivation_rate": derived / elapsed, "canonical_apply_rate": canonical_apply_events / max(1e-9, canonical_apply_seconds), "canonical_apply_latency_ms": 1000.0 * canonical_apply_seconds / max(1, canonical_apply_events), "canonical_batch_size": coordinator_batch_size, "actors_exited_without_done": len(clean_exit_without_done)}
        for key, value in gauges.items():
            runtime.set_telemetry_gauge(key, value)

    def progress() -> None:
        telemetry()
        diag = dict(runtime.unified_telemetry.diagnostic_metrics())
        pct = 100.0 * ingested / total_steps if total_steps else 100.0
        print(f"{time.strftime('[%H:%M]')} {pct:5.1f}% sampled={sampled}/{total_steps} ingested={ingested} rate={float(diag.get('ingestion_rate', 0.0)):.0f}/s backlog={max(0, sampled - ingested)}", flush=True)

    def drain_publication_queue() -> bool:
        progressed = False
        for _ in range(coordinator_batch_size):
            try:
                item = topology.publication_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == "transition":
                dispatch_transition(item[3])
                progressed = True
            elif item[0] == "shard_done":
                topology.publication_queue.put(item)
                break
        return progressed

    try:
        while active or pending:
            progressed = launch_one()
            progressed = drain_publication_queue() or progressed
            progressed = drain_results() or progressed
            progressed = drain_actor_results() or progressed
            missing = _reconcile_actor_liveness(active, clean_exit_without_done)
            runtime.set_telemetry_gauge("actors_exited_without_done", missing)
            now = time.monotonic()
            if now >= next_policy_publish:
                snapshot = runtime.actor_policy_snapshot()
                if int(snapshot.generation) > int(published_policy_generation):
                    topology.publish_policy_snapshot(tuple(slot for slot, _ in active.values()), snapshot)
                    published_policy_generation = int(snapshot.generation)
                    runtime.set_telemetry_gauge("policy_snapshot_generation", published_policy_generation)
                next_policy_publish = now + max(0.01, float(actor_view_refresh_ms) / 1000.0)
            if now >= next_progress:
                progress()
                next_progress = time.monotonic() + max(1.0, float(progress_interval_seconds))
            if not progressed:
                time.sleep(0.001)

        while any(process.is_alive() for process in topology.actor_processes):
            progressed = drain_publication_queue()
            progressed = drain_results() or progressed
            progressed = drain_actor_results() or progressed
            if not progressed:
                time.sleep(0.001)
        topology.join_actor_workers()

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

        for key, value in {"actor_processes": min(int(actor_limit), len(jobs)), "stage_worker_processes": int(stage_workers), "shard_worker_processes": int(shards), "ingest_worker_processes": int(ingest_workers), "derivation_worker_processes": int(derivation_workers), "multiprocess_transitions_published": int(ingested), "coordinator_action_requests": 0, "policy_snapshot_generation": int(published_policy_generation), "policy_snapshot_refreshes": sum(int(row.policy_refreshes) for row in results), "actors_exited_without_done": 0}.items():
            runtime.set_telemetry_gauge(key, value)
        progress()
        clean_shutdown = True
        return sorted(results, key=lambda row: row.actor_id)
    except BaseException:
        memory.terminate()
        topology.terminate()
        raise
    finally:
        memory.close(drain=clean_shutdown)
        topology.close(drain=clean_shutdown)
