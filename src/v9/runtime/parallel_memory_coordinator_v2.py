from __future__ import annotations

import queue
import time
from typing import Any

from .memory_worker_topology_v2 import MemoryWorkerTopologyV2
from .multiprocess import ActorDone, ActorError, ProcessTopology
from .parallel_memory_coordinator import ProcessActorResult, _reconcile_actor_liveness
from .pipeline_service_v2 import MemoryPipelineServiceV2


_PIPELINE_DRAIN_STALL_SECONDS = 30.0


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
    memory = MemoryWorkerTopologyV2(
        topology.ctx,
        ingest_workers=int(ingest_workers),
        derivation_workers=int(derivation_workers),
        ingest_queue_capacity=int(ingest_queue_capacity),
        derivation_queue_capacity=int(derivation_queue_capacity),
        ingest_result_queue_capacity=max(int(publication_queue_capacity), 1024),
        derivation_result_queue_capacity=max(1024, int(derivation_queue_capacity)),
    )
    memory.start()
    pipeline = MemoryPipelineServiceV2(
        runtime,
        memory,
        ingest_queue_capacity=int(ingest_queue_capacity),
    )

    pending = list(jobs)
    active: dict[int, tuple[int, Any]] = {}
    free_slots = list(range(topology.actors))
    results: list[ProcessActorResult] = []
    clean_exit_without_done: dict[int, float] = {}
    initial_policy = runtime.actor_policy_snapshot()
    published_policy_generation = int(initial_policy.generation)
    next_policy_publish = time.monotonic() + max(0.01, float(actor_view_refresh_ms) / 1000.0)
    run_nonce = int(runtime.watermark)
    requested_steps = sum(int(row[2]) for row in jobs)
    started_at = time.monotonic()
    next_progress = started_at + max(1.0, float(progress_interval_seconds))
    clean_shutdown = False

    def launch_one() -> bool:
        if not pending or not free_slots:
            return False
        actor_id, spec, steps, seed = pending.pop(0)
        slot = free_slots.pop(0)
        launch_started = time.perf_counter()
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
            initial_policy=runtime.actor_policy_snapshot(),
            epsilon=float(epsilon),
            policy_refresh_steps=int(actor_view_refresh_steps),
            policy_refresh_ms=float(actor_view_refresh_ms),
        )
        active[actor_id] = (slot, topology.actor_processes[-1])
        runtime.set_telemetry_gauge(
            "actor_launch_latency_ms", 1000.0 * (time.perf_counter() - launch_started)
        )
        runtime.set_telemetry_gauge("actors_launched", len(topology.actor_processes))
        return True

    def drain_publication_queue() -> bool:
        if pipeline.sampled - pipeline.ingested >= pipeline.ingest_local_high_water:
            return False
        progressed = False
        budget = min(512, pipeline.ingest_local_high_water - (pipeline.sampled - pipeline.ingested))
        for _ in range(max(0, budget)):
            try:
                item = topology.publication_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == "transition":
                pipeline.dispatch_transition(item[3])
                progressed = True
            elif item[0] == "shard_done":
                topology.publication_queue.put(item)
                break
        return progressed

    def drain_actor_results() -> bool:
        progressed = False
        for _ in range(256):
            try:
                done = topology.result_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(done, ActorError):
                raise RuntimeError(
                    f"actor {done.actor_id} ({done.game_id}) failed: {done.message}\n{done.traceback_text}"
                )
            if not isinstance(done, ActorDone):
                continue
            entry = active.pop(done.actor_id, None)
            if entry is None:
                continue
            slot, _ = entry
            clean_exit_without_done.pop(done.actor_id, None)
            free_slots.append(slot)
            free_slots.sort()
            results.append(
                ProcessActorResult(
                    done.actor_id,
                    done.game_id,
                    done.steps,
                    done.positive_boundaries,
                    done.negative_boundaries,
                    done.episode_boundaries,
                    done.resets,
                    done.policy_refreshes,
                    done.task_successes,
                    done.task_failures,
                    done.task_truncations,
                    done.levels_completed,
                )
            )
            progressed = True
        return progressed

    def telemetry() -> None:
        elapsed = max(1e-9, time.monotonic() - started_at)
        for key, value in memory.queue_depths().items():
            runtime.set_telemetry_gauge(key, value)
        gauges = pipeline.diagnostics()
        gauges.update(
            {
                "active_actor_processes": len(active),
                "coordinator_action_requests": 0,
                "policy_snapshot_generation": int(published_policy_generation),
                "active_ingest_workers": int(ingest_workers),
                "active_derivation_workers": int(derivation_workers),
                "sampling_rate": pipeline.sampled / elapsed,
                "ingestion_rate": pipeline.ingested / elapsed,
                "derivation_rate": pipeline.derived / elapsed,
                "actors_exited_without_done": len(clean_exit_without_done),
                "canonical_pipeline_version": 2,
            }
        )
        for key, value in gauges.items():
            runtime.set_telemetry_gauge(key, value)

    def progress() -> None:
        telemetry()
        diag = dict(runtime.unified_telemetry.diagnostic_metrics())
        pct = 100.0 * pipeline.ingested / requested_steps if requested_steps else 100.0
        print(
            f"{time.strftime('[%H:%M]')} {pct:5.1f}% sampled={pipeline.sampled}/{requested_steps} "
            f"ingested={pipeline.ingested} rate={float(diag.get('ingestion_rate', 0.0)):.0f}/s "
            f"backlog={max(0, pipeline.sampled - pipeline.ingested)}",
            flush=True,
        )

    def wait_for_published_transitions(expected: int) -> None:
        nonlocal next_progress
        last_progress_at = time.monotonic()
        last_state = (pipeline.sampled, pipeline.ingested)
        while pipeline.sampled < int(expected):
            progressed = drain_publication_queue()
            progressed = pipeline.service() or progressed
            state = (pipeline.sampled, pipeline.ingested)
            if progressed or state != last_state:
                last_progress_at = time.monotonic()
                last_state = state
            now = time.monotonic()
            if now >= next_progress:
                progress()
                next_progress = now + max(1.0, float(progress_interval_seconds))
            if now - last_progress_at >= _PIPELINE_DRAIN_STALL_SECONDS:
                raise RuntimeError(
                    f"actor transition drain stalled: expected={expected} sampled={pipeline.sampled} "
                    f"ingested={pipeline.ingested} pending_ingest={len(pipeline.pending_ingest)}"
                )
            if not progressed:
                time.sleep(0.001)

    try:
        while active or pending:
            progressed = launch_one()
            progressed = drain_publication_queue() or progressed
            progressed = pipeline.service() or progressed
            progressed = drain_actor_results() or progressed
            missing = _reconcile_actor_liveness(active, clean_exit_without_done)
            runtime.set_telemetry_gauge("actors_exited_without_done", missing)
            now = time.monotonic()
            if now >= next_policy_publish:
                snapshot = runtime.actor_policy_snapshot()
                if int(snapshot.generation) > int(published_policy_generation):
                    topology.publish_policy_snapshot(tuple(slot for slot, _ in active.values()), snapshot)
                    published_policy_generation = int(snapshot.generation)
                next_policy_publish = now + max(0.01, float(actor_view_refresh_ms) / 1000.0)
            if now >= next_progress:
                progress()
                next_progress = now + max(1.0, float(progress_interval_seconds))
            if not progressed:
                time.sleep(0.001)

        expected_transitions = sum(int(row.steps) for row in results)
        wait_for_published_transitions(expected_transitions)
        topology.join_actor_workers()
        topology.signal_stage_stop()
        topology.join_stage_workers()
        topology.stop_shard_workers()

        shard_done = 0
        while shard_done < int(shards):
            try:
                item = topology.publication_queue.get(timeout=0.05)
            except queue.Empty:
                pipeline.service()
                continue
            if item[0] == "transition":
                pipeline.dispatch_transition(item[3])
            elif item[0] == "shard_done":
                shard_done += 1
            pipeline.service()
        topology.join_shard_workers()

        last_progress_at = time.monotonic()
        while pipeline.ingested < pipeline.sampled or pipeline.pending_ingest or pipeline.ingest_results:
            progressed = pipeline.service()
            if progressed:
                last_progress_at = time.monotonic()
            elif not pipeline.block_for_result():
                if time.monotonic() - last_progress_at >= _PIPELINE_DRAIN_STALL_SECONDS:
                    raise RuntimeError("ingestion drain stalled")
        memory.signal_ingest_stop()
        memory.join_ingest()

        last_progress_at = time.monotonic()
        while pipeline.pending_derivation or pipeline.inflight or pipeline.derive_results:
            progressed = pipeline.service()
            if progressed:
                last_progress_at = time.monotonic()
            elif not pipeline.block_for_result():
                if time.monotonic() - last_progress_at >= _PIPELINE_DRAIN_STALL_SECONDS:
                    raise RuntimeError("derivation drain stalled")
        memory.signal_derivation_stop()
        memory.join_derivation()

        if pipeline.ingested != expected_transitions:
            raise RuntimeError(
                f"ingestion count mismatch at epoch boundary: expected={expected_transitions} ingested={pipeline.ingested}"
            )
        for key, value in {
            "actor_processes": min(int(actor_limit), len(jobs)),
            "stage_worker_processes": int(stage_workers),
            "shard_worker_processes": int(shards),
            "ingest_worker_processes": int(ingest_workers),
            "derivation_worker_processes": int(derivation_workers),
            "multiprocess_transitions_published": int(pipeline.ingested),
            "coordinator_action_requests": 0,
            "policy_snapshot_generation": int(published_policy_generation),
            "policy_snapshot_refreshes": sum(int(row.policy_refreshes) for row in results),
            "actors_exited_without_done": 0,
            "coordinator_pending_ingest": 0,
            "coordinator_pending_derivation": 0,
            "canonical_pipeline_version": 2,
        }.items():
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
