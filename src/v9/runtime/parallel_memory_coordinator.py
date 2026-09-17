from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import islice
import queue
import time
from typing import Any

from .canonical_commit import apply_canonical_commit_batch
from .memory_pipeline import DerivationBatchTask, DerivationResult, IngestionBatchTask, IngestionTask, PreparedCommitBatch
from .memory_worker_topology import MemoryWorkerTopology
from .multiprocess import ActorDone, ActorError, ProcessTopology
from .shared_batch_transport import consume_shared_batch


_ACTOR_COMPLETION_GRACE_SECONDS = 2.0
_PIPELINE_DRAIN_STALL_SECONDS = 30.0
_CANONICAL_BATCH_MIN = 256
_CANONICAL_BATCH_MAX = 4096
_INGEST_RESULT_DECODE_BATCHES = 2
_INGEST_RESULT_DECODE_BUDGET_SECONDS = 0.005


def _adaptive_canonical_batch_size(current: int, backlog: int) -> int:
    current = max(_CANONICAL_BATCH_MIN, min(_CANONICAL_BATCH_MAX, int(current)))
    backlog = max(0, int(backlog))
    if backlog >= current * 2 and current < _CANONICAL_BATCH_MAX:
        return min(_CANONICAL_BATCH_MAX, current * 2)
    if backlog <= current // 2 and current > _CANONICAL_BATCH_MIN:
        return max(_CANONICAL_BATCH_MIN, current // 2)
    return current


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
    task_successes: int = 0
    task_failures: int = 0
    task_truncations: int = 0
    levels_completed: int = 0


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


class MemoryPipelineService:
    def __init__(self, runtime: Any, memory: Any, *, ingest_queue_capacity: int) -> None:
        self.runtime = runtime
        self.memory = memory
        self.pending_ingest: deque[IngestionTask] = deque()
        self.pending_derivation: deque[Any] = deque()
        self.ingest_results: dict[int, PreparedCommitBatch] = {}
        self.derive_results: dict[int, DerivationResult] = {}
        self.inflight: set[int] = set()
        self.waiting_candidates: dict[int, Any] = {}
        self.last_support: dict[int, int] = {}
        self.sampled = 0
        self.ingested = 0
        self.derived = 0
        self.ingest_sequence = 1
        self.ingest_apply = 1
        self.derive_task_id = 1
        self.derive_apply = 1
        self.watermark_cursor = int(runtime.watermark)
        self.ingest_local_high_water = max(1024, int(ingest_queue_capacity) * 2)
        self.ipc_batch_size = 256
        self.canonical_batch_size = 256
        self.last_canonical_ingest_batch = 0
        self.canonical_apply_seconds = 0.0
        self.canonical_apply_events = 0
        self.ingest_result_drain_seconds = 0.0
        self.ingest_result_batches = 0
        self.ingest_result_rows = 0
        self.ingest_result_bytes = 0
        self.ingest_result_encode_ms = 0.0
        self.ingest_result_decode_ms = 0.0

    def dispatch_transition(self, transition: Any) -> None:
        self.sampled += 1
        canonical_sequence = self.runtime.reserve_producer_sequence(int(transition.actor_id), int(transition.producer_sequence))
        if canonical_sequence != int(transition.producer_sequence):
            transition = replace(transition, producer_sequence=canonical_sequence)
        scientific = self.runtime.config.scientific
        if not bool(scientific.symbolic_grounding_enabled) and tuple(transition.symbols):
            transition = replace(transition, symbols=())
        sequence = self.ingest_sequence
        self.ingest_sequence += 1
        self.watermark_cursor += 1
        symbol_limit = min(int(scientific.symbol_budget_per_window), int(scientific.max_symbol_facts_per_window))
        symbol_count = min(len(tuple(transition.symbols)), symbol_limit)
        self.pending_ingest.append(
            IngestionTask(
                sequence,
                self.watermark_cursor,
                transition,
                symbol_limit,
                int(scientific.symbol_payload_bytes),
                int(scientific.max_cross_modal_facts_per_macro_event),
                str(scientific.symbol_deduplication_policy),
                int(scientific.symbol_window_time_span),
                str(scientific.symbol_codec_name),
                int(scientific.symbol_codec_version),
            )
        )
        self.watermark_cursor += symbol_count

    def pump_ingest_tasks(self) -> bool:
        if not self.pending_ingest:
            return False
        count = min(self.ipc_batch_size, len(self.pending_ingest))
        tasks = tuple(islice(self.pending_ingest, 0, count))
        batch = IngestionBatchTask(tasks[0].sequence, tasks[-1].sequence, tasks)
        try:
            self.memory.ingest_queue.put_nowait(batch)
        except queue.Full:
            return False
        for _ in range(count):
            self.pending_ingest.popleft()
        return True

    def _consider_candidate(self, candidate: Any) -> None:
        signature = int(candidate.structural_signature)
        support = int(candidate.support)
        previous = int(self.last_support.get(signature, 0))
        if signature in self.inflight:
            current = self.waiting_candidates.get(signature)
            if current is None or int(current.support) < support:
                self.waiting_candidates[signature] = candidate
            return
        if support <= previous:
            return
        if previous >= 2 and support < previous * 2:
            return
        task = replace(candidate, task_id=self.derive_task_id)
        self.derive_task_id += 1
        self.inflight.add(signature)
        self.pending_derivation.append(task)

    def pump_derivation_tasks(self) -> bool:
        if not self.pending_derivation:
            return False
        count = min(64, len(self.pending_derivation))
        tasks = tuple(islice(self.pending_derivation, 0, count))
        batch = DerivationBatchTask(int(tasks[0].task_id), int(tasks[-1].task_id), tasks)
        try:
            self.memory.derivation_queue.put_nowait(batch)
        except queue.Full:
            return False
        for _ in range(count):
            self.pending_derivation.popleft()
        return True

    def drain_ingest_results(self, *, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = False
        started = time.perf_counter()
        first = True
        for _ in range(_INGEST_RESULT_DECODE_BATCHES):
            try:
                item = self.memory.ingest_result_queue.get(timeout=timeout) if block and first else self.memory.ingest_result_queue.get_nowait()
            except queue.Empty:
                break
            first = False
            if item[0] == "worker_error":
                raise RuntimeError(f"{item[1]} worker task {item[2]} failed: {item[3]}")
            if item[0] == "ingest_batch_shm":
                descriptor = item[3]
                batch, decode_ms = consume_shared_batch(descriptor)
                self.ingest_results[int(item[1])] = batch
                self.ingest_result_batches += 1
                self.ingest_result_rows += int(descriptor.rows)
                self.ingest_result_bytes += int(descriptor.size)
                self.ingest_result_encode_ms += float(descriptor.encode_ms)
                self.ingest_result_decode_ms += float(decode_ms)
                progressed = True
            elif item[0] == "ingest_batch":
                batch = item[3]
                self.ingest_results[int(item[1])] = batch
                self.ingest_result_batches += 1
                self.ingest_result_rows += len(batch.rows)
                progressed = True
            if progressed and time.perf_counter() - started >= _INGEST_RESULT_DECODE_BUDGET_SECONDS:
                break
        self.ingest_result_drain_seconds += time.perf_counter() - started
        return progressed

    def apply_ingest_ready(self) -> bool:
        plans: list[Any] = []
        while self.ingest_apply in self.ingest_results:
            batch = self.ingest_results[self.ingest_apply]
            if plans and len(plans) + len(batch.rows) > self.canonical_batch_size:
                break
            self.ingest_results.pop(self.ingest_apply)
            plans.extend(batch.rows)
            self.ingest_apply = int(batch.end_sequence) + 1
            if len(plans) >= self.canonical_batch_size:
                break
        if not plans:
            return False
        started = time.perf_counter()
        result = apply_canonical_commit_batch(self.runtime, plans)
        elapsed = time.perf_counter() - started
        self.canonical_apply_seconds += elapsed
        self.canonical_apply_events += len(plans)
        self.last_canonical_ingest_batch = len(plans)
        self.ingested += len(plans)
        for candidate in result.derivation_candidates:
            self._consider_candidate(candidate)
        backlog = max(len(self.pending_ingest), len(self.ingest_results), max(0, self.sampled - self.ingested))
        self.canonical_batch_size = _adaptive_canonical_batch_size(self.canonical_batch_size, backlog)
        return True

    def drain_derivation_results(self, *, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = False
        first = True
        for _ in range(16):
            try:
                item = self.memory.derivation_result_queue.get(timeout=timeout) if block and first else self.memory.derivation_result_queue.get_nowait()
            except queue.Empty:
                break
            first = False
            if item[0] == "worker_error":
                raise RuntimeError(f"{item[1]} worker task {item[2]} failed: {item[3]}")
            if item[0] == "derivation_batch_shm":
                descriptor = item[3]
                batch, _decode_ms = consume_shared_batch(descriptor)
                for row in batch:
                    self.derive_results[int(row.task_id)] = row
                progressed = True
            elif item[0] == "derivation_batch":
                for row in item[3]:
                    self.derive_results[int(row.task_id)] = row
                progressed = True
        return progressed

    def apply_derivation_ready(self) -> bool:
        rows: list[DerivationResult] = []
        while self.derive_apply in self.derive_results:
            row = self.derive_results.pop(self.derive_apply)
            rows.append(row)
            self.derive_apply += 1
        if not rows:
            return False
        self.runtime.apply_derivation_results_batch(rows)
        self.derived += len(rows)
        for row in rows:
            signature = int(row.structural_signature)
            self.inflight.discard(signature)
            self.last_support[signature] = int(row.support)
            waiting = self.waiting_candidates.pop(signature, None)
            if waiting is not None:
                self._consider_candidate(waiting)
        return True

    def service(self) -> bool:
        progressed = self.pump_ingest_tasks()
        progressed = self.pump_derivation_tasks() or progressed
        progressed = self.drain_ingest_results() or progressed
        progressed = self.apply_ingest_ready() or progressed
        progressed = self.drain_derivation_results() or progressed
        progressed = self.apply_derivation_ready() or progressed
        return progressed

    def block_for_result(self, timeout: float = 0.05) -> bool:
        if self.drain_ingest_results(block=True, timeout=float(timeout)):
            return True
        return self.drain_derivation_results(block=True, timeout=float(timeout))

    def diagnostics(self) -> dict[str, float | int]:
        rows = max(1, self.ingest_result_rows)
        return {
            "sampled_steps": self.sampled,
            "ingested_steps": self.ingested,
            "sampling_backlog": max(0, self.sampled - self.ingested),
            "canonical_apply_latency_ms": 1000.0 * self.canonical_apply_seconds / max(1, self.canonical_apply_events),
            "canonical_batch_size": self.last_canonical_ingest_batch,
            "canonical_batch_target": self.canonical_batch_size,
            "ingest_result_batches": self.ingest_result_batches,
            "ingest_result_rows": self.ingest_result_rows,
            "ingest_result_bytes": self.ingest_result_bytes,
            "ingest_result_encode_ms": self.ingest_result_encode_ms,
            "ingest_result_decode_ms": self.ingest_result_decode_ms,
            "ingest_result_drain_ms": 1000.0 * self.ingest_result_drain_seconds,
            "ipc_bytes_per_transition": self.ingest_result_bytes / rows,
            "decode_ms_per_transition": self.ingest_result_decode_ms / rows,
        }


def run_parallel_memory_jobs(
    runtime: Any,
    jobs: list[tuple[int, Any, int, int]],
    *,
    actor_limit: int,
    stage_workers: int,
    shards: int,
    queue_capacity: int,
    epsilon: float,
    stagnation_by_game: dict[str, float] | None = None,
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
    hgt_dataset: Any | None = None,
) -> list[ProcessActorResult]:
    target_actor_slots = min(int(actor_limit), len(jobs))
    topology = ProcessTopology(
        actors=target_actor_slots,
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
        ingest_result_queue_capacity=max(int(publication_queue_capacity), 1024),
        derivation_result_queue_capacity=max(1024, int(derivation_queue_capacity)),
    )
    memory.start()
    pipeline = MemoryPipelineService(runtime, memory, ingest_queue_capacity=int(ingest_queue_capacity))

    pending = deque(jobs)
    active: dict[int, tuple[int, Any]] = {}
    free_slots = list(range(topology.actors))
    results: list[ProcessActorResult] = []
    clean_exit_without_done: dict[int, float] = {}
    grounded_influence_total = 0
    peak_active_actors = 0
    initial_policy = runtime.actor_policy_snapshot()
    published_policy_generation = int(initial_policy.generation)
    next_policy_publish = time.monotonic() + max(0.01, float(actor_view_refresh_ms) / 1000.0)
    run_nonce = int(runtime.watermark)
    requested_steps = sum(int(row[2]) for row in jobs)
    started_at = time.monotonic()
    next_progress = started_at + max(1.0, float(progress_interval_seconds))
    clean_shutdown = False
    dataset_start_count = int(getattr(hgt_dataset, "count", 0)) if hgt_dataset is not None else 0

    runtime.set_telemetry_gauge("actor_slots_target", target_actor_slots)
    runtime.set_telemetry_gauge("actor_process_start_method", topology.actor_start_method)

    def actor_produced_steps() -> int:
        return int(getattr(topology, "produced_steps", pipeline.sampled))

    def _publish_actor_process_telemetry() -> None:
        runtime.set_telemetry_gauge("active_actor_processes", len(active))
        runtime.set_telemetry_gauge("peak_active_actor_processes", peak_active_actors)
        runtime.set_telemetry_gauge("actor_processes_total_launched", len(topology.actor_processes))
        runtime.set_telemetry_gauge("actor_slots_filled", len(active))
        runtime.set_telemetry_gauge("actor_process_pids", ",".join(str(process.pid) for process in topology.actor_processes if process.pid is not None))

    def launch_one() -> bool:
        nonlocal peak_active_actors
        if not pending or not free_slots:
            return False
        actor_id, spec, steps, seed = pending.popleft()
        slot = free_slots.pop(0)
        launch_started = time.perf_counter()
        game_name = str(getattr(spec, "display_name", getattr(spec, "game_id", "")))
        stagnation = float((stagnation_by_game or {}).get(game_name, 0.0))
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
            stagnation=stagnation,
            policy_refresh_steps=int(actor_view_refresh_steps),
            policy_refresh_ms=float(actor_view_refresh_ms),
        )
        process = topology.actor_processes[-1]
        if process.pid is None:
            raise RuntimeError(f"actor {actor_id} failed to start an OS process")
        active[actor_id] = (slot, process)
        peak_active_actors = max(peak_active_actors, len(active))
        runtime.set_telemetry_gauge("actor_launch_latency_ms", 1000.0 * (time.perf_counter() - launch_started))
        runtime.set_telemetry_gauge("actors_launched", len(topology.actor_processes))
        runtime.set_telemetry_gauge("live_environment_instances", len(active))
        runtime.set_telemetry_gauge("available_actor_slots", len(free_slots))
        runtime.set_telemetry_gauge("pending_environment_jobs", len(pending))
        _publish_actor_process_telemetry()
        return True

    def launch_available_slots() -> int:
        launched = 0
        while pending and free_slots:
            if not launch_one():
                break
            launched += 1
        return launched

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
                if hgt_dataset is not None:
                    hgt_dataset.append(item[3])
                pipeline.dispatch_transition(item[3])
                progressed = True
            elif item[0] == "shard_done":
                topology.publication_queue.put(item)
                break
        return progressed

    def drain_actor_results() -> bool:
        nonlocal grounded_influence_total
        progressed = False
        for _ in range(256):
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
            grounded_influence_total += int(getattr(done, "grounded_action_influence", 0))
            runtime.set_telemetry_gauge("grounded_action_influence", grounded_influence_total)
            runtime.telemetry["grounded_action_influence"] = grounded_influence_total
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
            runtime.set_telemetry_gauge("live_environment_instances", len(active))
            runtime.set_telemetry_gauge("available_actor_slots", len(free_slots))
            runtime.set_telemetry_gauge("pending_environment_jobs", len(pending))
            _publish_actor_process_telemetry()
            progressed = True
        return progressed

    def telemetry() -> None:
        if hgt_dataset is not None:
            runtime.set_telemetry_gauge("hgt_dataset_written_transitions", int(hgt_dataset.count) - dataset_start_count)
            runtime.set_telemetry_gauge("hgt_dataset_bytes", int(hgt_dataset.bytes_written))
        elapsed = max(1e-9, time.monotonic() - started_at)
        produced = actor_produced_steps()
        published = int(pipeline.sampled)
        ingested = int(pipeline.ingested)
        for key, value in memory.queue_depths().items():
            runtime.set_telemetry_gauge(key, value)
        gauges = pipeline.diagnostics()
        gauges.update(
            {
                "sampled_steps": produced,
                "actor_produced_steps": produced,
                "publication_drained_steps": published,
                "ingested_steps": ingested,
                "sampling_backlog": max(0, produced - ingested),
                "publication_backlog": max(0, produced - published),
                "canonical_ingest_backlog": max(0, published - ingested),
                "active_actor_processes": len(active),
                "peak_active_actor_processes": peak_active_actors,
                "actor_slots_target": target_actor_slots,
                "actor_slots_filled": len(active),
                "actor_processes_total_launched": len(topology.actor_processes),
                "live_environment_instances": len(active),
                "available_actor_slots": len(free_slots),
                "pending_environment_jobs": len(pending),
                "coordinator_action_requests": 0,
                "policy_snapshot_generation": int(published_policy_generation),
                "active_ingest_workers": int(ingest_workers),
                "active_derivation_workers": int(derivation_workers),
                "sampling_rate": produced / elapsed,
                "publication_rate": published / elapsed,
                "ingestion_rate": ingested / elapsed,
                "derivation_rate": pipeline.derived / elapsed,
                "actors_exited_without_done": len(clean_exit_without_done),
                "canonical_pipeline_version": 3,
            }
        )
        for key, value in gauges.items():
            runtime.set_telemetry_gauge(key, value)
        _publish_actor_process_telemetry()

    def progress() -> None:
        telemetry()
        diag = dict(runtime.unified_telemetry.diagnostic_metrics())
        produced = actor_produced_steps()
        published = int(pipeline.sampled)
        ingested = int(pipeline.ingested)
        pct = 100.0 * produced / requested_steps if requested_steps else 100.0
        print(
            f"{time.strftime('[%H:%M]')} {pct:5.1f}% sampled={produced}/{requested_steps} "
            f"published={published} ingested={ingested} "
            f"games_finished={len(results)}/{len(jobs)} actors_active={len(active)}/{target_actor_slots} "
            f"actors_peak={peak_active_actors} rate={float(diag.get('ingestion_rate', 0.0)):.0f}/s "
            f"pub_backlog={max(0, produced - published)} ingest_backlog={max(0, published - ingested)}",
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
                    f"actor transition drain stalled: expected={expected} produced={actor_produced_steps()} "
                    f"published={pipeline.sampled} ingested={pipeline.ingested} pending_ingest={len(pipeline.pending_ingest)}"
                )
            if not progressed:
                time.sleep(0.001)

    try:
        initial_launched = launch_available_slots()
        if target_actor_slots > 0 and len(active) != target_actor_slots:
            raise RuntimeError(
                f"failed to prefill actor slots: target={target_actor_slots} active={len(active)} launched={initial_launched}"
            )
        if target_actor_slots > 1 and len({process.pid for _, process in active.values()}) != target_actor_slots:
            raise RuntimeError("actor slots did not produce distinct OS processes")
        runtime.set_telemetry_gauge("actor_prefill_complete", 1)
        runtime.set_telemetry_gauge("actor_prefill_processes", len(active))
        _publish_actor_process_telemetry()

        while active or pending:
            progressed = drain_publication_queue()
            progressed = pipeline.service() or progressed
            progressed = drain_actor_results() or progressed
            launched = launch_available_slots()
            progressed = bool(launched) or progressed
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
        shard_drain_started = time.monotonic()
        last_shard_progress = shard_drain_started
        completed_shards: set[int] = set()
        while shard_done < int(shards):
            try:
                item = topology.publication_queue.get(timeout=0.05)
            except queue.Empty:
                pipeline.service()
                exited = {index: process.exitcode for index, process in enumerate(topology.shard_processes) if process.exitcode is not None}
                if exited and len(completed_shards) < int(shards):
                    missing = sorted(set(range(int(shards))) - completed_shards)
                    raise RuntimeError(
                        f"shard drain terminated before completion: completed={sorted(completed_shards)} missing={missing} exitcodes={exited}"
                    )
                if time.monotonic() - last_shard_progress >= _PIPELINE_DRAIN_STALL_SECONDS:
                    missing = sorted(set(range(int(shards))) - completed_shards)
                    raise RuntimeError(
                        f"shard drain stalled for {_PIPELINE_DRAIN_STALL_SECONDS:.0f}s: completed={sorted(completed_shards)} "
                        f"missing={missing} produced={actor_produced_steps()} published={pipeline.sampled} ingested={pipeline.ingested}"
                    )
                continue
            if item[0] == "transition":
                pipeline.dispatch_transition(item[3])
                last_shard_progress = time.monotonic()
            elif item[0] == "shard_done":
                completed_shards.add(int(item[1]))
                shard_done = len(completed_shards)
                last_shard_progress = time.monotonic()
            pipeline.service()
        runtime.set_telemetry_gauge("shard_drain_seconds", time.monotonic() - shard_drain_started)
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
            raise RuntimeError(f"ingestion count mismatch at epoch boundary: expected={expected_transitions} ingested={pipeline.ingested}")
        for key, value in {
            "actor_processes": target_actor_slots,
            "peak_active_actor_processes": peak_active_actors,
            "actor_processes_total_launched": len(topology.actor_processes),
            "stage_worker_processes": int(stage_workers),
            "shard_worker_processes": int(shards),
            "ingest_worker_processes": int(ingest_workers),
            "derivation_worker_processes": int(derivation_workers),
            "actor_produced_steps": int(actor_produced_steps()),
            "publication_drained_steps": int(pipeline.sampled),
            "multiprocess_transitions_published": int(pipeline.sampled),
            "coordinator_action_requests": 0,
            "policy_snapshot_generation": int(published_policy_generation),
            "policy_snapshot_refreshes": sum(int(row.policy_refreshes) for row in results),
            "actors_exited_without_done": 0,
            "coordinator_pending_ingest": 0,
            "coordinator_pending_derivation": 0,
            "canonical_pipeline_version": 3,
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
        shutdown_parallel_pipeline = getattr(pipeline, "shutdown_parallel_pipeline", None)
        if callable(shutdown_parallel_pipeline):
            shutdown_parallel_pipeline()
        memory.close(drain=clean_shutdown)
        topology.close(drain=clean_shutdown)
