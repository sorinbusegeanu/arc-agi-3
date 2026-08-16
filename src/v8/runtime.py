from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from v8.arena import SharedActionArena, SharedEdgeArena, SharedNodeArena
from v8.compaction import CompactionResult, compact_retired_memory as compact_retired_arenas
from v8.development import STAGES, stage_worker
from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.model import (
    CognitiveState,
    EventId,
    ExperienceEvent,
    MemoryLevel,
    MemoryProposal,
    MemoryUid,
    PIPELINE_PACKET_SIZE,
    PROPOSAL_PACKET_SIZE,
    PipelineEvent,
    encode_pipeline,
    encode_proposal,
    stable_u64,
)
from v8.peers import DevelopmentalPeerSupervisor
from v8.publication import LiveReadView, ShardReadDescriptor
from v8.reporting_snapshot import capture_reporting_cut
from v8.ring import SharedRingBuffer
from v8.scheduler import ResourceController
from v8.shard import ShardConfig, shard_worker
from v8.snapshot import (
    SnapshotResult,
    SnapshotService,
    load_latest_auxiliary_state,
    restore_latest_snapshot,
)


def _safe_mp_context(requested: str | None = None):
    methods = tuple(mp.get_all_start_methods())
    method = requested
    if method is None:
        method = "forkserver" if "forkserver" in methods else "spawn"
    if method not in methods:
        raise ValueError(
            f"multiprocessing start method {method!r} is unavailable; choices={methods}"
        )
    if method == "fork":
        raise ValueError("v8 does not allow fork because runtime peers use threads")
    return mp.get_context(method)


@dataclass(frozen=True, slots=True)
class V8RuntimeConfig:
    root: Path
    shards: int = 4
    stage_workers: int = 2
    stage_ring_capacity: int = 8192
    shard_ring_capacity: int = 8192
    node_capacity_per_shard: int = 250_000
    edge_capacity_per_shard: int = 500_000
    action_capacity_per_shard: int = 65_536
    shard_batch_size: int = 256
    snapshot_interval_seconds: float = 60.0
    enable_snapshots: bool = True
    restore: bool = True
    enable_peers: bool = True
    peer_interval_seconds: float = 0.5
    multiprocessing_start_method: str | None = None

    @classmethod
    def from_path(cls, root: str | Path, **kwargs) -> "V8RuntimeConfig":
        return cls(Path(root), **kwargs)

    def __post_init__(self) -> None:
        if self.shards <= 0 or self.stage_workers <= 0:
            raise ValueError("shards and stage_workers must be positive")
        if self.stage_ring_capacity <= 0 or self.shard_ring_capacity <= 0:
            raise ValueError("ring capacities must be positive")
        if self.snapshot_interval_seconds <= 0 or self.peer_interval_seconds <= 0:
            raise ValueError("snapshot and peer intervals must be positive")
        if self.multiprocessing_start_method is not None:
            _safe_mp_context(self.multiprocessing_start_method)


class ContinuousMemoryRuntime:
    """RAM-authoritative continuous developmental memory runtime."""

    def __init__(self, config: V8RuntimeConfig) -> None:
        self.config = config
        self.root = config.root
        self.root.mkdir(parents=True, exist_ok=True)
        self._mp_ctx = _safe_mp_context(config.multiprocessing_start_method)
        self._stop = self._mp_ctx.Event()
        self._snapshot_freeze = self._mp_ctx.Event()
        self._accepting = True
        self._started = False
        self._closed = False
        self._error_queue = self._mp_ctx.Queue()
        self._watermark = self._mp_ctx.Value("Q", 0)
        self._generation = self._mp_ctx.Value("Q", 0)
        self._actor_throttle = self._mp_ctx.Value("d", 0.0)
        self._snapshot_id = 0
        self._submit_lock = threading.Lock()
        self._maintenance_lock = threading.Lock()
        self._snapshot_error: str | None = None
        self._last_compaction = CompactionResult(0, 0, 0, 0)

        self._stage_rings = tuple(
            SharedRingBuffer(
                capacity=config.stage_ring_capacity,
                slot_size=PIPELINE_PACKET_SIZE,
                mp_context=self._mp_ctx,
            )
            for _ in STAGES
        )
        self._shard_rings = tuple(
            SharedRingBuffer(
                capacity=config.shard_ring_capacity,
                slot_size=PROPOSAL_PACKET_SIZE,
                mp_context=self._mp_ctx,
            )
            for _ in range(config.shards)
        )
        self._node_arenas = tuple(
            SharedNodeArena(capacity=config.node_capacity_per_shard) for _ in range(config.shards)
        )
        self._edge_arenas = tuple(
            SharedEdgeArena(capacity=config.edge_capacity_per_shard) for _ in range(config.shards)
        )
        self._action_arenas = tuple(
            SharedActionArena(capacity=config.action_capacity_per_shard) for _ in range(config.shards)
        )
        self.shard_descriptors = tuple(
            ShardReadDescriptor(nodes.descriptor, edges.descriptor, actions.descriptor)
            for nodes, edges, actions in zip(
                self._node_arenas, self._edge_arenas, self._action_arenas, strict=True
            )
        )

        restored = restore_latest_snapshot(self.root, self.shard_descriptors) if config.restore else None
        restored_aux = load_latest_auxiliary_state(self.root) if restored is not None else None
        if restored is not None:
            self._snapshot_id, restored_watermark = restored
            with self._watermark.get_lock():
                self._watermark.value = int(restored_watermark)
        if restored_aux is not None:
            with self._generation.get_lock():
                self._generation.value = int(restored_aux.get("generation", 0))

        self.read_view = LiveReadView(self.shard_descriptors)
        self._submitted_event_ids: set[tuple[int, int]] = set()
        if restored is not None:
            for record in self.read_view.node_records(level=MemoryLevel.M0):
                if len(record.key_parts) >= 2:
                    self._submitted_event_ids.add((int(record.key_parts[0]), int(record.key_parts[1])))

        self._stage_inflight = tuple(self._mp_ctx.Value("Q", 0) for _ in STAGES)
        self._shard_inflight = tuple(
            self._mp_ctx.Value("Q", 0) for _ in range(config.shards)
        )
        self._shard_watermarks = tuple(
            self._mp_ctx.Value("Q", 0) for _ in range(config.shards)
        )
        self._stage_processes = []
        self._shard_processes = [
            self._build_shard_process(shard_id) for shard_id in range(config.shards)
        ]

        shard_ring_args = tuple(ring.attachment_args() for ring in self._shard_rings)
        for stage_index, definition in enumerate(STAGES):
            next_args = (
                None
                if stage_index + 1 >= len(STAGES)
                else self._stage_rings[stage_index + 1].attachment_args()
            )
            for worker_id in range(config.stage_workers):
                self._stage_processes.append(
                    self._mp_ctx.Process(
                        target=stage_worker,
                        kwargs={
                            "level": int(definition.level),
                            "ingress_args": self._stage_rings[stage_index].attachment_args(),
                            "next_args": next_args,
                            "shard_ring_args": shard_ring_args,
                            "stop_event": self._stop,
                            "inflight": self._stage_inflight[stage_index],
                            "error_queue": self._error_queue,
                        },
                        name=f"v8-M{stage_index}-worker-{worker_id:02d}",
                        daemon=True,
                    )
                )

        self.snapshot_service = (
            SnapshotService(self.root, self.shard_descriptors) if config.enable_snapshots else None
        )
        if restored is not None and self.snapshot_service is not None:
            with self.snapshot_service.saved_watermark.get_lock():
                self.snapshot_service.saved_watermark.value = int(restored[1])
            with self.snapshot_service.saved_snapshot.get_lock():
                self.snapshot_service.saved_snapshot.value = int(restored[0])
        self._snapshot_thread: threading.Thread | None = None
        self._snapshot_thread_stop = threading.Event()

        self.resource_controller = ResourceController()
        self._resource_thread: threading.Thread | None = None
        self._resource_thread_stop = threading.Event()
        self.peers = (
            DevelopmentalPeerSupervisor(
                read_view=self.read_view,
                submit_proposal=self.submit_proposal,
                watermark=lambda: self.watermark,
                generation=lambda: self.generation,
                interval_seconds=config.peer_interval_seconds,
            )
            if config.enable_peers
            else None
        )
        if self.peers is not None and restored_aux is not None:
            peer_state = restored_aux.get("peers")
            if isinstance(peer_state, dict):
                self.peers.load_state(peer_state)
        self.hypothesis_evaluator = ScientificHypothesisEvaluator()

    def _build_shard_process(self, shard_id: int):
        return self._mp_ctx.Process(
            target=shard_worker,
            args=(
                ShardConfig(
                    shard_id,
                    self._node_arenas[shard_id].descriptor,
                    self._edge_arenas[shard_id].descriptor,
                    self._action_arenas[shard_id].descriptor,
                ),
                self._shard_rings[shard_id].attachment_args(),
                self._stop,
                self._shard_inflight[shard_id],
                self._shard_watermarks[shard_id],
                self._error_queue,
                self.config.shard_batch_size,
                self._generation,
            ),
            name=f"v8-shard-{shard_id:02d}",
            daemon=True,
        )

    @property
    def watermark(self) -> int:
        with self._watermark.get_lock():
            return int(self._watermark.value)

    @property
    def generation(self) -> int:
        with self._generation.get_lock():
            return int(self._generation.value)

    def start(self) -> None:
        if self._started:
            return
        for process in self._shard_processes:
            process.start()
        for process in self._stage_processes:
            process.start()
        if self.snapshot_service is not None:
            self.snapshot_service.start()
            self._snapshot_thread = threading.Thread(
                target=self._snapshot_cadence,
                name="v8-snapshot-cadence",
                daemon=True,
            )
            self._snapshot_thread.start()
        if self.peers is not None:
            self.peers.start()
        self._resource_thread = threading.Thread(
            target=self._resource_cadence,
            name="v8-resource-controller",
            daemon=True,
        )
        self._resource_thread.start()
        self._started = True

    def _auxiliary_state_json(self) -> str:
        payload: dict[str, object] = {
            "version": 1,
            "watermark": self.watermark,
            "generation": self.generation,
        }
        if self.peers is not None:
            payload["peers"] = self.peers.state_dict()
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _snapshot_cadence(self) -> None:
        interval = float(self.config.snapshot_interval_seconds)
        while not self._snapshot_thread_stop.wait(interval):
            if self._closed or self.snapshot_service is None:
                return
            try:
                self.request_consistent_snapshot(timeout=max(10.0, interval))
            except BaseException as exc:
                self._snapshot_error = f"{type(exc).__name__}: {exc}"
                return

    def _resource_cadence(self) -> None:
        while not self._resource_thread_stop.wait(0.5):
            if self._closed:
                return
            memory_capacity = self.config.node_capacity_per_shard * self.config.shards
            edge_capacity = self.config.edge_capacity_per_shard * self.config.shards
            decision = self.resource_controller.decide(
                stage_depths=tuple(ring.qsize for ring in self._stage_rings),
                shard_depths=tuple(ring.qsize for ring in self._shard_rings),
                stage_capacity=self.config.stage_ring_capacity,
                shard_capacity=self.config.shard_ring_capacity,
                memory_count=self.read_view.memory_count,
                memory_capacity=memory_capacity,
            )
            with self._actor_throttle.get_lock():
                self._actor_throttle.value = float(decision.actor_throttle_seconds)
            if self.peers is not None:
                self.peers.set_interval(decision.peer_interval_seconds)
                self.peers.set_candidate_budget(decision.candidate_budget)
            node_ratio = self.read_view.memory_count / max(1, memory_capacity)
            edge_ratio = self.read_view.edge_count / max(1, edge_capacity)
            if max(node_ratio, edge_ratio) >= 0.80 and any(
                int(row.cognitive_state) == int(CognitiveState.RETIRED)
                for row in self.read_view.node_records()
            ):
                try:
                    self.compact_retired_memory(timeout=30.0)
                except BaseException as exc:
                    self._snapshot_error = f"compaction {type(exc).__name__}: {exc}"
                    return

    def make_experience(
        self,
        *,
        producer_id: int,
        producer_sequence: int,
        source_game_hash: int,
        global_step: int,
        context_signature: int,
        action_id: int,
        outcome_signature: int,
        family_signature: int,
        carrier_signature: int = 0,
        future_option_delta: float = 0.0,
        changed_cells: int = 0,
        terminal_polarity: int = 0,
        trajectory_signature: int = 0,
        next_context_signature: int = 0,
        prediction_error: float = 0.0,
    ) -> ExperienceEvent:
        return ExperienceEvent(
            event_id=EventId.from_producer(producer_id, producer_sequence),
            watermark=0,
            producer_id=int(producer_id),
            producer_sequence=int(producer_sequence),
            source_game_hash=int(source_game_hash),
            global_step=int(global_step),
            context_signature=int(context_signature),
            action_id=int(action_id),
            outcome_signature=int(outcome_signature),
            family_signature=int(family_signature),
            carrier_signature=int(carrier_signature),
            future_option_delta=float(future_option_delta),
            changed_cells=int(changed_cells),
            terminal_polarity=int(terminal_polarity),
            trajectory_signature=int(trajectory_signature),
            next_context_signature=int(next_context_signature),
            prediction_error=float(prediction_error),
        )

    def submit(self, event: ExperienceEvent, *, timeout: float = 1.0) -> None:
        if not self._started:
            self.start()
        if not self._accepting:
            raise RuntimeError("v8 runtime is draining and no longer accepts experiences")
        while self._snapshot_freeze.is_set() and not self._stop.is_set():
            time.sleep(0.002)
        self.raise_worker_errors()
        event_key = (int(event.event_id.hi), int(event.event_id.lo))
        with self._submit_lock:
            if event_key in self._submitted_event_ids:
                return
            with self._watermark.get_lock():
                current = int(self._watermark.value)
                assigned = int(event.watermark)
                if assigned <= 0:
                    assigned = current + 1
                accepted = replace(event, watermark=assigned)
                packet = encode_pipeline(PipelineEvent(accepted))
                if not self._stage_rings[0].put(packet, timeout=timeout):
                    raise TimeoutError("M0 experience ring remained full")
                self._watermark.value = max(current, assigned)
            self._submitted_event_ids.add(event_key)

    def submit_proposal(self, proposal: MemoryProposal, *, timeout: float = 0.25) -> None:
        if self._closed or self._stop.is_set():
            return
        shard = proposal.uid.shard(len(self._shard_rings))
        deadline = time.monotonic() + float(timeout)
        payload = encode_proposal(proposal)
        while not self._stop.is_set():
            if self._shard_rings[shard].put(payload, timeout=0.05):
                return
            if time.monotonic() >= deadline:
                return

    def raise_worker_errors(self) -> None:
        errors = []
        while True:
            try:
                errors.append(self._error_queue.get_nowait())
            except queue.Empty:
                break
        if errors:
            raise RuntimeError(f"v8 worker failure: {errors}")
        for process in (*self._stage_processes, *self._shard_processes):
            if process.exitcode not in (None, 0):
                raise RuntimeError(f"v8 worker exited unexpectedly: {process.name}={process.exitcode}")
        if self.peers is not None:
            self.peers.raise_if_failed()
        if self._snapshot_error is not None:
            message = self._snapshot_error
            self._snapshot_error = None
            raise RuntimeError(f"v8 snapshot failure: {message}")

    def _is_quiescent(self) -> bool:
        if any(not ring.empty for ring in (*self._stage_rings, *self._shard_rings)):
            return False
        if any(int(value.value) != 0 for value in (*self._stage_inflight, *self._shard_inflight)):
            return False
        return True

    def wait_quiescent(
        self,
        *,
        timeout: float = 60.0,
        stable_checks: int = 5,
        resume_peers: bool = True,
        settle_peers: bool = True,
    ) -> None:
        """Drain canonical queues and advance peer operators to a fixed point."""
        deadline = time.monotonic() + float(timeout)
        stable = 0
        if self.peers is not None:
            self.peers.pause()
        try:
            while time.monotonic() < deadline:
                self.raise_worker_errors()
                if not self._is_quiescent():
                    stable = 0
                    time.sleep(0.01)
                    continue
                peer_changed = False
                if self.peers is not None and settle_peers:
                    before = self.peers.metrics().proposals
                    self.peers.run_once()
                    after = self.peers.metrics().proposals
                    peer_changed = after > before
                if peer_changed or not self._is_quiescent():
                    stable = 0
                else:
                    stable += 1
                    if stable >= int(stable_checks):
                        return
                time.sleep(0.01)
            raise TimeoutError("v8 did not reach quiescence; " + json.dumps(self.metrics(), sort_keys=True))
        finally:
            if self.peers is not None and resume_peers and not self._snapshot_freeze.is_set():
                self.peers.resume()

    def compact_retired_memory(self, *, timeout: float = 30.0) -> CompactionResult:
        """Archive and physically reclaim RETIRED RAM rows without losing provenance."""
        if not self._started:
            self.start()
        with self._maintenance_lock:
            self._snapshot_freeze.set()
            if self.peers is not None:
                self.peers.pause()
            try:
                self.wait_quiescent(timeout=timeout, resume_peers=False)
                if not any(
                    int(row.cognitive_state) == int(CognitiveState.RETIRED)
                    for row in self.read_view.node_records()
                ):
                    self._last_compaction = CompactionResult(
                        0, 0, self.read_view.memory_count, self.read_view.edge_count
                    )
                    return self._last_compaction

                old_processes = tuple(self._shard_processes)
                self._shard_processes = []
                for process in old_processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=2.0)

                result = compact_retired_arenas(
                    self.shard_descriptors,
                    archive_path=self.root / "archive" / "retired_memory.jsonl",
                )
                self._shard_processes = [
                    self._build_shard_process(shard_id) for shard_id in range(self.config.shards)
                ]
                for process in self._shard_processes:
                    process.start()
                with self._generation.get_lock():
                    self._generation.value += 1
                self._last_compaction = result
                return result
            finally:
                if self.peers is not None:
                    self.peers.resume()
                self._snapshot_freeze.clear()

    def request_consistent_snapshot(self, *, timeout: float = 30.0) -> None:
        if self.snapshot_service is None:
            return
        with self._maintenance_lock:
            deadline = time.monotonic() + float(timeout)
            self._snapshot_freeze.set()
            if self.peers is not None:
                self.peers.pause()
            try:
                if self.peers is not None and not self.peers.wait_idle(
                    max(0.0, deadline - time.monotonic())
                ):
                    raise TimeoutError("v8 peers did not pause for consistent snapshot")
                self.wait_quiescent(
                    timeout=max(0.0, deadline - time.monotonic()),
                    resume_peers=False,
                    settle_peers=False,
                )
                self._snapshot_id += 1
                self.snapshot_service.request_consistent_capture(
                    self._snapshot_id,
                    self.watermark,
                    generation=self.generation,
                    auxiliary_state=self._auxiliary_state_json(),
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            finally:
                if self.peers is not None:
                    self.peers.resume()
                self._snapshot_freeze.clear()

    def scientific_statuses(self) -> dict[str, str]:
        if self.peers is None:
            return {f"H{i:02d}": "INSUFFICIENT_EVIDENCE" for i in range(1, 16)}
        decisions = self.hypothesis_evaluator.evaluate(self.peers.ledger.cut(self.watermark))
        return self.hypothesis_evaluator.status_map(decisions)

    def write_scientific_report(self) -> None:
        if self.peers is None:
            return
        cut = capture_reporting_cut(
            self.read_view,
            self.peers.ledger,
            self.watermark,
            generation=self.generation,
        )
        self.peers.ledger.export_jsonl(
            self.root / "evidence" / "v8_evidence.jsonl", watermark=cut.watermark
        )
        decisions = self.hypothesis_evaluator.write_report(
            self.root / "reports" / "h01_h15.json", cut.evidence
        )
        (self.root / "reports" / "reporting_cut.json").write_text(
            json.dumps(
                {
                    "watermark": cut.watermark,
                    "generation": cut.generation,
                    "graph_digest": cut.graph_digest,
                    "evidence_count": len(cut.evidence),
                    "decisions": [asdict(row) for row in decisions],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def record_actor_results(self, results) -> None:
        if self.peers is None:
            return
        for result in results:
            game_hash = stable_u64(result.game_id, person=b"v8-game")
            for stat in getattr(result, "strategy_stats", ()):
                self.peers.record_strategy_statistics(
                    stat.strategy_uid,
                    attempts=stat.attempts,
                    successes=stat.successes,
                    cost=stat.cost,
                    source_game_hash=game_hash,
                )
            for probe in getattr(result, "preference_probes", ()):
                self.peers.record_preference_probe(
                    outcome_a=probe.outcome_a,
                    outcome_b=probe.outcome_b,
                    context_bucket=probe.context_bucket,
                    chosen_outcome=probe.chosen_outcome,
                    both_reachable=True,
                    preference_influenced=probe.preference_influenced,
                )
            for trial in getattr(result, "replanning_trials", ()):
                recorded = self.peers.record_replanning_trial(
                    primary_strategy_uid=trial.primary_strategy_uid,
                    alternative_strategy_uid=trial.alternative_strategy_uid,
                    outcome_uid=trial.outcome_uid,
                    primary_invalidated=True,
                    alternative_selected=True,
                    recovery_succeeded=trial.recovery_succeeded,
                )
                if not recorded.valid_recovery:
                    alternative = next(
                        (
                            row
                            for row in self.read_view.node_records(level=MemoryLevel.M7)
                            if row.uid == trial.alternative_strategy_uid
                        ),
                        None,
                    )
                    if alternative is not None:
                        self.peers._append_evidence(
                            "replanning_recovery_fail",
                            alternative,
                            1.0,
                            unique=True,
                            causal_intervention="strategy_ablation_recovery",
                            effect_direction=-1,
                        )
            if int(getattr(result, "replans", 0)) > 0:
                self.peers.ledger.append(
                    EvidenceRecord.for_uid(
                        f"replanning-observed:{result.actor_id}:{self.watermark}",
                        MemoryUid.zero(),
                        evidence_kind="replanning_observed",
                        watermark=self.watermark,
                        raw_value=float(result.replans),
                        normalized_value=min(1.0, float(result.replans)),
                        developmental_stage=int(MemoryLevel.M7),
                        validation_state=3,
                        source_game_hash=game_hash,
                        graph_generation=self.generation,
                    )
                )

    def request_snapshot(self) -> None:
        self.request_consistent_snapshot()

    def final_snapshot(self, *, timeout: float = 120.0) -> SnapshotResult:
        if self.snapshot_service is None:
            raise RuntimeError("snapshots are disabled")
        self._snapshot_id += 1
        return self.snapshot_service.request_final(
            self._snapshot_id,
            self.watermark,
            generation=self.generation,
            auxiliary_state=self._auxiliary_state_json(),
            timeout=timeout,
        )

    def metrics(self) -> dict[str, object]:
        saved_watermark = (
            0 if self.snapshot_service is None else int(self.snapshot_service.saved_watermark.value)
        )
        peer_metrics = None
        if self.peers is not None:
            peer = self.peers.metrics()
            peer_metrics = {
                "cycles": peer.cycles,
                "proposals": peer.proposals,
                "evidence_records": peer.evidence_records,
                "interval_seconds": peer.interval_seconds,
                "candidate_budget": peer.candidate_budget,
                "failures": peer.failures,
            }
        with self._actor_throttle.get_lock():
            throttle = float(self._actor_throttle.value)
        return {
            "watermark": self.watermark,
            "generation": self.generation,
            "multiprocessing_start_method": self._mp_ctx.get_start_method(),
            "saved_watermark": saved_watermark,
            "unsaved_tail": max(0, self.watermark - saved_watermark),
            "memories": self.read_view.memory_count,
            "edges": self.read_view.edge_count,
            "level_counts": self.read_view.level_counts(),
            "stage_queue_depths": [ring.qsize for ring in self._stage_rings],
            "shard_queue_depths": [ring.qsize for ring in self._shard_rings],
            "stage_inflight": [int(value.value) for value in self._stage_inflight],
            "shard_inflight": [int(value.value) for value in self._shard_inflight],
            "shard_watermarks": [int(value.value) for value in self._shard_watermarks],
            "actor_throttle_seconds": throttle,
            "last_compaction": asdict(self._last_compaction),
            "peers": peer_metrics,
        }

    def _cleanup(self) -> None:
        self._stop.set()
        self._snapshot_freeze.clear()
        for process in (*self._stage_processes, *self._shard_processes):
            process.join(timeout=5.0)
        for process in (*self._stage_processes, *self._shard_processes):
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        if self.snapshot_service is not None:
            self.snapshot_service.close()
        try:
            self.read_view.close()
        finally:
            for ring in (*self._stage_rings, *self._shard_rings):
                ring.dispose()
            for arena in (*self._node_arenas, *self._edge_arenas, *self._action_arenas):
                arena.dispose()
        self._closed = True

    def close(self, *, normal: bool = True, timeout: float = 120.0) -> SnapshotResult | None:
        if self._closed:
            return None
        if not self._started:
            self.start()
        self._accepting = False
        self._snapshot_thread_stop.set()
        self._resource_thread_stop.set()
        if self._snapshot_thread is not None:
            self._snapshot_thread.join(timeout=2.0)
        if self._resource_thread is not None:
            self._resource_thread.join(timeout=2.0)

        final_result: SnapshotResult | None = None
        try:
            if normal:
                self._snapshot_freeze.set()
                self.wait_quiescent(timeout=timeout, resume_peers=False)
                if self.peers is not None:
                    self.peers.close()
                if any(
                    int(row.cognitive_state) == int(CognitiveState.RETIRED)
                    for row in self.read_view.node_records()
                ):
                    old_processes = tuple(self._shard_processes)
                    self._shard_processes = []
                    for process in old_processes:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=2.0)
                    self._last_compaction = compact_retired_arenas(
                        self.shard_descriptors,
                        archive_path=self.root / "archive" / "retired_memory.jsonl",
                    )
                    self._shard_processes = [
                        self._build_shard_process(shard_id) for shard_id in range(self.config.shards)
                    ]
                    for process in self._shard_processes:
                        process.start()
                    with self._generation.get_lock():
                        self._generation.value += 1
                self.write_scientific_report()
                if self.snapshot_service is not None:
                    final_result = self.final_snapshot(timeout=timeout)
                    if int(final_result.watermark) != int(self.watermark):
                        raise RuntimeError("final persisted watermark does not match final RAM watermark")
                    (self.root / "RUN_COMPLETE.json").write_text(
                        json.dumps(
                            {
                                "final_watermark": self.watermark,
                                "final_generation": self.generation,
                                "snapshot_id": final_result.snapshot_id,
                                "snapshot_digest": final_result.digest,
                                "snapshot_path": final_result.path,
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
            elif self.peers is not None:
                self.peers.close()
            return final_result
        finally:
            self._cleanup()

    def __enter__(self) -> "ContinuousMemoryRuntime":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(normal=exc is None)
