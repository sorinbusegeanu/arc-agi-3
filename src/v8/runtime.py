from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from v8.arena import SharedActionArena, SharedEdgeArena, SharedNodeArena
from v8.development import STAGES, stage_worker
from v8.model import (
    EventId,
    ExperienceEvent,
    PIPELINE_PACKET_SIZE,
    PROPOSAL_PACKET_SIZE,
    PipelineEvent,
    encode_pipeline,
)
from v8.publication import LiveReadView, ShardReadDescriptor
from v8.ring import SharedRingBuffer
from v8.shard import ShardConfig, shard_worker
from v8.snapshot import SnapshotResult, SnapshotService, restore_latest_snapshot


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

    @classmethod
    def from_path(cls, root: str | Path, **kwargs) -> "V8RuntimeConfig":
        return cls(Path(root), **kwargs)

    def __post_init__(self) -> None:
        if self.shards <= 0 or self.stage_workers <= 0:
            raise ValueError("shards and stage_workers must be positive")
        if self.stage_ring_capacity <= 0 or self.shard_ring_capacity <= 0:
            raise ValueError("ring capacities must be positive")
        if self.snapshot_interval_seconds <= 0:
            raise ValueError("snapshot_interval_seconds must be positive")


class ContinuousMemoryRuntime:
    """RAM-authoritative continuous developmental memory runtime."""

    def __init__(self, config: V8RuntimeConfig) -> None:
        self.config = config
        self.root = config.root
        self.root.mkdir(parents=True, exist_ok=True)
        self._stop = mp.Event()
        self._accepting = True
        self._started = False
        self._closed = False
        self._error_queue: mp.Queue = mp.Queue()
        self._watermark = mp.Value("Q", 0)
        self._snapshot_id = 0

        self._stage_rings = tuple(
            SharedRingBuffer(capacity=config.stage_ring_capacity, slot_size=PIPELINE_PACKET_SIZE)
            for _ in STAGES
        )
        self._shard_rings = tuple(
            SharedRingBuffer(capacity=config.shard_ring_capacity, slot_size=PROPOSAL_PACKET_SIZE)
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
        if restored is not None:
            self._snapshot_id, restored_watermark = restored
            with self._watermark.get_lock():
                self._watermark.value = int(restored_watermark)

        self.read_view = LiveReadView(self.shard_descriptors)
        self._stage_inflight = tuple(mp.Value("Q", 0) for _ in STAGES)
        self._shard_inflight = tuple(mp.Value("Q", 0) for _ in range(config.shards))
        self._shard_watermarks = tuple(mp.Value("Q", 0) for _ in range(config.shards))
        self._stage_processes: list[mp.Process] = []
        self._shard_processes: list[mp.Process] = []

        for shard_id in range(config.shards):
            self._shard_processes.append(
                mp.Process(
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
                        config.shard_batch_size,
                    ),
                    name=f"v8-shard-{shard_id:02d}",
                    daemon=True,
                )
            )

        shard_ring_args = tuple(ring.attachment_args() for ring in self._shard_rings)
        for stage_index, definition in enumerate(STAGES):
            next_args = (
                None
                if stage_index + 1 >= len(STAGES)
                else self._stage_rings[stage_index + 1].attachment_args()
            )
            for worker_id in range(config.stage_workers):
                self._stage_processes.append(
                    mp.Process(
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

    @property
    def watermark(self) -> int:
        with self._watermark.get_lock():
            return int(self._watermark.value)

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
        self._started = True

    def _snapshot_cadence(self) -> None:
        interval = float(self.config.snapshot_interval_seconds)
        while not self._snapshot_thread_stop.wait(interval):
            if self._closed or self.snapshot_service is None:
                return
            self._snapshot_id += 1
            self.snapshot_service.request_latest(self._snapshot_id, self.watermark)

    def _next_watermark(self) -> int:
        with self._watermark.get_lock():
            self._watermark.value += 1
            return int(self._watermark.value)

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
    ) -> ExperienceEvent:
        return ExperienceEvent(
            event_id=EventId.from_producer(producer_id, producer_sequence),
            watermark=self._next_watermark(),
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
        )

    def submit(self, event: ExperienceEvent, *, timeout: float = 1.0) -> None:
        if not self._started:
            self.start()
        if not self._accepting:
            raise RuntimeError("v8 runtime is draining and no longer accepts experiences")
        self.raise_worker_errors()
        with self._watermark.get_lock():
            self._watermark.value = max(int(self._watermark.value), int(event.watermark))
        if not self._stage_rings[0].put(encode_pipeline(PipelineEvent(event)), timeout=timeout):
            raise TimeoutError("M0 experience ring remained full")

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

    def _is_quiescent(self) -> bool:
        if any(not ring.empty for ring in (*self._stage_rings, *self._shard_rings)):
            return False
        if any(int(value.value) != 0 for value in (*self._stage_inflight, *self._shard_inflight)):
            return False
        return True

    def wait_quiescent(self, *, timeout: float = 60.0, stable_checks: int = 5) -> None:
        deadline = time.monotonic() + float(timeout)
        stable = 0
        while time.monotonic() < deadline:
            self.raise_worker_errors()
            if self._is_quiescent():
                stable += 1
                if stable >= int(stable_checks):
                    return
            else:
                stable = 0
            time.sleep(0.01)
        raise TimeoutError("v8 did not reach quiescence; " + json.dumps(self.metrics(), sort_keys=True))

    def request_snapshot(self) -> None:
        if self.snapshot_service is None:
            return
        self._snapshot_id += 1
        self.snapshot_service.request_latest(self._snapshot_id, self.watermark)

    def final_snapshot(self, *, timeout: float = 120.0) -> SnapshotResult:
        if self.snapshot_service is None:
            raise RuntimeError("snapshots are disabled")
        self._snapshot_id += 1
        return self.snapshot_service.request_final(self._snapshot_id, self.watermark, timeout=timeout)

    def metrics(self) -> dict[str, object]:
        saved_watermark = (
            0 if self.snapshot_service is None else int(self.snapshot_service.saved_watermark.value)
        )
        return {
            "watermark": self.watermark,
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
        }

    def close(self, *, normal: bool = True, timeout: float = 120.0) -> SnapshotResult | None:
        if self._closed:
            return None
        if not self._started:
            self.start()
        self._accepting = False
        self._snapshot_thread_stop.set()
        if self._snapshot_thread is not None:
            self._snapshot_thread.join(timeout=2.0)

        final_result = None
        if normal:
            self.wait_quiescent(timeout=timeout)
            if self.snapshot_service is not None:
                final_result = self.final_snapshot(timeout=timeout)
                if int(final_result.watermark) != int(self.watermark):
                    raise RuntimeError("final persisted watermark does not match final RAM watermark")
                (self.root / "RUN_COMPLETE.json").write_text(
                    json.dumps(
                        {
                            "final_watermark": self.watermark,
                            "snapshot_id": final_result.snapshot_id,
                            "snapshot_digest": final_result.digest,
                            "snapshot_path": final_result.path,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

        self._stop.set()
        for process in (*self._stage_processes, *self._shard_processes):
            process.join(timeout=5.0)
        for process in (*self._stage_processes, *self._shard_processes):
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        if self.snapshot_service is not None:
            self.snapshot_service.close()
        self.read_view.close()
        for ring in (*self._stage_rings, *self._shard_rings):
            ring.dispose()
        for arena in (*self._node_arenas, *self._edge_arenas, *self._action_arenas):
            arena.dispose()
        self._closed = True
        return final_result

    def __enter__(self) -> "ContinuousMemoryRuntime":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(normal=exc is None)
